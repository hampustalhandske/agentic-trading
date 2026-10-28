# heston_vol_surface_analyst.py

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import Field

from agentic_trading.agents._market_math import clamp, market_slice, raw_spot
from agentic_trading.agents.analysts.cvar_pod._options_data import (
    annualized_garch_volatility,
    options_chain_requirements,
    real_atm_options_by_expiration,
)
from agentic_trading.agents.base import AnalystAgent, DataRequirement
from agentic_trading.agents.schemas import AgentSignal, SignalStatus
from agentic_trading.config import settings
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

DAYS_PER_YEAR = 365.0
TARGET_DTE = 30
# At least two expiries for a slope; an expiry inside five days is dominated
# by expiry-day distortions and is ignored as the nearest point.
MIN_EXPIRIES = 2
MIN_NEAREST_DTE = 5
# Two expiries fewer than 20 days apart carry no usable term information.
MIN_SLOPE_SPAN_DAYS = 20
# Backwardation of 3 vol points between nearest and longest expiry is a
# stress signature regardless of the premium.
BACKWARDATION_SLOPE = -0.03
# A 6-vol-point term slope adds the full 50% confidence uplift.
SLOPE_FULL_UPLIFT = 0.06
SLOPE_UPLIFT_WEIGHT = 0.5
# Beyond 30 vol points the "premium" is a data error, not a signal.
MAX_ABS_VRP = 0.30
MAX_CONFIDENCE = 0.75
# GARCH needs a reasonable sample; its failure falls back to trailing
# realized vol over this many days.
MIN_RETURN_OBSERVATIONS = 30
REALIZED_FALLBACK_WINDOW = 60


def _interpolated_atm_iv(days: np.ndarray, ivs: np.ndarray, target_days: int) -> float:
    """ATM IV at `target_days`, interpolated linearly in total variance
    between the two bracketing expiries; the nearest expiry's IV when no
    pair brackets the target."""
    lower = np.where(days <= target_days)[0]
    upper = np.where(days >= target_days)[0]
    if lower.size == 0 or upper.size == 0:
        nearest = int(np.argmin(np.abs(days - target_days)))
        return float(ivs[nearest])
    lo, hi = int(lower[-1]), int(upper[0])
    if lo == hi:
        return float(ivs[lo])
    t_lo, t_hi, t_target = (
        days[lo] / DAYS_PER_YEAR,
        days[hi] / DAYS_PER_YEAR,
        (target_days / DAYS_PER_YEAR),
    )
    var_lo, var_hi = float(ivs[lo]) ** 2 * t_lo, float(ivs[hi]) ** 2 * t_hi
    var_target = var_lo + (var_hi - var_lo) * (t_target - t_lo) / (t_hi - t_lo)
    return float(np.sqrt(max(var_target, 0.0) / t_target))


class HestonVolSurfaceAnalyst(AnalystAgent):
    """Model-free variance-risk-premium read on the real ATM implied-vol
    term structure. The lakehouse chain carries a handful of expiries out
    to about two months, which cannot identify a stochastic-volatility
    term-structure model, so the signal uses the observed surface
    directly: `vrp = IV_30 - GARCH vol` (30-day ATM IV interpolated in
    total variance, against this analyst's own annualized GARCH(1,1) read
    on a year of returns) and `slope = IV_longest - IV_nearest`.

    A rich premium (`vrp >= vrp_threshold`) in contango is long
    (Bollerslev-Tauchen-Zhou: a positive variance risk premium predicts
    positive returns); implied vol below the model forecast, or a
    backwardated curve, is short. Confidence grows from zero at the
    threshold to full `vrp_full_confidence_span` beyond it, with up to a
    50% uplift for a steep term slope, capped at 0.75. Flat with zero
    confidence when fewer than two usable expiries exist for the date.
    The agent id is kept so scoreboard history stays continuous."""

    agent_id: str = "heston_vol_surface_analyst"
    data_requirements: tuple[DataRequirement, ...] = Field(
        default_factory=options_chain_requirements
    )

    def _flat(
        self,
        asset: str,
        as_of_date: pd.Timestamp,
        rationale: str,
        status: SignalStatus = "active",
    ) -> AgentSignal:
        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction="flat",
            confidence=0.0,
            rationale=rationale,
            status=status,
        )

    def generate_signal(
        self,
        asset: str,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> AgentSignal:
        returns = market_slice(market_data, as_of_date, asset).log_returns[
            -settings.analysts.risk_window_long :
        ]
        spot = raw_spot(market_data, as_of_date, asset)
        if spot is None or returns.size < MIN_RETURN_OBSERVATIONS:
            return self._flat(
                asset, as_of_date, "insufficient history for a volatility read", "unavailable"
            )

        real_options = real_atm_options_by_expiration(asset, as_of_date, spot)
        if real_options is None or real_options.empty:
            return self._flat(
                asset,
                as_of_date,
                "no real option-chain snapshot in the lakehouse for this date",
                "unavailable",
            )
        usable = real_options[real_options["days_to_expiration"] >= MIN_NEAREST_DTE]
        if len(usable) < MIN_EXPIRIES:
            return self._flat(
                asset,
                as_of_date,
                f"fewer than {MIN_EXPIRIES} usable expiries in the chain",
                "unavailable",
            )
        days = usable["days_to_expiration"].to_numpy(dtype=np.float64)
        ivs = usable["implied_volatility"].to_numpy(dtype=np.float64)

        iv_30 = _interpolated_atm_iv(days, ivs, TARGET_DTE)
        try:
            model_vol = annualized_garch_volatility(returns, self.agent_id, asset, as_of_date)
        except (RuntimeError, ValueError):
            model_vol = float(np.std(returns[-REALIZED_FALLBACK_WINDOW:], ddof=1)) * np.sqrt(
                settings.firm.trading_days_per_year
            )
        vrp = iv_30 - model_vol
        if abs(vrp) > MAX_ABS_VRP:
            return self._flat(
                asset,
                as_of_date,
                f"variance risk premium {vrp:.3f} outside plausible range",
                "unavailable",
            )
        slope = float(ivs[-1] - ivs[0]) if days[-1] - days[0] >= MIN_SLOPE_SPAN_DAYS else 0.0

        threshold = settings.analysts.vrp_threshold
        direction: Literal["long", "short", "flat"] = "flat"
        if vrp >= threshold and slope >= 0.0:
            direction = "long"
        elif vrp <= -threshold or slope <= BACKWARDATION_SLOPE:
            direction = "short"

        confidence = 0.0
        if direction != "flat":
            premium_strength = clamp(
                (abs(vrp) - threshold) / settings.analysts.vrp_full_confidence_span
            )
            slope_uplift = 1.0 + SLOPE_UPLIFT_WEIGHT * clamp(abs(slope) / SLOPE_FULL_UPLIFT)
            confidence = clamp(premium_strength * slope_uplift, 0.0, MAX_CONFIDENCE)

        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction=direction,
            confidence=confidence,
            rationale=(
                f"variance risk premium {vrp:+.4f} (IV_30={iv_30:.4f} vs model vol "
                f"{model_vol:.4f}), term slope {slope:+.4f} over {days[0]:.0f}-{days[-1]:.0f} DTE"
            ),
        )
