# greeks_delta_hedge_analyst.py

from datetime import date
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import PrivateAttr

from agentic_trading.agents._market_math import clamp, market_slice, raw_spot
from agentic_trading.agents.analysts.cvar_pod._options_data import (
    real_atm_options_by_expiration,
)
from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.schemas import AgentSignal, SignalStatus
from agentic_trading.config import settings
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

TRADING_DAYS_PER_YEAR = 252
# Front-month band: inside a week gamma explodes mechanically, beyond
# ~six weeks the expiry is not where hedging flow concentrates.
MIN_DTE = 7
MAX_DTE = 45
MOMENTUM_DAYS = 5
MOMENTUM_VOL_WINDOW = 60
# A 5-day move smaller than this many sigmas is noise, not a move to fade
# or follow.
MOMENTUM_NOISE_FRACTION = 0.25
MIN_RETURN_OBSERVATIONS = 10
# |z| this far beyond the threshold earns full confidence.
CONFIDENCE_ZSCORE_SPAN = 2.0
MAX_CONFIDENCE = 0.75


class GreeksDeltaHedgeAnalyst(AnalystAgent):
    """Gamma-regime read from the real ATM gamma of the nearest front-month
    expiry (`MIN_DTE`-`MAX_DTE` days) in the lakehouse's options-chain
    snapshot (`_options_data.real_atm_options_by_expiration`).

    The intensity g = S * gamma * sqrt(DTE / 252) is a DTE-normalised ATM
    dollar gamma, with S the unadjusted close (`_market_math.raw_spot`) —
    the basis the chain's strikes and gamma are quoted on. Without open
    interest, ATM gamma is a deterministic function of spot, implied vol
    and time to expiry, so g says nothing about dealer positioning by
    itself; the only information is how it
    compares to its own history. Each distinct snapshot's log g is kept
    in-process per asset and z-scored against the trailing
    `settings.analysts.gamma_history_window` points once at least
    `settings.analysts.gamma_min_history` exist.

    A high z (gamma-heavy, pinning regime) fades the trailing 5-day move; a
    low z (gamma-light regime) follows it; a 5-day move smaller than a
    fraction of its own 60-day sigma is neither. Confidence scales with
    |z| beyond `settings.analysts.gamma_zscore_threshold`, capped at
    `MAX_CONFIDENCE`."""

    agent_id: str = "greeks_delta_hedge_analyst"

    _log_gamma_history: dict[str, dict[date, float]] = PrivateAttr(default_factory=dict)

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

    def _record_log_gamma(self, asset: str, as_of_date: pd.Timestamp, log_gamma: float) -> None:
        history = self._log_gamma_history.setdefault(asset, {})
        if history and abs(next(reversed(history.values())) - log_gamma) < 1e-12:
            return
        history[as_of_date.date()] = log_gamma

    def generate_signal(
        self,
        asset: str,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> AgentSignal:
        returns = market_slice(market_data, as_of_date, asset).log_returns[-MOMENTUM_VOL_WINDOW:]
        spot = raw_spot(market_data, as_of_date, asset)
        if spot is None or returns.size < MIN_RETURN_OBSERVATIONS:
            return self._flat(
                asset, as_of_date, "insufficient history for gamma-regime read", "unavailable"
            )

        real_options = real_atm_options_by_expiration(asset, as_of_date, spot)
        if real_options is None or real_options.empty:
            return self._flat(
                asset,
                as_of_date,
                "no real option-chain snapshot in the lakehouse for this date",
                "unavailable",
            )
        front = real_options[
            (real_options["days_to_expiration"] >= MIN_DTE)
            & (real_options["days_to_expiration"] <= MAX_DTE)
        ]
        if front.empty:
            return self._flat(
                asset, as_of_date, f"no {MIN_DTE}-{MAX_DTE} DTE expiry", "unavailable"
            )
        near = front.iloc[0]
        dte = int(near["days_to_expiration"])
        gamma_intensity = spot * float(near["gamma"]) * float(np.sqrt(dte / TRADING_DAYS_PER_YEAR))
        if not np.isfinite(gamma_intensity) or gamma_intensity <= 0.0:
            return self._flat(
                asset, as_of_date, "non-positive ATM gamma in the snapshot", "unavailable"
            )
        log_gamma = float(np.log(gamma_intensity))
        self._record_log_gamma(asset, as_of_date, log_gamma)

        history = list(self._log_gamma_history[asset].values())
        if len(history) < settings.analysts.gamma_min_history:
            return self._flat(
                asset,
                as_of_date,
                f"gamma history {len(history)} < {settings.analysts.gamma_min_history} snapshots",
                "unavailable",
            )
        window = np.asarray(history[-settings.analysts.gamma_history_window :], dtype=np.float64)
        window_std = float(np.std(window, ddof=1))
        if window_std < 1e-12:
            return self._flat(asset, as_of_date, "gamma history has no dispersion", "unavailable")
        zscore = (log_gamma - float(np.mean(window))) / window_std

        momentum = float(np.sum(returns[-MOMENTUM_DAYS:]))
        noise_floor = (
            MOMENTUM_NOISE_FRACTION * float(np.std(returns, ddof=1)) * float(np.sqrt(MOMENTUM_DAYS))
        )
        if abs(momentum) < noise_floor:
            return self._flat(
                asset,
                as_of_date,
                f"no move to fade/follow ({MOMENTUM_DAYS}-day momentum {momentum:+.4f} "
                f"within noise {noise_floor:.4f}; g={gamma_intensity:.4f}, z={zscore:+.2f})",
            )
        momentum_direction: Literal["long", "short"] = "long" if momentum > 0.0 else "short"
        against_momentum: Literal["long", "short"] = (
            "short" if momentum_direction == "long" else "long"
        )

        threshold = settings.analysts.gamma_zscore_threshold
        direction: Literal["long", "short", "flat"]
        if zscore > threshold:
            direction = against_momentum
            regime = "pinning regime: fading"
        elif zscore < -threshold:
            direction = momentum_direction
            regime = "gamma-light regime: following"
        else:
            direction = "flat"
            regime = "no gamma regime"
        confidence = (
            clamp((abs(zscore) - threshold) / CONFIDENCE_ZSCORE_SPAN, 0.0, MAX_CONFIDENCE)
            if direction != "flat"
            else 0.0
        )

        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction=direction,
            confidence=confidence,
            rationale=(
                f"{regime} {MOMENTUM_DAYS}-day momentum {momentum:+.4f}; "
                f"ATM gamma intensity g={gamma_intensity:.4f} at {dte} DTE, z={zscore:+.2f}"
            ),
        )
