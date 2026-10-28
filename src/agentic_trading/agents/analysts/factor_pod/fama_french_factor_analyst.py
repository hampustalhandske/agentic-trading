# fama_french_factor_analyst.py

import math
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import Field
from quantcore.statistics.regression import factor_loadings

from agentic_trading.agents._market_math import clamp, log_return_series
from agentic_trading.agents.base import AnalystAgent, DataRequirement, calendar_days_for_sessions
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.config import settings
from agentic_trading.data.loaders.lakehouse_loader import load_fama_french_factors_from_lakehouse
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

TRADING_DAYS_PER_MONTH = 21
# A trailing year gives 3-5 loadings enough degrees of freedom for
# standard errors around 0.1; the last month is dropped from the momentum
# leg (12-1) to avoid short-term reversal.
MIN_REGRESSION_OBSERVATIONS = 120
MOMENTUM_LOOKBACK_DAYS = 252
MOMENTUM_SKIP_DAYS = 21
# Confidence cap so one factor view never dominates a pod on its own.
MAX_CONFIDENCE = 0.8


def _expected_factor_premia(factor_returns: pd.DataFrame) -> np.ndarray:
    """Daily expected premium per factor: half 12-1-month factor momentum
    (mean over days [-252, -21)), half the unconditional full-history
    mean — momentum shrunk toward the long-run premium."""
    history = factor_returns.to_numpy(dtype=np.float64)
    unconditional = history.mean(axis=0)
    momentum_window = history[-MOMENTUM_LOOKBACK_DAYS:-MOMENTUM_SKIP_DAYS]
    momentum = momentum_window.mean(axis=0) if len(momentum_window) else unconditional
    return np.asarray(0.5 * momentum + 0.5 * unconditional)


class FamaFrenchFactorAnalyst(AnalystAgent):
    """Expected return from factor exposures: regresses the asset's
    trailing-year excess returns on the Fama-French factors, joined by
    date (only days with both a return and a factor row enter; that
    joined row count is the signal's `sample_count`), keeps only
    loadings with a Newey-West |t| at or above
    `settings.analysts.t_stat_threshold`, and multiplies them by each
    factor's expected premium — half its 12-1-month momentum (Ehsani &
    Linnainmaa), half its full-history mean. The monthly expected return
    over the asset's own monthly vol is the signal: its sign is the
    direction once above `factor_zscore_flat`, and confidence scales to
    `factor_zscore_full` (an information-ratio-style ratio, so it does
    not saturate the way a t-stat does)."""

    agent_id: str = "fama_french_factor_analyst"

    model: str = "3factor"
    # The regression's trailing `risk_window_long` days and the 12-1
    # momentum leg's `MOMENTUM_LOOKBACK_DAYS`; the unconditional premium
    # averages whatever older history exists.
    data_requirements: tuple[DataRequirement, ...] = Field(
        default_factory=lambda: (
            DataRequirement(
                source="ken_french_factors",
                identifier="3factor",
                asset_scoped=False,
                description="Ken French 3-factor returns",
                lookback_days=calendar_days_for_sessions(
                    max(settings.analysts.risk_window_long, MOMENTUM_LOOKBACK_DAYS)
                ),
            ),
        )
    )

    def generate_signal(
        self,
        asset: str,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> AgentSignal:
        returns = log_return_series(market_data, as_of_date, asset).iloc[
            -settings.analysts.risk_window_long :
        ]
        factors = load_fama_french_factors_from_lakehouse(as_of_date.date(), model=self.model)
        if factors is None:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="no Ken French factor coverage in the lakehouse",
                status="unavailable",
            )

        factor_columns = [column for column in factors.columns if column != "RF"]
        joined = pd.concat(
            {"asset": returns.to_frame("return"), "factors": factors}, axis=1, join="inner"
        ).dropna()
        if len(joined) < MIN_REGRESSION_OBSERVATIONS:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="insufficient overlapping history for a factor regression",
                status="unavailable",
                sample_count=len(joined),
            )

        aligned_factors = joined["factors"]
        excess_returns = (joined[("asset", "return")] - aligned_factors["RF"]).to_numpy(
            dtype=np.float64
        )
        factor_matrix = aligned_factors[factor_columns].to_numpy(dtype=np.float64)

        # quantcore returns the intercept (alpha) first, then one loading
        # per factor column — drop the intercept so positions line up with
        # `factor_columns`, then gate each loading on its own Newey-West
        # t-stat (lag count from quantcore's sample-size rule).
        regression = factor_loadings(excess_returns, factor_matrix)
        betas = np.where(
            np.abs(regression.t_stats[1:]) >= settings.analysts.t_stat_threshold,
            regression.loadings[1:],
            0.0,
        )
        if not np.any(betas):
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="no factor loading is statistically significant",
                sample_count=len(joined),
            )

        premia = _expected_factor_premia(factors[factor_columns])
        monthly_alpha = TRADING_DAYS_PER_MONTH * float(np.dot(betas, premia))
        monthly_vol = float(np.std(excess_returns, ddof=1)) * math.sqrt(TRADING_DAYS_PER_MONTH)
        z_score = monthly_alpha / max(monthly_vol, 1e-12)

        direction: Literal["long", "short", "flat"] = "flat"
        confidence = 0.0
        if abs(z_score) >= settings.analysts.factor_zscore_flat:
            direction = "long" if z_score > 0.0 else "short"
            confidence = clamp(
                abs(z_score) / settings.analysts.factor_zscore_full, 0.0, MAX_CONFIDENCE
            )
        loadings_text = ", ".join(
            f"{name}={beta:.2f}" for name, beta in zip(factor_columns, betas, strict=True)
        )
        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction=direction,
            confidence=confidence,
            rationale=(
                f"factor expected return {monthly_alpha:.4f}/month vs vol {monthly_vol:.4f} "
                f"(z={z_score:.3f}; significant loadings: {loadings_text})"
            ),
            sample_count=len(joined),
        )
