# newey_west_significance_analyst.py

from typing import Literal

import numpy as np
import pandas as pd
from quantcore.statistics.regression import newey_west_cov, newey_west_optimal_lags, ols

from agentic_trading.agents._market_math import clamp, market_slice
from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.config import settings
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_MOMENTUM_WINDOW = 5


class NeweyWestSignificanceAnalyst(AnalystAgent):
    """General "does X actually predict this asset" analyst: regresses the
    asset's forward 1-day return on a configurable lagged predictor
    (default: the asset's own trailing momentum) via OLS + a Newey-West
    HAC covariance, only signaling when the predictor's HAC t-stat clears
    `t_stat_threshold`. The HAC bandwidth (`n_lags`) is selected fresh
    each call from the fit's own residuals via
    `quantcore.statistics.regression.newey_west_optimal_lags` (the
    Newey-West/Andrews plug-in rule) unless explicitly pinned."""

    agent_id: str = "newey_west_significance_analyst"

    momentum_window: int = DEFAULT_MOMENTUM_WINDOW
    t_stat_threshold: float = settings.analysts.t_stat_threshold
    # None -> selected fresh each call via newey_west_optimal_lags; pin to
    # a fixed int to skip selection entirely.
    n_lags: int | None = None

    def generate_signal(
        self,
        asset: str,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> AgentSignal:
        returns = market_slice(market_data, as_of_date, asset).log_returns[
            -settings.analysts.risk_window_medium :
        ]
        if returns.size < self.momentum_window + 30:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="insufficient history for a Newey-West significance test",
                status="unavailable",
            )

        # The full convolution's last element is today's own trailing momentum
        # window (through returns[-1]); it has no forward return yet, so it's
        # excluded from the regression fit below, but it's what the live
        # decision must condition on — keep it separately rather than reading
        # it back off the already-trimmed `predictor` array.
        full_predictor = np.convolve(returns, np.ones(self.momentum_window), mode="valid")
        predictor = full_predictor[:-1]
        forward_returns = returns[self.momentum_window :]
        length = min(predictor.size, forward_returns.size)
        predictor = predictor[-length:]
        forward_returns = forward_returns[-length:]

        design_matrix = np.column_stack([np.ones(length), predictor])
        coefs, _std_errors, _r_squared = ols(forward_returns, design_matrix)
        residuals = forward_returns - design_matrix @ coefs
        n_lags = self.n_lags if self.n_lags is not None else newey_west_optimal_lags(residuals)
        cov_matrix = newey_west_cov(design_matrix, residuals, n_lags)

        slope = float(coefs[1])
        slope_std_error = float(np.sqrt(max(cov_matrix[1, 1], 0.0)))
        t_stat = slope / slope_std_error if slope_std_error > 0 else 0.0

        if abs(t_stat) < self.t_stat_threshold:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale=f"predictor not significant (HAC t={t_stat:.2f})",
            )

        current_predictor_value = float(full_predictor[-1])
        predicted_move = slope * current_predictor_value
        direction: Literal["long", "short", "flat"] = "long" if predicted_move >= 0.0 else "short"
        confidence = clamp(abs(t_stat) / (2 * self.t_stat_threshold))

        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction=direction,
            confidence=confidence,
            rationale=f"predictor HAC t-stat={t_stat:.2f}, slope={slope:.4f}",
        )
