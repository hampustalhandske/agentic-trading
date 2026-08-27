# newey_west_significance_analyst.py

from typing import Literal

import numpy as np
import pandas as pd
from quantcore.statistics.regression import newey_west_cov, ols

from agentic_trading.agents._market_math import clamp, market_slice
from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_MOMENTUM_WINDOW = 5
DEFAULT_T_STAT_THRESHOLD = 2.0
DEFAULT_NW_LAGS = 4


class NeweyWestSignificanceAnalyst(AnalystAgent):
    """General "does X actually predict this asset" analyst: regresses the
    asset's forward 1-day return on a configurable lagged predictor
    (default: the asset's own trailing momentum) via OLS + a Newey-West
    HAC covariance, only signaling when the predictor's HAC t-stat clears
    `t_stat_threshold`."""

    agent_id: str = "newey_west_significance_analyst"

    momentum_window: int = DEFAULT_MOMENTUM_WINDOW
    t_stat_threshold: float = DEFAULT_T_STAT_THRESHOLD
    n_lags: int = DEFAULT_NW_LAGS
    _risk_window: int = 120

    def generate_signal(
        self,
        asset: str,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> AgentSignal:
        returns = market_slice(market_data, as_of_date, asset).log_returns[-self._risk_window :]
        if returns.size < self.momentum_window + 30:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="insufficient history for a Newey-West significance test",
            )

        predictor = np.convolve(returns, np.ones(self.momentum_window), mode="valid")[:-1]
        forward_returns = returns[self.momentum_window :]
        length = min(predictor.size, forward_returns.size)
        predictor = predictor[-length:]
        forward_returns = forward_returns[-length:]

        design_matrix = np.column_stack([np.ones(length), predictor])
        coefs, _std_errors, _r_squared = ols(forward_returns, design_matrix)
        residuals = forward_returns - design_matrix @ coefs
        cov_matrix = newey_west_cov(design_matrix, residuals, self.n_lags)

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

        current_predictor_value = float(predictor[-1])
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
