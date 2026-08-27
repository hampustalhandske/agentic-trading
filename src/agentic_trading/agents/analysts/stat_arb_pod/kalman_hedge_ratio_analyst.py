# kalman_hedge_ratio_analyst.py

from typing import Literal

import numpy as np
import pandas as pd
from quantcore.filtering.kalman import dynamic_hedge_ratio
from quantcore.statistics.cointegration import spread_zscore

from agentic_trading.agents._market_math import clamp, market_slice
from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

Z_ENTRY_THRESHOLD = 1.0
Z_WINDOW = 60


class KalmanHedgeRatioAnalyst(AnalystAgent):
    """Same partner-ticker spread idea as PairsCointegrationAnalyst, but
    the hedge ratio is time-varying via a Kalman filter rather than a
    static OLS beta — the spread is built from the *latest filtered*
    hedge ratio, which is this analyst's differentiator."""

    agent_id: str = "kalman_hedge_ratio_analyst"

    partner_ticker: str | None = None
    _risk_window: int = 250

    def generate_signal(
        self,
        asset: str,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> AgentSignal:
        if self.partner_ticker is None:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="no partner ticker configured",
            )

        y_returns = market_slice(market_data, as_of_date, asset).log_returns[-self._risk_window :]
        x_returns = market_slice(market_data, as_of_date, self.partner_ticker).log_returns[
            -self._risk_window :
        ]
        length = min(y_returns.size, x_returns.size)
        if length < 30:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="insufficient overlapping history for a Kalman hedge ratio",
            )

        y_prices = np.cumsum(y_returns[-length:])
        x_prices = np.cumsum(x_returns[-length:])
        hedge_ratios = dynamic_hedge_ratio(y_prices, x_prices)
        latest_beta = float(hedge_ratios[-1])
        spread = y_prices - latest_beta * x_prices

        z_window = min(Z_WINDOW, spread.size)
        current_z = float(spread_zscore(spread, z_window)[-1])

        direction: Literal["long", "short", "flat"]
        if current_z < -Z_ENTRY_THRESHOLD:
            direction = "long"
        elif current_z > Z_ENTRY_THRESHOLD:
            direction = "short"
        else:
            direction = "flat"
        confidence = clamp((abs(current_z) - Z_ENTRY_THRESHOLD) / 2.0)

        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction=direction,
            confidence=confidence,
            rationale=(
                f"Kalman hedge ratio vs {self.partner_ticker}={latest_beta:.3f}, "
                f"spread z={current_z:.2f}"
            ),
        )
