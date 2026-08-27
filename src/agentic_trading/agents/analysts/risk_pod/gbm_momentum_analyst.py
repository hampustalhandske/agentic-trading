# gbm_momentum_analyst.py

from typing import Literal

import numpy as np
import pandas as pd
from quantcore.core.sde_solver import simulate_gbm_paths

from agentic_trading.agents._market_math import clamp, market_slice
from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)


class GBMMomentumAnalyst(AnalystAgent):
    """Simulates the asset forward under GBM using its own trailing drift
    and volatility, and reads direction off the simulated probability that
    the terminal price ends up above today's price."""

    agent_id: str = "gbm_momentum_analyst"

    _risk_window: int = 60
    _num_steps: int = 5
    _num_paths: int = 2_000
    _time_to_maturity: float = 5 / 252

    def generate_signal(
        self,
        asset: str,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> AgentSignal:
        slice_ = market_slice(market_data, as_of_date, asset)
        returns = slice_.log_returns[-self._risk_window :]
        if slice_.close is None or returns.size < 2:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="insufficient history for a GBM momentum simulation",
            )

        mu = float(np.mean(returns))
        sigma = float(np.std(returns, ddof=1))
        terminal_prices = simulate_gbm_paths(
            slice_.close,
            mu,
            sigma,
            self._time_to_maturity,
            self._num_steps,
            self._num_paths,
            seed=42,
        )[:, -1]
        probability_up = float(np.mean(terminal_prices > slice_.close))

        direction: Literal["long", "short", "flat"]
        if probability_up > 0.55:
            direction = "long"
        elif probability_up < 0.45:
            direction = "short"
        else:
            direction = "flat"

        confidence = clamp(abs(probability_up - 0.5) * 2)
        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction=direction,
            confidence=confidence,
            rationale=(
                f"GBM simulation P(terminal > spot)={probability_up:.3f} "
                f"(mu={mu:.4f}, sigma={sigma:.4f})"
            ),
        )
