# gbm_momentum_analyst.py

from typing import Literal

import numpy as np
import pandas as pd
from quantcore.core.sde_solver import simulate_gbm_paths

from agentic_trading.agents._market_math import clamp, derived_seed, market_slice
from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.config import settings
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

TRADING_DAYS_PER_YEAR = 252


class GBMMomentumAnalyst(AnalystAgent):
    """Simulates the asset forward under GBM using its own trailing drift
    and volatility, and reads direction off the simulated probability that
    the terminal price ends up above today's price."""

    agent_id: str = "gbm_momentum_analyst"

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
        returns = slice_.log_returns[-settings.analysts.risk_window_short :]
        if slice_.close is None or returns.size < 2:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="insufficient history for a GBM momentum simulation",
                status="unavailable",
            )

        mu = float(np.mean(returns))
        sigma = float(np.std(returns, ddof=1))
        # simulate_gbm_paths takes annualized drift/volatility over
        # time_to_maturity in years — mu/sigma above are daily-scale
        # statistics of daily log returns and must be annualized before
        # crossing that boundary (see quantcore's sde_solver docstring).
        terminal_prices = simulate_gbm_paths(
            slice_.close,
            mu * TRADING_DAYS_PER_YEAR,
            sigma * TRADING_DAYS_PER_YEAR**0.5,
            self._time_to_maturity,
            self._num_steps,
            self._num_paths,
            seed=derived_seed(self.agent_id, asset, as_of_date.date()),
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
