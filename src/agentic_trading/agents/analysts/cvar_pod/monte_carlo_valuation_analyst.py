# valuation_monte_carlo.py

from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import PrivateAttr
from quantcore.core.sde_solver import simulate_gbm_paths

from agentic_trading.agents._market_math import (
    RefitCache,
    clamp,
    conditional_daily_volatility,
    derived_seed,
    market_slice,
)
from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.config import settings
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

TRADING_DAYS_PER_YEAR = 252


class MonteCarloValuationAnalyst(AnalystAgent):
    """Real-world-drift valuation: projects the asset forward via
    quantcore's GBM Monte Carlo path simulator using the asset's own
    historical mean log-return as drift (not the risk-free rate) and a
    GARCH-fitted volatility, discounts the expected terminal price back
    to today as a 'fair value', and compares it against the current
    market price — proposes long when fair value exceeds market price,
    short when market price exceeds fair value. The GARCH fit is only
    re-run at most once every `_refit_cache.refit_interval_days` simulated
    days per asset — see `RefitCache`'s docstring."""

    agent_id: str = "monte_carlo_valuation_analyst"

    _num_paths: int = 2_000
    _num_steps: int = 32
    _time_to_maturity: float = 0.25
    _refit_cache: RefitCache = PrivateAttr(default_factory=RefitCache)

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
                rationale="insufficient history for quantcore Monte Carlo valuation",
                status="unavailable",
            )

        mean_return = float(returns.mean())
        volatility = conditional_daily_volatility(
            returns, self._refit_cache, asset, as_of_date, self.agent_id
        )

        # simulate_gbm_paths takes annualized rate/volatility over
        # time_to_maturity in years (see quantcore's sde_solver docstring);
        # mean_return/volatility above are daily-scale statistics of daily
        # log returns and must be annualized before crossing that boundary.
        terminal_prices = simulate_gbm_paths(
            slice_.close,
            mean_return * TRADING_DAYS_PER_YEAR,
            volatility * TRADING_DAYS_PER_YEAR**0.5,
            self._time_to_maturity,
            self._num_steps,
            self._num_paths,
            seed=derived_seed(self.agent_id, asset, as_of_date.date()),
        )[:, -1]
        fair_value = float(terminal_prices.mean())
        relative_edge = (fair_value - slice_.close) / slice_.close
        direction: Literal["long", "short", "flat"] = "long" if relative_edge >= 0.0 else "short"
        confidence = clamp(abs(relative_edge))
        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction=direction,
            confidence=confidence,
            rationale=(
                f"quantcore Monte Carlo fair value {fair_value:.4f} vs spot {slice_.close:.4f}"
            ),
        )
