# valuation_monte_carlo.py

from __future__ import annotations

from typing import Literal

import pandas as pd
from quantcore.core.sde_solver import simulate_gbm_paths
from quantcore.risk import fit_garch_11, garch_11_variance

from agentic_trading.agents._market_math import clamp, market_slice
from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)


class MonteCarloValuationAnalyst(AnalystAgent):
    """Real-world-drift valuation: projects the asset forward via
    quantcore's GBM Monte Carlo path simulator using the asset's own
    historical mean log-return as drift (not the risk-free rate) and a
    GARCH-fitted volatility, discounts the expected terminal price back
    to today as a 'fair value', and compares it against the current
    market price — proposes long when fair value exceeds market price,
    short when market price exceeds fair value."""

    agent_id: str = "monte_carlo_valuation_analyst"

    _risk_window: int = 60
    _num_paths: int = 2_000
    _num_steps: int = 32
    _time_to_maturity: float = 0.25

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
                confidence=0.5,
                rationale="insufficient history for quantcore Monte Carlo valuation",
            )

        mean_return = float(returns.mean())
        volatility = float(returns.std(ddof=1))
        try:
            omega, alpha, beta = fit_garch_11(returns)
            centered = returns - mean_return
            variance_path = garch_11_variance(centered, omega, alpha, beta)
            volatility = float((max(float(variance_path[-1]), 0.0)) ** 0.5)
        except (RuntimeError, ValueError) as exc:
            logger.debug(
                "%s: GARCH fit failed for %s on %s (%s) — falling back to sample stdev "
                "volatility=%.4f.",
                self.agent_id,
                asset,
                as_of_date.date(),
                exc,
                volatility,
            )

        terminal_prices = simulate_gbm_paths(
            slice_.close,
            mean_return,
            volatility,
            self._time_to_maturity,
            self._num_steps,
            self._num_paths,
            seed=42,
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
