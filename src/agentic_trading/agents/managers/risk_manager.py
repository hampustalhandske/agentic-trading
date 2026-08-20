# risk_manager.py

from __future__ import annotations

import numpy as np
import pandas as pd
from quantcore.risk import fit_garch_11, garch_11_variance, value_at_risk

from agentic_trading.agents._market_math import clamp, market_slice
from agentic_trading.agents.base import ManagerAgent
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)


class RiskManager(ManagerAgent):
    """Scales analyst confidence down for assets with elevated quantcore
    VaR/CVaR or an unfavorable GARCH volatility regime, and closer to 1.0
    for calm assets. Does not decide direction — only how much confidence
    an analyst's signal deserves once risk is accounted for."""

    agent_id: str = "risk_manager"

    _risk_window: int = 60
    _confidence_level: float = 0.95

    def _risk_load(
        self, signal: AgentSignal, market_data: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> float:
        slice_ = market_slice(market_data, as_of_date, signal.asset)
        returns = slice_.log_returns[-self._risk_window :]
        if returns.size < 2:
            return 0.0

        volatility = float(np.std(returns, ddof=1))
        try:
            omega, alpha, beta = fit_garch_11(returns)
            centered = returns - float(np.mean(returns))
            variance_path = garch_11_variance(centered, omega, alpha, beta)
            volatility = float(np.sqrt(max(float(variance_path[-1]), 0.0)))
        except (RuntimeError, ValueError) as exc:
            logger.debug(
                "%s: GARCH fit failed for %s on %s (%s) — falling back to sample stdev "
                "volatility=%.4f.",
                self.agent_id,
                signal.asset,
                as_of_date.date(),
                exc,
                volatility,
            )

        try:
            var = value_at_risk(returns, self._confidence_level, volatility=volatility)
        except ValueError as exc:
            logger.debug(
                "%s: VaR computation failed for %s on %s (%s) — falling back to volatility=%.4f.",
                self.agent_id,
                signal.asset,
                as_of_date.date(),
                exc,
                volatility,
            )
            var = volatility

        return max(0.0, float(var)) + volatility

    def scale_factor(
        self,
        signal: AgentSignal,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> float:
        """Computes this manager's dampening multiplier for one analyst
        signal, using quantcore's VaR and GARCH volatility regime for
        `signal.asset` as of `as_of_date`. Must be >= 0."""
        risk_load = self._risk_load(signal, market_data, as_of_date)
        return clamp(1.0 / (1.0 + risk_load))
