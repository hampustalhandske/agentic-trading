# cvar_manager.py

from __future__ import annotations

import numpy as np
import pandas as pd
from quantcore.risk import conditional_value_at_risk, fit_garch_11, garch_11_variance

from agentic_trading.agents._market_math import clamp, market_slice
from agentic_trading.agents.base import ManagerAgent
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)


class ConditionalValueAtRiskManager(ManagerAgent):
    """Scales analyst confidence down when an asset's tail loss profile is
    elevated. This is the CVaR counterpart to RiskManager's VaR-based
    dampening."""

    agent_id: str = "cvar_manager"

    _risk_window: int = 60
    _confidence_level: float = 0.95

    def _tail_risk(
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
            cvar = conditional_value_at_risk(returns, self._confidence_level, volatility=volatility)
        except ValueError as exc:
            logger.debug(
                "%s: CVaR computation failed for %s on %s (%s) — falling back to volatility=%.4f.",
                self.agent_id,
                signal.asset,
                as_of_date.date(),
                exc,
                volatility,
            )
            cvar = volatility

        return max(0.0, float(cvar)) + volatility

    def scale_factor(
        self,
        signal: AgentSignal,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> float:
        tail_risk = self._tail_risk(signal, market_data, as_of_date)
        return clamp(1.0 / (1.0 + tail_risk))


CVaRManager = ConditionalValueAtRiskManager
