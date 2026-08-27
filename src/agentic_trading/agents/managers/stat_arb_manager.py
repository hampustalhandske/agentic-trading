# stat_arb_manager.py

import numpy as np
import pandas as pd
from quantcore.risk.egarch import ewma_variance

from agentic_trading.agents._market_math import clamp, market_slice
from agentic_trading.agents.base import ManagerAgent
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

EWMA_LAMBDA = 0.94


class StatArbManager(ManagerAgent):
    """Scales analyst confidence down for elevated spread volatility. Uses
    the asset's own recent return series as an EWMA-variance proxy for
    spread vol — the manager doesn't have direct access to each analyst's
    specific spread construction (OU spread, cointegration spread, Kalman
    hedge-ratio spread), so it reads risk at the manager level instead."""

    agent_id: str = "stat_arb_manager"

    _risk_window: int = 60

    def _spread_vol_load(
        self, signal: AgentSignal, market_data: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> float:
        slice_ = market_slice(market_data, as_of_date, signal.asset)
        returns = slice_.log_returns[-self._risk_window :]
        if returns.size < 2:
            return 0.0

        volatility = float(np.std(returns, ddof=1))
        try:
            variance_path = ewma_variance(returns, EWMA_LAMBDA)
            volatility = float(np.sqrt(max(float(variance_path[-1]), 0.0)))
        except (RuntimeError, ValueError) as exc:
            logger.debug(
                "%s: EWMA variance failed for %s on %s (%s) — falling back to sample stdev "
                "volatility=%.4f.",
                self.agent_id,
                signal.asset,
                as_of_date.date(),
                exc,
                volatility,
            )

        return volatility

    def scale_factor(
        self,
        signal: AgentSignal,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> float:
        volatility = self._spread_vol_load(signal, market_data, as_of_date)
        return clamp(1.0 / (1.0 + volatility))
