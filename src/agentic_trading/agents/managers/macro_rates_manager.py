# macro_rates_manager.py

import numpy as np
import pandas as pd
from quantcore.risk.egarch import egarch_11_variance

from agentic_trading.agents._market_math import clamp, market_slice
from agentic_trading.agents.base import ManagerAgent
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

# quantcore has no EGARCH MLE fitter — these are documented fixed
# parameters (same values EGARCHAsymmetricVolAnalyst uses), not a fitted
# estimate. A real fitter would be quantcore-shaped future work.
FIXED_OMEGA = -0.1
FIXED_ALPHA = 0.15
FIXED_GAMMA = 0.1
FIXED_BETA = 0.9


class MacroRatesManager(ManagerAgent):
    """Scales analyst confidence down for an elevated EGARCH-implied
    asymmetric volatility regime, using the same fixed-parameter EGARCH(1,1)
    approach as EGARCHAsymmetricVolAnalyst."""

    agent_id: str = "macro_rates_manager"

    _risk_window: int = 60

    def _asymmetric_vol_load(
        self, signal: AgentSignal, market_data: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> float:
        slice_ = market_slice(market_data, as_of_date, signal.asset)
        returns = slice_.log_returns[-self._risk_window :]
        if returns.size < 10:
            return 0.0

        try:
            variance_path = egarch_11_variance(
                returns, FIXED_OMEGA, FIXED_ALPHA, FIXED_GAMMA, FIXED_BETA
            )
        except (RuntimeError, ValueError) as exc:
            fallback = float(np.std(returns, ddof=1))
            logger.debug(
                "%s: EGARCH variance failed for %s on %s (%s) — falling back to sample stdev "
                "volatility=%.4f.",
                self.agent_id,
                signal.asset,
                as_of_date.date(),
                exc,
                fallback,
            )
            return fallback

        return float(np.sqrt(max(float(np.exp(variance_path[-1])), 0.0)))

    def scale_factor(
        self,
        signal: AgentSignal,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> float:
        volatility = self._asymmetric_vol_load(signal, market_data, as_of_date)
        return clamp(1.0 / (1.0 + volatility))
