# factor_manager.py

"""FactorManager is deliberately different in shape from every other
manager in this project. Every other manager reads a fresh risk metric off
market data and uses it to dampen an analyst's confidence. This pod's
analysts (see `agents/analysts/factor_pod/`) already encode their own
statistical confidence in `signal.confidence` at signal-generation time —
by convention, `confidence = clamp(abs(t_stat) / 4.0)` — so the real
judgment call already happened inside the analyst, not this manager.
Re-deriving a risk metric here and multiplying it in again would double-
count the same statistical-significance information. Instead,
`scale_factor` returns a factor that stays near 1.0 (never dampening
confidence more than lightly) and is only pulled down by a genuine VaR
tail-risk check, reusing the same VaR approach `RiskManager` uses, as a
secondary safety net — not this manager's primary judgment."""

import numpy as np
import pandas as pd
from quantcore.risk import fit_garch_11, garch_11_variance, value_at_risk

from agentic_trading.agents._market_math import clamp, market_slice
from agentic_trading.agents.base import ManagerAgent
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

MIN_SCALE_FACTOR = 0.9
MAX_SCALE_FACTOR = 1.0


class FactorManager(ManagerAgent):
    """See module docstring — a light secondary VaR check, not a duplicate
    of the factor-pod analysts' own confidence."""

    agent_id: str = "factor_manager"

    _risk_window: int = 60
    _confidence_level: float = 0.95

    def _significance_load(
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

        return max(0.0, float(var))

    def scale_factor(
        self,
        signal: AgentSignal,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> float:
        tail_risk = self._significance_load(signal, market_data, as_of_date)
        span = MAX_SCALE_FACTOR - MIN_SCALE_FACTOR
        return clamp(MAX_SCALE_FACTOR - span * clamp(tail_risk), MIN_SCALE_FACTOR, MAX_SCALE_FACTOR)
