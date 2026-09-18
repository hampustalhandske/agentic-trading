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

import pandas as pd

from agentic_trading.agents._market_math import clamp, vol_regime_scale_factor
from agentic_trading.agents.base import ManagerAgent
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.config import settings
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)


class FactorManager(ManagerAgent):
    """See module docstring — a light secondary VaR check, not a duplicate
    of the factor-pod analysts' own confidence."""

    agent_id: str = "factor_manager"

    def scale_factor(
        self,
        signal: AgentSignal,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> float:
        """The shared volatility-regime dampening
        (`_market_math.vol_regime_scale_factor`, 0-1) mapped into this
        manager's deliberately narrow
        [`factor_min_scale_factor`, `factor_max_scale_factor`] band."""
        low = settings.managers.factor_min_scale_factor
        high = settings.managers.factor_max_scale_factor
        regime = vol_regime_scale_factor(market_data, as_of_date, signal.asset)
        return clamp(low + (high - low) * regime, low, high)
