# risk_manager.py

from __future__ import annotations

import pandas as pd

from agentic_trading.agents._market_math import vol_regime_scale_factor
from agentic_trading.agents.base import ManagerAgent
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)


class RiskManager(ManagerAgent):
    """Scales analyst confidence down when the asset's own volatility is
    elevated relative to its trailing history, and leaves it near 1.0 in
    a normal regime. Does not decide direction — only how much confidence
    an analyst's signal deserves once risk is accounted for."""

    agent_id: str = "risk_manager"

    def scale_factor(
        self,
        signal: AgentSignal,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> float:
        """Volatility-regime dampening for `signal.asset` (see
        `_market_math.vol_regime_scale_factor`): near 1.0 in a normal
        regime, halved at a 1.5-sigma vol spike."""
        return vol_regime_scale_factor(market_data, as_of_date, signal.asset)
