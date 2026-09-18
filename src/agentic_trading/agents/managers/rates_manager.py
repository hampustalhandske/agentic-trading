# rates_manager.py

import pandas as pd

from agentic_trading.agents._market_math import vol_regime_scale_factor
from agentic_trading.agents.base import ManagerAgent
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

RATES_ASSETS: tuple[str, ...] = ("SHY", "IEF", "TLT")


class RatesManager(ManagerAgent):
    """Scoped (`self.assets = RATES_ASSETS`, via `ManagerAgent`'s own
    asset-scoping mechanism) to a real duration ladder of Treasury ETFs —
    this is the pod `cir_rate_regime_analyst`'s CIR-simulated rate call
    was always meant to land on: a genuinely rate-sensitive tradeable
    instrument, not an arbitrary equity via a fixed sign flip (see the
    analyst's own docstring). Uses `PortfolioConstructor`'s default
    (Black-Litterman + mean-variance) sizing — a bond-ETF ladder has a
    real, duration-weighted equilibrium concept, unlike the pairs/vol/
    FX/crypto pods elsewhere in this roster."""

    agent_id: str = "rates_manager"

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
