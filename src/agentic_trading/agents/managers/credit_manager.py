# credit_manager.py

import pandas as pd

from agentic_trading.agents._market_math import vol_regime_scale_factor
from agentic_trading.agents.base import ManagerAgent
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

# HYG/LQD (high-yield / investment-grade corporate bond ETFs) are the
# tradeable instruments credit_spread_signal_analyst's market-wide
# risk-appetite read attaches to — no per-issuer credit ratings or CDS
# data exists for free (see CLAUDE.md), so this pod's whole book is these
# two market-wide credit proxies, not individual names.
CREDIT_ASSETS: tuple[str, ...] = ("HYG", "LQD")


class CreditManager(ManagerAgent):
    """Scoped (`self.assets = CREDIT_ASSETS`) to the two market-wide
    credit-proxy ETFs `credit_spread_signal_analyst` is built for — the
    "Credit" half of the original "Event/Credit" pod name that pod never
    actually had any credit data for. Wired with
    `PortfolioConstructor(method="risk_parity")` in `roster.py` — a
    2-name book has no need for a market-cap equilibrium prior."""

    agent_id: str = "credit_manager"

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
