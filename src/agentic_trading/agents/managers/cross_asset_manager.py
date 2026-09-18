# cross_asset_manager.py

import pandas as pd

from agentic_trading.agents._market_math import vol_regime_scale_factor
from agentic_trading.agents.base import ManagerAgent
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

COMMODITIES_ASSETS: tuple[str, ...] = ("GC=F", "CL=F", "SI=F", "NG=F", "HG=F")
FX_ASSETS: tuple[str, ...] = ("EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X")
CRYPTO_ASSETS: tuple[str, ...] = ("BTC-USD", "ETH-USD")
CROSS_ASSET_ASSETS: tuple[str, ...] = COMMODITIES_ASSETS + FX_ASSETS + CRYPTO_ASSETS


class CrossAssetManager(ManagerAgent):
    """Scoped (`self.assets = CROSS_ASSET_ASSETS`) to commodity futures,
    major currency pairs, and crypto — three asset classes with the same
    sizing answer, so they share one pod rather than three thin ones.
    Wired with `PortfolioConstructor(method="risk_parity")` in
    `roster.py`, not Black-Litterman: none of gold/crude futures,
    currency pairs, or BTC/ETH has a market-cap-style equilibrium a BL
    prior could be built from, so every name in this basket sizes on
    volatility alone."""

    agent_id: str = "cross_asset_manager"

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
