# cot_positioning_analyst.py

from typing import Literal

import numpy as np
import pandas as pd

from agentic_trading.agents._market_math import clamp
from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.data.loaders.lakehouse_loader import load_cot_positioning_from_lakehouse
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

_RISK_WINDOW = 52  # ~1 year of weekly COT reports
_MIN_OBSERVATIONS = 12
# COT's `ticker` column already matches this project's own convention for
# commodities and FX/rate futures (e.g. GC=F, EURUSD=X) — only Treasury
# ETFs need remapping to the futures contract COT actually tracks
# positioning for (no COT category exists for a bond ETF itself).
DEFAULT_COT_TICKER_BY_ASSET: dict[str, str] = {"SHY": "ZT=F", "IEF": "ZN=F", "TLT": "ZB=F"}


class COTPositioningAnalyst(AnalystAgent):
    """The standard "smart money vs. crowded trade" contrarian read real
    macro/commodity desks build from CFTC Commitment of Traders reports:
    z-scores net speculative positioning (managed-money net for
    commodities, leveraged-funds net for FX/rate futures, as a fraction
    of open interest — see `load_cot_positioning_from_lakehouse`) against
    its own trailing window. Extreme net-long positioning is treated as
    contrarian-bearish (a crowded long is exposed to unwind risk), extreme
    net-short as contrarian-bullish — not a trend-following read.

    `cot_ticker_by_asset` remaps `asset` to whatever ticker COT actually
    tracks positioning for (only needed for the Rates pod's Treasury
    ETFs, which have no COT category of their own — see
    `DEFAULT_COT_TICKER_BY_ASSET`); defaults to `asset` itself, which
    already matches COT's own ticker convention for commodities and FX."""

    agent_id: str = "cot_positioning_analyst"

    cot_ticker_by_asset: dict[str, str] = DEFAULT_COT_TICKER_BY_ASSET
    # Restricts this instance to one asset class so two instances in the
    # same pod (commodities and FX) never read the same COT series twice;
    # `None` covers whatever the pod hands it.
    covered_assets: tuple[str, ...] | None = None

    def generate_signal(
        self,
        asset: str,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> AgentSignal:
        cot_ticker = self.cot_ticker_by_asset.get(asset, asset)
        if self.covered_assets is not None and asset not in self.covered_assets:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale=f"{asset} is outside this instance's COT asset class",
                status="unavailable",
            )
        series = load_cot_positioning_from_lakehouse(cot_ticker, as_of_date.date())
        if series is None:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale=f"no COT positioning coverage for {cot_ticker}",
                status="unavailable",
            )

        window = series.to_numpy(dtype=np.float64)[-_RISK_WINDOW:]
        if window.size < _MIN_OBSERVATIONS:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="insufficient COT history for a positioning z-score",
                status="unavailable",
            )

        mean = float(np.mean(window))
        std = float(np.std(window, ddof=1))
        if std <= 0.0:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="zero trailing variance in COT net positioning",
                status="unavailable",
            )

        z_score = (float(window[-1]) - mean) / std
        direction: Literal["long", "short", "flat"]
        if z_score > 0.0:
            direction = "short"  # crowded long -> contrarian bearish
        elif z_score < 0.0:
            direction = "long"  # crowded short -> contrarian bullish
        else:
            direction = "flat"
        confidence = clamp(abs(z_score) / 3.0)

        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction=direction,
            confidence=confidence,
            rationale=(
                f"COT net positioning z-score={z_score:.2f} for {cot_ticker} "
                f"vs its own {_RISK_WINDOW}-report trailing window"
            ),
        )
