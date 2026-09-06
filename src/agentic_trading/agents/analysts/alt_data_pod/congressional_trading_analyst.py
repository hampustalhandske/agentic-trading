# congressional_trading_analyst.py

from typing import Literal

import pandas as pd

from agentic_trading.agents._market_math import clamp
from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.data.loaders.congress_trading_loader import load_congressional_trades
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

LOOKBACK_DAYS = 90
MAX_CONFIDENCE = 0.7


class CongressionalTradingAnalyst(AnalystAgent):
    """Cluster-buying/selling signal from Senate/House Stock Watcher
    disclosures (`data/loaders/congress_trading_loader.py`): direction from
    the net buy/sell imbalance among an asset's disclosed trades in the
    trailing window, confidence scaled by how many distinct filers
    contributed to that imbalance (one member's trade is a weak signal;
    several independently disclosing the same direction is stronger).
    Appends `[source_as_of=...]` (latest disclosure date) and
    `[sample_count=...]` (distinct filers) for `AltDataManager`'s
    staleness/sparsity dampening."""

    agent_id: str = "congressional_trading_analyst"

    def generate_signal(
        self,
        asset: str,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> AgentSignal:
        try:
            trades = load_congressional_trades(asset, as_of_date)
        except Exception as exc:
            logger.debug(
                "%s: Stock Watcher lookup failed for %s (%s) — holding.",
                self.agent_id,
                asset,
                exc,
            )
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="congressional trading lookup failed — holding.",
            )

        window_start = as_of_date - pd.Timedelta(days=LOOKBACK_DAYS)
        recent = trades.loc[trades["disclosure_date"] >= window_start]
        if recent.empty:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="no recent congressional disclosures — holding.",
            )

        transaction_type = recent["transaction_type"].astype(str).str.lower()
        buys = int(transaction_type.str.contains("purchase").sum())
        sells = int(transaction_type.str.contains("sale").sum())
        total = buys + sells
        if total == 0:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="disclosures found but none classify as a purchase or sale — holding.",
            )

        net_imbalance = (buys - sells) / total
        distinct_filers = int(recent["filer"].nunique())
        latest_disclosure = recent["disclosure_date"].max().date()

        direction: Literal["long", "short", "flat"]
        if net_imbalance > 0:
            direction = "long"
        elif net_imbalance < 0:
            direction = "short"
        else:
            direction = "flat"

        confidence = clamp(abs(net_imbalance)) * MAX_CONFIDENCE

        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction=direction,
            confidence=confidence,
            rationale=(
                f"{buys} purchase / {sells} sale disclosures across {distinct_filers} distinct "
                f"filers in the trailing {LOOKBACK_DAYS}d "
                f"[source_as_of={latest_disclosure.isoformat()}] "
                f"[sample_count={distinct_filers}]"
            ),
        )
