# insider_cluster_buying_analyst.py

from typing import Literal

import pandas as pd

from agentic_trading.agents._market_math import clamp
from agentic_trading.agents.base import AnalystAgent, DataRequirement, is_likely_equity
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.config import settings
from agentic_trading.data.loaders.lakehouse_loader import load_insider_transactions_from_lakehouse
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

# Form 4 code for an open-market purchase; grants, exercises, tax
# withholding and gifts say nothing about an insider's view.
OPEN_MARKET_PURCHASE = "P"


class InsiderClusterBuyingAnalyst(AnalystAgent):
    """Cluster buying by corporate insiders: the number of distinct
    insiders who made open-market purchases (Form 4 code `P`) filed within
    the trailing `settings.analysts.insider_window_days`. At least
    `insider_min_buyers` distinct buyers is a long view, with confidence
    rising toward `insider_full_confidence_buyers` and capped at
    `insider_max_confidence`. Sales are ignored: they are dominated by
    compensation and diversification and carry little information. Gated
    on the filing date, so a trade only counts once it was public."""

    agent_id: str = "insider_cluster_buying_analyst"

    data_requirements: tuple[DataRequirement, ...] = (
        DataRequirement(
            source="sec_insider_transactions",
            identifier="{ticker}",
            asset_scoped=True,
            description="SEC Form 4 insider transactions",
        ),
    )

    def _flat(
        self,
        asset: str,
        as_of_date: pd.Timestamp,
        rationale: str,
        status: Literal["active", "unavailable", "error"],
    ) -> AgentSignal:
        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction="flat",
            confidence=0.0,
            rationale=rationale,
            status=status,
        )

    def generate_signal(
        self,
        asset: str,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> AgentSignal:
        if not is_likely_equity(asset):
            return self._flat(
                asset, as_of_date, "not an equity — Form 4 filings don't apply", "unavailable"
            )
        filings = load_insider_transactions_from_lakehouse(asset, as_of_date.date())
        if filings is None or filings.empty:
            return self._flat(
                asset, as_of_date, "no Form 4 coverage in the lakehouse", "unavailable"
            )

        window_start = as_of_date - pd.Timedelta(days=settings.analysts.insider_window_days)
        recent = filings[filings["filed_date"] > window_start]
        purchases = recent[recent["transaction_code"] == OPEN_MARKET_PURCHASE]
        buyers = int(purchases["owner_cik"].nunique())
        latest_filing = (
            recent["filed_date"].max() if not recent.empty else filings["filed_date"].max()
        )

        cfg = settings.analysts
        direction: Literal["long", "short", "flat"] = "flat"
        confidence = 0.0
        if buyers >= cfg.insider_min_buyers:
            direction = "long"
            confidence = (
                clamp(buyers / cfg.insider_full_confidence_buyers, 0.0, 1.0)
                * cfg.insider_max_confidence
            )
        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction=direction,
            confidence=confidence,
            rationale=(
                f"{buyers} distinct insider open-market buyer(s) across {len(purchases)} "
                f"purchase(s) filed in the last {cfg.insider_window_days} days"
            ),
            source_as_of=latest_filing.date(),
            sample_count=len(purchases),
        )
