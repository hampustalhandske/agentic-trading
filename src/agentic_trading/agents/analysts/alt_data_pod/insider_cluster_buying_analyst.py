# insider_cluster_buying_analyst.py

import pandas as pd

from agentic_trading.agents._market_math import clamp
from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.data.loaders.sec_fulltext_loader import load_filings
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

LOOKBACK_DAYS = 14
MIN_FILINGS_FOR_SIGNAL = 2
FULL_CONFIDENCE_FILING_COUNT = 5
MAX_CONFIDENCE = 0.5


class InsiderClusterBuyingAnalyst(AnalystAgent):
    """SEC Form 4 insider-transaction filings
    (`data/loaders/sec_fulltext_loader.py`) — distinct from
    `sec_fundamentals_analyst`, which reads financial statement facts, not
    insider transactions. Detects cluster filing: several distinct Form 4s
    for the same issuer in a short trailing window is read as multiple
    insiders transacting independently, a stronger signal than a single
    filing, per the standard finding that clustered insider activity is
    more informative than an isolated one.

    Data-source honesty note: SEC's full-text search index (what
    `load_filings` wraps) returns filing-level metadata only — form type,
    filer/issuer display names, filed date — not the per-transaction
    acquired/disposed code buried in each filing's own XML, which would
    require fetching and parsing every filing individually. Since
    purchases and sales can't be told apart from this metadata, this
    analyst never signals `short`: it only reads filing-cluster density as
    an attention/informativeness proxy and biases `long`, consistent with
    the academic asymmetry that insider *purchases* (not routine sales,
    e.g. scheduled 10b5-1 plans) are the more informative half of insider
    activity — with confidence capped well below what a manager with
    verified transaction direction would earn."""

    agent_id: str = "insider_cluster_buying_analyst"

    def generate_signal(
        self,
        asset: str,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> AgentSignal:
        try:
            filings = load_filings(asset, as_of_date, forms="4")
        except Exception as exc:
            logger.debug(
                "%s: SEC full-text search failed for %s (%s) — holding.",
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
                rationale="SEC full-text search lookup failed — holding.",
            )

        if filings.empty:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="no recent Form 4 filings found — holding.",
            )

        window_start = as_of_date - pd.Timedelta(days=LOOKBACK_DAYS)
        recent = filings.loc[filings["filed"] >= window_start]
        cluster_size = int(recent["accession_no"].nunique())

        if cluster_size < MIN_FILINGS_FOR_SIGNAL:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale=(
                    f"only {cluster_size} Form 4 filing(s) in the trailing {LOOKBACK_DAYS}d — "
                    "no cluster — holding."
                ),
            )

        latest_filed = recent["filed"].max().date()
        confidence = clamp(cluster_size / FULL_CONFIDENCE_FILING_COUNT) * MAX_CONFIDENCE

        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction="long",
            confidence=confidence,
            rationale=(
                f"{cluster_size} distinct Form 4 filings in the trailing {LOOKBACK_DAYS}d — "
                "insider filing cluster, direction unverified (metadata-only) so capped long bias "
                f"[source_as_of={latest_filed.isoformat()}] [sample_count={cluster_size}]"
            ),
        )
