# anomaly_network_manager.py

import re

import pandas as pd

from agentic_trading.agents._market_math import clamp
from agentic_trading.agents.base import ManagerAgent
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

_SAMPLE_COUNT_PATTERN = re.compile(r"\[sample_count=(\d+)\]")

MIN_SAMPLE_COUNT_FOR_FULL_CONFIDENCE = 20.0


class AnomalyNetworkManager(ManagerAgent):
    """Dampens confidence outside of the well-studied regimes each
    anomaly/network analyst depends on: a calendar effect estimated from
    a handful of historical observations, or a lead-lag correlation
    computed from a short return window, is far less trustworthy than the
    same estimate backed by a deep sample. Analysts in this pod encode
    their sample size as a `[sample_count=N]` rationale tag (the same
    convention AltDataManager uses for sparsity); this manager scales
    confidence down when N falls short of what the underlying technique
    needs to be statistically meaningful. No staleness dampening here —
    unlike alt_data_pod's filings/disclosures, every source in this pod
    is either pure price history or a same-day-gated market-wide series,
    so there's no publication-lag axis to dampen against."""

    agent_id: str = "anomaly_network_manager"

    def _sample_count(self, signal: AgentSignal) -> float | None:
        match = _SAMPLE_COUNT_PATTERN.search(signal.rationale)
        if match is None:
            return None
        return float(match.group(1))

    def scale_factor(
        self,
        signal: AgentSignal,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> float:
        sample_count = self._sample_count(signal)
        if sample_count is None:
            return 1.0
        return clamp(sample_count / MIN_SAMPLE_COUNT_FOR_FULL_CONFIDENCE)
