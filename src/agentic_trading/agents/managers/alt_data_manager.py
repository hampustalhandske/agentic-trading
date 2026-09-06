# alt_data_manager.py

import re
from datetime import date

import pandas as pd

from agentic_trading.agents._market_math import clamp
from agentic_trading.agents.base import ManagerAgent
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

_SOURCE_AS_OF_PATTERN = re.compile(r"\[source_as_of=(\d{4}-\d{2}-\d{2})\]")
_SAMPLE_COUNT_PATTERN = re.compile(r"\[sample_count=(\d+)\]")

STALENESS_HALF_LIFE_DAYS = 14.0
MIN_SAMPLE_COUNT_FOR_FULL_CONFIDENCE = 5.0


class AltDataManager(ManagerAgent):
    """Dampens alt_data_pod confidence along two independent axes, both
    encoded via [tag=value] rationale conventions (staleness follows
    EventCreditManager's `[source_as_of=...]` convention, with a shorter
    half-life since attention/disclosure signals go stale faster than
    filings; sparsity is a new `[sample_count=...]` tag — e.g. distinct
    congressional filers, distinct insider filings, or data points behind
    a z-score). Alt-data sources are noisier per-observation than priced
    markets, so a thin sample is dampened regardless of how strong its
    point estimate looks. A missing tag defaults to no dampening on that
    axis rather than raising, matching EventCreditManager's convention
    that a formatting slip in an analyst's rationale must never crash a
    run."""

    agent_id: str = "alt_data_manager"

    def _staleness_days(self, signal: AgentSignal, as_of_date: pd.Timestamp) -> float:
        match = _SOURCE_AS_OF_PATTERN.search(signal.rationale)
        if match is None:
            return 0.0
        try:
            source_as_of = date.fromisoformat(match.group(1))
        except ValueError:
            return 0.0
        return max(0.0, float((as_of_date.date() - source_as_of).days))

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
        staleness_days = self._staleness_days(signal, as_of_date)
        staleness_factor = clamp(1.0 / (1.0 + staleness_days / STALENESS_HALF_LIFE_DAYS))

        sample_count = self._sample_count(signal)
        sparsity_factor = (
            1.0
            if sample_count is None
            else clamp(sample_count / MIN_SAMPLE_COUNT_FOR_FULL_CONFIDENCE)
        )
        return staleness_factor * sparsity_factor
