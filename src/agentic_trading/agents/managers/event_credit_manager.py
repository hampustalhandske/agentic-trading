# event_credit_manager.py

import re
from datetime import date

import pandas as pd

from agentic_trading.agents._market_math import clamp
from agentic_trading.agents.base import ManagerAgent
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

_SOURCE_AS_OF_PATTERN = re.compile(r"\[source_as_of=(\d{4}-\d{2}-\d{2})\]")
STALENESS_HALF_LIFE_DAYS = 30.0


class EventCreditManager(ManagerAgent):
    """Scales analyst confidence down as its underlying event/filing/
    sentiment data ages. This pod's analysts have no schema field for
    their source data's as-of date, so by convention they append
    `[source_as_of=YYYY-MM-DD]` to `rationale` (see
    `agents/analysts/event_credit_pod/`); this manager parses it back out.
    A missing or unparseable tag is treated as zero staleness rather than
    raised on — a manager must never crash the run over a formatting slip
    in an analyst's rationale string."""

    agent_id: str = "event_credit_manager"

    def _staleness_load(
        self, signal: AgentSignal, market_data: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> float:
        match = _SOURCE_AS_OF_PATTERN.search(signal.rationale)
        if match is None:
            return 0.0

        try:
            source_as_of = date.fromisoformat(match.group(1))
        except ValueError:
            return 0.0

        staleness_days = (as_of_date.date() - source_as_of).days
        return max(0.0, float(staleness_days))

    def scale_factor(
        self,
        signal: AgentSignal,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> float:
        staleness_days = self._staleness_load(signal, market_data, as_of_date)
        return clamp(1.0 / (1.0 + staleness_days / STALENESS_HALF_LIFE_DAYS))
