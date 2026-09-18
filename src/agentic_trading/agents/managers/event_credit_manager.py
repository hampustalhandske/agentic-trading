# event_credit_manager.py

import pandas as pd

from agentic_trading.agents._market_math import clamp
from agentic_trading.agents.base import ManagerAgent
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.config import settings
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)


class EventCreditManager(ManagerAgent):
    """Scales analyst confidence down as its underlying event/filing/
    sentiment data ages, measured from the signal's `source_as_of` field
    (the publication date of the latest filing, dissemination, or article
    the read rests on). A signal without `source_as_of` is treated as
    zero staleness."""

    agent_id: str = "event_credit_manager"

    def _staleness_load(
        self, signal: AgentSignal, market_data: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> float:
        if signal.source_as_of is None:
            return 0.0
        return max(0.0, float((as_of_date.date() - signal.source_as_of).days))

    def scale_factor(
        self,
        signal: AgentSignal,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> float:
        staleness_days = self._staleness_load(signal, market_data, as_of_date)
        return clamp(
            1.0 / (1.0 + staleness_days / settings.managers.event_staleness_half_life_days)
        )
