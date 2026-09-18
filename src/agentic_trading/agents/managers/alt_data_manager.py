# alt_data_manager.py

import pandas as pd

from agentic_trading.agents._market_math import clamp
from agentic_trading.agents.base import ManagerAgent
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.config import settings
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)


class AltDataManager(ManagerAgent):
    """Dampens alt_data_pod confidence along two independent axes read
    from the signal's typed provenance fields: staleness of the source
    data (`source_as_of`, with a shorter half-life than filings since
    attention/disclosure signals go stale faster) and sparsity of the
    sample behind the read (`sample_count` — e.g. distinct insider
    filings, or data points behind a z-score). Alt-data sources are
    noisier per-observation than priced markets, so a thin sample is
    dampened regardless of how strong its point estimate looks. A field
    left `None` means no dampening on that axis."""

    agent_id: str = "alt_data_manager"

    def _staleness_days(self, signal: AgentSignal, as_of_date: pd.Timestamp) -> float:
        if signal.source_as_of is None:
            return 0.0
        return max(0.0, float((as_of_date.date() - signal.source_as_of).days))

    def _sample_count(self, signal: AgentSignal) -> float | None:
        if signal.sample_count is None:
            return None
        return float(signal.sample_count)

    def scale_factor(
        self,
        signal: AgentSignal,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> float:
        staleness_days = self._staleness_days(signal, as_of_date)
        staleness_factor = clamp(
            1.0 / (1.0 + staleness_days / settings.managers.alt_data_staleness_half_life_days)
        )

        sample_count = self._sample_count(signal)
        sparsity_factor = (
            1.0
            if sample_count is None
            else clamp(sample_count / settings.managers.alt_data_min_sample_count)
        )
        return staleness_factor * sparsity_factor
