# short_interest_squeeze_analyst.py

import numpy as np
import pandas as pd
from quantcore.statistics.cointegration import spread_zscore

from agentic_trading.agents._market_math import clamp
from agentic_trading.agents.base import AnalystAgent, DataRequirement, is_likely_equity
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.config import settings
from agentic_trading.data.loaders.lakehouse_loader import load_short_interest_from_lakehouse
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

MAX_CONFIDENCE = 0.7


class ShortInterestSqueezeAnalyst(AnalystAgent):
    """A high positive z-score on days-to-cover (unusually elevated vs. its
    own trailing history) signals squeeze risk — a contrarian long bias,
    since a crowded short position is itself fuel for a squeeze. Confidence
    is capped since short-interest extremes are rare and the z-score can
    spike hard on thin data. Reports the latest dissemination date as
    `source_as_of` so EventCreditManager can dampen on staleness."""

    agent_id: str = "short_interest_squeeze_analyst"

    z_window: int = 6
    data_requirements: tuple[DataRequirement, ...] = (
        DataRequirement(
            source="finra_short_interest",
            identifier="{ticker}",
            asset_scoped=True,
            description="FINRA short interest",
        ),
    )

    def generate_signal(
        self,
        asset: str,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> AgentSignal:
        if not is_likely_equity(asset):
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="not an equity — FINRA short interest doesn't apply",
                status="unavailable",
            )
        history = load_short_interest_from_lakehouse(asset, as_of_date.date())
        if history is None:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="no FINRA short-interest coverage in the lakehouse",
                status="unavailable",
            )

        if len(history) < self.z_window + 1:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="insufficient short interest history for a z-score read",
                status="unavailable",
            )

        sorted_history = history.sort_values("settlement_date")
        days_to_cover = sorted_history["days_to_cover"].to_numpy(dtype=np.float64)
        current_z = float(spread_zscore(days_to_cover, min(self.z_window, days_to_cover.size))[-1])
        latest_dissemination = sorted_history["dissemination_date"].max().date()

        if current_z <= 0:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale=f"days-to-cover not elevated (z={current_z:.2f})",
                source_as_of=latest_dissemination,
                sample_count=len(history),
            )

        confidence = (
            clamp(min(current_z, settings.analysts.z_cap) / settings.analysts.z_cap)
            * MAX_CONFIDENCE
        )
        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction="long",
            confidence=confidence,
            rationale=(
                f"days-to-cover elevated (z={current_z:.2f}) — contrarian squeeze-risk long bias"
            ),
            source_as_of=latest_dissemination,
            sample_count=len(history),
        )
