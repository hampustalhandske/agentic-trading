# short_interest_squeeze_analyst.py

import numpy as np
import pandas as pd
from quantcore.statistics.cointegration import spread_zscore

from agentic_trading.agents._market_math import clamp
from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.data.loaders.short_interest_loader import load_short_interest
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

MAX_CONFIDENCE = 0.7
Z_CAP = 3.0


class ShortInterestSqueezeAnalyst(AnalystAgent):
    """A high positive z-score on days-to-cover (unusually elevated vs. its
    own trailing history) signals squeeze risk — a contrarian long bias,
    since a crowded short position is itself fuel for a squeeze. Confidence
    is capped since short-interest extremes are rare and the z-score can
    spike hard on thin data. Appends `[source_as_of=...]` per
    EventCreditManager's staleness convention."""

    agent_id: str = "short_interest_squeeze_analyst"

    z_window: int = 6

    def generate_signal(
        self,
        asset: str,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> AgentSignal:
        try:
            history = load_short_interest(asset, as_of_date)
        except Exception as exc:
            logger.debug(
                "%s: FINRA lookup failed for %s (%s) — returning flat.", self.agent_id, asset, exc
            )
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="FINRA short interest lookup failed",
            )

        if len(history) < self.z_window + 1:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="insufficient short interest history for a z-score read",
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
                rationale=(
                    f"days-to-cover not elevated (z={current_z:.2f}) "
                    f"[source_as_of={latest_dissemination.isoformat()}]"
                ),
            )

        confidence = clamp(min(current_z, Z_CAP) / Z_CAP) * MAX_CONFIDENCE
        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction="long",
            confidence=confidence,
            rationale=(
                f"days-to-cover elevated (z={current_z:.2f}) — contrarian squeeze-risk long bias "
                f"[source_as_of={latest_dissemination.isoformat()}]"
            ),
        )
