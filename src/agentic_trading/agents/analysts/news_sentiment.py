# news_sentiment.py

import pandas as pd

from agentic_trading.agents.base import (
    SAFE_FALLBACK_CONFIDENCE,
    SAFE_FALLBACK_DIRECTION,
    AnalystAgent,
)
from agentic_trading.agents.schemas import AgentSignal


class NewsSentimentAnalyst(AnalystAgent):
    """Text-driven, not numeric — proposes directional views from news
    sentiment rather than quantcore price/risk data. Needs an external
    news/text data source that is not yet integrated into this project.

    Uses the same flat/0-confidence fallback as a crashed agent (see
    agents/base.py's SAFE_FALLBACK_*) rather than a 0.5 "neutral" stance —
    an unimplemented agent has no real view to back a 50/50 claim with, and
    0.5 confidence would still show up as "this agent has an opinion" in
    the scoreboard/GUI even though it contributes nothing to target_weight
    either way (direction="flat" already zeroes it out mathematically)."""

    agent_id: str = "news_sentiment_analyst"

    def generate_signal(
        self,
        asset: str,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> AgentSignal:
        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction=SAFE_FALLBACK_DIRECTION,
            confidence=SAFE_FALLBACK_CONFIDENCE,
            rationale="news sentiment unavailable; no data source wired in yet — holding.",
        )
