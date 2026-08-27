# news_sentiment.py

from typing import Literal

import pandas as pd

from agentic_trading.agents._market_math import clamp
from agentic_trading.agents.base import (
    SAFE_FALLBACK_CONFIDENCE,
    SAFE_FALLBACK_DIRECTION,
    AnalystAgent,
)
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.data.loaders.news_loader import load_news_sentiment
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

MAX_ARTICLE_COUNT_FOR_CONFIDENCE = 20


class NewsSentimentAnalyst(AnalystAgent):
    """Wired to Finnhub via `data/loaders/news_loader.py`: direction from
    the aggregate sentiment score's sign, confidence from the score's
    magnitude and article count (more articles → more confidence, up to a
    cap). Appends `[source_as_of=...]` to `rationale` using the most recent
    article's date, per EventCreditManager's staleness convention."""

    agent_id: str = "news_sentiment_analyst"

    def generate_signal(
        self,
        asset: str,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> AgentSignal:
        try:
            news = load_news_sentiment(asset, as_of_date)
        except Exception as exc:
            logger.debug(
                "%s: Finnhub lookup failed for %s (%s) — holding.", self.agent_id, asset, exc
            )
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction=SAFE_FALLBACK_DIRECTION,
                confidence=SAFE_FALLBACK_CONFIDENCE,
                rationale="news sentiment lookup failed — holding.",
            )

        if news.empty:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction=SAFE_FALLBACK_DIRECTION,
                confidence=SAFE_FALLBACK_CONFIDENCE,
                rationale="no recent news found — holding.",
            )

        sentiment_score = float(news["sentiment_score"].iloc[0])
        article_count = len(news)
        latest_article_date = news["datetime"].max().date()

        direction: Literal["long", "short", "flat"]
        if sentiment_score > 0.0:
            direction = "long"
        elif sentiment_score < 0.0:
            direction = "short"
        else:
            direction = "flat"

        confidence = clamp(abs(sentiment_score)) * clamp(
            article_count / MAX_ARTICLE_COUNT_FOR_CONFIDENCE
        )

        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction=direction,
            confidence=clamp(confidence),
            rationale=(
                f"Finnhub aggregate sentiment={sentiment_score:.3f} across "
                f"{article_count} articles "
                f"[source_as_of={latest_article_date.isoformat()}]"
            ),
        )
