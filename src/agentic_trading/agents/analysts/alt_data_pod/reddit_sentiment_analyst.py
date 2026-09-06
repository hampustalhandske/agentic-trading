# reddit_sentiment_analyst.py

from typing import Literal

import numpy as np
import pandas as pd

from agentic_trading.agents._market_math import clamp
from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.data.loaders.reddit_loader import load_mention_counts
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

LOOKBACK_DAYS = 7
MIN_MENTIONS = 5
SCORE_CAP = 50.0


class RedditSentimentAnalyst(AnalystAgent):
    """Mention-volume and sentiment from finance subreddits via
    `data/loaders/reddit_loader.py`. That loader returns post metadata
    only (score, comment count, timestamp) — no title/selftext — since
    Reddit's free search API doesn't surface either cheaply at scale, so a
    lexicon-based text sentiment scorer isn't applicable here. This
    analyst uses each post's own community `score` (net upvotes) as a
    lightweight, deterministic sentiment proxy instead: positive average
    score across recent mentions reads as bullish community reception,
    negative as bearish — still a classical, non-LLM read, just over a
    different signal than headline text. Confidence scales with both
    sentiment magnitude and mention volume, since a single post's score is
    noisy."""

    agent_id: str = "reddit_sentiment_analyst"

    def generate_signal(
        self,
        asset: str,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> AgentSignal:
        try:
            mentions = load_mention_counts(asset, as_of_date)
        except Exception as exc:
            logger.debug(
                "%s: Reddit lookup failed for %s (%s) — holding.", self.agent_id, asset, exc
            )
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="Reddit lookup failed — holding.",
            )

        if mentions.empty:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="no recent Reddit mentions — holding.",
            )

        window_start = as_of_date - pd.Timedelta(days=LOOKBACK_DAYS)
        recent = mentions.loc[mentions["created_utc"] >= window_start]
        mention_count = len(recent)
        if mention_count < MIN_MENTIONS:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale=(
                    f"only {mention_count} mention(s) in the trailing {LOOKBACK_DAYS}d — "
                    "too sparse — holding."
                ),
            )

        avg_score = float(np.mean(recent["score"].to_numpy(dtype=np.float64)))
        latest_mention = recent["created_utc"].max().date()

        direction: Literal["long", "short", "flat"]
        if avg_score > 0.0:
            direction = "long"
        elif avg_score < 0.0:
            direction = "short"
        else:
            direction = "flat"

        confidence = clamp(abs(avg_score) / SCORE_CAP) * clamp(mention_count / MIN_MENTIONS)

        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction=direction,
            confidence=confidence,
            rationale=(
                f"{mention_count} mentions, avg community score={avg_score:.1f} in the trailing "
                f"{LOOKBACK_DAYS}d [source_as_of={latest_mention.isoformat()}] "
                f"[sample_count={mention_count}]"
            ),
        )
