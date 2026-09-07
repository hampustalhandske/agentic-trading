# news_loader.py

"""Finnhub company news. Point-in-time is enforced by the query itself —
`company-news` is bounded by `to=as_of_date`, so nothing published after
`as_of_date` can come back.

Finnhub's `news-sentiment` endpoint (an aggregate, pre-computed score) is
a paid-tier-only endpoint — it returns 403 on a free API key regardless of
account age or usage. Per-headline sentiment is instead scored locally
with a small finance-domain keyword lexicon (see `_score_headline`), so
this loader stays usable on the free tier rather than depending on an
endpoint that free keys can never reach."""

from datetime import timedelta

import pandas as pd
import requests

from agentic_trading.data.cache import cached_fetch
from agentic_trading.data.loaders._env import require_env
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

_BASE_URL = "https://finnhub.io/api/v1"
DEFAULT_LOOKBACK_DAYS = 30

_POSITIVE_WORDS: frozenset[str] = frozenset(
    {
        "beat",
        "beats",
        "surge",
        "surges",
        "soar",
        "soars",
        "rally",
        "rallies",
        "upgrade",
        "upgraded",
        "outperform",
        "growth",
        "profit",
        "profits",
        "record",
        "gain",
        "gains",
        "strong",
        "buyback",
        "expansion",
    }
)
_NEGATIVE_WORDS: frozenset[str] = frozenset(
    {
        "miss",
        "misses",
        "plunge",
        "plunges",
        "slump",
        "slumps",
        "downgrade",
        "downgraded",
        "underperform",
        "loss",
        "losses",
        "lawsuit",
        "investigation",
        "recall",
        "layoff",
        "layoffs",
        "cut",
        "cuts",
        "weak",
        "decline",
        "declines",
        "fraud",
        "warning",
    }
)


def _score_headline(headline: str) -> float:
    """A lightweight, free-tier-compatible substitute for Finnhub's paid
    aggregate sentiment score: (positive - negative) keyword hits over
    total words, in [-1, 1]. Not a substitute for a real NLP sentiment
    model — a small, transparent proxy chosen specifically because it
    needs no paid API and no ML dependency."""
    words = [word.strip(".,!?()").lower() for word in headline.split()]
    if not words:
        return 0.0
    positive_hits = sum(1 for word in words if word in _POSITIVE_WORDS)
    negative_hits = sum(1 for word in words if word in _NEGATIVE_WORDS)
    if positive_hits == 0 and negative_hits == 0:
        return 0.0
    return (positive_hits - negative_hits) / len(words)


def _fetch_news_sentiment(
    ticker: str, as_of_date: pd.Timestamp, lookback_days: int
) -> pd.DataFrame:
    token = require_env("FINNHUB_API_KEY")
    to_date = as_of_date.date()
    from_date = to_date - timedelta(days=lookback_days)

    logger.debug("Loading Finnhub news for %s from %s to %s", ticker, from_date, to_date)
    news_response = requests.get(
        f"{_BASE_URL}/company-news",
        params={
            "symbol": ticker,
            "from": from_date.isoformat(),
            "to": to_date.isoformat(),
            "token": token,
        },
        timeout=30,
    )
    news_response.raise_for_status()
    articles = news_response.json()

    if not articles:
        return pd.DataFrame(columns=["datetime", "headline", "sentiment_score"])

    headlines = [article["headline"] for article in articles]
    return pd.DataFrame(
        {
            "datetime": [pd.Timestamp(article["datetime"], unit="s") for article in articles],
            "headline": headlines,
            "sentiment_score": [_score_headline(headline) for headline in headlines],
        }
    )


def load_news_sentiment(
    ticker: str, as_of_date: pd.Timestamp, lookback_days: int = DEFAULT_LOOKBACK_DAYS
) -> pd.DataFrame:
    """Recent news headlines for `ticker` up to `as_of_date`, each tagged
    with a locally-computed per-headline sentiment score (see
    `_score_headline`). Returns `datetime, headline, sentiment_score`."""
    return cached_fetch(
        source="finnhub_news",
        identifier=f"{ticker.upper()}_{lookback_days}d",
        as_of_date=as_of_date.date(),
        fetch_fn=lambda: _fetch_news_sentiment(ticker, as_of_date, lookback_days),
    )
