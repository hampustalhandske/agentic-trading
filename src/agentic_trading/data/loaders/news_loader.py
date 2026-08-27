# news_loader.py

"""Finnhub company news + aggregate sentiment. Point-in-time is enforced by
the query itself — `company-news` is bounded by `to=as_of_date`, so nothing
published after `as_of_date` can come back."""

from datetime import timedelta

import pandas as pd
import requests

from agentic_trading.data.loaders._env import require_env
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

_BASE_URL = "https://finnhub.io/api/v1"
DEFAULT_LOOKBACK_DAYS = 30


def load_news_sentiment(
    ticker: str, as_of_date: pd.Timestamp, lookback_days: int = DEFAULT_LOOKBACK_DAYS
) -> pd.DataFrame:
    """Recent news headlines for `ticker` up to `as_of_date`, each tagged
    with Finnhub's aggregate company sentiment score (per-article sentiment
    isn't available on Finnhub's free tier, so every row shares the same
    aggregate score). Returns `datetime, headline, sentiment_score`."""
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

    sentiment_response = requests.get(
        f"{_BASE_URL}/news-sentiment",
        params={"symbol": ticker, "token": token},
        timeout=30,
    )
    sentiment_response.raise_for_status()
    sentiment_payload = sentiment_response.json()
    aggregate_score = float(sentiment_payload.get("companyNewsScore", 0.0))

    if not articles:
        return pd.DataFrame(columns=["datetime", "headline", "sentiment_score"])

    return pd.DataFrame(
        {
            "datetime": [pd.Timestamp(article["datetime"], unit="s") for article in articles],
            "headline": [article["headline"] for article in articles],
            "sentiment_score": aggregate_score,
        }
    )
