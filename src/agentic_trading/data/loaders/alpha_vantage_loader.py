# alpha_vantage_loader.py

"""Alpha Vantage daily EOD prices — a documented alternative price source
for names where Stooq/Tiingo have gaps. Free tier is heavily rate-limited
(~25 requests/day), so `data/loader.py` only reaches this as its last-resort
tier, after both yfinance and Stooq have already failed for a ticker —
common-path runs never touch it, and the on-disk cache (`data/cache.py`)
keeps a repeat run of the same failing ticker from re-spending quota.

The free tier's `outputsize=full` still returns the entire available daily
history in one call, so a single request captures as much usable data as
the API allows rather than a short recent window.

Gates on the bar's own date: the full daily series is fetched once and
filtered to `<= as_of_date`, the same bulk-then-filter pattern
`factor_loader.py` uses for a bulk source."""

from datetime import date

import pandas as pd
import requests

from agentic_trading.data.cache import cached_fetch
from agentic_trading.data.loaders._env import require_env
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

_BASE_URL = "https://www.alphavantage.co/query"


def _fetch_alpha_vantage_series(ticker: str) -> pd.DataFrame:
    api_key = require_env("ALPHAVANTAGE_API_KEY")
    logger.info("Downloading Alpha Vantage daily series for %s", ticker)
    response = requests.get(
        _BASE_URL,
        params={
            "function": "TIME_SERIES_DAILY",
            "symbol": ticker,
            "outputsize": "full",
            "apikey": api_key,
        },
        timeout=30,
    )
    response.raise_for_status()
    raw = response.json()
    series = raw.get("Time Series (Daily)")
    if not series:
        raise ValueError(
            f"Alpha Vantage returned no daily series for {ticker!r}: {raw.get('Note') or raw}"
        )
    frame = pd.DataFrame.from_dict(series, orient="index")
    frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index().rename(columns={"4. close": "close"})
    frame["close"] = frame["close"].astype(float)
    return frame[["close"]]


def load_alpha_vantage_price_history(ticker: str, as_of_date: pd.Timestamp) -> pd.DataFrame:
    """Alpha Vantage close-price history for `ticker`, filtered to
    `<= as_of_date`. Meant as a low-volume alternative source, not a
    high-frequency fallback — the free tier allows roughly 25 requests/day."""
    frame = cached_fetch(
        source="alpha_vantage",
        identifier=ticker.upper(),
        as_of_date=date.today(),
        fetch_fn=lambda: _fetch_alpha_vantage_series(ticker),
    )
    return frame.loc[frame.index <= as_of_date]
