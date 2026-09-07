# tiingo_loader.py

"""Tiingo EOD prices, an independent IEX-sourced source used as a
cross-check against the primary yfinance/Stooq price chain (see
`data/loader.py`) rather than another link in that fallback chain — a
divergence between Tiingo's close and the primary source's close is a
data-quality signal in its own right, not just another way to fill a gap.

Fetches through the real wall-clock date and caches under that real date,
then filters to `<= as_of_date` locally — the same "fetch the bulk series
once per real day, filter by simulated date after" pattern
`wikipedia_loader.py`/`trends_loader.py` use. Still structurally
incapable of returning a row dated after `as_of_date`, since that filter
is the only way a row reaches the caller; caching on the real date rather
than the simulated one is what keeps a multi-day backtest from
re-downloading the ticker's full history from Tiingo on every simulated
day."""

from datetime import date

import pandas as pd
import requests

from agentic_trading.data.cache import cached_fetch
from agentic_trading.data.loaders._env import require_env
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

_BASE_URL = "https://api.tiingo.com/tiingo/daily"


def _fetch_tiingo_prices(ticker: str, end: date) -> pd.DataFrame:
    token = require_env("TIINGO_API_KEY")
    logger.info("Downloading Tiingo EOD prices for %s through %s", ticker, end)
    response = requests.get(
        f"{_BASE_URL}/{ticker}/prices",
        params={"endDate": end.isoformat(), "token": token, "format": "json"},
        timeout=30,
    )
    response.raise_for_status()
    raw = response.json()
    frame = pd.DataFrame(raw)
    if frame.empty:
        return pd.DataFrame(columns=["close"], index=pd.DatetimeIndex([], name="date"))
    frame["date"] = pd.to_datetime(frame["date"], utc=True).dt.tz_localize(None)
    frame = frame.set_index("date").sort_index()
    return frame[["close"]]


def load_tiingo_price_history(ticker: str, as_of_date: pd.Timestamp) -> pd.DataFrame:
    """Tiingo close-price history for `ticker` through `as_of_date`, for
    cross-checking against the primary yfinance/Stooq price series — not a
    replacement for it."""
    today = date.today()
    frame = cached_fetch(
        source="tiingo",
        identifier=ticker.upper(),
        as_of_date=today,
        fetch_fn=lambda: _fetch_tiingo_prices(ticker, today),
    )
    return frame.loc[frame.index <= as_of_date]
