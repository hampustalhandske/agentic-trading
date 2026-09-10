# wikipedia_loader.py

"""Wikimedia Pageviews REST API — daily page-view counts per Wikipedia
article, no key required. Wikimedia publishes pageview data with a lag of
roughly 24-48 hours, so a page-view count for calendar day D only becomes
available on D+1 in practice; this loader gates on that availability date
(`date + 1 day <= as_of_date`), not the pageview date itself, the same
"published later than it happened" pattern as SEC/FRED/FINRA.

Fetches through the real wall-clock date, not `as_of_date`, and caches
under that real date rather than the simulated one — the same "fetch the
bulk series once per real day, filter by simulated date after" pattern
`tiingo_loader.py` uses. A backtest calls this once per
simulated day per ticker; keying the cache on `as_of_date` the way
FRED/SEC do (where the *query result itself* legitimately differs by
as_of_date) would re-hit Wikimedia on every single simulated day for a
series that hasn't actually changed."""

from datetime import date

import pandas as pd
import requests

from agentic_trading.data.cache import cached_fetch
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

_BASE_URL = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user"
)
_PUBLICATION_LAG_DAYS = 2
DEFAULT_LOOKBACK_DAYS = 90


def _fetch_pageviews(article: str, start: date, end: date) -> pd.DataFrame:
    logger.info("Downloading Wikimedia pageviews for %r (%s to %s)", article, start, end)
    response = requests.get(
        f"{_BASE_URL}/{article}/daily/{start.strftime('%Y%m%d')}/{end.strftime('%Y%m%d')}",
        headers={"User-Agent": "agentic-trading-research (no contact configured)"},
        timeout=30,
    )
    response.raise_for_status()
    items = response.json().get("items", [])
    frame = pd.DataFrame(items)
    if frame.empty:
        return pd.DataFrame(
            columns=["views", "available_date"], index=pd.DatetimeIndex([], name="date")
        )
    frame["date"] = pd.to_datetime(frame["timestamp"], format="%Y%m%d00")
    frame["available_date"] = frame["date"] + pd.Timedelta(days=_PUBLICATION_LAG_DAYS)
    frame = frame.set_index("date").sort_index()
    return frame[["views", "available_date"]]


def load_pageviews(
    article: str, as_of_date: pd.Timestamp, lookback_days: int = DEFAULT_LOOKBACK_DAYS
) -> pd.DataFrame:
    """Daily Wikipedia page-view counts for `article` (URL-safe article
    title, e.g. `"Apple_Inc."`), over the trailing `lookback_days` through
    `as_of_date`, filtered to rows whose publication-lag-adjusted
    `available_date` is `<= as_of_date`."""
    today = date.today()
    start = (as_of_date - pd.Timedelta(days=lookback_days)).date()
    frame = cached_fetch(
        source="wikimedia_pageviews",
        identifier=f"{article}_{start.isoformat()}_{today.isoformat()}",
        as_of_date=today,
        fetch_fn=lambda: _fetch_pageviews(article, start, today),
    )
    if frame.empty:
        return frame
    return frame.loc[frame["available_date"] <= as_of_date]
