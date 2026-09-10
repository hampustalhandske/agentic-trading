# trends_loader.py

"""Google Trends search-interest via pytrends (unofficial scraping client,
no key). Trends publishes a rolling window where the most recent point(s)
are provisional — flagged `isPartial` — and can still change; gating here
drops any `isPartial` row in addition to filtering `<= as_of_date`, so a
provisional point that later gets revised can never leak into a backtest
as if it were final.

Fetches through the real wall-clock date, not `as_of_date`, and caches
under that real date rather than the simulated one — the same "fetch the
bulk series once per real day, filter by simulated date after" pattern
`tiingo_loader.py` uses. A backtest calls this once per
simulated day per ticker; keying the cache on `as_of_date` the way
FRED/SEC do (where the *query result itself* legitimately differs by
as_of_date) would re-hit Trends on every single simulated day for a
series that hasn't actually changed."""

from datetime import date

import pandas as pd
from pytrends.request import TrendReq

from agentic_trading.data.cache import cached_fetch
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_LOOKBACK_DAYS = 90


def _fetch_trends(keyword: str, start: date, end: date) -> pd.DataFrame:
    logger.info("Downloading Google Trends series for %r (%s to %s)", keyword, start, end)
    pytrends = TrendReq(hl="en-US", tz=360)
    pytrends.build_payload([keyword], timeframe=f"{start.isoformat()} {end.isoformat()}")
    frame = pytrends.interest_over_time()
    if frame.empty:
        return pd.DataFrame(columns=["interest"], index=pd.DatetimeIndex([], name="date"))
    frame = frame.loc[~frame["isPartial"]]
    result: pd.DataFrame = frame[[keyword]].rename(columns={keyword: "interest"})
    return result


def load_search_interest(
    keyword: str, as_of_date: pd.Timestamp, lookback_days: int = DEFAULT_LOOKBACK_DAYS
) -> pd.DataFrame:
    """Daily Google Trends search-interest index (0-100) for `keyword`,
    over the trailing `lookback_days` through `as_of_date`, with any
    still-provisional (`isPartial`) point dropped."""
    today = date.today()
    start = (as_of_date - pd.Timedelta(days=lookback_days)).date()
    frame = cached_fetch(
        source="google_trends",
        identifier=f"{keyword}_{start.isoformat()}_{today.isoformat()}",
        as_of_date=today,
        fetch_fn=lambda: _fetch_trends(keyword, start, today),
    )
    return frame.loc[frame.index <= as_of_date]
