# cache.py

"""On-disk cache for external data-source calls, keyed by
`(source, identifier, as_of_date)`. Cache misses call the given fetch
function and persist the result as parquet; cache hits read from disk and
never re-fetch. This is what keeps a backtest re-run from re-hitting
rate-limited free APIs on every simulated day."""

from collections.abc import Callable
from datetime import date
from pathlib import Path

import pandas as pd

from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

CACHE_ROOT = Path(__file__).parent / ".cache"


def _cache_path(source: str, identifier: str, as_of_date: date) -> Path:
    safe_identifier = identifier.replace("/", "_")
    return CACHE_ROOT / source / f"{safe_identifier}_{as_of_date.isoformat()}.parquet"


def cached_fetch(
    source: str,
    identifier: str,
    as_of_date: date,
    fetch_fn: Callable[[], pd.DataFrame],
) -> pd.DataFrame:
    """Returns the cached frame for `(source, identifier, as_of_date)` if
    present on disk; otherwise calls `fetch_fn()`, persists the result, and
    returns it."""
    path = _cache_path(source, identifier, as_of_date)
    if path.exists():
        logger.debug("Cache hit for %s/%s as of %s", source, identifier, as_of_date)
        return pd.read_parquet(path)

    logger.debug("Cache miss for %s/%s as of %s — fetching", source, identifier, as_of_date)
    frame = fetch_fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
    return frame
