# cache.py

"""On-disk cache for external data-source calls, keyed by
`(source, identifier, as_of_date)`. Cache misses call the given fetch
function and persist the result as parquet; cache hits read from disk and
never re-fetch. This is what keeps a backtest re-run from re-hitting
rate-limited free APIs on every simulated day.

Every cache-miss fetch is also recorded in a local SQLite database (see
`db.py`) alongside the parquet files, so the data survives as a queryable,
durable record independent of the per-file cache layout — write-once,
never overwritten, since `db.record_fetch` only ever inserts a
`(source, identifier, as_of_date)` key it hasn't seen before.

A source that's currently unreachable (a dead endpoint, a network outage)
is remembered for `_FAILURE_TTL` so that a single backtest — which may call
`cached_fetch` for the same flaky `(source, identifier)` on every simulated
day — pays that source's connection/timeout cost once per outage, not once
per simulated day. This does not affect point-in-time correctness (the
failure marker is keyed by real wall-clock time, never `as_of_date`) — it
only avoids re-paying a network timeout for a source that's already known
to be down right now."""

from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd

from agentic_trading.data import db
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

CACHE_ROOT = Path(__file__).parent / ".cache"

_FAILURE_TTL = timedelta(minutes=5)
_recent_failures: dict[tuple[str, str], datetime] = {}


def _now() -> datetime:
    """A seam for tests to override so failure-TTL behavior doesn't depend
    on real sleeps."""
    return datetime.now(UTC)


def _cache_path(source: str, identifier: str, as_of_date: date) -> Path:
    safe_identifier = identifier.replace("/", "_")
    return CACHE_ROOT / source / f"{safe_identifier}_{as_of_date.isoformat()}.parquet"


def _db_path() -> Path:
    return CACHE_ROOT / "fetch_log.db"


def cached_fetch(
    source: str,
    identifier: str,
    as_of_date: date,
    fetch_fn: Callable[[], pd.DataFrame],
    legacy_as_of_dates: tuple[date, ...] = (),
) -> pd.DataFrame:
    """Returns the cached frame for `(source, identifier, as_of_date)` if
    present on disk; otherwise calls `fetch_fn()`, persists the result (to
    both the parquet cache and the SQLite fetch log), and returns it.

    `legacy_as_of_dates` allows a caller with immutable historical data to
    reuse cache files written under an older cache-date convention.

    Raises immediately, without calling `fetch_fn()`, if `(source,
    identifier)` failed within the last `_FAILURE_TTL` — see module
    docstring."""
    path = _cache_path(source, identifier, as_of_date)
    if path.exists():
        logger.debug("Cache hit for %s/%s as of %s", source, identifier, as_of_date)
        return pd.read_parquet(path)

    for legacy_as_of_date in legacy_as_of_dates:
        legacy_path = _cache_path(source, identifier, legacy_as_of_date)
        if legacy_path.exists():
            logger.debug(
                "Legacy cache hit for %s/%s as of %s (requested %s)",
                source,
                identifier,
                legacy_as_of_date,
                as_of_date,
            )
            return pd.read_parquet(legacy_path)

    failure_key = (source, identifier)
    last_failure = _recent_failures.get(failure_key)
    if last_failure is not None and _now() - last_failure < _FAILURE_TTL:
        raise RuntimeError(
            f"{source}/{identifier} failed within the last {_FAILURE_TTL} — "
            "skipping retry until the failure window elapses."
        )

    logger.debug("Cache miss for %s/%s as of %s — fetching", source, identifier, as_of_date)
    try:
        frame = fetch_fn()
    except Exception:
        _recent_failures[failure_key] = _now()
        raise
    _recent_failures.pop(failure_key, None)

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
    db.record_fetch(_db_path(), source, identifier, as_of_date, frame)
    return frame
