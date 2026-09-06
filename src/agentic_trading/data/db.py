# db.py

"""Durable, queryable record of every external fetch `data/cache.py` has
ever made — a SQLite index alongside the per-file parquet cache.

The parquet cache under `data/.cache/<source>/` is fast and simple but not
queryable as a whole (answering "what data do we actually have across every
source?" means listing directories and parsing filenames). This module adds
a single SQLite file next to it recording the same fetches in one place, so
that data survives independently of the cache layout and can be inspected
with plain SQL. It's an additive record, not a replacement for the parquet
cache: `cache.py` still reads/writes parquet for the hot path and only
calls into this module when a fetch actually happened (a cache miss) —
never on a cache hit, which is what gives "store it once, keep it forever"
semantics for free.

`record_fetch` is idempotent: a `(source, identifier, as_of_date)` row is
only ever written once, enforced by a primary key plus `INSERT OR IGNORE`,
so re-running the same backtest never duplicates or overwrites what's
already stored."""

import io
import sqlite3
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS fetched_data (
    source TEXT NOT NULL,
    identifier TEXT NOT NULL,
    as_of_date TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    row_count INTEGER NOT NULL,
    payload BLOB NOT NULL,
    PRIMARY KEY (source, identifier, as_of_date)
)
"""
_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_fetched_data_source
    ON fetched_data (source, as_of_date)
"""


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.execute(_CREATE_TABLE)
    connection.execute(_CREATE_INDEX)
    return connection


def record_fetch(
    db_path: Path,
    source: str,
    identifier: str,
    as_of_date: date,
    frame: pd.DataFrame,
) -> None:
    """Persists `frame` for `(source, identifier, as_of_date)` if — and only
    if — that key isn't already stored. Never raises: a database write is a
    durability nice-to-have, not something that should take down a fetch
    that already succeeded and is about to be returned to the caller."""
    buffer = io.BytesIO()
    frame.to_parquet(buffer)
    try:
        with _connect(db_path) as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO fetched_data
                    (source, identifier, as_of_date, fetched_at, row_count, payload)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    source,
                    identifier,
                    as_of_date.isoformat(),
                    datetime.now(UTC).isoformat(),
                    len(frame),
                    buffer.getvalue(),
                ),
            )
    except sqlite3.Error:
        logger.warning(
            "Failed to record fetch of %s/%s as of %s in the local database — "
            "the parquet cache still has it, continuing.",
            source,
            identifier,
            as_of_date,
            exc_info=True,
        )


def read_fetch_log(db_path: Path, source: str | None = None) -> pd.DataFrame:
    """Metadata (never payload) for every recorded fetch, optionally
    filtered to one `source` — the data-source status strip in the GUI
    dashboard is the intended consumer, not raw payload retrieval."""
    if not db_path.exists():
        return pd.DataFrame(
            columns=["source", "identifier", "as_of_date", "fetched_at", "row_count"]
        )

    query = "SELECT source, identifier, as_of_date, fetched_at, row_count FROM fetched_data"
    params: tuple[str, ...] = ()
    if source is not None:
        query += " WHERE source = ?"
        params = (source,)

    with _connect(db_path) as connection:
        return pd.read_sql_query(query, connection, params=params)
