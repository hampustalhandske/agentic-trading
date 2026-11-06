# lakehouse_loader.py

"""Every data read a run makes, from the lakehouse's HTTP API (`POST
/v1/sql`, through `data/lakehouse_client.py`): equities/ETFs/indices/
FX/crypto prices, continuous futures prices and roll days, point-in-time
S&P 500 membership, option chains, dividends, CFTC positioning, earnings,
SIC codes, FRED series, BIS central-bank policy rates, Ken French factors,
FINRA short interest and SEC company facts. The service
URL and token come from `LAKEHOUSE_API_URL` and `LAKEHOUSE_API_TOKEN`
(process environment or the repo-root `.env`). Every query passes its
values as `?` parameters.

Inside a `lake_session` every read sees the lake as it stood at one lake
time T. Each query names its tables through `_table_sql`, which turns a
table into the relation it held at T, using the lake's own metadata for
the table (`Lake.describe`, read once when the session opens):

- a "current" table reads `<table>_log` filtered on `scraped_at <= T` and
  deduplicated on the table's key columns (the latest write at or before
  T wins); the loader's own filters apply after that dedupe;
- an append-only table reads `<table>_log` filtered on `scraped_at <= T`
  with no dedupe, and `crypto_klines_1m`, which has no `scraped_at`, is
  filtered on its own row time `open_time <= T`;
- a log view read directly (`sec_company_facts_log`,
  `sec_insider_transactions_log`) is filtered on `scraped_at <= T`;
- a rebuilt-whole table holds only its current build, so it is read
  unpinned and its build time is recorded in the session.

Outside a session each query reads the lake's current, deduplicated
views (latest write wins) as the lake stands at that moment.

This is the only market/alternative-data source a run reads:
`data/loader.py::load_price_history_with_fallback` raises for any ticker
the lake doesn't cover rather than falling back to a live API.

A loader returns `None` only when its query succeeds and finds no rows. A
failed request (`LakehouseError`, connection or timeout errors) propagates
to the caller; inside a session the lake's refusal of a pinned read
(HTTP 409) and a change of lake version raise `LakeTimeError`.

A table in `ALWAYS_POPULATED_TABLES` is rebuilt whole by the lake and
always holds rows, yet a read landing on a rebuild can find it empty
without an error. Every read of such a table as a whole goes through
`_query_table`, which sends an empty read once more after
`EMPTY_READ_RETRY_SECONDS` and raises `EmptyTableReadError` when the
table is empty again. A read filtered to one security, ticker or market
that finds no rows is an answer, not a rebuild, and is never retried."""

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from functools import cache
from typing import Any, Literal, cast, get_args
from weakref import WeakKeyDictionary

import numpy as np
import pandas as pd
from pydantic import BaseModel, PrivateAttr

from agentic_trading.agents.schemas import UnfactoredSplitMove
from agentic_trading.config import _env_value
from agentic_trading.data.instruments import futures_root
from agentic_trading.data.lakehouse_client import Lake, LakehouseError
from agentic_trading.data.point_in_time import assert_point_in_time
from agentic_trading.data.trading_calendar import add_sessions, is_session, next_session
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

_URL_ENV_VAR = "LAKEHOUSE_API_URL"
_TOKEN_ENV_VAR = "LAKEHOUSE_API_TOKEN"


class LakeProvenance(BaseModel):
    """Which lake build a run read, and when: `lake_version` is the git
    commit of the running service, `read_at` the UTC time the status was
    read, and `status_generated_at` the timestamp the lake stamped on its
    own status document (`None` when the lake has not written one yet).
    `adjustment_table_rows` maps each price-adjustment table
    (`equity_adjustment_factors`, `yahoo_split_close_basis`) to its total
    row count.

    When read inside a `lake_session`, `lake_time` is the session's lake
    time T, `rebuilt_whole_builds` maps each rebuilt-whole table the loader
    reads to the build time (its `scraped_at`) the run read, and
    `row_time_pinned_tables` lists the tables pinned on their own row time
    rather than on `scraped_at`. Outside a session they are `None`, `{}`
    and `[]`."""

    lake_version: str
    read_at: datetime
    status_generated_at: datetime | None
    adjustment_table_rows: dict[str, int] = {}
    lake_time: datetime | None = None
    rebuilt_whole_builds: dict[str, datetime] = {}
    row_time_pinned_tables: list[str] = []


class LakeTimeError(RuntimeError):
    """A read cannot be made at the session's lake time: a conflicting
    lake time or `data_end` when joining an open session, a lake time
    before a rebuilt-whole table's build, the lake refusing a pinned read
    (HTTP 409), a lake version change during the session, or a checkpoint
    written at another lake time."""


class HoldoutReadError(RuntimeError):
    """A loader was asked for data known after the open session's
    `data_end`: its knowledge-time argument is later than the cut (see
    `_guard_data_end`). Raised before any query is sent."""


def _required_env(name: str) -> str:
    """The value of environment variable `name`; raises naming the
    variable (never its value) when it is unset or empty."""
    value = _env_value(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set; add it to the process environment or the repo-root .env"
        )
    return value


@cache
def _lake() -> Lake:
    """The process-wide lake client, built on first use from
    `LAKEHOUSE_API_URL` and `LAKEHOUSE_API_TOKEN`. The underlying HTTP
    client is safe to share across threads."""
    return Lake(base_url=_required_env(_URL_ENV_VAR), token=_required_env(_TOKEN_ENV_VAR))


# Every table the loader's queries read.
READ_TABLES: tuple[str, ...] = (
    "bis_observations",
    "cftc_cot_disaggregated",
    "cftc_cot_publication",
    "cftc_cot_tff",
    "crypto_klines_1m",
    "dolthub_earnings_calendar",
    "dolthub_eps_history",
    "dolthub_option_chain",
    "equity_adjustment_factors",
    "equity_daily_from_bars",
    "equity_distributions",
    "famafrench_returns",
    "finra_short_interest",
    "fred_vintages",
    "futures_continuous_daily",
    "instrument_prices",
    "option_chain_history",
    "sec_company_facts_log",
    "sec_filings",
    "sec_insider_transactions_log",
    "sec_sic_classification",
    "sp500_member_count",
    "sp500_membership",
    "sp500_price_coverage",
    "yahoo_split_close_basis",
)

# Tables with no `scraped_at`, pinned on their own row-time column (naive
# UTC) instead.
_ROW_TIME_PIN_COLUMNS = {"crypto_klines_1m": "open_time"}

# Prefix of the lake client's error text for an HTTP 409, the status the
# lake answers a pin it cannot serve with.
_HTTP_409_PREFIX = "HTTP 409:"


class LakeTableMetadata(BaseModel):
    """The lake's description of one table (`Lake.describe`): its `kind`
    ("current", "append-only", "log" or "compatibility view"), the
    `key_columns` a "current" table deduplicates on, whether it is
    `rebuilt_whole` (holding only its latest build), and `last_write`, the
    newest write the lake's status file records (rounded up to the second;
    it can lag the data). `None` when the lake reports no write."""

    kind: str = "current"
    key_columns: list[str] = []
    rebuilt_whole: bool = False
    last_write: datetime | None = None


class LakeSession(BaseModel):
    """One lake time for every read made while the session is open.
    `lake_time` is T (UTC), `data_end` the last knowledge-time date a
    loader may be asked for (`None`: no cut; see `_guard_data_end`),
    `lake_version` the `X-Lakehouse-Version` the session first saw
    (`None` until a response carries one), `rebuilt_whole_builds` maps
    each rebuilt-whole table in `READ_TABLES` that holds rows to its build
    time, and `holdout_violations` lists the message of every read refused
    past `data_end`, kept even when the caller catches the error. The
    table metadata the reads are shaped from, the set of tables a query
    has read, and the always-populated tables `_require_rows` has found
    holding rows are held privately for the life of the session."""

    lake_time: datetime
    data_end: date | None = None
    lake_version: str | None = None
    rebuilt_whole_builds: dict[str, datetime] = {}
    holdout_violations: list[str] = []
    _tables: dict[str, LakeTableMetadata] = PrivateAttr(default_factory=dict)
    _read_tables: set[str] = PrivateAttr(default_factory=set)
    _populated_tables: set[str] = PrivateAttr(default_factory=set)

    def row_time_pinned_tables(self) -> list[str]:
        """The tables in `READ_TABLES` this session pins on a row-time
        column instead of `scraped_at`, sorted."""
        return sorted(
            table
            for table in _ROW_TIME_PIN_COLUMNS
            if table in self._tables and not self._tables[table].rebuilt_whole
        )


# The open session, shared by every thread: analysts read the lake from
# `ThreadPoolExecutor` workers, which a context variable would not reach.
_SESSION: LakeSession | None = None
_SESSION_LOCK = threading.Lock()


def _current_session() -> LakeSession | None:
    with _SESSION_LOCK:
        return _SESSION


def current_lake_time() -> datetime | None:
    """The open `lake_session`'s lake time T (UTC), or `None` when no
    session is open."""
    session = _current_session()
    return None if session is None else session.lake_time


def _guard_data_end(knowledge_time: date, loader: str) -> None:
    """Raises `HoldoutReadError` when a `lake_session` with a `data_end`
    is open and `knowledge_time` is later than it; otherwise does nothing.

    The cut is on knowledge time: `knowledge_time` is the loader's
    `as_of`/`as_of_date`, or `end` for a bar-dated range, the last date
    whose data the read may rest on. Each loader gates its rows on that
    argument by source, so a read at or before `data_end` returns nothing
    known after it. Every public loader with a knowledge-time argument
    calls this first, before any early return or query, so a read past the
    cut never reaches the lake. A `datetime` (including `pd.Timestamp`) is
    compared by its calendar date. `loader` names the caller in the
    error."""
    session = _current_session()
    if session is None or session.data_end is None:
        return
    day = knowledge_time.date() if isinstance(knowledge_time, datetime) else knowledge_time
    if day > session.data_end:
        message = (
            f"{loader} was asked for data known at {day.isoformat()}, after the session's "
            f"data_end {session.data_end.isoformat()}"
        )
        with _SESSION_LOCK:
            session.holdout_violations.append(message)
        raise HoldoutReadError(message)


def _utc(value: Any) -> datetime | None:
    """`value` (a timestamp from a query result; naive means UTC) as an
    aware UTC `datetime`, or `None` when it is null."""
    if value is None or pd.isna(value):
        return None
    stamp = pd.Timestamp(value)
    stamp = stamp.tz_localize(UTC) if stamp.tzinfo is None else stamp.tz_convert(UTC)
    return stamp.to_pydatetime().astimezone(UTC)


def _rebuilt_whole_build(lake: Lake, table: str) -> datetime | None:
    """The build time of rebuilt-whole `table`: the newest `scraped_at` of
    its current rows, `None` when it holds none."""
    frame = cast(pd.DataFrame, lake.sql(f"SELECT max(scraped_at) AS build FROM {table}", []))
    return None if frame.empty else _utc(frame.iloc[0]["build"])


def _open_session(lake_time: datetime | None, data_end: date | None) -> LakeSession:
    """Reads the metadata of every table in `READ_TABLES` and the build of
    each rebuilt-whole one, then fixes T: `lake_time` when given, else 1 s
    after the newest of the tables' `last_write` and the builds. The
    session carries `data_end` as given. Raises
    `LakeTimeError` when an explicit `lake_time` precedes a build, or when
    no default can be formed because the lake reports no write for any of
    the tables."""
    lake = _lake()
    tables = {
        table: LakeTableMetadata.model_validate(lake.describe(table)) for table in READ_TABLES
    }
    builds: dict[str, datetime] = {}
    for table, metadata in tables.items():
        if metadata.rebuilt_whole:
            build = _rebuilt_whole_build(lake, table)
            if build is not None:
                builds[table] = build
    if lake_time is None:
        writes = [
            metadata.last_write.astimezone(UTC)
            for metadata in tables.values()
            if metadata.last_write is not None
        ]
        newest = max([*writes, *builds.values()], default=None)
        if newest is None:
            raise LakeTimeError(
                "cannot choose a default lake time: the lake reports no write for any table the "
                "loader reads; pass an explicit lake time"
            )
        lake_time = newest + timedelta(seconds=1)
    else:
        for table, build in sorted(builds.items()):
            if lake_time < build:
                raise LakeTimeError(
                    f"lake time {lake_time.isoformat()} precedes the build of rebuilt-whole table "
                    f"{table} ({build.isoformat()}); the lake keeps only that build, so a read at "
                    "this lake time cannot be reproduced"
                )
    session = LakeSession(
        lake_time=lake_time,
        data_end=data_end,
        lake_version=lake.last_version,
        rebuilt_whole_builds=builds,
    )
    session._tables = tables
    return session


@contextmanager
def lake_session(
    lake_time: datetime | None = None, *, data_end: date | None = None
) -> Iterator[LakeSession]:
    """Opens a process-wide session in which every loader read sees the
    lake as it stood at one lake time T, and yields it.

    `lake_time` must be timezone-aware (a naive value raises `ValueError`)
    and is held in UTC; when `None`, T is 1 s after the newest `last_write`
    of `READ_TABLES` and the newest rebuilt-whole build, so the same lake
    state gives the same T.

    `data_end` is a hard cut on knowledge time, a separate axis from T:
    T bounds which writes of the lake are visible, `data_end` bounds which
    dates a loader may be asked about. Inside the session every public
    loader with a knowledge-time argument (`as_of`/`as_of_date`, or `end`
    for a bar-dated range) raises `HoldoutReadError` before querying when
    that argument is later than `data_end`; an argument equal to
    `data_end` reads as it would without the cut. Each loader already gates
    its rows on that argument by source (bar date, SEC `filed`, FRED
    vintage release, FINRA estimated dissemination, COT release, option
    snapshot date), so no row known after `data_end` is returned. `None`
    means no cut.

    A call while a session is open joins it and yields the open session.
    A joining call whose `lake_time` differs from the open T raises
    `LakeTimeError`; a joining call without a `data_end` inherits the open
    session's, and one whose `data_end` differs from the open session's
    (including a session opened without one) raises `LakeTimeError`. Only
    the call that opened the session closes it, and when its block ends
    without an error it first raises `HoldoutReadError` if any read was
    refused past `data_end` (a caller's `except` cannot hide one), then
    runs `verify_rebuilt_whole_builds`."""
    global _SESSION
    if lake_time is not None and lake_time.utcoffset() is None:
        raise ValueError(
            f"lake time {lake_time.isoformat()} has no UTC offset; give one, e.g. "
            "2024-06-01T12:00:00+00:00"
        )
    requested = None if lake_time is None else lake_time.astimezone(UTC)
    with _SESSION_LOCK:
        session = _SESSION
        opened = session is None
        if session is None:
            session = _open_session(requested, data_end)
            _SESSION = session
        elif requested is not None and requested != session.lake_time:
            raise LakeTimeError(
                f"a lake session at {session.lake_time.isoformat()} is open; cannot read at "
                f"{requested.isoformat()} inside it"
            )
        elif data_end is not None and data_end != session.data_end:
            open_end = "no data_end" if session.data_end is None else session.data_end.isoformat()
            raise LakeTimeError(
                f"a lake session with {open_end} is open; cannot join it with data_end "
                f"{data_end.isoformat()}"
            )
    try:
        yield session
        if opened:
            if session.holdout_violations:
                raise HoldoutReadError(
                    f"{len(session.holdout_violations)} read(s) past data_end in this session: "
                    + "; ".join(session.holdout_violations)
                )
            verify_rebuilt_whole_builds()
    finally:
        if opened:
            with _SESSION_LOCK:
                _SESSION = None


def verify_rebuilt_whole_builds() -> None:
    """Checks that no table a query of the open session read at its
    current state changed during the session, and raises `LakeTimeError`
    naming each one that did. Those tables carry no `scraped_at` pin, so a
    change would otherwise alter the data a run reads without an error: a
    rebuilt-whole table is compared by its build (re-read from the table),
    a row-time pinned table (`_ROW_TIME_PIN_COLUMNS`) by the `last_write`
    the lake reports for it. A table no query read is not checked. Does
    nothing outside a session. Not subject to the session's `data_end`
    (`_guard_data_end`): it takes no knowledge-time argument and returns
    no market data, only checks build and write times."""
    session = _current_session()
    if session is None:
        return
    with _SESSION_LOCK:
        read = set(session._read_tables)
    lake = _lake()
    changed = []
    for table, metadata in sorted(session._tables.items()):
        if table not in read:
            continue
        if metadata.rebuilt_whole:
            recorded = session.rebuilt_whole_builds.get(table)
            current = _rebuilt_whole_build(lake, table)
            label = "build"
        elif table in _ROW_TIME_PIN_COLUMNS:
            recorded = metadata.last_write
            current = LakeTableMetadata.model_validate(lake.describe(table)).last_write
            label = "last write"
        else:
            continue
        if current != recorded:
            before = "none" if recorded is None else recorded.isoformat()
            after = "none" if current is None else current.isoformat()
            changed.append(f"{table} ({label} {before} when the session opened, now {after})")
    if changed:
        raise LakeTimeError(
            f"tables read at their current state changed during the session at "
            f"{session.lake_time.isoformat()}: {'; '.join(changed)}; the run read more than one "
            "state of them"
        )


def _quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _scraped_at_pin(lake_time: datetime) -> str:
    """The `scraped_at <= T` condition, with T as a literal compared
    directly (the form the lake detects as a pin)."""
    return f"scraped_at <= CAST('{lake_time.astimezone(UTC).isoformat()}' AS TIMESTAMPTZ)"


def _deduplicated_log_sql(table: str, key_columns: list[str], lake_time: datetime) -> str:
    """`table` as it stood at `lake_time`: the rows of `<table>_log` with
    `scraped_at <= lake_time`, keeping the latest write per `key_columns`."""
    if not key_columns:
        raise LakeTimeError(f"the lake reports no key columns for {table}; cannot read it as of T")
    keys = ", ".join(_quote_identifier(column) for column in key_columns)
    return (
        "SELECT * EXCLUDE (_rn) FROM ("
        f"SELECT *, row_number() OVER (PARTITION BY {keys} ORDER BY scraped_at DESC) AS _rn "
        f"FROM {table}_log WHERE {_scraped_at_pin(lake_time)}"
        ") WHERE _rn = 1"
    )


def _pinned_table_sql(session: LakeSession, table: str) -> str:
    """The relation `_table_sql` gives for `table` inside `session`."""
    lake_time = session.lake_time
    if table not in session._tables:
        raise KeyError(f"{table} is not in READ_TABLES")
    metadata = session._tables[table]
    if metadata.rebuilt_whole:
        return table
    row_time_column = _ROW_TIME_PIN_COLUMNS.get(table)
    if row_time_column is not None:
        naive_utc = lake_time.astimezone(UTC).replace(tzinfo=None).isoformat()
        pin = f"{row_time_column} <= CAST('{naive_utc}' AS TIMESTAMP)"
        return f"(SELECT * FROM {table} WHERE {pin})"
    if table.endswith("_log"):
        return f"(SELECT * FROM {table} WHERE {_scraped_at_pin(lake_time)})"
    if metadata.kind == "append-only":
        return f"(SELECT * FROM {table}_log WHERE {_scraped_at_pin(lake_time)})"
    return f"({_deduplicated_log_sql(table, metadata.key_columns, lake_time)})"


def _table_sql(table: str) -> str:
    """The relation a query reads for `table`, to place after `FROM` or
    `JOIN`: the bare table name outside a session, and inside one the
    table as of the session's lake time (see the module docstring for the
    read shape per table). A pinned relation is a parenthesized subquery,
    so a caller that qualifies columns gives it an alias. Inside a session
    the table is recorded as read, for `verify_rebuilt_whole_builds`."""
    session = _current_session()
    if session is None:
        return table
    relation = _pinned_table_sql(session, table)
    with _SESSION_LOCK:
        session._read_tables.add(table)
    return relation


def _record_version(session: LakeSession, version: str | None) -> None:
    """Records `version` as the session's lake version when it has none,
    and raises `LakeTimeError` when it differs from the recorded one."""
    if version is None:
        return
    with _SESSION_LOCK:
        if session.lake_version is None:
            session.lake_version = version
            return
        recorded = session.lake_version
    if version != recorded:
        raise LakeTimeError(
            f"the lake version changed from {recorded} to {version} during the session at "
            f"{session.lake_time.isoformat()}; its reads no longer come from one lake build"
        )


def _query(query: str, params: list[Any]) -> pd.DataFrame:
    """Runs one read-only `SELECT` with `?` placeholders bound to `params`
    and returns every row as a DataFrame. Request failures propagate.
    Inside a session, the lake refusing the read with HTTP 409 raises
    `LakeTimeError` naming T and quoting the lake's detail (the read is
    never retried unpinned), and a response from a different lake version
    than the session's raises `LakeTimeError`."""
    session = _current_session()
    lake = _lake()
    try:
        frame = cast(pd.DataFrame, lake.sql(query, params))
    except LakehouseError as error:
        if session is not None and str(error).startswith(_HTTP_409_PREFIX):
            raise LakeTimeError(
                f"the lake refused a read pinned at lake time {session.lake_time.isoformat()}: "
                f"{error}"
            ) from error
        raise
    if session is not None:
        _record_version(session, lake.last_version)
    return frame


# Tables the lake rebuilds whole that always hold rows: an empty read of
# one as a whole is a read that landed on a rebuild, not an answer.
ALWAYS_POPULATED_TABLES: frozenset[str] = frozenset(
    {
        "cftc_cot_publication",
        "equity_adjustment_factors",
        "sp500_membership",
        "yahoo_split_close_basis",
    }
)

# Seconds `_query_table` waits before sending an empty read of a table in
# `ALWAYS_POPULATED_TABLES` again; a rebuild's swap takes far less.
EMPTY_READ_RETRY_SECONDS = 1.0


class EmptyTableReadError(RuntimeError):
    """A read of a table in `ALWAYS_POPULATED_TABLES` as a whole came back
    with no rows twice, `EMPTY_READ_RETRY_SECONDS` apart, so the table is
    empty or still being rebuilt and no answer read from it can be
    trusted."""


def _query_table(table: str, query: str, params: list[Any]) -> pd.DataFrame:
    """Runs `query` with `params` through `_query`, as a read of `table`
    as a whole, and returns its rows.

    When `table` is in `ALWAYS_POPULATED_TABLES` and the result has no
    rows, waits `EMPTY_READ_RETRY_SECONDS` (read at call time) and sends
    the same query once more; a second empty result raises
    `EmptyTableReadError` naming the table. For any other table an empty
    result is returned from the one read. A query against an
    always-populated table must therefore return no rows only when the
    table is empty (an aggregate over it adds `HAVING count(*) > 0`). An
    exception from `_query` propagates as is and is never retried."""
    frame = _query(query, params)
    if not frame.empty or table not in ALWAYS_POPULATED_TABLES:
        return frame
    time.sleep(EMPTY_READ_RETRY_SECONDS)
    frame = _query(query, params)
    if frame.empty:
        raise EmptyTableReadError(
            f"{table} read empty twice, {EMPTY_READ_RETRY_SECONDS} s apart; it always holds "
            "rows, so it is empty or still being rebuilt"
        )
    return frame


def _require_rows(table: str) -> None:
    """Checks that `table`, one of `ALWAYS_POPULATED_TABLES`, holds rows,
    with one `SELECT 1 ... LIMIT 1` read through `_query_table` (so an
    empty table is read again once and then raises
    `EmptyTableReadError`). Inside a `lake_session` a table found holding
    rows is not read again for the rest of the session; outside one every
    call reads. Returns nothing."""
    session = _current_session()
    if session is not None:
        with _SESSION_LOCK:
            if table in session._populated_tables:
                return
    _query_table(table, f"SELECT 1 AS present FROM {_table_sql(table)} LIMIT 1", [])
    if session is not None:
        with _SESSION_LOCK:
            session._populated_tables.add(table)


def lake_provenance() -> LakeProvenance:
    """The lake's version and the UTC time its status was read, from
    `Lake.status()` and the service's health payload, plus the row count
    of each adjustment table from `adjustment_table_row_count`. Inside a
    `lake_session` it also carries the session's lake time, rebuilt-whole
    builds and row-time-pinned tables. Raises `EmptyAdjustmentTableError`
    when either adjustment table is empty. Not subject to the session's
    `data_end` (`_guard_data_end`): it takes no knowledge-time argument and
    returns provenance, not market data."""
    lake = _lake()
    status = lake.status()
    read_at = datetime.now(UTC)
    generated = status.get("generated_at")
    session = _current_session()
    return LakeProvenance(
        lake_version=str(lake.health()["version"]),
        read_at=read_at,
        status_generated_at=datetime.fromisoformat(generated) if generated else None,
        adjustment_table_rows={
            table: adjustment_table_row_count(table) for table in get_args(AdjustmentTable)
        },
        lake_time=None if session is None else session.lake_time,
        rebuilt_whole_builds={} if session is None else dict(session.rebuilt_whole_builds),
        row_time_pinned_tables=[] if session is None else session.row_time_pinned_tables(),
    )


class _LakeTimeMemo[V]:
    """Values read from one lake client at one lake time, keyed by name.
    `values()` returns the dictionary for the current client and the open
    session's lake time (`None` outside a session); a read at a different
    lake time than the stored one starts from an empty dictionary, so no
    value is reused across lake times."""

    def __init__(self) -> None:
        self._entries: WeakKeyDictionary[Lake, tuple[datetime | None, dict[str, V]]] = (
            WeakKeyDictionary()
        )

    def values(self) -> dict[str, V]:
        lake = _lake()
        lake_time = current_lake_time()
        entry = self._entries.get(lake)
        if entry is None or entry[0] != lake_time:
            entry = (lake_time, {})
            self._entries[lake] = entry
        return entry[1]


AdjustmentTable = Literal["equity_adjustment_factors", "yahoo_split_close_basis"]

# Row count per adjustment table, for the lake client and lake time that answered.
_ADJUSTMENT_TABLE_ROWS: _LakeTimeMemo[int] = _LakeTimeMemo()


class EmptyAdjustmentTableError(EmptyTableReadError):
    """An adjustment table (`equity_adjustment_factors` or
    `yahoo_split_close_basis`) read empty twice (see `_query_table`), so
    every price read through it would come back unadjusted."""


def adjustment_table_row_count(table: AdjustmentTable) -> int:
    """The total row count of adjustment table `table`, from a `count(*)`
    query sent through `_query_table`, which returns no row for an empty
    table, so an empty read is sent once more after
    `EMPTY_READ_RETRY_SECONDS`. Read once per table, lake client and lake
    time (see `_LakeTimeMemo`) and kept while reads stay at that lake
    time. Raises `EmptyAdjustmentTableError` when the table reads empty
    twice (that result is not kept, so a later call queries again) and
    `ValueError` when `table` is not an adjustment table. Not subject to
    the session's `data_end` (`_guard_data_end`): it takes no
    knowledge-time argument and returns a row count recorded as
    provenance, never row values."""
    if table not in get_args(AdjustmentTable):
        raise ValueError(f"{table!r} is not an adjustment table")
    cached = _ADJUSTMENT_TABLE_ROWS.values()
    if table not in cached:
        query = f"SELECT count(*) AS row_count FROM {_table_sql(table)} HAVING count(*) > 0"
        try:
            frame = _query_table(table, query, [])
        except EmptyTableReadError as error:
            raise EmptyAdjustmentTableError(
                f"{error}; prices read through it would be unadjusted"
            ) from error
        cached[table] = int(frame.iloc[0]["row_count"])
    return cached[table]


class IndexMember(BaseModel):
    """One S&P 500 member on a date: the lake's `security_id` for the
    company and the `ticker` it traded under on that date."""

    security_id: int
    ticker: str


def load_sp500_member_securities_asof(as_of_date: date) -> list[IndexMember] | None:
    """Point-in-time S&P 500 membership on `as_of_date` as securities, one
    `IndexMember` per `security_id`, sorted by ticker.

    Reads `sp500_price_coverage`, one row per piece of a membership span:
    every piece carries its span's `start_date` and `end_date`, and its
    own `security_id` with the half-open range it covers, `piece_start`
    to `piece_end` (null: still a member). A row counts when
    `piece_start <= as_of_date < piece_end`, so a span whose member
    security changes counts one piece on any date. When several counted
    rows share one security (a rename recorded as two overlapping spans),
    the row with the latest `start_date` names the ticker. A counted span without a
    `security_id` is left out and named in one WARNING. Returns `None`
    when no row counts."""
    _guard_data_end(as_of_date, "load_sp500_member_securities_asof")
    query = f"""
        SELECT security_id, ticker
        FROM (
            SELECT
                security_id,
                ticker,
                ROW_NUMBER() OVER (
                    PARTITION BY security_id,
                        CASE WHEN security_id IS NULL THEN ticker END
                    ORDER BY start_date DESC, ticker
                ) AS security_rank
            FROM {_table_sql("sp500_price_coverage")}
            WHERE piece_start <= ?
              AND (piece_end IS NULL OR piece_end > ?)
        ) AS ranked
        WHERE security_rank = 1
        ORDER BY ticker, security_id
    """
    as_of = as_of_date.isoformat()
    result = _query(query, [as_of, as_of])
    if result.empty:
        return None
    unresolved = result["security_id"].isna()
    if unresolved.any():
        logger.warning(
            "S&P 500 members on %s without a security_id, left out: %s",
            as_of,
            sorted(result.loc[unresolved, "ticker"].tolist()),
        )
    resolved = result.loc[~unresolved]
    members = [
        IndexMember(security_id=int(security_id), ticker=str(ticker))
        for security_id, ticker in zip(resolved["security_id"], resolved["ticker"], strict=True)
    ]
    return members or None


# The count columns `load_sp500_member_count` returns.
_MEMBER_COUNT_COLUMNS = ("member_lines", "nominal_members", "shortfall")


def load_sp500_member_count(start: date, end: date) -> pd.DataFrame | None:
    """The lake's daily S&P 500 member count from `sp500_member_count`,
    one row per date from `start` to `end` inclusive, on an ascending
    `DatetimeIndex` named `date`, with the lake's `member_lines`,
    `nominal_members` and `shortfall` columns as nullable integers
    (`Int64`).

    For reporting only: it states each day's shortfall of priced members
    against the nominal 500 and gates nothing. Returns `None` when no row
    is dated in the range. Self-asserts point-in-time on the index against
    `end` before returning."""
    _guard_data_end(end, "load_sp500_member_count")
    query = f"""
        SELECT date, {", ".join(_MEMBER_COUNT_COLUMNS)}
        FROM {_table_sql("sp500_member_count")}
        WHERE date BETWEEN ? AND ?
        ORDER BY date
    """
    frame = _query(query, [start.isoformat(), end.isoformat()])
    if frame.empty:
        return None
    result = frame.set_index(pd.DatetimeIndex(pd.to_datetime(frame["date"]), name="date"))
    result = result[list(_MEMBER_COUNT_COLUMNS)].astype("Int64").sort_index()
    assert_point_in_time(result, end)
    return result


# Shares received per share held in the split sizes the lake records; a
# reverse split is the reciprocal. Splits smaller than 3-for-2 move the
# price by less than many ordinary trading days, so they are not listed.
_SPLIT_SHARE_RATIOS = (1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 8.0, 10.0, 15.0, 20.0)
# Largest distance, in log price, between a one-day move and a split
# ratio for the move to count as split-sized: room for the day's ordinary
# market move on top of the split.
_SPLIT_MOVE_LOG_TOLERANCE = 0.05

# Every split-sized move `_warn_unfactored_split_moves` has flagged in this
# process, keyed by (security_id, date) to its price ratio; analysts read
# prices from worker threads, so every access holds the lock.
_UNFACTORED_SPLIT_MOVES: dict[tuple[int, date], float] = {}
_UNFACTORED_SPLIT_MOVES_LOCK = threading.Lock()


def unfactored_split_move_registry() -> list[UnfactoredSplitMove]:
    """Every distinct (security_id, date) split-sized move flagged by a
    `load_security_log_returns` or `load_security_daily_bars` read since
    the registry was last cleared, sorted by security then date. A move
    flagged again by a later read keeps the ratio it was first flagged
    with."""
    with _UNFACTORED_SPLIT_MOVES_LOCK:
        entries = sorted(_UNFACTORED_SPLIT_MOVES.items())
    return [
        UnfactoredSplitMove(security_id=security_id, as_of_date=day, ratio=ratio)
        for (security_id, day), ratio in entries
    ]


def clear_unfactored_split_move_registry() -> None:
    """Empties the registry `unfactored_split_move_registry` reads."""
    with _UNFACTORED_SPLIT_MOVES_LOCK:
        _UNFACTORED_SPLIT_MOVES.clear()


def unfactored_split_moves(
    returns: dict[int, pd.Series], factor_dates: dict[int, set[date]]
) -> list[tuple[int, date, float]]:
    """One-day moves in `returns` (log returns per `security_id`, on a
    `DatetimeIndex`) whose size matches a forward or reverse split in
    `_SPLIT_SHARE_RATIOS` within `_SPLIT_MOVE_LOG_TOLERANCE`, on a date
    that is not in the security's `factor_dates` (the `effective_date`s of
    its adjustment-factor rows). Each entry is `(security_id, date,
    ratio)`, where `ratio` is the day's price over the previous day's
    (0.5 for an unadjusted 2-for-1 split), ordered by security and date.
    A move on a factor date is never returned: the factor row is the
    split's adjustment."""
    split_logs = np.log(np.array(_SPLIT_SHARE_RATIOS))
    flagged: list[tuple[int, date, float]] = []
    for security_id in sorted(returns):
        covered = factor_dates.get(security_id, set())
        series = returns[security_id].sort_index()
        for stamp, log_return in series.items():
            day = pd.Timestamp(cast(Any, stamp)).date()
            if day in covered or not np.isfinite(log_return):
                continue
            distance = np.min(np.abs(abs(float(log_return)) - split_logs))
            if distance <= _SPLIT_MOVE_LOG_TOLERANCE:
                flagged.append((security_id, day, float(np.exp(log_return))))
    return flagged


def _adjusted_security_bars(
    security_ids: list[int], start: date, end: date, with_open_and_volume: bool
) -> pd.DataFrame:
    """The `equity_daily_from_bars` bars of `security_ids` dated `start`
    to `end` inclusive, adjusted with `equity_adjustment_factors`, as one
    row per (`security_id`, `date`) sorted by security then date. One
    query serves every id; it reads the bars' `open` and `volume` only
    when `with_open_and_volume` is true.

    A bar's raw price is `official_close`, or `close` when that is null;
    a bar whose raw price is null or not positive is left out. Each bar
    takes the `cum_factor` of the security's factor row with the smallest
    `effective_date` strictly after the bar's date (1.0 when there is
    none); bars and factors are joined on `security_id`, never on ticker.
    Columns:

    - `security_id`, `date` (a `Timestamp`);
    - `close`: the raw price times the factor;
    - `raw_close`: the raw price;
    - `open` (only with `with_open_and_volume`): the bar's `open` times
      the factor, NaN when that is null or not positive;
    - `volume` (only with `with_open_and_volume`): the bar's `volume`
      unadjusted, NaN when null or not positive;
    - `factor_date`: the `effective_date` of the factor row joined (null
      when none);
    - `log_return`: the log of the bar's `close` over the previous kept
      bar's, only when that bar is on the NYSE session immediately before
      (a gap never yields a multi-day return, and the first bar in the
      range has none), NaN otherwise.

    The frame is empty when no bar is kept. Raises
    `EmptyAdjustmentTableError` when `equity_adjustment_factors` has no
    rows at all. Logs and registers split-sized moves through
    `_warn_unfactored_split_moves`; the frame is not changed by them."""
    adjustment_table_row_count("equity_adjustment_factors")
    placeholders = ", ".join("?" for _ in security_ids)
    bar_columns = ", open, volume" if with_open_and_volume else ""
    open_and_volume = (
        """
               CASE WHEN bars.open > 0
                    THEN bars.open * COALESCE(factors.cum_factor, 1.0) END AS open,
               CASE WHEN bars.volume > 0 THEN bars.volume END AS volume,"""
        if with_open_and_volume
        else ""
    )
    query = f"""
        SELECT bars.security_id, bars.date,
               bars.raw_close * COALESCE(factors.cum_factor, 1.0) AS close,
               bars.raw_close,{open_and_volume}
               factors.effective_date AS factor_date
        FROM (
            SELECT security_id, date, COALESCE(official_close, close) AS raw_close{bar_columns}
            FROM {_table_sql("equity_daily_from_bars")}
            WHERE security_id IN ({placeholders})
              AND date BETWEEN ? AND ?
              AND COALESCE(official_close, close) > 0
        ) AS bars
        ASOF LEFT JOIN (
            SELECT security_id, effective_date, cum_factor
            FROM {_table_sql("equity_adjustment_factors")}
            WHERE security_id IN ({placeholders})
              AND effective_date > ?
        ) AS factors
          ON factors.security_id = bars.security_id
         AND bars.date < factors.effective_date
        ORDER BY bars.security_id, bars.date
    """
    ids = [int(security_id) for security_id in security_ids]
    params = [*ids, start.isoformat(), end.isoformat(), *ids, start.isoformat()]
    frame = _query(query, params)
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.drop_duplicates(["security_id", "date"]).reset_index(drop=True)
    previous_date = frame.groupby("security_id")["date"].shift(1)
    session_after = {
        stamp: pd.Timestamp(next_session(stamp.date()))
        for stamp in pd.DatetimeIndex(previous_date.dropna().unique())
    }
    consecutive = previous_date.map(session_after) == frame["date"]
    log_price = pd.Series(np.log(frame["close"].to_numpy(dtype=np.float64)), index=frame.index)
    log_return = log_price - log_price.groupby(frame["security_id"]).shift(1)
    frame["log_return"] = log_return.where(consecutive)
    _warn_unfactored_split_moves(_log_return_series(frame), frame)
    return frame


def _log_return_series(frame: pd.DataFrame) -> dict[int, pd.Series]:
    """The non-NaN `log_return` values of an `_adjusted_security_bars`
    frame, one series per `security_id` on an ascending `DatetimeIndex`.
    A security with no returns has no key."""
    returns = frame.loc[frame["log_return"].notna()]
    result: dict[int, pd.Series] = {}
    for security_id, group in returns.groupby("security_id"):
        result[int(cast(int, security_id))] = pd.Series(
            group["log_return"].to_numpy(dtype=np.float64),
            index=pd.DatetimeIndex(group["date"]),
            name="log_return",
        ).sort_index()
    return result


def load_security_log_returns(
    security_ids: list[int], start: date, end: date
) -> dict[int, pd.Series]:
    """Daily split- and distribution-adjusted log returns per security,
    keyed by `security_id`, from `equity_daily_from_bars` bars dated
    `start` to `end` inclusive. One query serves every id in
    `security_ids`.

    A bar's price is `official_close`, or `close` when that is null,
    times the `cum_factor` of the security's `equity_adjustment_factors`
    row with the smallest `effective_date` strictly after the bar's date
    (1.0 when there is none); bars and factors are joined on
    `security_id`, never on ticker. A bar whose price is null or not
    positive counts as missing. The return on day d is the log of d's
    price over the previous bar's price, and exists only when the
    security has bars on d and on the NYSE session immediately before d,
    both inside the range; a gap never yields a multi-day return, and the
    first bar in the range has none.

    Each series has an ascending `DatetimeIndex`. A security with no
    returns in the range has no key. Factor rows dated after `end` set
    the price level but cancel in every return, so no return depends on
    data after `end`. Self-asserts point-in-time on every series against
    `end` before returning.

    Raises `EmptyAdjustmentTableError` when `equity_adjustment_factors`
    has no rows at all; a security with no factor rows of its own is read
    unadjusted. Logs one WARNING listing every split-sized move
    `unfactored_split_moves` finds on a date with no factor row for the
    security, using the factor `effective_date`s the price read joined,
    and adds each to `unfactored_split_move_registry`; the returned series
    are not changed."""
    _guard_data_end(end, "load_security_log_returns")
    if not security_ids:
        return {}
    frame = _adjusted_security_bars(security_ids, start, end, with_open_and_volume=False)
    if frame.empty:
        return {}
    result = _log_return_series(frame)
    for series in result.values():
        assert_point_in_time(series, end)
    return result


# The fields `load_security_daily_bars` returns per security, in column order.
_DAILY_BAR_FIELDS = ("close", "raw_close", "open", "volume", "log_return")


def load_security_daily_bars(
    security_ids: list[int], start: date, end: date
) -> pd.DataFrame | None:
    """Daily bars per security from `equity_daily_from_bars`, dated
    `start` to `end` inclusive, on the price basis and consecutive-session
    rule of `load_security_log_returns`. One query serves every id in
    `security_ids`.

    The frame has an ascending `DatetimeIndex` named `date` (every date
    on which at least one of the securities has a kept bar) and
    `MultiIndex` columns named `security_id` and `field`, with one group
    per security that has a kept bar in the range and these fields:

    - `close`: the bar's `official_close` (or `close` when that is null)
      times the `cum_factor` of the security's first
      `equity_adjustment_factors` row with an `effective_date` strictly
      after the bar's date (1.0 when there is none), joined on
      `security_id`;
    - `raw_close`: the same price before the factor;
    - `open`: the bar's `open` times the same factor, NaN when the open
      is null or not positive;
    - `volume`: the bar's volume, unadjusted, NaN when null or not
      positive;
    - `log_return`: the value `load_security_log_returns` gives for that
      security and day, NaN on every day it gives none.

    A bar whose price is null or not positive is missing: every field of
    that security is NaN on that day, as on a day it has no bar.

    Back-adjustment with the factors known at the lake time puts every
    later split and distribution, including those after `end`, into the
    earlier `close` and `open` levels, so those levels are not the prices
    that traded. Returns and ratios therefore use `close` and
    `log_return`; any level-based rule (a price filter, a round-number
    level) uses `raw_close`. `volume` is raw.

    Returns `None` when `security_ids` is empty or no security has a kept
    bar in the range. Raises `EmptyAdjustmentTableError` when
    `equity_adjustment_factors` has no rows at all; a security with no
    factor rows of its own is read unadjusted. Split-sized moves on dates
    without a factor row are logged and registered as in
    `load_security_log_returns`. Self-asserts point-in-time on the index
    against `end` before returning."""
    _guard_data_end(end, "load_security_daily_bars")
    if not security_ids:
        return None
    frame = _adjusted_security_bars(security_ids, start, end, with_open_and_volume=True)
    if frame.empty:
        return None
    per_security = {
        int(cast(int, security_id)): group.set_index("date")[list(_DAILY_BAR_FIELDS)].astype(
            np.float64
        )
        for security_id, group in frame.groupby("security_id")
    }
    result = pd.concat(per_security, axis=1, names=["security_id", "field"], sort=True)
    result.index = pd.DatetimeIndex(result.index, name="date")
    assert_point_in_time(result, end)
    return result


def _warn_unfactored_split_moves(returns: dict[int, pd.Series], frame: pd.DataFrame) -> None:
    """Logs one WARNING naming each `(security_id, date, ratio)` that
    `unfactored_split_moves` flags in `returns`, and adds each to the
    registry `unfactored_split_move_registry` reads. `frame` is the joined
    bar/factor read, whose non-null `factor_date` values are each
    security's factor `effective_date`s."""
    dated = frame.dropna(subset=["factor_date"])
    factor_dates: dict[int, set[date]] = {
        int(cast(int, security_id)): {pd.Timestamp(day).date() for day in group["factor_date"]}
        for security_id, group in dated.groupby("security_id")
    }
    flagged = unfactored_split_moves(returns, factor_dates)
    if flagged:
        with _UNFACTORED_SPLIT_MOVES_LOCK:
            for security_id, day, ratio in flagged:
                _UNFACTORED_SPLIT_MOVES.setdefault((security_id, day), ratio)
        listed = ", ".join(
            f"({security_id}, {day.isoformat()}, {ratio:.4f})"
            for security_id, day, ratio in flagged
        )
        logger.warning(
            "split-sized one-day moves with no adjustment factor on that date "
            "(security_id, date, ratio): %s",
            listed,
        )


def split_adjustment_factor(ticker: str, quote_date: date) -> float:
    """Factor that moves an option-chain row quoted on `quote_date` onto
    the basis of `ticker`'s `instrument_prices` close: the product of
    `1 / ratio` over the ticker's rows in the lake's
    `yahoo_split_close_basis` with an ex-date after `quote_date` (`ratio`
    is old shares per new share, so a 4-for-1 split is 0.25 and
    contributes 4.0). Returns 1.0 when no such row exists; a missing or
    non-positive `ratio` contributes nothing.

    `dolthub_option_chain` strikes and premiums are quoted as traded and
    never adjusted, while `instrument_prices` closes are split-adjusted by
    Yahoo; dividing a chain row by this factor puts its strike and
    premiums on the same basis as the underlying's close. A row whose
    `in_close_basis` is false is a split Yahoo's close does not reflect,
    so it is skipped and the strike lands on the close's actual basis;
    rows where it is true or null count. Spin-offs Yahoo writes as
    fractional splits count too, because the close is adjusted for them.

    Raises `EmptyAdjustmentTableError` when `yahoo_split_close_basis` has
    no rows at all; a ticker with no rows of its own gets 1.0.

    Not subject to the session's `data_end` (`_guard_data_end`):
    `quote_date` is not a knowledge time but the date of a chain row an
    option loader already read under its guarded `as_of_date`. The split
    ratios it reads have ex-dates after `quote_date`, up to the lake time
    T, and serve only to put that row on the split-adjusted close basis;
    they carry no signal value of their own."""
    adjustment_table_row_count("yahoo_split_close_basis")
    query = f"""
        SELECT ratio
        FROM {_table_sql("yahoo_split_close_basis")}
        WHERE ticker = ?
          AND ex_date > ?
          AND in_close_basis IS NOT FALSE
    """
    frame = _query(query, [ticker, quote_date.isoformat()])
    factor = 1.0
    for ratio in frame["ratio"]:
        if ratio and ratio > 0:
            factor /= float(ratio)
    return factor


def _apply_split_adjustment(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Rescales an option-chain frame's price-denominated columns
    (`strike`, `bid`, `ask`, `gamma`, `vega`, `theta`, `rho`) onto
    `instrument_prices`' continuously split-adjusted basis, per each
    row's own `date`. `delta` and `implied_volatility` are dimensionless
    and untouched. A no-op for any ticker/date with no later split."""
    frame = frame.copy()
    for quote_date in frame["date"].dt.date.unique():
        factor = split_adjustment_factor(ticker, quote_date)
        if factor == 1.0:
            continue
        mask = frame["date"].dt.date == quote_date
        frame.loc[mask, ["strike", "bid", "ask"]] /= factor
        frame.loc[mask, "gamma"] /= factor**2
        frame.loc[mask, ["vega", "theta", "rho"]] /= factor
    return frame


# Underlyings whose end-of-day chains `option_chain_history` holds from
# 2008, covering the years before `dolthub_option_chain` begins.
_OPTION_CHAIN_HISTORY_UNDERLYINGS = frozenset({"SPY", "QQQ", "IWM"})

# First `dolthub_option_chain` date per underlying, for the lake client and
# lake time that answered; a `None` value records an underlying with no rows.
_FIRST_DOLTHUB_CHAIN_DATES: _LakeTimeMemo[date | None] = _LakeTimeMemo()


def _first_dolthub_chain_date(underlying: str) -> date | None:
    """The earliest snapshot `date` in `dolthub_option_chain` for
    `underlying` as of the session's lake time (the current table outside
    a session), or `None` when it has no rows. Read once per underlying,
    lake client and lake time (see `_LakeTimeMemo`) and kept while reads
    stay at that lake time."""
    cached = _FIRST_DOLTHUB_CHAIN_DATES.values()
    if underlying not in cached:
        query = f"""
            SELECT MIN(date) AS first
            FROM {_table_sql("dolthub_option_chain")}
            WHERE underlying = ?
        """
        frame = _query(query, [underlying])
        first = None if frame.empty else frame.iloc[0]["first"]
        cached[underlying] = None if pd.isna(first) else pd.Timestamp(first).date()
    return cached[underlying]


def _option_chain_source(underlying: str, as_of_date: date) -> tuple[str, str]:
    """The chain table a read of `underlying` as of `as_of_date` queries,
    and that table's snapshot-date column. An underlying in
    `_OPTION_CHAIN_HISTORY_UNDERLYINGS` reads `option_chain_history`
    (`snapshot_date`) when `as_of_date` precedes its first
    `dolthub_option_chain` date or it has no dolthub rows; every other
    read uses `dolthub_option_chain` (`date`)."""
    if underlying in _OPTION_CHAIN_HISTORY_UNDERLYINGS:
        first = _first_dolthub_chain_date(underlying)
        if first is None or as_of_date < first:
            return "option_chain_history", "snapshot_date"
    return "dolthub_option_chain", "date"


# Calendar days before `as_of_date` within which
# `load_options_snapshot_from_lakehouse` looks for the latest snapshot by
# default: chain coverage is not daily-dense.
DEFAULT_OPTIONS_SNAPSHOT_LOOKBACK_DAYS = 10


def load_options_snapshot_from_lakehouse(
    ticker: str, as_of_date: date, lookback_days: int = DEFAULT_OPTIONS_SNAPSHOT_LOOKBACK_DAYS
) -> pd.DataFrame | None:
    """Real historical options-chain rows (strike, expiration, bid, ask,
    implied_volatility, delta, gamma, theta, vega, rho — no open interest,
    no volume, confirmed absent from the source) for `ticker`, from the
    most recent available snapshot date on or before `as_of_date` within
    `lookback_days` — coverage is real but not daily-dense (roughly
    every-other-trading-day for AAPL), so a same-day match isn't
    guaranteed even for a covered ticker.

    Rows come from the table `_option_chain_source` picks for `ticker`
    and `as_of_date`: `dolthub_option_chain`, or `option_chain_history`
    for SPY, QQQ and IWM before dolthub coverage begins. A history row's
    `snapshot_date` (a 16:00 ET end-of-day snapshot) is returned as
    `date`, and its `bid`/`ask` are the prices; its `mark` and
    `last_trade_price` are never read. Both tables yield the same
    columns and dtypes.

    Returns `None` if `ticker` has no coverage at all or the nearest
    snapshot is more than `lookback_days` stale — the caller reports a
    flat, zero-confidence signal in that case rather than treat a miss as
    fatal. `strike`/`bid`/`ask`/`gamma`/`vega`/`theta`/`rho` are rescaled
    onto `instrument_prices`' continuously split-adjusted basis via
    `_apply_split_adjustment` before returning, since the raw chain is
    archival and never itself adjusted for a later split. Rows are sorted
    on every returned column. Self-asserts point-in-time on `date` before
    returning."""
    _guard_data_end(as_of_date, "load_options_snapshot_from_lakehouse")
    window_start = (as_of_date - pd.Timedelta(days=lookback_days)).isoformat()
    window_end = as_of_date.isoformat()
    table, date_column = _option_chain_source(ticker, as_of_date)
    query = f"""
        SELECT {date_column} AS date, expiration, option_type, strike, bid, ask,
               implied_volatility, delta, gamma, theta, vega, rho
        FROM {_table_sql(table)}
        WHERE underlying = ?
          AND {date_column} BETWEEN ? AND ?
          AND {date_column} = (
              SELECT MAX({date_column})
              FROM {_table_sql(table)}
              WHERE underlying = ?
                AND {date_column} BETWEEN ? AND ?
          )
        ORDER BY expiration, option_type, strike, bid, ask, implied_volatility, delta, gamma,
                 theta, vega, rho
    """
    frame = _query(query, [ticker, window_start, window_end, ticker, window_start, window_end])
    if frame.empty:
        return None
    frame["date"] = pd.to_datetime(frame["date"])
    frame["expiration"] = pd.to_datetime(frame["expiration"])
    frame = _apply_split_adjustment(frame, ticker)
    assert_point_in_time(frame, as_of_date, date_column="date")
    return frame


# CFTC contract-market codes for the tickers the loader covers, keyed by
# the `cftc_cot_disaggregated` (commodities) and `cftc_cot_tff` (FX and
# rate futures) tables.
_COT_DISAGGREGATED_CODES = {
    "GC=F": "088691",
    "CL=F": "067651",
    "SI=F": "084691",
    "NG=F": "023651",
    "HG=F": "085692",
}
_COT_TFF_CODES = {
    "EURUSD=X": "099741",
    "GBPUSD=X": "096742",
    "USDJPY=X": "097741",
    "AUDUSD=X": "232741",
    "ZT=F": "042601",
    "ZN=F": "043602",
    "ZB=F": "020601",
}
# Treasury ETFs have no COT category of their own; each reads the
# positioning of the Treasury future it tracks. Every other ticker the COT
# tables cover (commodities, FX and rate futures) is its own COT ticker.
DEFAULT_COT_TICKER_BY_ASSET: dict[str, str] = {"SHY": "ZT=F", "IEF": "ZN=F", "TLT": "ZB=F"}
# Calendar days of reports before `as_of_date` that
# `load_cot_positioning_from_lakehouse` returns by default: a little over
# a year of weekly reports.
DEFAULT_COT_LOOKBACK_DAYS = 400


def load_cot_positioning_from_lakehouse(
    ticker: str, as_of_date: date, lookback_days: int = DEFAULT_COT_LOOKBACK_DAYS
) -> pd.Series | None:
    """Net speculative positioning as a fraction of open interest, from
    CFTC Commitment of Traders reports: the `net_positioning_ratio`
    column of `load_cot_reports_from_lakehouse`, indexed by
    `report_date`. Returns `None` if `ticker` has no COT report available
    by `as_of_date`."""
    _guard_data_end(as_of_date, "load_cot_positioning_from_lakehouse")
    reports = _cot_reports(ticker, as_of_date, lookback_days)
    return None if reports is None else reports["net_positioning_ratio"]


def load_cot_reports_from_lakehouse(
    ticker: str, as_of_date: date, lookback_days: int = DEFAULT_COT_LOOKBACK_DAYS
) -> pd.DataFrame | None:
    """The CFTC Commitment of Traders reports for `ticker` available at
    the close of `as_of_date`, dated within `lookback_days` calendar days
    before it. Net speculative positioning is `managed_money_long -
    managed_money_short` (the disaggregated report's commodity category)
    for commodities, `leveraged_funds_long - leveraged_funds_short` (the
    TFF report's closest equivalent) for FX/rates, divided by
    `open_interest_all`. A given `ticker` only ever appears in one of the
    two tables (commodities in `cftc_cot_disaggregated`, FX/rate-futures
    in `cftc_cot_tff`), selected by its CFTC contract-market code.

    Point-in-time: each report is gated on its row in
    `cftc_cot_publication`, joined on `report_date`. A report whose
    `published_at` is set (always 15:30 New York time, before the close)
    is available from its `published_date`; one without a `published_at`
    is available only from the day after its `published_date`. A report
    with no publication row, or a null `published_date` (every report
    before 2000), is never available.

    Returns a frame indexed by `report_date` (datetime64, ascending) with
    columns `net_positioning_ratio` and `published_date` (the CFTC
    release date, datetime64), or `None` if `ticker` has no COT coverage
    at all — callers should treat that as "no positioning read
    available," not an empty-but-valid frame. Self-asserts point-in-time
    on the `report_date` index before returning."""
    _guard_data_end(as_of_date, "load_cot_reports_from_lakehouse")
    return _cot_reports(ticker, as_of_date, lookback_days)


def _cot_reports(ticker: str, as_of_date: date, lookback_days: int) -> pd.DataFrame | None:
    """The query behind `load_cot_reports_from_lakehouse`, without the
    `data_end` guard its public callers apply first. For a ticker with a
    contract-market code, `_require_rows` first checks that
    `cftc_cot_publication` holds rows, so a read landing on its rebuild
    raises `EmptyTableReadError` instead of finding no available report;
    inside a `lake_session` that check is made once per session."""
    window_start = (as_of_date - pd.Timedelta(days=lookback_days)).isoformat()
    availability_cutoff = as_of_date.isoformat()
    queries = [
        (
            "cftc_cot_disaggregated",
            _COT_DISAGGREGATED_CODES,
            "(managed_money_long - managed_money_short)",
        ),
        (
            "cftc_cot_tff",
            _COT_TFF_CODES,
            "(leveraged_funds_long - leveraged_funds_short)",
        ),
    ]
    for table, codes, net_position_expr in queries:
        code = codes.get(ticker)
        if code is None:
            continue
        _require_rows("cftc_cot_publication")
        query = f"""
            SELECT cot.report_date,
                   {net_position_expr} / cot.open_interest_all AS net_positioning_ratio,
                   pub.published_date
            FROM {_table_sql(table)} AS cot
            JOIN {_table_sql("cftc_cot_publication")} AS pub ON pub.report_date = cot.report_date
            WHERE cot.cftc_contract_market_code = ?
              AND cot.report_date BETWEEN ? AND ?
              AND cot.open_interest_all > 0
              AND (
                  (pub.published_at IS NOT NULL AND pub.published_date <= ?)
                  OR (pub.published_at IS NULL AND pub.published_date < ?)
              )
            ORDER BY cot.report_date
        """
        frame = _query(
            query,
            [code, window_start, availability_cutoff, availability_cutoff, availability_cutoff],
        )
        if not frame.empty:
            frame["report_date"] = pd.to_datetime(frame["report_date"])
            frame["published_date"] = pd.to_datetime(frame["published_date"])
            reports = frame.set_index("report_date")[["net_positioning_ratio", "published_date"]]
            assert_point_in_time(reports, as_of_date)
            return reports
    return None


# `equity_distributions.distribution_type` of an ordinary recurring cash
# dividend; specials, spin-offs and unclassified large payments carry
# other types.
_REGULAR_CASH = "regular_cash"


def load_dividends_from_lakehouse(ticker: str, as_of_date: date) -> pd.Series | None:
    """Regular cash dividends per share for `ticker`, from the
    `equity_distributions` table: rows whose `distribution_type` is
    `regular_cash`, as `amount_split_adjusted` indexed by ex-date
    (datetime64, ascending), with `ex_date <= as_of_date`.

    Every amount is on one split-consistent basis, so a stock split
    never reads as a dividend cut. Special cash distributions, spin-offs
    and other non-recurring payments are excluded. Gating on the ex-date
    is safe because a dividend is declared before its ex-date, so it is
    public by then. Returns `None` when no rows qualify. Self-asserts
    point-in-time on the ex-date index before returning."""
    _guard_data_end(as_of_date, "load_dividends_from_lakehouse")
    query = f"""
        SELECT ex_date, amount_split_adjusted
        FROM {_table_sql("equity_distributions")}
        WHERE ticker = ? AND distribution_type = ? AND ex_date <= ?
        ORDER BY ex_date
    """
    frame = _query(query, [ticker, _REGULAR_CASH, as_of_date.isoformat()])
    if frame.empty:
        return None
    frame["ex_date"] = pd.to_datetime(frame["ex_date"])
    series = frame.set_index("ex_date")["amount_split_adjusted"]
    assert_point_in_time(series, as_of_date)
    return series


def load_earnings_surprise_from_lakehouse(
    ticker: str, as_of_date: date
) -> tuple[float, float, pd.Timestamp] | None:
    """The most recent real EPS surprise for `ticker` known at the close
    of `as_of_date`: `(reported, estimate, announcement_date)` for the
    quarter with the latest announcement (`dolthub_earnings_calendar.
    date`) available by then. Every announcement is available from the
    next NYSE session after its date, whatever its `when` ("Before
    market open" included): `earnings` can publish after the close
    (`stage_a.decisions.SOURCES`), and the announcement date is the
    signal's date-only publication date, which must fall before the
    decision date. The returned `announcement_date` is the calendar date
    itself. Joins the two tables on a 75-day window after
    `dolthub_eps_history.period_end_date` — a generous margin around the
    typical ~29-30 day real reporting lag, not a tight estimate.

    Returns `None` if `ticker` has no earnings history at all, or no
    quarter's announcement is available yet as of `as_of_date`.
    Self-asserts point-in-time on the availability date before
    returning."""
    _guard_data_end(as_of_date, "load_earnings_surprise_from_lakehouse")
    query = f"""
        SELECT h.reported, h.estimate, c.date AS announcement_date
        FROM (
            SELECT *
            FROM {_table_sql("dolthub_eps_history")}
            WHERE act_symbol = ? AND reported IS NOT NULL AND estimate IS NOT NULL
        ) h
        JOIN (
            SELECT *
            FROM {_table_sql("dolthub_earnings_calendar")}
            WHERE act_symbol = ?
        ) c
        ON c.act_symbol = h.act_symbol
           AND CAST(c.date AS DATE) BETWEEN CAST(h.period_end_date AS DATE)
               AND date_add(CAST(h.period_end_date AS DATE), INTERVAL 75 DAY)
        WHERE CAST(c.date AS DATE) < ?
        ORDER BY c.date DESC
    """
    frame = _query(query, [ticker, ticker, as_of_date.isoformat()])
    if frame.empty:
        return None
    frame["announcement_date"] = pd.to_datetime(frame["announcement_date"])
    frame["available_date"] = [
        pd.Timestamp(next_session(announced.date())) for announced in frame["announcement_date"]
    ]
    frame = frame[frame["available_date"] <= pd.Timestamp(as_of_date)]
    if frame.empty:
        return None
    assert_point_in_time(frame, as_of_date, date_column="available_date")
    row = frame.iloc[0]
    return float(row["reported"]), float(row["estimate"]), pd.Timestamp(row["announcement_date"])


def load_sic_code_from_lakehouse(ticker: str) -> str | None:
    """SIC (Standard Industrial Classification) code for `ticker` from
    SEC EDGAR, via `sec_sic_classification`; `None` if `ticker` has no SIC
    record (mostly non-equity ETF/future/FX tickers, which have no SEC
    filing to classify).

    Not point-in-time: the lake holds one current classification per
    ticker and no history of reclassifications, so a backtest date sees
    the code as of the lake's last scrape (at or before the session's lake
    time inside a `lake_session`). A company reclassified after
    that date is grouped by its later industry. This is a known look-ahead
    limit of SIC-based pair screening, with no `as_of_date` gate possible
    until the lake carries classification history.

    Not subject to the session's `data_end` (`_guard_data_end`): it takes
    no knowledge-time argument. The read is current-state, bounded only by
    the lake time T, so inside a session with a `data_end` it can return a
    classification made after `data_end`."""
    frame = _query(
        f"SELECT sic FROM {_table_sql('sec_sic_classification')} WHERE ticker = ?", [ticker]
    )
    if frame.empty or pd.isna(frame.iloc[0]["sic"]):
        return None
    return str(frame.iloc[0]["sic"])


def same_sic_candidate_pairs(tickers: list[str]) -> list[tuple[str, str]]:
    """Every pair within `tickers` that share a real SEC SIC code — a
    genuine industry-classification candidate list for pairs-cointegration
    screening, replacing a hand-picked "looks same-sector to me" list.
    Tickers with no SIC record (non-equities, or an equity outside SEC
    EDGAR's coverage) can't form a pair here, so this only ever
    under-generates candidates. The codes are current, not point-in-time
    (see `load_sic_code_from_lakehouse`). Not subject to the session's
    `data_end` (`_guard_data_end`): it reads only through
    `load_sic_code_from_lakehouse`, which takes no knowledge-time
    argument, so the pairs can reflect classifications made after
    `data_end`."""
    sic_by_ticker: dict[str, str] = {}
    for ticker in tickers:
        sic = load_sic_code_from_lakehouse(ticker)
        if sic is not None:
            sic_by_ticker[ticker] = sic

    pairs: list[tuple[str, str]] = []
    tickers_with_sic = sorted(sic_by_ticker)
    for i, ticker_a in enumerate(tickers_with_sic):
        for ticker_b in tickers_with_sic[i + 1 :]:
            if sic_by_ticker[ticker_a] == sic_by_ticker[ticker_b]:
                pairs.append((ticker_a, ticker_b))
    return pairs


# Lake symbol in `crypto_klines_1m` for each crypto ticker the loader
# serves: Binance USDT spot one-minute klines. USDT trades within a few
# basis points of USD, so the USDT price stands in for the USD price.
_CRYPTO_KLINE_SYMBOLS = {
    "BTC-USD": "BTCUSDT",
    "ETH-USD": "ETHUSDT",
}
# The NYSE regular session in America/New_York: the first and last
# one-minute kline start times of each session.
_SESSION_FIRST_MINUTE = "09:30:00"
_SESSION_LAST_MINUTE = "15:59:00"


def _equity_price_rows(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    """Daily `ticker`/`date`/`open`/`close`/`raw_close`/`volume` rows from
    `instrument_prices` for `tickers` between `start` and `end`
    inclusive: `close` is `adj_close`, `open` is the raw open scaled by
    `adj_close / close`, `raw_close` the unadjusted close.

    A row whose raw open lies outside its raw `[low, high]` range
    (boundaries included as valid), or whose raw low exceeds its raw
    high, has an inconsistent bar, so its `open` is NaN; the rest of the
    row is kept. A row with a null raw low or high keeps its open. Logs
    one WARNING naming each ticker with masked rows and its count when
    any row is masked."""
    placeholders = ", ".join("?" for _ in tickers)
    query = f"""
        SELECT ticker, date,
               CASE WHEN close > 0 THEN open * adj_close / close ELSE NULL END AS open,
               open AS raw_open,
               high AS raw_high,
               low AS raw_low,
               adj_close AS close,
               close AS raw_close,
               volume
        FROM {_table_sql("instrument_prices")}
        WHERE ticker IN ({placeholders})
          AND date BETWEEN ? AND ?
        ORDER BY ticker, date
    """
    frame = _query(query, [*tickers, start.isoformat(), end.isoformat()])
    raw_open = frame["raw_open"].astype(float)
    raw_high = frame["raw_high"].astype(float)
    raw_low = frame["raw_low"].astype(float)
    inconsistent = (raw_open < raw_low) | (raw_open > raw_high) | (raw_low > raw_high)
    if inconsistent.any():
        frame.loc[inconsistent, "open"] = np.nan
        masked_by_ticker = frame.loc[inconsistent, "ticker"].value_counts().sort_index()
        logger.warning(
            "Masked the open of rows outside their raw low-high range: %s",
            ", ".join(f"{ticker}: {count}" for ticker, count in masked_by_ticker.items()),
        )
    return frame[["ticker", "date", "open", "close", "raw_close", "volume"]]


def _crypto_price_rows(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    """One `ticker`/`date`/`open`/`close`/`raw_close`/`volume` row per
    NYSE session between `start` and `end` inclusive for each crypto
    ticker in `tickers` (keys of `_CRYPTO_KLINE_SYMBOLS`), built from
    `crypto_klines_1m` minutes whose `open_time` (naive UTC, the minute's
    start) falls in the session in America/New_York: `open` is the open
    of the minute starting at `_SESSION_FIRST_MINUTE`, `close` and
    `raw_close` the close of the minute starting at
    `_SESSION_LAST_MINUTE`, `volume` the summed volume of the minutes
    from `_SESSION_FIRST_MINUTE` to `_SESSION_LAST_MINUTE`. A day without
    a `_SESSION_LAST_MINUTE` kline, or that is not an NYSE session, has
    no row. Only minutes inside the session on days in the range are
    read; the aggregation runs in the lake."""
    symbols = [_CRYPTO_KLINE_SYMBOLS[ticker] for ticker in tickers]
    placeholders = ", ".join("?" for _ in symbols)
    query = f"""
        SELECT symbol,
               CAST(local_time AS DATE) AS date,
               max(open) FILTER (WHERE CAST(local_time AS TIME) = CAST(? AS TIME)) AS open,
               max(close) FILTER (WHERE CAST(local_time AS TIME) = CAST(? AS TIME)) AS close,
               sum(volume) AS volume
        FROM (
            SELECT symbol, open, close, volume,
                   timezone(?, open_time AT TIME ZONE 'UTC') AS local_time
            FROM {_table_sql("crypto_klines_1m")}
            WHERE symbol IN ({placeholders})
              AND open_time >= CAST(? AS TIMESTAMP)
              AND open_time < CAST(? AS TIMESTAMP)
        )
        WHERE CAST(local_time AS TIME) BETWEEN CAST(? AS TIME) AND CAST(? AS TIME)
        GROUP BY symbol, CAST(local_time AS DATE)
        HAVING max(close) FILTER (WHERE CAST(local_time AS TIME) = CAST(? AS TIME)) IS NOT NULL
        ORDER BY symbol, date
    """
    params = [
        _SESSION_FIRST_MINUTE,
        _SESSION_LAST_MINUTE,
        _DECISION_TIMEZONE,
        *symbols,
        start.isoformat(),
        (end + pd.Timedelta(days=1)).isoformat(),
        _SESSION_FIRST_MINUTE,
        _SESSION_LAST_MINUTE,
        _SESSION_LAST_MINUTE,
    ]
    frame = _query(query, params)
    if frame.empty:
        return frame
    ticker_by_symbol = {symbol: ticker for ticker, symbol in _CRYPTO_KLINE_SYMBOLS.items()}
    frame["ticker"] = frame["symbol"].map(ticker_by_symbol)
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame[[is_session(stamp.date()) for stamp in frame["date"]]]
    frame["raw_close"] = frame["close"]
    return frame[["ticker", "date", "open", "close", "raw_close", "volume"]]


def load_price_history_from_lakehouse(
    tickers: list[str], start: date, end: date
) -> pd.DataFrame | None:
    """Returns a `MultiIndex(ticker, field)` frame (`open`/`close`/
    `raw_close`/`volume`/`log_return`) for every ticker in `tickers` that the lake actually
    covers, or `None` if the lake covers none of them —
    `load_price_history_with_fallback` is what turns an incomplete result
    into a clear error naming the uncovered ticker(s). A failed request
    propagates.

    Crypto tickers (`_CRYPTO_KLINE_SYMBOLS`) are built from one-minute
    klines by `_crypto_price_rows`, one bar per NYSE session. Decisions
    are taken at the US close, so crypto is sampled at 16:00
    America/New_York (the close of the 15:59 minute) rather than at
    Yahoo's 23:59 UTC daily close, keeping it synchronous with every
    other asset's close.

    Every other ticker reads `instrument_prices` and uses `adj_close`
    (dividend/split-adjusted) rather than raw `close` — correct for a
    multi-year backtest where corporate actions actually occur — and
    scales the raw `open` by the same `adj_close / close` factor so the
    open the engine fills at and the close it marks at are on one price
    basis (an unadjusted open against an adjusted close would book the
    whole dividend-adjustment gap as a loss on every fill). A row whose
    raw open lies outside its raw low-high range, or whose low exceeds
    its high, gets a NaN `open` and keeps every other field (see
    `_equity_price_rows`, which logs one WARNING per call with the masked
    count per ticker); the engine does not fill on such a day.
    Self-asserts point-in-time on the combined frame's date index against
    `end` before returning.
    """
    _guard_data_end(end, "load_price_history_from_lakehouse")
    crypto = [ticker for ticker in tickers if ticker in _CRYPTO_KLINE_SYMBOLS]
    others = [ticker for ticker in tickers if ticker not in _CRYPTO_KLINE_SYMBOLS]
    parts = []
    if others:
        parts.append(_equity_price_rows(others, start, end))
    if crypto:
        parts.append(_crypto_price_rows(crypto, start, end))
    parts = [part for part in parts if not part.empty]
    if not parts:
        return None
    frame = pd.concat(parts, ignore_index=True)

    frame["date"] = pd.to_datetime(frame["date"])
    covered_tickers = set(frame["ticker"].unique())
    missing = [ticker for ticker in tickers if ticker not in covered_tickers]
    if missing:
        logger.info("Lakehouse has no price coverage for %s in this range", missing)

    per_ticker: dict[str, pd.DataFrame] = {}
    for ticker, group in frame.groupby("ticker"):
        indexed = (
            group.set_index("date")[["open", "close", "raw_close", "volume"]].sort_index().copy()
        )
        # Some lakehouse rows have a real, non-zero
        # `close` but a corrupted `adj_close` of exactly 0.0 — a
        # zero/negative price here would silently produce
        # log(0) = -inf rather than NaN, which .dropna() downstream would
        # NOT catch. Mask non-positive prices to NaN before taking the
        # log, so a corrupted row degrades to "missing," the same way a
        # genuine data gap already does, not to a silent -inf/NaN
        # poisoning every consumer's arithmetic.
        indexed.loc[indexed["close"] <= 0.0, "close"] = np.nan
        indexed.loc[indexed["open"] <= 0.0, "open"] = np.nan
        # `raw_close` is the unadjusted print: option strikes, marks and
        # settlement live on that basis, never the dividend-adjusted one.
        indexed.loc[indexed["raw_close"] <= 0.0, "raw_close"] = np.nan
        # Spot FX and indices carry no volume; a zero means "no ADV", not
        # "no trading", so it becomes NaN and impact/participation skip it.
        indexed["volume"] = indexed["volume"].astype(float)
        indexed.loc[indexed["volume"] <= 0.0, "volume"] = np.nan
        indexed["log_return"] = np.log(indexed["close"] / indexed["close"].shift(1))
        per_ticker[str(ticker)] = indexed

    combined = pd.concat(per_ticker, axis=1)
    assert_point_in_time(combined, end)
    return combined


def load_futures_price_history_from_lakehouse(
    tickers: list[str], start: date, end: date
) -> pd.DataFrame | None:
    """A `MultiIndex(ticker, field)` frame (`open`/`close`/`raw_close`/
    `volume`/`log_return`/`roll`) for every futures ticker in `tickers`
    (`data/instruments.futures_root`; `CL=F` reads root `CL`) that
    `futures_continuous_daily` covers between `start` and `end` inclusive,
    or `None` if it covers none of them. Raises `ValueError` for a ticker
    that is not a future.

    The frame holds the kept bars: those with a positive
    `ratio_adjusted_close`. `close` is a level built by chaining the
    returns of `ratio_adjusted_close` from one kept bar to the next, from
    the first kept bar's `unadjusted_close`. The return into a kept bar is
    set to 0 when any row since the previous kept bar, this bar included,
    is a roll day whose `roll_gap_known` is false, since that change of the
    adjusted close still holds the gap between the two contracts. The
    level's returns are therefore the returns of the contract held, and a
    later roll's back-adjustment of earlier rows does not change it.
    `open` is that same level: the continuous series carries closes only,
    so a future fills at the close of its fill day. `raw_close` is the
    front contract's `unadjusted_close`, `volume` is NaN (the series has
    none), and `log_return` is the log of each level over the previous
    one (NaN on the first bar).

    `roll` is 1.0 on a kept bar whose `front_contract` differs from the
    previous kept bar's, and 0.0 otherwise, so a roll on a dropped row
    lands on the next kept bar together with that bar's zeroed return. The
    first bar compares against the last kept bar before `start`, read for
    that purpose, and takes 0.0 when there is none.

    Point-in-time on the bar date, as prices are: only bars dated on or
    before `end` are read. Self-asserts point-in-time on the frame's date
    index against `end` before returning."""
    _guard_data_end(end, "load_futures_price_history_from_lakehouse")
    ticker_by_root: dict[str, str] = {}
    for ticker in tickers:
        root = futures_root(ticker)
        if root is None:
            raise ValueError(f"{ticker} is not a futures ticker")
        ticker_by_root[root] = ticker
    placeholders = ", ".join("?" for _ in ticker_by_root)
    query = f"""
        SELECT root, date, front_contract, unadjusted_close, ratio_adjusted_close,
               is_roll_day, roll_gap_known
        FROM (
            SELECT root, date, front_contract, unadjusted_close, ratio_adjusted_close,
                   is_roll_day, roll_gap_known,
                   max(CASE WHEN date < ? AND ratio_adjusted_close > 0 THEN date END)
                       OVER (PARTITION BY root) AS previous_kept
            FROM {_table_sql("futures_continuous_daily")}
            WHERE root IN ({placeholders})
              AND date <= ?
        ) AS daily
        WHERE date >= COALESCE(previous_kept, ?)
        ORDER BY root, date
    """
    frame = _query(query, [start.isoformat(), *ticker_by_root, end.isoformat(), start.isoformat()])
    if frame.empty:
        return None
    frame["date"] = pd.to_datetime(frame["date"])

    per_ticker: dict[str, pd.DataFrame] = {}
    for lake_root, group in frame.groupby("root", sort=True):
        adjusted_all = group["ratio_adjusted_close"].astype(float)
        kept = (adjusted_all > 0.0).to_numpy(dtype=bool)
        unknown_gap = group["is_roll_day"].eq(True).to_numpy(dtype=bool) & group[
            "roll_gap_known"
        ].eq(False).fillna(False).to_numpy(dtype=bool)
        # Unknown-gap rows up to each kept bar; a rise between two kept bars
        # means the span into the later one holds an unknown gap.
        unknown_gaps_so_far = np.cumsum(unknown_gap)[kept]
        bars = group.loc[kept]
        adjusted = adjusted_all.to_numpy(dtype=np.float64)[kept]
        daily = np.zeros(len(adjusted))
        daily[1:] = adjusted[1:] / adjusted[:-1] - 1.0
        daily[1:][np.diff(unknown_gaps_so_far) > 0] = 0.0
        fronts = bars["front_contract"]
        previous_fronts = fronts.shift(1)
        roll = fronts.ne(previous_fronts) & fronts.notna() & previous_fronts.notna()

        in_range = (bars["date"] >= pd.Timestamp(start)).to_numpy(dtype=bool)
        if not in_range.any():
            continue
        bars = bars.loc[in_range]
        daily = daily[in_range]
        daily[0] = 0.0
        index = pd.DatetimeIndex(bars["date"])
        unadjusted = bars["unadjusted_close"].astype(float)
        unadjusted = unadjusted.where(unadjusted > 0.0)
        first_adjusted = float(adjusted[in_range][0])
        base = float(unadjusted.iloc[0]) if pd.notna(unadjusted.iloc[0]) else first_adjusted
        close = pd.Series(base * np.cumprod(1.0 + daily), index=index)
        per_ticker[ticker_by_root[str(lake_root)]] = pd.DataFrame(
            {
                "open": close,
                "close": close,
                "raw_close": unadjusted.to_numpy(),
                "volume": np.nan,
                "log_return": np.log(close / close.shift(1)),
                "roll": roll.loc[in_range].astype(float).to_numpy(),
            },
            index=index,
        )

    if not per_ticker:
        return None
    combined = pd.concat(per_ticker, axis=1)
    assert_point_in_time(combined, end)
    return combined


def load_raw_close_from_lakehouse(
    ticker: str, as_of_date: date, lookback_days: int
) -> float | None:
    """`ticker`'s unadjusted `instrument_prices` close (the `raw_close`
    basis of `load_price_history_from_lakehouse`, which option strikes are
    rescaled onto) on its last session on or before `as_of_date` within
    `lookback_days` calendar days. `None` when the lake has no positive
    close for `ticker` in that window. Self-asserts point-in-time on
    `date` before returning."""
    _guard_data_end(as_of_date, "load_raw_close_from_lakehouse")
    window_start = (as_of_date - pd.Timedelta(days=lookback_days)).isoformat()
    query = f"""
        SELECT date, close
        FROM {_table_sql("instrument_prices")}
        WHERE ticker = ?
          AND date BETWEEN ? AND ?
          AND close > 0
        ORDER BY date DESC
        LIMIT 1
    """
    frame = _query(query, [ticker, window_start, as_of_date.isoformat()])
    if frame.empty:
        return None
    assert_point_in_time(frame, as_of_date, date_column="date")
    return float(frame["close"].iloc[0])


def load_fred_series_from_lakehouse(series_id: str, as_of_date: date) -> pd.Series | None:
    """A FRED macro series (`DGS3MO`/`DGS2`/`DGS10`/`VIXCLS`/`CPIAUCSL`/
    `BAMLH0A0HYM2`/`BAMLC0A0CM` — the same series `macro_loader.
    SUPPORTED_SERIES` names) as it stood at the close of `as_of_date`:
    the `value` column of `load_fred_vintages_from_lakehouse`, indexed by
    observation date. A version is known at `known_at`, the later of its
    vintage date (`realtime_start`) and its observation date, because a
    value cannot be known before its own day ends; only versions with
    `known_at` strictly before `as_of_date` are used. Returns `None` if
    no observation is known by then."""
    _guard_data_end(as_of_date, "load_fred_series_from_lakehouse")
    vintages = _fred_vintages(series_id, as_of_date)
    return None if vintages is None else vintages["value"]


def load_fred_vintages_from_lakehouse(series_id: str, as_of_date: date) -> pd.DataFrame | None:
    """The FRED series `series_id` as it stood at the close of
    `as_of_date`, from the `fred_vintages` table (`series_id`, `date`,
    `realtime_start`, `value`; one row per published version of an
    observation).

    A version is known at `known_at`, the later of its vintage date
    (`realtime_start`) and its observation date, because a value cannot
    be known before its own day ends. Each observation date carries its
    latest version (by `realtime_start`) among those with `known_at`
    strictly before `as_of_date`: FRED can publish after the 16:00 ET
    close (the H.15 daily rates around 16:15 ET, VIXCLS the next
    morning), so a version known on `as_of_date` counts from the next
    day. An observation with no version known by then is left out.

    Returns a frame indexed by observation date (sorted and unique) with
    columns `value`, `realtime_start` (the date that version was
    published) and `known_at` (the date it became known), both
    datetime64, or `None` if no rows remain. Self-asserts point-in-time
    on the date index before returning."""
    _guard_data_end(as_of_date, "load_fred_vintages_from_lakehouse")
    return _fred_vintages(series_id, as_of_date)


def _fred_vintages(series_id: str, as_of_date: date) -> pd.DataFrame | None:
    """The query behind `load_fred_vintages_from_lakehouse`, without the
    `data_end` guard its public callers apply first."""
    query = f"""
        SELECT date, value, realtime_start, known_at
        FROM (
            SELECT date, value, realtime_start, known_at, ROW_NUMBER() OVER (
                PARTITION BY date
                ORDER BY realtime_start DESC
            ) AS rn
            FROM (
                SELECT date, value, realtime_start,
                    GREATEST(realtime_start, date) AS known_at
                FROM {_table_sql("fred_vintages")}
                WHERE series_id = ?
            )
            WHERE known_at < ?
        )
        WHERE rn = 1
        ORDER BY date
    """
    frame = _query(query, [series_id, as_of_date.isoformat()])
    if frame.empty:
        return None
    frame["date"] = pd.to_datetime(frame["date"])
    frame["realtime_start"] = pd.to_datetime(frame["realtime_start"])
    frame["known_at"] = pd.to_datetime(frame["known_at"])
    vintages = frame.set_index("date")[["value", "realtime_start", "known_at"]]
    assert_point_in_time(vintages, as_of_date)
    return vintages


# `famafrench_returns` holds every Ken French dataset as long rows
# (`dataset`, `frequency`, `date`, `portfolio`, `value`); each model is one
# daily US factor dataset and the portfolios that make up its factors, in
# the column order the factor-pod analysts consume.
_KEN_FRENCH_DATASET_BY_MODEL = {
    "3factor": "F-F_Research_Data_Factors_daily",
    "5factor": "F-F_Research_Data_5_Factors_2x3_daily",
}
_KEN_FRENCH_FACTORS_BY_MODEL = {
    "3factor": ("Mkt-RF", "SMB", "HML", "RF"),
    "5factor": ("Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"),
}
_KEN_FRENCH_FREQUENCY = "daily"
# Ken French posts each month's daily factors after that month closes, so
# the end of the following month is the conservative public date.
_KEN_FRENCH_PUBLICATION_LAG_MONTHS = 1


# BIS reference area of each currency's central-bank policy rate in the
# `WS_CBPOL` dataflow (`XM` is the euro area, the ECB's rate).
_BIS_POLICY_RATE_AREAS = {
    "AUD": "AU",
    "CAD": "CA",
    "CHF": "CH",
    "EUR": "XM",
    "GBP": "GB",
    "JPY": "JP",
    "NOK": "NO",
    "NZD": "NZ",
    "SEK": "SE",
    "USD": "US",
}
_BIS_POLICY_RATE_DATAFLOW = "WS_CBPOL"
_BIS_DAILY = "D"
# Calendar days before `start` that `load_policy_rates_from_lakehouse`
# also reads, so the rate in force on `start` is there when `start` has
# no observation of its own (BIS stores no weekend or holiday rows).
POLICY_RATE_LOOKBACK_DAYS = 14


def load_policy_rates_from_lakehouse(
    currencies: list[str], start: date, end: date
) -> pd.DataFrame | None:
    """Central-bank policy rates for `currencies` (ISO codes) as annual
    decimals, from the daily (`freq = 'D'`) `WS_CBPOL` observations in
    `bis_observations` (per cent per year, divided by 100), dated from
    `POLICY_RATE_LOOKBACK_DAYS` calendar days before `start` to `end`
    inclusive: one column per currency in the order given, one row per
    observation date on an ascending `DatetimeIndex` named `date`, NaN
    where a currency has no observation that day. A daily policy rate is
    the rate in force on its date. Returns `None` when no currency has an
    observation in the window; raises `ValueError` for a currency without
    a BIS reference area here (`_BIS_POLICY_RATE_AREAS`).

    Point-in-time on the observation date: BIS publishes no release date
    and a rate is announced before it takes effect, so an observation
    dated on or before `end` is read as known on its date; inside a
    `lake_session` the value is the one the lake held at its lake time
    (BIS revisions are kept as later versions). Self-asserts
    point-in-time on the index against `end` before returning."""
    _guard_data_end(end, "load_policy_rates_from_lakehouse")
    wanted = list(dict.fromkeys(currencies))
    currency_by_area: dict[str, str] = {}
    for currency in wanted:
        area = _BIS_POLICY_RATE_AREAS.get(currency)
        if area is None:
            raise ValueError(f"no BIS policy-rate reference area for currency {currency}")
        currency_by_area[area] = currency
    placeholders = ", ".join("?" for _ in currency_by_area)
    query = f"""
        SELECT ref_area, date, value
        FROM {_table_sql("bis_observations")}
        WHERE dataflow = ?
          AND freq = ?
          AND ref_area IN ({placeholders})
          AND date BETWEEN ? AND ?
          AND value IS NOT NULL
        ORDER BY date, ref_area
    """
    window_start = start - timedelta(days=POLICY_RATE_LOOKBACK_DAYS)
    frame = _query(
        query,
        [
            _BIS_POLICY_RATE_DATAFLOW,
            _BIS_DAILY,
            *currency_by_area,
            window_start.isoformat(),
            end.isoformat(),
        ],
    )
    if frame.empty:
        return None
    frame["date"] = pd.to_datetime(frame["date"])
    frame["currency"] = frame["ref_area"].map(currency_by_area)
    rates = frame.pivot(index="date", columns="currency", values="value").astype(float) / 100.0
    rates = rates.reindex(columns=wanted).sort_index()
    rates.index.name = "date"
    rates.columns.name = None
    assert_point_in_time(rates, end)
    return rates


def ken_french_public_date(day: date) -> date:
    """The date the Ken French daily factor row for `day` counts as
    public: the last day of the month `_KEN_FRENCH_PUBLICATION_LAG_MONTHS`
    after `day`'s month, the date `load_fama_french_factors_from_lakehouse`
    gates each row on."""
    month = pd.Period(day, freq="M") + _KEN_FRENCH_PUBLICATION_LAG_MONTHS
    return month.end_time.date()


def load_fama_french_factors_from_lakehouse(
    as_of_date: date, model: str = "3factor"
) -> pd.DataFrame | None:
    """Daily Fama-French factor returns for `model` ("3factor" or
    "5factor"), from the `famafrench_returns` table's matching daily US
    factor dataset. A day's values are returned only once public, which
    is the last day of the month after that day's month
    (`_KEN_FRENCH_PUBLICATION_LAG_MONTHS`) falling on or before
    `as_of_date`. The values are today's revised history, because the
    source keeps no daily vintages. The lake stores returns as plain
    percentage points (0.09 meaning 0.09%, Ken French's own
    convention); the result is on a 0-1 return scale with Ken French's
    `Mkt-RF`/`SMB`/`HML`/`RF`(/`RMW`/`CMA`) column names. Returns `None`
    if the table has no rows or `model` isn't one of the two supported
    names. Self-asserts point-in-time on the date index before
    returning."""
    _guard_data_end(as_of_date, "load_fama_french_factors_from_lakehouse")
    dataset = _KEN_FRENCH_DATASET_BY_MODEL.get(model)
    if dataset is None:
        return None
    factors = _KEN_FRENCH_FACTORS_BY_MODEL[model]

    placeholders = ", ".join("?" for _ in factors)
    query = f"""
        SELECT date, portfolio, value
        FROM {_table_sql("famafrench_returns")}
        WHERE dataset = ? AND frequency = ?
          AND portfolio IN ({placeholders})
          AND last_day(
                date_add(CAST(date AS DATE), INTERVAL {_KEN_FRENCH_PUBLICATION_LAG_MONTHS} MONTH)
              ) <= ?
        ORDER BY date
    """
    long = _query(query, [dataset, _KEN_FRENCH_FREQUENCY, *factors, as_of_date.isoformat()])
    if long.empty:
        return None
    long["date"] = pd.to_datetime(long["date"])
    wide = long.pivot_table(index="date", columns="portfolio", values="value", aggfunc="max")
    frame = wide[[factor for factor in factors if factor in wide.columns]]
    result = (frame.astype(float) / 100.0).sort_index()
    assert_point_in_time(result, as_of_date)
    return result


# The columns identifying one reported period of one concept; a later
# filing restating the same period replaces the earlier value.
_COMPANY_FACT_PERIOD_KEY = ["concept", "unit", "period_start", "period_end"]


def load_company_facts_from_lakehouse(ticker: str, as_of_date: date) -> pd.DataFrame | None:
    """Filed SEC EDGAR company facts for `ticker` known at the close of
    `as_of_date`, one row per reported period: columns `concept`, `unit`,
    `period_start`, `period_end`, `value`, `fiscal_year`,
    `fiscal_period`, `form`, `filed` and `available_date` (the four date
    columns datetime64), sorted by `filed` then `period_end`.

    Reads the `sec_company_facts_log` view, which keeps every scrape,
    because the deduplicated `sec_company_facts` view drops one of two
    facts that differ only in `period_start` until lake issue LK-13
    lands. Inside a `lake_session` only scrapes at or before its lake time
    are read. Rows identical apart from their scrape (`scraped_at`, `dt`,
    `table`) count once.

    A filing reports several period windows for one concept (current
    quarter, year-to-date, prior-year comparatives) as separate rows, and
    later filings repeat or restate earlier periods. Gated on when the
    filing became public, never the period it describes:
    `available_date` is set from the filing's latest acceptance time in
    `sec_filings` (`_sec_acceptance_query`) by `_sec_available_dates`,
    and only rows with `available_date <= as_of_date` are kept. Of
    those, each (`concept`, `unit`, `period_start`, `period_end`) keeps
    the row with the latest `available_date`, ties broken by the latest
    `filed`, then the latest EDGAR acceptance; the query sorts on every
    column it returns, so which of two tied rows is kept never depends on
    the order the lake stores them in.
    Returns `None` if no rows remain. Self-asserts point-in-time on
    `filed` (the SQL gate) and on `available_date` before returning."""
    _guard_data_end(as_of_date, "load_company_facts_from_lakehouse")
    query = f"""
        SELECT x.concept, x.unit, x.period_start, x.period_end, x.value, x.fiscal_year,
               x.fiscal_period, x.form, x.filed_date AS filed, f.accepted_at
        FROM (
            SELECT DISTINCT * EXCLUDE (scraped_at, dt, "table")
            FROM {_table_sql("sec_company_facts_log")}
            WHERE ticker = ? AND filed_date <= ?
        ) x
        LEFT JOIN ({_sec_acceptance_query()}) f ON f.accession_number = x.accession_number
        ORDER BY filed, concept, unit, period_start, period_end, accepted_at, fiscal_year,
                 fiscal_period, form, value
    """
    frame = _query(query, [ticker, as_of_date.isoformat()])
    if frame.empty:
        return None
    frame["filed"] = pd.to_datetime(frame["filed"])
    assert_point_in_time(frame, as_of_date, date_column="filed")
    frame["period_start"] = pd.to_datetime(frame["period_start"])
    frame["period_end"] = pd.to_datetime(frame["period_end"])
    frame["available_date"] = _sec_available_dates(frame["filed"], frame["accepted_at"])
    frame = frame[frame["available_date"] <= pd.Timestamp(as_of_date)].drop(columns="accepted_at")
    if frame.empty:
        return None
    frame = (
        frame.sort_values(["available_date", "filed"], kind="stable")
        .drop_duplicates(subset=_COMPANY_FACT_PERIOD_KEY, keep="last")
        .sort_values(["filed", "period_end"], kind="stable")
        .reset_index(drop=True)
    )
    assert_point_in_time(frame, as_of_date, date_column="available_date")
    return frame


def _sec_acceptance_query() -> str:
    """One acceptance time per filing, as a query over `sec_filings`:
    the table lists a filing once per issuer ticker, sometimes with
    different `accepted_at` values, and the latest one is the conservative
    availability."""
    return f"""
        SELECT accession_number, max(accepted_at) AS accepted_at
        FROM {_table_sql("sec_filings")}
        GROUP BY accession_number
    """


# Decisions are taken at the NYSE close (16:00 America/New_York); an SEC
# filing accepted before that time is public for that day's decision.
_DECISION_TIMEZONE = "America/New_York"
_DECISION_TIME = pd.Timedelta(hours=16)


def _sec_available_dates(filed: pd.Series, accepted_at: pd.Series) -> pd.Series:
    """The day each SEC filing is known at the decision time, from its
    `filed` date and EDGAR `accepted_at` timestamp (time-zone aware, any
    zone, or missing): the filed date when acceptance falls strictly
    before 16:00 America/New_York on that date, otherwise — including a
    missing `accepted_at` — the next NYSE session after the filed date.
    Returns a datetime64 series aligned with `filed`."""
    filed_days = pd.to_datetime(filed).dt.normalize()
    cutoff = (filed_days + _DECISION_TIME).dt.tz_localize(_DECISION_TIMEZONE)
    accepted = pd.to_datetime(accepted_at, utc=True).dt.tz_convert(_DECISION_TIMEZONE)
    before_close = accepted < cutoff
    following = filed_days.map(lambda day: pd.Timestamp(next_session(day.date())))
    return filed_days.where(before_close, following)


# FINRA publishes short interest about seven business days after the
# settlement date, after the close; eight NYSE sessions after settlement
# and then the next session is the conservative day it is known at the
# close.
_FINRA_DISSEMINATION_LAG_SESSIONS = 8
# FINRA short interest settles twice a month (mid-month and month-end).
FINRA_SETTLEMENTS_PER_YEAR = 24
# `finra_short_interest.revision_flag` value marking a row FINRA revised
# after first publication.
_FINRA_REVISED = "R"


def _finra_available_date(settlement: date) -> date:
    """The day a settlement's short interest is known at the close:
    `next_session(add_sessions(settlement, _FINRA_DISSEMINATION_LAG_SESSIONS))`."""
    return next_session(add_sessions(settlement, _FINRA_DISSEMINATION_LAG_SESSIONS))


def load_short_interest_from_lakehouse(ticker: str, as_of_date: date) -> pd.DataFrame | None:
    """FINRA short-interest history for `ticker`, from the
    `finra_short_interest` table (`symbol`, `settlement_date`,
    `revision_flag`, `current_short_position`, `days_to_cover`,
    `average_daily_volume`, ...), as `settlement_date`/
    `dissemination_date`/`short_interest_shares`/`days_to_cover`/
    `avg_daily_volume` rows sorted by settlement date.

    `dissemination_date` holds this loader's own availability estimate,
    `_finra_available_date(settlement_date)`, not the lake's
    `dissemination_date`, which counts weekdays rather than NYSE sessions
    (lake issue LK-47). A revised row (`revision_flag` 'R') carries a
    value published with the symbol's next settlement, so it takes that
    settlement's availability; a revised row with no later settlement is
    dropped, because the lake keeps no pre-revision value. Only rows
    available on or before `as_of_date` are returned; `None` if none are.
    Self-asserts point-in-time on `dissemination_date` before returning."""
    _guard_data_end(as_of_date, "load_short_interest_from_lakehouse")
    query = f"""
        SELECT settlement_date, revision_flag,
               current_short_position AS short_interest_shares,
               days_to_cover, average_daily_volume AS avg_daily_volume
        FROM {_table_sql("finra_short_interest")}
        WHERE symbol = ?
        ORDER BY settlement_date
    """
    frame = _query(query, [ticker])
    if frame.empty:
        return None
    frame["settlement_date"] = pd.to_datetime(frame["settlement_date"])
    settlements = sorted({stamp.date() for stamp in frame["settlement_date"]})
    next_settlement: dict[date, date | None] = dict(
        zip(settlements, [*settlements[1:], None], strict=True)
    )
    revised = frame["revision_flag"].eq(_FINRA_REVISED).fillna(False).astype(bool)
    available: list[pd.Timestamp | None] = []
    for stamp, is_revised in zip(frame["settlement_date"], revised, strict=True):
        published_with = next_settlement[stamp.date()] if is_revised else stamp.date()
        available.append(
            None if published_with is None else pd.Timestamp(_finra_available_date(published_with))
        )
    frame["dissemination_date"] = pd.to_datetime(pd.Series(available, index=frame.index))
    frame = frame[frame["dissemination_date"] <= pd.Timestamp(as_of_date)]
    if frame.empty:
        return None
    frame = frame[
        [
            "settlement_date",
            "dissemination_date",
            "short_interest_shares",
            "days_to_cover",
            "avg_daily_volume",
        ]
    ].reset_index(drop=True)
    assert_point_in_time(frame, as_of_date, date_column="dissemination_date")
    return frame


def load_insider_transactions_from_lakehouse(ticker: str, as_of_date: date) -> pd.DataFrame | None:
    """SEC Form 4 insider transactions for `ticker` (`owner_cik`,
    `owner_name`, `is_officer`, `is_director`, `is_ten_percent_owner`,
    `transaction_date`, `transaction_code`, `shares`, `price_per_share`,
    `acquired_disposed`, `filed_date`, `available_date`), sorted by
    filed and transaction date.

    Reads the `sec_insider_transactions_log` view, which keeps every
    scrape, because the deduplicated `sec_insider_transactions` view
    collapses distinct lines of one filing into one row until lake issue
    LK-14 lands. Inside a `lake_session` only scrapes at or before its
    lake time are read. Rows identical apart from their scrape
    (`scraped_at`, `dt`, `table`) count once, so two genuinely identical
    lines in one filing also count once.

    The trade date is not when a trade became public; the filing is.
    `available_date` (datetime64) is set from the filing's latest EDGAR
    acceptance time in `sec_filings` (`_sec_acceptance_query`) by
    `_sec_available_dates` (16:00 America/New_York, the close, is the
    decision time), and only rows with `available_date <= as_of_date`
    are returned, sorted on every returned column. Returns `None` if no
    rows remain. Self-asserts point-in-time on `filed_date` (the SQL gate)
    and on `available_date` before returning."""
    _guard_data_end(as_of_date, "load_insider_transactions_from_lakehouse")
    query = f"""
        SELECT t.owner_cik, t.owner_name, t.is_officer, t.is_director, t.is_ten_percent_owner,
               t.transaction_date, t.transaction_code, t.shares, t.price_per_share,
               t.acquired_disposed, t.filed_date, f.accepted_at
        FROM (
            SELECT DISTINCT * EXCLUDE (scraped_at, dt, "table")
            FROM {_table_sql("sec_insider_transactions_log")}
            WHERE issuer_ticker = ? AND filed_date <= ?
        ) t
        LEFT JOIN ({_sec_acceptance_query()}) f ON f.accession_number = t.accession_number
        ORDER BY t.filed_date, t.transaction_date, t.owner_cik, t.owner_name, t.transaction_code,
                 t.acquired_disposed, t.shares, t.price_per_share, t.is_officer, t.is_director,
                 t.is_ten_percent_owner, f.accepted_at
    """
    frame = _query(query, [ticker, as_of_date.isoformat()])
    if frame.empty:
        return None
    frame["filed_date"] = pd.to_datetime(frame["filed_date"])
    frame["transaction_date"] = pd.to_datetime(frame["transaction_date"])
    assert_point_in_time(frame, as_of_date, date_column="filed_date")
    frame["available_date"] = _sec_available_dates(frame["filed_date"], frame["accepted_at"])
    frame = frame[frame["available_date"] <= pd.Timestamp(as_of_date)].drop(columns="accepted_at")
    if frame.empty:
        return None
    frame = frame.reset_index(drop=True)
    assert_point_in_time(frame, as_of_date, date_column="available_date")
    return frame


# Two chain rows quote the same strike when they differ by less than half a
# cent: a split rescale leaves a strike up to a third of a cent from its
# listed post-split value (625 / 3 = 208.333... against a listed 208.33),
# and listed strikes are at least 0.5 apart, so no two distinct strikes
# fall within this tolerance.
STRIKE_MATCH_TOLERANCE = 0.005


def load_option_quote_from_lakehouse(
    underlying: str,
    expiration: date,
    option_type: str,
    strike: float,
    as_of_date: date,
    lookback_days: int,
) -> pd.DataFrame | None:
    """The latest chain row quoting one contract on or before
    `as_of_date` within `lookback_days`, regardless of whether the latest
    snapshot still lists that expiry. Same source table, columns and
    prices as `load_options_snapshot_from_lakehouse` (the table
    `_option_chain_source` picks for `underlying` and `as_of_date`),
    including the `_apply_split_adjustment` rescaling. A row matches on
    `expiration` and `option_type` exactly and on a strike within
    `STRIKE_MATCH_TOLERANCE` of `strike`; the strike is compared in the
    adjusted basis rather than in SQL against the raw column, since the
    two differ for a contract quoted before a later split. Of several
    matching rows the one with the latest `date` is returned. `None`
    when never quoted in the window. Self-asserts point-in-time on
    `date` before returning."""
    _guard_data_end(as_of_date, "load_option_quote_from_lakehouse")
    window_start = (as_of_date - pd.Timedelta(days=lookback_days)).isoformat()
    table, date_column = _option_chain_source(underlying, as_of_date)
    query = f"""
        SELECT {date_column} AS date, expiration, option_type, strike, bid, ask,
               implied_volatility, delta, gamma, theta, vega, rho
        FROM {_table_sql(table)}
        WHERE underlying = ?
          AND expiration = ?
          AND option_type = ?
          AND {date_column} BETWEEN ? AND ?
        ORDER BY {date_column} DESC
    """
    frame = _query(
        query,
        [underlying, expiration.isoformat(), option_type, window_start, as_of_date.isoformat()],
    )
    if frame.empty:
        return None
    frame["date"] = pd.to_datetime(frame["date"])
    frame["expiration"] = pd.to_datetime(frame["expiration"])
    frame = _apply_split_adjustment(frame, underlying)
    frame = frame[(frame["strike"] - strike).abs() < STRIKE_MATCH_TOLERANCE]
    if frame.empty:
        return None
    frame = frame.iloc[[0]]
    assert_point_in_time(frame, as_of_date, date_column="date")
    return frame


CoverageSource = Literal["options_chain", "cftc_cot", "earnings", "dividends", "sp500_membership"]


def _first_and_last(frame: pd.DataFrame) -> tuple[date, date] | None:
    """The `first`/`last` dates of a one-row `MIN`/`MAX` result, or `None`
    when the aggregate matched no rows (both values null)."""
    if frame.empty or pd.isna(frame.iloc[0]["first"]) or pd.isna(frame.iloc[0]["last"]):
        return None
    return pd.Timestamp(frame.iloc[0]["first"]).date(), pd.Timestamp(frame.iloc[0]["last"]).date()


def source_coverage(source: CoverageSource, identifier: str | None) -> tuple[date, date] | None:
    """The first and last date the lake holds for `source`, read from the
    tables the matching loader reads, with no point-in-time gate on the
    data's own dates (inside a `lake_session`, the tables as they stood at
    its lake time). `identifier` is a ticker for the per-ticker sources and `None` for a
    check over the whole table:

    - `options_chain`: snapshot `date`s in `dolthub_option_chain` for
      underlying `identifier`; for an underlying in
      `_OPTION_CHAIN_HISTORY_UNDERLYINGS`, also the `option_chain_history`
      `snapshot_date`s before its first dolthub date (all of them when it
      has no dolthub rows), the two tables' ranges joined as the option
      loaders read them.
    - `cftc_cot`: `cftc_cot_publication.published_date`s of the rows
      with positive open interest and a non-null `published_date` for the
      COT ticker of `identifier` (its `DEFAULT_COT_TICKER_BY_ASSET`
      future, else itself), from whichever of `cftc_cot_disaggregated` /
      `cftc_cot_tff` holds that ticker's contract-market code, joined on
      `report_date`; `None` when it has no code.
    - `earnings`: announcement `date`s in `dolthub_earnings_calendar`,
      over every symbol when `identifier` is `None`, else for that symbol.
    - `dividends`: ex-dates of regular cash distributions in
      `equity_distributions`, over every ticker when `identifier` is
      `None`, else for that ticker.
    - `sp500_membership`: the earliest `start_date` and the latest
      `end_date` in `sp500_membership`, an open membership span counting
      as today; `identifier` is not used.

    Returns `None` when the lake holds no matching rows. The
    `sp500_membership` read, and for `cftc_cot` a check that
    `cftc_cot_publication` holds rows (`_require_rows`) made before the
    per-market read, go through `_query_table`, so a table those reads
    find empty twice raises `EmptyTableReadError` rather than reporting
    no coverage. Not subject to the session's `data_end`
    (`_guard_data_end`): it takes no knowledge-time argument and returns
    coverage dates for the preflight, not data, so the last date can fall
    after `data_end`."""
    if source == "options_chain":
        query = f"""
            SELECT MIN(date) AS first, MAX(date) AS last
            FROM {_table_sql("dolthub_option_chain")}
            WHERE underlying = ?
        """
        dolthub = _first_and_last(_query(query, [identifier]))
        if identifier not in _OPTION_CHAIN_HISTORY_UNDERLYINGS:
            return dolthub
        history_query = f"""
            SELECT MIN(snapshot_date) AS first, MAX(snapshot_date) AS last
            FROM {_table_sql("option_chain_history")}
            WHERE underlying = ?
        """
        if dolthub is None:
            return _first_and_last(_query(history_query, [identifier]))
        history = _first_and_last(
            _query(history_query + " AND snapshot_date < ?", [identifier, dolthub[0].isoformat()])
        )
        return dolthub if history is None else (history[0], dolthub[1])
    if source == "cftc_cot":
        cot_ticker = DEFAULT_COT_TICKER_BY_ASSET.get(identifier or "", identifier or "")
        for table, codes in (
            ("cftc_cot_disaggregated", _COT_DISAGGREGATED_CODES),
            ("cftc_cot_tff", _COT_TFF_CODES),
        ):
            code = codes.get(cot_ticker)
            if code is None:
                continue
            _require_rows("cftc_cot_publication")
            query = f"""
                SELECT MIN(pub.published_date) AS first, MAX(pub.published_date) AS last
                FROM {_table_sql(table)} AS cot
                JOIN {_table_sql("cftc_cot_publication")} AS pub
                  ON pub.report_date = cot.report_date
                WHERE cot.cftc_contract_market_code = ?
                  AND cot.open_interest_all > 0
                  AND pub.published_date IS NOT NULL
            """
            return _first_and_last(_query(query, [code]))
        return None
    if source == "earnings":
        query = f"""
            SELECT MIN(date) AS first, MAX(date) AS last
            FROM {_table_sql("dolthub_earnings_calendar")}
        """
        if identifier is None:
            return _first_and_last(_query(query, []))
        return _first_and_last(_query(query + " WHERE act_symbol = ?", [identifier]))
    if source == "dividends":
        query = f"""
            SELECT MIN(ex_date) AS first, MAX(ex_date) AS last
            FROM {_table_sql("equity_distributions")}
            WHERE distribution_type = ?
        """
        if identifier is None:
            return _first_and_last(_query(query, [_REGULAR_CASH]))
        return _first_and_last(_query(query + " AND ticker = ?", [_REGULAR_CASH, identifier]))
    if source == "sp500_membership":
        query = f"""
            SELECT MIN(start_date) AS first,
                   MAX(COALESCE(end_date, CAST(? AS DATE))) AS last
            FROM {_table_sql("sp500_membership")}
            HAVING count(*) > 0
        """
        return _first_and_last(_query_table("sp500_membership", query, [date.today().isoformat()]))
    raise ValueError(f"no coverage query for source {source!r}")
