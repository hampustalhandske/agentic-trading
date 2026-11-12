# checkpoint_store.py

"""Append-only SQLite-backed backtest checkpoint.

Each simulated day appends exactly that day's new rows to eight tables
(`track_record`, `decisions`, `trades`, `unfilled_orders`, `capital_usage`,
`sizing_diagnostics`, `pod_returns`, `pod_cost_days`) via `INSERT` — O(1) per day, O(n)
total for a run of length n — with the track record's day row
(`track_record_days`) and any source or asset it sees for the first time
(`track_record_sources`, `track_record_assets`), and the day's signal
tally (`signal_tally`), and overwrites two things each day: a single
`cursor` row (next day index, portfolio state, signal health, the
pending decisions when they are not the day's own, the Commander's
serialized risk state, the previous close mid
of every option contract the run has marked, the lake time the run reads
at, and the run's own config for `_restore_checkpoint`'s match-check:
dates, assets, settings fingerprint and roster fingerprint), and the
`carried_state` table, one row per agent holding its `carried_state()`
snapshot, of which only the rows whose bytes changed since the last write
through the same connection are rewritten. Both hold one day's state; the
cursor grows only with the number of option contracts the run has marked.
Mirrors the append-only idiom `data/db.py` uses
for the fetch log.

`read_all` reads the whole state back; `read_recent` reads it with only
the recent track record and decisions a horizon-bounded engine holds;
`read_state` reads it without the track record and the decisions;
`read_last_day`, `read_unfilled_orders`, `iter_decision_payloads`,
`iter_source_returns`, `decision_days` and `iter_signal_tally` read single
columns of one table each.

Each day's decisions are stored as their JSON text, UTF-8 encoded and
zlib-compressed (`DECISIONS_COMPRESSION_LEVEL`); every reader gives back
the JSON text itself. The file records its layout as
`CHECKPOINT_FORMAT_VERSION` in SQLite's `user_version`; `connect` with
`resume=True` refuses a file of any other version with
`CheckpointFormatError`, so a run is resumed only from a checkpoint of the
layout this module writes. `connect_read_only` opens a file of that layout
for the readers above without changing anything in it, and `snapshot`
reads a copy of it, so a run writing the file is never held up.

A `carried_state` blob is a pickle: open with `connect` only checkpoint
files this program wrote, never one from an untrusted source."""

import hashlib
import json
import sqlite3
import tempfile
import zlib
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any, cast

from agentic_trading.agents.schemas import (
    CapitalUsage,
    PerformanceRecord,
    PodCostDay,
    PortfolioDecision,
    SignalStatus,
    SizingDiagnostics,
    TradeRecord,
    UnfilledOrder,
    VolatilityEstimator,
)
from agentic_trading.firm_record.availability import day_signal_counts

# The layout version this module writes into SQLite's `user_version`; a
# file of any other version, 0 included, is refused for a resume.
CHECKPOINT_FORMAT_VERSION = 5
# zlib level of each day's stored decisions: level 6 compresses a firm
# record's day of decisions about 14x in about 10 ms; the higher levels
# shrink it little more for several times the time.
DECISIONS_COMPRESSION_LEVEL = 6


class CheckpointFormatError(ValueError):
    """A checkpoint file of another layout version than
    `CHECKPOINT_FORMAT_VERSION`, which this module cannot resume from."""


# The track record, one row per active signal of a day: the analyst's
# record and the manager's record of that signal share every field but
# their source, so a row carries both source ids (`track_record_sources`
# ids; either NULL when the record has no partner) and is read back as the
# analyst's record followed by the manager's, the order they were appended
# in. `asset` is a `track_record_assets` id and `as_of_date` the date's
# proleptic Gregorian ordinal (`date.toordinal`). `id` is the row's
# position: every row of an append has a larger id than every row before.
_CREATE_TRACK_RECORD = """
CREATE TABLE IF NOT EXISTS track_record (
    id INTEGER PRIMARY KEY,
    analyst_source INTEGER,
    manager_source INTEGER,
    asset INTEGER NOT NULL,
    as_of_date INTEGER NOT NULL,
    attributable_return REAL NOT NULL,
    raw_attributable_return REAL NOT NULL,
    signal_score REAL NOT NULL,
    realized_return REAL NOT NULL,
    CHECK (analyst_source IS NOT NULL OR manager_source IS NOT NULL)
)
"""
# Each source_id of the track record with its integer id and the ordinal of
# its earliest record's date.
_CREATE_TRACK_RECORD_SOURCES = """
CREATE TABLE IF NOT EXISTS track_record_sources (
    id INTEGER PRIMARY KEY,
    source_id TEXT NOT NULL UNIQUE,
    first_date INTEGER NOT NULL
)
"""
# Each asset of the track record with its integer id.
_CREATE_TRACK_RECORD_ASSETS = """
CREATE TABLE IF NOT EXISTS track_record_assets (
    id INTEGER PRIMARY KEY,
    asset TEXT NOT NULL UNIQUE
)
"""
# One row per `append_day` that added track-record rows: the ids of its
# first and last row, the ordinals of the earliest and latest date among
# them and the number of records they hold, so a read by date seeks to the
# ids it needs instead of scanning the table.
_CREATE_TRACK_RECORD_DAYS = """
CREATE TABLE IF NOT EXISTS track_record_days (
    first_id INTEGER NOT NULL,
    last_id INTEGER NOT NULL,
    min_date INTEGER NOT NULL,
    max_date INTEGER NOT NULL,
    records INTEGER NOT NULL
)
"""
# Each day's signals as the firm record's availability report counts them
# (`firm_record.availability.day_signal_counts` of the day's decisions):
# one row per analyst id, status and volatility estimator (NULL for none)
# with its number of signals, written with the day's decisions.
_CREATE_SIGNAL_TALLY = """
CREATE TABLE IF NOT EXISTS signal_tally (
    as_of_date TEXT NOT NULL,
    analyst_id TEXT NOT NULL,
    status TEXT NOT NULL,
    volatility_estimator TEXT,
    signals INTEGER NOT NULL
)
"""
_CREATE_DECISIONS = """
CREATE TABLE IF NOT EXISTS decisions (
    as_of_date TEXT NOT NULL,
    payload BLOB NOT NULL
)
"""
_CREATE_TRADES = """
CREATE TABLE IF NOT EXISTS trades (
    as_of_date TEXT NOT NULL,
    payload TEXT NOT NULL
)
"""
_CREATE_UNFILLED_ORDERS = """
CREATE TABLE IF NOT EXISTS unfilled_orders (
    as_of_date TEXT NOT NULL,
    payload TEXT NOT NULL
)
"""
_CREATE_CAPITAL_USAGE = """
CREATE TABLE IF NOT EXISTS capital_usage (
    as_of_date TEXT NOT NULL,
    payload TEXT NOT NULL
)
"""
_CREATE_SIZING_DIAGNOSTICS = """
CREATE TABLE IF NOT EXISTS sizing_diagnostics (
    as_of_date TEXT NOT NULL,
    payload TEXT NOT NULL
)
"""
_CREATE_POD_RETURNS = """
CREATE TABLE IF NOT EXISTS pod_returns (
    as_of_date TEXT NOT NULL,
    pod_id TEXT NOT NULL,
    value REAL NOT NULL
)
"""
_CREATE_POD_COST_DAYS = """
CREATE TABLE IF NOT EXISTS pod_cost_days (
    as_of_date TEXT NOT NULL,
    payload TEXT NOT NULL
)
"""
_CREATE_PORTFOLIO_VALUE = """
CREATE TABLE IF NOT EXISTS portfolio_value (
    as_of_date TEXT NOT NULL,
    value REAL NOT NULL
)
"""
# Single-row table (id is always 0) — the only thing this module rewrites
# in place rather than appends, since it's small regardless of run length.
_CREATE_CURSOR = """
CREATE TABLE IF NOT EXISTS cursor (
    id INTEGER PRIMARY KEY CHECK (id = 0),
    next_day_index INTEGER NOT NULL,
    start_date TEXT NOT NULL,
    end_date TEXT NOT NULL,
    assets TEXT NOT NULL,
    portfolio TEXT NOT NULL,
    signal_health TEXT NOT NULL,
    pending_decisions TEXT,
    commander_risk_state TEXT,
    total_financing_cost REAL NOT NULL DEFAULT 0.0,
    total_option_premium_paid REAL NOT NULL DEFAULT 0.0,
    option_last_mid TEXT,
    lake_time TEXT,
    settings_fingerprint TEXT,
    roster_fingerprint TEXT
)
"""
# One row per agent: the agent's `carried_state()` snapshot (a pickle) keyed
# by its agent_id, rewritten on a day its bytes change.
_CREATE_CARRIED_STATE = """
CREATE TABLE IF NOT EXISTS carried_state (
    agent_id TEXT PRIMARY KEY,
    state BLOB NOT NULL
)
"""
# Columns added to a table after its first release, applied with `ALTER
# TABLE` on connect so a checkpoint file created by an older schema can
# still be resumed. Each entry is (table, column, column definition).
_ADDITIVE_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("cursor", "commander_risk_state", "TEXT"),
    ("cursor", "total_financing_cost", "REAL NOT NULL DEFAULT 0.0"),
    ("cursor", "total_option_premium_paid", "REAL NOT NULL DEFAULT 0.0"),
    ("cursor", "option_last_mid", "TEXT"),
    ("cursor", "lake_time", "TEXT"),
    ("cursor", "settings_fingerprint", "TEXT"),
    ("cursor", "roster_fingerprint", "TEXT"),
)


def _add_missing_columns(connection: sqlite3.Connection) -> None:
    for table, column, definition in _ADDITIVE_COLUMNS:
        existing = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
        if column not in existing:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


# Every table this module creates, in creation order.
_TABLES: tuple[tuple[str, str], ...] = (
    ("track_record", _CREATE_TRACK_RECORD),
    ("track_record_sources", _CREATE_TRACK_RECORD_SOURCES),
    ("track_record_assets", _CREATE_TRACK_RECORD_ASSETS),
    ("track_record_days", _CREATE_TRACK_RECORD_DAYS),
    ("decisions", _CREATE_DECISIONS),
    ("signal_tally", _CREATE_SIGNAL_TALLY),
    ("trades", _CREATE_TRADES),
    ("unfilled_orders", _CREATE_UNFILLED_ORDERS),
    ("capital_usage", _CREATE_CAPITAL_USAGE),
    ("sizing_diagnostics", _CREATE_SIZING_DIAGNOSTICS),
    ("pod_returns", _CREATE_POD_RETURNS),
    ("pod_cost_days", _CREATE_POD_COST_DAYS),
    ("portfolio_value", _CREATE_PORTFOLIO_VALUE),
    ("cursor", _CREATE_CURSOR),
    ("carried_state", _CREATE_CARRIED_STATE),
)


class CheckpointConnection(sqlite3.Connection):
    """The connection `connect` returns: a `sqlite3.Connection` that also
    remembers the SHA-256 digest of each `carried_state` row this
    connection last wrote (`carried_state_digests`, `None` until its first
    write, or after a write that failed), so `append_day` rewrites only the
    snapshots that changed. A new connection knows no digest and rewrites
    every snapshot on its first write, so what a resume reads never depends
    on a digest held in memory."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.carried_state_digests: dict[str, bytes] | None = None


def connect(db_path: Path, *, resume: bool = False) -> sqlite3.Connection:
    """Opens (creating tables if needed) the checkpoint database at
    `db_path`. When `resume` is `False`, every table of this module already
    in the file is dropped and created anew first — a fresh (non-resumed)
    run always starts from an empty checkpoint of the current layout, even
    if `db_path` points at a file left over from an earlier, unrelated
    run, rather than silently appending its rows on top of that leftover
    state. When `resume` is `True`, a file that already holds tables must
    be of `CHECKPOINT_FORMAT_VERSION`; one of another version raises
    `CheckpointFormatError`, naming the file and both versions, before
    anything in it changes. The drops, creates and version stamp run in
    one explicit transaction (SQLite's DDL is transactional, and Python's
    sqlite3 opens no transaction of its own before DDL), so a reset that
    fails or is killed part way leaves the file as it was."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path, factory=CheckpointConnection)
    (version,) = connection.execute("PRAGMA user_version").fetchone()
    (tables,) = connection.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table'"
    ).fetchone()
    if resume and tables and version != CHECKPOINT_FORMAT_VERSION:
        connection.close()
        raise CheckpointFormatError(
            f"checkpoint {db_path} is of checkpoint format {version}, but this program "
            f"reads only format {CHECKPOINT_FORMAT_VERSION}; it cannot be resumed: remove it, "
            "and the trial it belongs to, and start the run over"
        )
    with connection:
        connection.execute("BEGIN")
        if not resume:
            for name, _ in _TABLES:
                connection.execute(f"DROP TABLE IF EXISTS {name}")
        for _, create in _TABLES:
            connection.execute(create)
        _add_missing_columns(connection)
        connection.execute(f"PRAGMA user_version = {CHECKPOINT_FORMAT_VERSION}")
    return connection


def connect_read_only(db_path: Path) -> sqlite3.Connection:
    """Opens the checkpoint database at `db_path` for reading only (SQLite's
    `mode=ro`): nothing in the file is created, altered or written, and a
    write through the connection fails. Raises `FileNotFoundError` when
    the file does not exist, and `CheckpointFormatError` (closing the
    connection) when it is not of `CHECKPOINT_FORMAT_VERSION`, naming the
    file and both versions."""
    if not db_path.is_file():
        raise FileNotFoundError(f"no checkpoint at {db_path}")
    connection = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    (version,) = connection.execute("PRAGMA user_version").fetchone()
    if version != CHECKPOINT_FORMAT_VERSION:
        connection.close()
        raise CheckpointFormatError(
            f"checkpoint {db_path} is of checkpoint format {version}, but this program "
            f"reads only format {CHECKPOINT_FORMAT_VERSION}"
        )
    return connection


@contextmanager
def snapshot(db_path: Path, *, directory: Path | None = None) -> Iterator[sqlite3.Connection]:
    """Copies the checkpoint at `db_path` into a temporary file and yields
    a read-only connection (`connect_read_only`) to the copy; the copy is
    deleted when the block ends. The copy is taken in one step of SQLite's
    online backup from a read-only connection, so it is one committed
    state of the file, and the file is held only while it is copied: a
    run writing the checkpoint meanwhile commits as it would without the
    reader. The temporary file lies in a new directory under `directory`
    (the system's temporary directory by default), which must lie outside
    the directory holding `db_path`. Raises `ValueError` when it does not,
    `FileNotFoundError` when `db_path` does not exist and
    `CheckpointFormatError` when it is not of `CHECKPOINT_FORMAT_VERSION`,
    before anything is copied."""
    base = Path(tempfile.gettempdir() if directory is None else directory).resolve()
    if base.is_relative_to(db_path.resolve().parent):
        raise ValueError(
            f"a snapshot of {db_path} would be written under {base}, inside the checkpoint's "
            "directory; give a directory outside it"
        )
    source = connect_read_only(db_path)
    try:
        with tempfile.TemporaryDirectory(dir=directory, prefix="checkpoint-snapshot-") as folder:
            copy_path = Path(folder) / db_path.name
            target = sqlite3.connect(copy_path)
            try:
                source.backup(target)
            finally:
                target.close()
            source.close()
            connection = connect_read_only(copy_path)
            try:
                yield connection
            finally:
                connection.close()
    finally:
        source.close()


def append_day(
    connection: sqlite3.Connection,
    as_of_date: date,
    portfolio_value: float,
    new_track_record: list[PerformanceRecord],
    decisions: list[PortfolioDecision],
    new_trades: list[TradeRecord],
    next_day_index: int,
    start_date: date,
    end_date: date,
    assets: list[str],
    portfolio: dict[str, Any],
    signal_health: dict[str, dict[str, int]],
    pending_decisions: list[PortfolioDecision] | None = None,
    commander_risk_state: dict[str, Any] | None = None,
    total_financing_cost: float = 0.0,
    total_option_premium_paid: float = 0.0,
    new_unfilled_orders: list[UnfilledOrder] | None = None,
    capital_usage: CapitalUsage | None = None,
    sizing_diagnostics: list[SizingDiagnostics] | None = None,
    new_pod_returns: list[tuple[str, date, float]] | None = None,
    pod_cost_day: PodCostDay | None = None,
    lake_time: datetime | None = None,
    settings_fingerprint: str | None = None,
    roster_fingerprint: str | None = None,
    carried_states: dict[str, bytes] | None = None,
    option_last_mid: dict[str, float] | None = None,
) -> None:
    """Appends exactly this simulated day's new rows — never touches a
    prior day's already-written rows — plus overwrites the single small
    `cursor` row with this day's resumability state. `commander_risk_state`
    is the Commander's `risk_state()` already dumped to JSON-compatible
    form, or `None` for a commander that carries no risk state.
    `lake_time` is the lake time the run reads at (`None` for a run outside
    a lake session), stored as ISO 8601.
    `sizing_diagnostics` are the day's per-pod sizing diagnostics, in pod
    order. `new_pod_returns` are the (pod id, date, return) entries first seen
    this day. `pod_cost_day` is the day's pod contributions and the fill
    costs charged to each pod. `settings_fingerprint` and `roster_fingerprint`
    are recorded with the dates and assets as the run's config, on the day
    the cursor row is first written. `carried_states` is each agent's
    `carried_state()` keyed by agent_id; when given, the stored snapshots
    become exactly these, and when `None` they are left as they are. On a
    `CheckpointConnection` that wrote the snapshots before, only the rows
    whose bytes changed are rewritten and those of agents no longer given
    deleted; otherwise every row is replaced.
    `option_last_mid` is the engine's previous close mid per option
    contract, stored as JSON (`None` stores none). `pending_decisions` are
    the decisions still to execute; `None`, or a list whose JSON is that of
    `decisions`, stores none in the cursor, and a read gives back the
    decisions of the last day appended in their place."""
    decisions_json = json.dumps([decision.model_dump(mode="json") for decision in decisions])
    pending_json = None
    if pending_decisions is not None and pending_decisions is not decisions:
        pending_json = json.dumps([d.model_dump(mode="json") for d in pending_decisions])
        if pending_json == decisions_json:
            pending_json = None
    with connection:
        connection.execute(
            "INSERT INTO portfolio_value (as_of_date, value) VALUES (?, ?)",
            (as_of_date.isoformat(), portfolio_value),
        )
        _append_track_record(connection, new_track_record)
        connection.execute(
            "INSERT INTO decisions (as_of_date, payload) VALUES (?, ?)",
            (
                as_of_date.isoformat(),
                _compressed(decisions_json),
            ),
        )
        connection.executemany(
            "INSERT INTO signal_tally (as_of_date, analyst_id, status, volatility_estimator, "
            "signals) VALUES (?, ?, ?, ?, ?)",
            [
                (as_of_date.isoformat(), analyst_id, status, estimator, signals)
                for (analyst_id, status, estimator), signals in sorted(
                    day_signal_counts(as_of_date, decisions).items(),
                    key=lambda item: (item[0][0], item[0][1], item[0][2] or ""),
                )
            ],
        )
        connection.executemany(
            "INSERT INTO trades (as_of_date, payload) VALUES (?, ?)",
            [
                (as_of_date.isoformat(), json.dumps(trade.model_dump(mode="json")))
                for trade in new_trades
            ],
        )
        connection.executemany(
            "INSERT INTO unfilled_orders (as_of_date, payload) VALUES (?, ?)",
            [
                (as_of_date.isoformat(), json.dumps(order.model_dump(mode="json")))
                for order in new_unfilled_orders or []
            ],
        )
        if capital_usage is not None:
            connection.execute(
                "INSERT INTO capital_usage (as_of_date, payload) VALUES (?, ?)",
                (as_of_date.isoformat(), json.dumps(capital_usage.model_dump(mode="json"))),
            )
        connection.executemany(
            "INSERT INTO sizing_diagnostics (as_of_date, payload) VALUES (?, ?)",
            [
                (as_of_date.isoformat(), json.dumps(entry.model_dump(mode="json")))
                for entry in sizing_diagnostics or []
            ],
        )
        connection.executemany(
            "INSERT INTO pod_returns (as_of_date, pod_id, value) VALUES (?, ?, ?)",
            [
                (return_date.isoformat(), pod_id, value)
                for pod_id, return_date, value in new_pod_returns or []
            ],
        )
        if pod_cost_day is not None:
            connection.execute(
                "INSERT INTO pod_cost_days (as_of_date, payload) VALUES (?, ?)",
                (as_of_date.isoformat(), json.dumps(pod_cost_day.model_dump(mode="json"))),
            )
        connection.execute(
            """
            INSERT INTO cursor
                (id, next_day_index, start_date, end_date, assets, portfolio,
                 signal_health, pending_decisions, commander_risk_state,
                 total_financing_cost, total_option_premium_paid, option_last_mid,
                 lake_time, settings_fingerprint, roster_fingerprint)
            VALUES (0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                next_day_index = excluded.next_day_index,
                portfolio = excluded.portfolio,
                signal_health = excluded.signal_health,
                pending_decisions = excluded.pending_decisions,
                commander_risk_state = excluded.commander_risk_state,
                total_financing_cost = excluded.total_financing_cost,
                total_option_premium_paid = excluded.total_option_premium_paid,
                option_last_mid = excluded.option_last_mid,
                lake_time = excluded.lake_time
            """,
            (
                next_day_index,
                start_date.isoformat(),
                end_date.isoformat(),
                json.dumps(assets),
                json.dumps(portfolio),
                json.dumps(signal_health),
                pending_json,
                None if commander_risk_state is None else json.dumps(commander_risk_state),
                total_financing_cost,
                total_option_premium_paid,
                None if option_last_mid is None else json.dumps(option_last_mid),
                None if lake_time is None else lake_time.isoformat(),
                settings_fingerprint,
                roster_fingerprint,
            ),
        )
        digests = (
            None if carried_states is None else _write_carried_states(connection, carried_states)
        )
    if digests is not None and isinstance(connection, CheckpointConnection):
        connection.carried_state_digests = digests


def _write_carried_states(
    connection: sqlite3.Connection, carried_states: dict[str, bytes]
) -> dict[str, bytes]:
    """Makes the `carried_state` rows exactly `carried_states`, inside the
    caller's transaction, writing only what changed since the digests a
    `CheckpointConnection` holds (see `append_day`), and returns the
    digests of `carried_states` for the caller to record once the
    transaction commits. A `CheckpointConnection`'s digests are forgotten
    here, so a transaction that fails leaves it knowing none."""
    digests = {
        agent_id: hashlib.sha256(state).digest() for agent_id, state in carried_states.items()
    }
    known = (
        connection.carried_state_digests if isinstance(connection, CheckpointConnection) else None
    )
    if isinstance(connection, CheckpointConnection):
        connection.carried_state_digests = None
    if known is None:
        connection.execute("DELETE FROM carried_state")
        changed = sorted(carried_states.items())
    else:
        connection.executemany(
            "DELETE FROM carried_state WHERE agent_id = ?",
            [(agent_id,) for agent_id in sorted(known.keys() - digests.keys())],
        )
        changed = [
            (agent_id, state)
            for agent_id, state in sorted(carried_states.items())
            if known.get(agent_id) != digests[agent_id]
        ]
    connection.executemany(
        "INSERT INTO carried_state (agent_id, state) VALUES (?, ?) "
        "ON CONFLICT(agent_id) DO UPDATE SET state = excluded.state",
        changed,
    )
    return digests


def _compressed(payload_json: str) -> bytes:
    """`payload_json` as the bytes the `decisions` table stores."""
    return zlib.compress(payload_json.encode("utf-8"), DECISIONS_COMPRESSION_LEVEL)


def _decompressed(stored: bytes) -> str:
    """The JSON text `_compressed` stored as `stored`."""
    return zlib.decompress(stored).decode("utf-8")


def _outcome(entry: PerformanceRecord) -> tuple[Any, ...]:
    """Every field of `entry` but its source."""
    return (
        entry.asset,
        entry.as_of_date,
        entry.attributable_return,
        entry.raw_attributable_return,
        entry.signal_score,
        entry.realized_return,
    )


def _append_track_record(
    connection: sqlite3.Connection, records: Sequence[PerformanceRecord]
) -> None:
    """Appends `records`, in order, inside the caller's transaction: an
    analyst record directly followed by a manager record of the same
    outcome (`_outcome`) as one row carrying both sources, every other
    record as a row of its own source; then the append's
    `track_record_days` row."""
    if not records:
        return
    first_dates: dict[str, int] = {}
    for entry in records:
        ordinal = entry.as_of_date.toordinal()
        first_dates[entry.source_id] = min(first_dates.get(entry.source_id, ordinal), ordinal)
    connection.executemany(
        "INSERT INTO track_record_sources (source_id, first_date) VALUES (?, ?) "
        "ON CONFLICT(source_id) DO UPDATE SET first_date = MIN(first_date, excluded.first_date)",
        sorted(first_dates.items()),
    )
    connection.executemany(
        "INSERT INTO track_record_assets (asset) VALUES (?) ON CONFLICT(asset) DO NOTHING",
        [(asset,) for asset in sorted({entry.asset for entry in records})],
    )
    sources: dict[str, int] = dict(
        connection.execute("SELECT source_id, id FROM track_record_sources")
    )
    assets: dict[str, int] = dict(connection.execute("SELECT asset, id FROM track_record_assets"))
    (last_id,) = connection.execute("SELECT COALESCE(MAX(id), 0) FROM track_record").fetchone()
    rows: list[tuple[Any, ...]] = []
    index = 0
    while index < len(records):
        entry = records[index]
        partner = records[index + 1] if index + 1 < len(records) else None
        analyst: int | None = None
        manager: int | None = None
        if (
            entry.source_type == "analyst"
            and partner is not None
            and partner.source_type == "manager"
            and _outcome(partner) == _outcome(entry)
        ):
            analyst, manager = sources[entry.source_id], sources[partner.source_id]
            index += 2
        elif entry.source_type == "analyst":
            analyst = sources[entry.source_id]
            index += 1
        else:
            manager = sources[entry.source_id]
            index += 1
        rows.append(
            (
                last_id + 1 + len(rows),
                analyst,
                manager,
                assets[entry.asset],
                entry.as_of_date.toordinal(),
                entry.attributable_return,
                entry.raw_attributable_return,
                entry.signal_score,
                entry.realized_return,
            )
        )
    connection.executemany(
        "INSERT INTO track_record (id, analyst_source, manager_source, asset, as_of_date, "
        "attributable_return, raw_attributable_return, signal_score, realized_return) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    ordinals = [entry.as_of_date.toordinal() for entry in records]
    connection.execute(
        "INSERT INTO track_record_days (first_id, last_id, min_date, max_date, records) "
        "VALUES (?, ?, ?, ?, ?)",
        (last_id + 1, last_id + len(rows), min(ordinals), max(ordinals), len(records)),
    )


# The `track_record` columns `_track_record_rows` reads, in its order.
_TRACK_RECORD_COLUMNS = (
    "analyst_source, manager_source, asset, as_of_date, attributable_return, "
    "raw_attributable_return, signal_score, realized_return"
)


def _track_record_rows(
    connection: sqlite3.Connection, rows: Iterable[tuple[Any, ...]]
) -> list[dict[str, Any]]:
    """`track_record` rows of `_TRACK_RECORD_COLUMNS` as `PerformanceRecord`
    payload dicts, each row as its analyst's record then its manager's,
    the ones it carries."""
    sources = _source_names(connection)
    assets: dict[int, str] = {
        asset_id: asset
        for asset_id, asset in connection.execute("SELECT id, asset FROM track_record_assets")
    }
    dates: dict[int, str] = {}
    records: list[dict[str, Any]] = []
    for analyst, manager, asset, ordinal, attributable, raw, score, realized in rows:
        day = dates.get(ordinal)
        if day is None:
            day = dates[ordinal] = date.fromordinal(ordinal).isoformat()
        for source, source_type in ((analyst, "analyst"), (manager, "manager")):
            if source is None:
                continue
            records.append(
                {
                    "source_id": sources[source],
                    "source_type": source_type,
                    "asset": assets[asset],
                    "as_of_date": day,
                    "attributable_return": attributable,
                    "raw_attributable_return": raw,
                    "signal_score": score,
                    "realized_return": realized,
                }
            )
    return records


def _source_names(connection: sqlite3.Connection) -> dict[int, str]:
    """Each `track_record_sources` id to its source_id."""
    return {
        source: source_id
        for source, source_id in connection.execute(
            "SELECT id, source_id FROM track_record_sources"
        )
    }


def read_all(connection: sqlite3.Connection) -> dict[str, Any] | None:
    """Bulk-reads the full checkpoint state in the shape
    `BacktestEngine._restore_checkpoint()` (and any offline reader, e.g. the
    GUI's saved-run browser) needs — one read at startup, not per-day.
    Returns `None` if there's no cursor row yet (a fresh, never-checkpointed
    run). `lake_time` is the stored lake time as an aware `datetime`, or
    `None` when the run read outside a lake session. `config` holds the
    dates, the assets and the two fingerprints (`None` in a checkpoint
    written without them). `carried_states` is each agent's stored
    `carried_state()` snapshot keyed by agent_id, still pickled. Rows of
    one day keep the order they were appended in."""
    payload = _read_state(connection)
    if payload is None:
        return None
    payload["track_record"] = _track_record_rows(
        connection,
        connection.execute(
            f"SELECT {_TRACK_RECORD_COLUMNS} FROM track_record ORDER BY as_of_date, id"
        ),
    )
    payload["decisions_by_day"] = {
        as_of_date_str: json.loads(payload_json)
        for as_of_date_str, payload_json in iter_decision_payloads(connection)
    }
    return payload


def read_recent(
    connection: sqlite3.Connection, track_record_from: date, decision_days: int
) -> dict[str, Any] | None:
    """`read_all`'s payload with only the recent history a horizon-bounded
    engine holds in memory: `track_record` is the rows dated on or after
    `track_record_from`, in the order they were appended, and
    `decisions_by_day` the last `decision_days` days of decisions. Two
    keys describe the records left out: `track_record_count`, the number
    of records the track record holds, and
    `track_record_dropped_first_dates`, each source_id's earliest ISO date
    among the records dated before `track_record_from`. Every other key is as `read_all` gives it.
    Returns `None` when there is no cursor row yet. Raises `ValueError`
    when a row dated before `track_record_from` was appended after one
    dated on or after it, since the rows read would then not be the last
    ones appended. Reads only the track-record rows appended from the
    first day that holds one dated on or after `track_record_from`
    (`track_record_days`)."""
    payload = _read_state(connection)
    if payload is None:
        return None
    cutoff = track_record_from.toordinal()
    (start,) = connection.execute(
        "SELECT MIN(first_id) FROM track_record_days WHERE max_date >= ?", (cutoff,)
    ).fetchone()
    interleaved = 0
    if start is not None:
        (interleaved,) = connection.execute(
            "SELECT COALESCE(SUM((analyst_source IS NOT NULL) + (manager_source IS NOT NULL)), 0) "
            "FROM track_record WHERE as_of_date < ? AND id > "
            "(SELECT MIN(id) FROM track_record WHERE id >= ? AND as_of_date >= ?)",
            (cutoff, start, cutoff),
        ).fetchone()
    if interleaved:
        raise ValueError(
            f"the checkpoint's track record has {interleaved} row(s) dated before "
            f"{track_record_from.isoformat()} appended after a row dated on or after it"
        )
    payload["track_record"] = (
        []
        if start is None
        else _track_record_rows(
            connection,
            connection.execute(
                f"SELECT {_TRACK_RECORD_COLUMNS} FROM track_record "
                "WHERE id >= ? AND as_of_date >= ? ORDER BY id",
                (start, cutoff),
            ),
        )
    )
    (payload["track_record_count"],) = connection.execute(
        "SELECT COALESCE(SUM(records), 0) FROM track_record_days"
    ).fetchone()
    payload["track_record_dropped_first_dates"] = {
        source_id: date.fromordinal(first_date).isoformat()
        for source_id, first_date in connection.execute(
            "SELECT source_id, first_date FROM track_record_sources WHERE first_date < ? "
            "ORDER BY source_id",
            (cutoff,),
        )
    }
    payload["decisions_by_day"] = {
        as_of_date_str: json.loads(_decompressed(stored))
        for as_of_date_str, stored in connection.execute(
            "SELECT as_of_date, payload FROM ("
            "SELECT as_of_date, payload, rowid AS position FROM decisions "
            "ORDER BY as_of_date DESC, rowid DESC LIMIT ?"
            ") ORDER BY as_of_date, position",
            (decision_days,),
        )
    }
    return payload


def read_state(connection: sqlite3.Connection) -> dict[str, Any] | None:
    """Everything `read_all` returns but `track_record` and
    `decisions_by_day`: the run's config, its book and cursor values, and
    every per-day history but those two, which are the bulk of a long
    run's checkpoint. `None` when there is no cursor row yet."""
    return _read_state(connection)


def _read_state(connection: sqlite3.Connection) -> dict[str, Any] | None:
    """Everything `read_all` returns but `track_record` and
    `decisions_by_day`, the two histories a horizon-bounded engine keeps
    only in part; `None` when there is no cursor row yet."""
    cursor_row = connection.execute(
        "SELECT next_day_index, start_date, end_date, assets, portfolio, "
        "signal_health, pending_decisions, commander_risk_state, total_financing_cost, "
        "total_option_premium_paid, option_last_mid, lake_time, settings_fingerprint, "
        "roster_fingerprint "
        "FROM cursor WHERE id = 0"
    ).fetchone()
    if cursor_row is None:
        return None
    (
        next_day_index,
        start_date,
        end_date,
        assets_json,
        portfolio_json,
        signal_health_json,
        pending_decisions_json,
        commander_risk_state_json,
        total_financing_cost,
        total_option_premium_paid,
        option_last_mid_json,
        lake_time,
        settings_fingerprint,
        roster_fingerprint,
    ) = cursor_row

    portfolio_value_history = [
        (as_of_date_str, value)
        for as_of_date_str, value in connection.execute(
            "SELECT as_of_date, value FROM portfolio_value ORDER BY as_of_date"
        )
    ]
    trade_log = [
        json.loads(payload)
        for _as_of_date_str, payload in connection.execute(
            "SELECT as_of_date, payload FROM trades ORDER BY as_of_date, rowid"
        )
    ]
    unfilled_orders = read_unfilled_orders(connection)
    capital_usage = [
        json.loads(payload)
        for _as_of_date_str, payload in connection.execute(
            "SELECT as_of_date, payload FROM capital_usage ORDER BY as_of_date"
        )
    ]
    sizing_diagnostics = [
        json.loads(payload)
        for _as_of_date_str, payload in connection.execute(
            "SELECT as_of_date, payload FROM sizing_diagnostics ORDER BY as_of_date, rowid"
        )
    ]
    pod_cost_days = [
        json.loads(payload)
        for _as_of_date_str, payload in connection.execute(
            "SELECT as_of_date, payload FROM pod_cost_days ORDER BY as_of_date, rowid"
        )
    ]
    pod_return_history: dict[str, list[tuple[str, float]]] = {}
    for pod_id, return_date, value in connection.execute(
        "SELECT pod_id, as_of_date, value FROM pod_returns ORDER BY as_of_date, rowid"
    ):
        pod_return_history.setdefault(pod_id, []).append((return_date, value))
    carried_states = {
        agent_id: bytes(state)
        for agent_id, state in connection.execute(
            "SELECT agent_id, state FROM carried_state ORDER BY agent_id"
        )
    }
    return {
        "next_day_index": next_day_index,
        "config": {
            "start_date": start_date,
            "end_date": end_date,
            "assets": json.loads(assets_json),
            "settings_fingerprint": settings_fingerprint,
            "roster_fingerprint": roster_fingerprint,
        },
        "portfolio": json.loads(portfolio_json),
        "signal_health": json.loads(signal_health_json),
        "pending_decisions": json.loads(
            _last_decisions_json(connection)
            if pending_decisions_json is None
            else pending_decisions_json
        ),
        "commander_risk_state": (
            None if commander_risk_state_json is None else json.loads(commander_risk_state_json)
        ),
        "total_financing_cost": float(total_financing_cost),
        "total_option_premium_paid": float(total_option_premium_paid),
        "option_last_mid": (
            None if option_last_mid_json is None else json.loads(option_last_mid_json)
        ),
        "lake_time": None if lake_time is None else datetime.fromisoformat(lake_time),
        "trade_log": trade_log,
        "unfilled_orders": unfilled_orders,
        "capital_usage": capital_usage,
        "sizing_diagnostics": sizing_diagnostics,
        "pod_return_history": pod_return_history,
        "pod_cost_days": pod_cost_days,
        "portfolio_value_history": portfolio_value_history,
        "carried_states": carried_states,
    }


def _last_decisions_json(connection: sqlite3.Connection) -> str:
    """The decisions JSON of the last day appended, `"[]"` when none is."""
    row = connection.execute(
        "SELECT payload FROM decisions WHERE rowid = (SELECT MAX(rowid) FROM decisions)"
    ).fetchone()
    return "[]" if row is None else _decompressed(row[0])


def read_last_day(connection: sqlite3.Connection) -> date | None:
    """The last day the checkpoint holds, `None` when there is no cursor
    row yet. Reads two single values, not the history."""
    if connection.execute("SELECT 1 FROM cursor WHERE id = 0").fetchone() is None:
        return None
    (last,) = connection.execute("SELECT MAX(as_of_date) FROM portfolio_value").fetchone()
    return None if last is None else date.fromisoformat(last)


def read_unfilled_orders(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    """Every stored `UnfilledOrder` payload, in day order and, within a
    day, in the order appended."""
    return [
        json.loads(payload)
        for (payload,) in connection.execute(
            "SELECT payload FROM unfilled_orders ORDER BY as_of_date, rowid"
        )
    ]


def iter_decision_payloads(connection: sqlite3.Connection) -> Iterator[tuple[str, str]]:
    """Each stored day's ISO date and its decisions as the JSON text
    `append_day` wrote (a list of `PortfolioDecision` payloads), in day
    order, read and decompressed one row at a time."""
    for as_of_date_str, stored in connection.execute(
        "SELECT as_of_date, payload FROM decisions ORDER BY as_of_date, rowid"
    ):
        yield as_of_date_str, _decompressed(stored)


def decision_days(connection: sqlite3.Connection) -> list[str]:
    """The ISO date of each stored day of decisions, in day order; reads
    the date column only."""
    return [
        as_of_date_str
        for (as_of_date_str,) in connection.execute(
            "SELECT as_of_date FROM decisions ORDER BY as_of_date, rowid"
        )
    ]


def iter_signal_tally(
    connection: sqlite3.Connection,
) -> Iterator[tuple[str, str, SignalStatus, VolatilityEstimator | None, int]]:
    """Each stored `signal_tally` row, `(ISO date, analyst id, status,
    volatility estimator or None, signals)`, in day order and, within a
    day, in the order written."""
    for row in connection.execute(
        "SELECT as_of_date, analyst_id, status, volatility_estimator, signals FROM signal_tally "
        "ORDER BY as_of_date, rowid"
    ):
        yield cast(tuple[str, str, SignalStatus, VolatilityEstimator | None, int], row)


def iter_source_returns(
    connection: sqlite3.Connection, start_date: date, end_date: date
) -> Iterator[tuple[str, float, float]]:
    """`(source_id, attributable_return, raw_attributable_return)` of each
    track-record record dated from `start_date` to `end_date` inclusive, in
    the order the records were appended, read one row at a time from the
    rows of the days whose dates overlap that window (`track_record_days`)."""
    start, end = start_date.toordinal(), end_date.toordinal()
    (low, high) = connection.execute(
        "SELECT MIN(first_id), MAX(last_id) FROM track_record_days "
        "WHERE max_date >= ? AND min_date <= ?",
        (start, end),
    ).fetchone()
    if low is None:
        return
    sources = _source_names(connection)
    for analyst, manager, attributable, raw in connection.execute(
        "SELECT analyst_source, manager_source, attributable_return, raw_attributable_return "
        "FROM track_record WHERE id BETWEEN ? AND ? AND as_of_date BETWEEN ? AND ? ORDER BY id",
        (low, high, start, end),
    ):
        if analyst is not None:
            yield sources[analyst], attributable, raw
        if manager is not None:
            yield sources[manager], attributable, raw


def read_lake_time(db_path: Path) -> datetime | None:
    """The lake time stored in the checkpoint file at `db_path`, read
    without creating or changing the file. `None` when the file does not
    exist, holds no cursor row yet, predates the `lake_time` column, or
    records a run made outside a lake session."""
    if not db_path.exists():
        return None
    connection = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(cursor)")}
        if "lake_time" not in columns:
            return None
        row = connection.execute("SELECT lake_time FROM cursor WHERE id = 0").fetchone()
    finally:
        connection.close()
    if row is None or row[0] is None:
        return None
    return datetime.fromisoformat(row[0])
