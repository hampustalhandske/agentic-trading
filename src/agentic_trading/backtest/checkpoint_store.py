# checkpoint_store.py

"""Append-only SQLite-backed backtest checkpoint.

Each simulated day appends exactly that day's new rows to eight tables
(`track_record`, `decisions`, `trades`, `unfilled_orders`, `capital_usage`,
`sizing_diagnostics`, `pod_returns`, `pod_cost_days`) via `INSERT` — O(1) per day, O(n)
total for a run of length n — and overwrites two things each day: a single
`cursor` row (next day index, portfolio state, signal health, pending
decisions, the Commander's serialized risk state, the previous close mid
of every option contract the run has marked, the lake time the run reads
at, and the run's own config for `_restore_checkpoint`'s match-check:
dates, assets, settings fingerprint and roster fingerprint), and the
`carried_state` table, one row per agent holding its `carried_state()`
snapshot. Both hold one day's state; the cursor grows only with the
number of option contracts the run has marked. Mirrors the append-only idiom `data/db.py` uses
for the fetch log.

`read_all` reads the whole state back; `read_recent` reads it with only
the recent track record and decisions a horizon-bounded engine holds;
`read_last_day`, `read_unfilled_orders`, `iter_decision_payloads` and
`iter_source_returns` read single columns of one table each.

Each day's decisions are stored as their JSON text, UTF-8 encoded and
zlib-compressed (`DECISIONS_COMPRESSION_LEVEL`); every reader gives back
the JSON text itself. The file records its layout as
`CHECKPOINT_FORMAT_VERSION` in SQLite's `user_version`; `connect` with
`resume=True` refuses a file of any other version with
`CheckpointFormatError`, so a run is resumed only from a checkpoint of the
layout this module writes.

A `carried_state` blob is a pickle: open with `connect` only checkpoint
files this program wrote, never one from an untrusted source."""

import json
import sqlite3
import zlib
from collections.abc import Iterable, Iterator
from datetime import date, datetime
from pathlib import Path
from typing import Any

from agentic_trading.agents.schemas import (
    CapitalUsage,
    PerformanceRecord,
    PodCostDay,
    PortfolioDecision,
    SizingDiagnostics,
    TradeRecord,
    UnfilledOrder,
)

# The layout version this module writes into SQLite's `user_version`; a
# file of any other version (0 for a checkpoint written before versions
# were recorded) is refused for a resume.
CHECKPOINT_FORMAT_VERSION = 2
# zlib level of each day's stored decisions: level 6 compresses a firm
# record's day of decisions about 14x in about 10 ms; the higher levels
# shrink it little more for several times the time.
DECISIONS_COMPRESSION_LEVEL = 6


class CheckpointFormatError(ValueError):
    """A checkpoint file of another layout version than
    `CHECKPOINT_FORMAT_VERSION`, which this module cannot resume from."""


_CREATE_TRACK_RECORD = """
CREATE TABLE IF NOT EXISTS track_record (
    source_id TEXT NOT NULL,
    source_type TEXT NOT NULL,
    asset TEXT NOT NULL,
    as_of_date TEXT NOT NULL,
    attributable_return REAL NOT NULL,
    raw_attributable_return REAL NOT NULL,
    signal_score REAL NOT NULL DEFAULT 0.0,
    realized_return REAL NOT NULL DEFAULT 0.0
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
    pending_decisions TEXT NOT NULL,
    commander_risk_state TEXT,
    total_financing_cost REAL NOT NULL DEFAULT 0.0,
    total_option_premium_paid REAL NOT NULL DEFAULT 0.0,
    option_last_mid TEXT,
    lake_time TEXT,
    settings_fingerprint TEXT,
    roster_fingerprint TEXT
)
"""
# One row per agent, rewritten each day: the agent's `carried_state()`
# snapshot (a pickle) keyed by its agent_id.
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
    ("track_record", "signal_score", "REAL NOT NULL DEFAULT 0.0"),
    ("track_record", "realized_return", "REAL NOT NULL DEFAULT 0.0"),
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
    ("decisions", _CREATE_DECISIONS),
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
    anything in it changes."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
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
        if not resume:
            for name, _ in _TABLES:
                connection.execute(f"DROP TABLE IF EXISTS {name}")
        for _, create in _TABLES:
            connection.execute(create)
        _add_missing_columns(connection)
        connection.execute(f"PRAGMA user_version = {CHECKPOINT_FORMAT_VERSION}")
    return connection


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
    pending_decisions: list[PortfolioDecision],
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
    `carried_state()` keyed by agent_id; when given it replaces every stored
    snapshot, and when `None` the stored snapshots are left as they are.
    `option_last_mid` is the engine's previous close mid per option
    contract, stored as JSON (`None` stores none)."""
    with connection:
        connection.execute(
            "INSERT INTO portfolio_value (as_of_date, value) VALUES (?, ?)",
            (as_of_date.isoformat(), portfolio_value),
        )
        connection.executemany(
            "INSERT INTO track_record (source_id, source_type, asset, as_of_date, "
            "attributable_return, raw_attributable_return, signal_score, realized_return) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    entry.source_id,
                    entry.source_type,
                    entry.asset,
                    entry.as_of_date.isoformat(),
                    entry.attributable_return,
                    entry.raw_attributable_return,
                    entry.signal_score,
                    entry.realized_return,
                )
                for entry in new_track_record
            ],
        )
        connection.execute(
            "INSERT INTO decisions (as_of_date, payload) VALUES (?, ?)",
            (
                as_of_date.isoformat(),
                _compressed(
                    json.dumps([decision.model_dump(mode="json") for decision in decisions])
                ),
            ),
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
                json.dumps([d.model_dump(mode="json") for d in pending_decisions]),
                None if commander_risk_state is None else json.dumps(commander_risk_state),
                total_financing_cost,
                total_option_premium_paid,
                None if option_last_mid is None else json.dumps(option_last_mid),
                None if lake_time is None else lake_time.isoformat(),
                settings_fingerprint,
                roster_fingerprint,
            ),
        )
        if carried_states is not None:
            connection.execute("DELETE FROM carried_state")
            connection.executemany(
                "INSERT INTO carried_state (agent_id, state) VALUES (?, ?)",
                sorted(carried_states.items()),
            )


def _compressed(payload_json: str) -> bytes:
    """`payload_json` as the bytes the `decisions` table stores."""
    return zlib.compress(payload_json.encode("utf-8"), DECISIONS_COMPRESSION_LEVEL)


def _decompressed(stored: bytes) -> str:
    """The JSON text `_compressed` stored as `stored`."""
    return zlib.decompress(stored).decode("utf-8")


_TRACK_RECORD_COLUMNS = (
    "source_id, source_type, asset, as_of_date, attributable_return, "
    "raw_attributable_return, signal_score, realized_return"
)


def _track_record_rows(rows: Iterable[tuple[Any, ...]]) -> list[dict[str, Any]]:
    """`track_record` rows of `_TRACK_RECORD_COLUMNS` as `PerformanceRecord`
    payload dicts."""
    return [
        {
            "source_id": row[0],
            "source_type": row[1],
            "asset": row[2],
            "as_of_date": row[3],
            "attributable_return": row[4],
            "raw_attributable_return": row[5],
            "signal_score": row[6],
            "realized_return": row[7],
        }
        for row in rows
    ]


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
        connection.execute(
            f"SELECT {_TRACK_RECORD_COLUMNS} FROM track_record ORDER BY as_of_date, rowid"
        )
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
    keys describe the rows left out: `track_record_count`, the number of
    rows the table holds, and `track_record_dropped_first_dates`, each
    source_id's earliest ISO date among the rows dated before
    `track_record_from`. Every other key is as `read_all` gives it.
    Returns `None` when there is no cursor row yet. Raises `ValueError`
    when a row dated before `track_record_from` was appended after one
    dated on or after it, since the rows read would then not be the last
    ones appended."""
    payload = _read_state(connection)
    if payload is None:
        return None
    cutoff = track_record_from.isoformat()
    (interleaved,) = connection.execute(
        "SELECT COUNT(*) FROM track_record WHERE as_of_date < ? AND rowid > "
        "(SELECT MIN(rowid) FROM track_record WHERE as_of_date >= ?)",
        (cutoff, cutoff),
    ).fetchone()
    if interleaved:
        raise ValueError(
            f"the checkpoint's track record has {interleaved} row(s) dated before {cutoff} "
            "appended after a row dated on or after it"
        )
    payload["track_record"] = _track_record_rows(
        connection.execute(
            f"SELECT {_TRACK_RECORD_COLUMNS} FROM track_record WHERE as_of_date >= ? "
            "ORDER BY rowid",
            (cutoff,),
        )
    )
    (payload["track_record_count"],) = connection.execute(
        "SELECT COUNT(*) FROM track_record"
    ).fetchone()
    payload["track_record_dropped_first_dates"] = dict(
        connection.execute(
            "SELECT source_id, MIN(as_of_date) FROM track_record WHERE as_of_date < ? "
            "GROUP BY source_id ORDER BY source_id",
            (cutoff,),
        )
    )
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
        "pending_decisions": json.loads(pending_decisions_json),
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


def iter_source_returns(
    connection: sqlite3.Connection, start_date: date, end_date: date
) -> Iterator[tuple[str, float, float]]:
    """`(source_id, attributable_return, raw_attributable_return)` of each
    track-record row dated from `start_date` to `end_date` inclusive, in
    the order the rows were appended, read one row at a time."""
    yield from connection.execute(
        "SELECT source_id, attributable_return, raw_attributable_return FROM track_record "
        "WHERE as_of_date BETWEEN ? AND ? ORDER BY rowid",
        (start_date.isoformat(), end_date.isoformat()),
    )


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
