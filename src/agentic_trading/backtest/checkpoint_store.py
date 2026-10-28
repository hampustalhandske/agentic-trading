# checkpoint_store.py

"""Append-only SQLite-backed backtest checkpoint.

Each simulated day appends exactly that day's new rows to six tables
(`track_record`, `decisions`, `trades`, `unfilled_orders`, `capital_usage`,
`pod_returns`) via `INSERT` — O(1) per day, O(n) total for a run of length n — and keeps a
single small `cursor` row (next
day index, portfolio state, signal health, pending decisions, the
Commander's serialized risk state, and the run's own config for
`_restore_checkpoint`'s match-check) as the only
thing overwritten each day, since that piece stays small regardless of
run length. Mirrors the append-only idiom `data/db.py` uses for the fetch
log."""

import json
import sqlite3
from datetime import date
from pathlib import Path
from typing import Any

from agentic_trading.agents.schemas import (
    CapitalUsage,
    PerformanceRecord,
    PortfolioDecision,
    TradeRecord,
    UnfilledOrder,
)

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
    payload TEXT NOT NULL
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
_CREATE_POD_RETURNS = """
CREATE TABLE IF NOT EXISTS pod_returns (
    as_of_date TEXT NOT NULL,
    pod_id TEXT NOT NULL,
    value REAL NOT NULL
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
    total_option_premium_paid REAL NOT NULL DEFAULT 0.0
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
)


def _add_missing_columns(connection: sqlite3.Connection) -> None:
    for table, column, definition in _ADDITIVE_COLUMNS:
        existing = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
        if column not in existing:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def connect(db_path: Path, *, resume: bool = False) -> sqlite3.Connection:
    """Opens (creating tables if needed) the checkpoint database at
    `db_path`. When `resume` is `False`, any rows already in the file are
    cleared first — a fresh (non-resumed) run always starts from an empty
    checkpoint, even if `db_path` points at a file left over from an
    earlier, unrelated run, rather than silently appending its rows on
    top of that leftover state."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.execute(_CREATE_TRACK_RECORD)
    connection.execute(_CREATE_DECISIONS)
    connection.execute(_CREATE_TRADES)
    connection.execute(_CREATE_UNFILLED_ORDERS)
    connection.execute(_CREATE_CAPITAL_USAGE)
    connection.execute(_CREATE_POD_RETURNS)
    connection.execute(_CREATE_PORTFOLIO_VALUE)
    connection.execute(_CREATE_CURSOR)
    _add_missing_columns(connection)
    if not resume:
        with connection:
            connection.execute("DELETE FROM track_record")
            connection.execute("DELETE FROM decisions")
            connection.execute("DELETE FROM trades")
            connection.execute("DELETE FROM unfilled_orders")
            connection.execute("DELETE FROM capital_usage")
            connection.execute("DELETE FROM pod_returns")
            connection.execute("DELETE FROM portfolio_value")
            connection.execute("DELETE FROM cursor")
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
    new_pod_returns: list[tuple[str, date, float]] | None = None,
) -> None:
    """Appends exactly this simulated day's new rows — never touches a
    prior day's already-written rows — plus overwrites the single small
    `cursor` row with this day's resumability state. `commander_risk_state`
    is the Commander's `risk_state()` already dumped to JSON-compatible
    form, or `None` for a commander that carries no risk state.
    `new_pod_returns` are the (pod id, date, return) entries first seen
    this day."""
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
                json.dumps([decision.model_dump(mode="json") for decision in decisions]),
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
            "INSERT INTO pod_returns (as_of_date, pod_id, value) VALUES (?, ?, ?)",
            [
                (return_date.isoformat(), pod_id, value)
                for pod_id, return_date, value in new_pod_returns or []
            ],
        )
        connection.execute(
            """
            INSERT INTO cursor
                (id, next_day_index, start_date, end_date, assets, portfolio,
                 signal_health, pending_decisions, commander_risk_state,
                 total_financing_cost, total_option_premium_paid)
            VALUES (0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                next_day_index = excluded.next_day_index,
                portfolio = excluded.portfolio,
                signal_health = excluded.signal_health,
                pending_decisions = excluded.pending_decisions,
                commander_risk_state = excluded.commander_risk_state,
                total_financing_cost = excluded.total_financing_cost,
                total_option_premium_paid = excluded.total_option_premium_paid
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
            ),
        )


def read_all(connection: sqlite3.Connection) -> dict[str, Any] | None:
    """Bulk-reads the full checkpoint state in the shape
    `BacktestEngine._restore_checkpoint()` (and any offline reader, e.g. the
    GUI's saved-run browser) needs — one read at startup, not per-day.
    Returns `None` if there's no cursor row yet (a fresh, never-checkpointed
    run)."""
    cursor_row = connection.execute(
        "SELECT next_day_index, start_date, end_date, assets, portfolio, "
        "signal_health, pending_decisions, commander_risk_state, total_financing_cost, "
        "total_option_premium_paid FROM cursor WHERE id = 0"
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
    ) = cursor_row

    portfolio_value_history = [
        (as_of_date_str, value)
        for as_of_date_str, value in connection.execute(
            "SELECT as_of_date, value FROM portfolio_value ORDER BY as_of_date"
        )
    ]
    track_record = [
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
        for row in connection.execute(
            "SELECT source_id, source_type, asset, as_of_date, attributable_return, "
            "raw_attributable_return, signal_score, realized_return FROM track_record "
            "ORDER BY as_of_date"
        )
    ]
    decisions_by_day = {
        as_of_date_str: json.loads(payload)
        for as_of_date_str, payload in connection.execute(
            "SELECT as_of_date, payload FROM decisions ORDER BY as_of_date"
        )
    }
    trade_log = [
        json.loads(payload)
        for _as_of_date_str, payload in connection.execute(
            "SELECT as_of_date, payload FROM trades ORDER BY as_of_date"
        )
    ]
    unfilled_orders = [
        json.loads(payload)
        for _as_of_date_str, payload in connection.execute(
            "SELECT as_of_date, payload FROM unfilled_orders ORDER BY as_of_date"
        )
    ]
    capital_usage = [
        json.loads(payload)
        for _as_of_date_str, payload in connection.execute(
            "SELECT as_of_date, payload FROM capital_usage ORDER BY as_of_date"
        )
    ]
    pod_return_history: dict[str, list[tuple[str, float]]] = {}
    for pod_id, return_date, value in connection.execute(
        "SELECT pod_id, as_of_date, value FROM pod_returns ORDER BY as_of_date, rowid"
    ):
        pod_return_history.setdefault(pod_id, []).append((return_date, value))
    return {
        "next_day_index": next_day_index,
        "config": {
            "start_date": start_date,
            "end_date": end_date,
            "assets": json.loads(assets_json),
        },
        "portfolio": json.loads(portfolio_json),
        "signal_health": json.loads(signal_health_json),
        "pending_decisions": json.loads(pending_decisions_json),
        "commander_risk_state": (
            None if commander_risk_state_json is None else json.loads(commander_risk_state_json)
        ),
        "total_financing_cost": float(total_financing_cost),
        "total_option_premium_paid": float(total_option_premium_paid),
        "track_record": track_record,
        "decisions_by_day": decisions_by_day,
        "trade_log": trade_log,
        "unfilled_orders": unfilled_orders,
        "capital_usage": capital_usage,
        "pod_return_history": pod_return_history,
        "portfolio_value_history": portfolio_value_history,
    }
