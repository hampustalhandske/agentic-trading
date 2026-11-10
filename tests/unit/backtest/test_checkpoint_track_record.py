"""The checkpoint's track record reads back exactly what one row per record gave.

`checkpoint_store` stores an analyst record and the manager record that follows it with the same
outcome as one row, with integer-coded sources, assets and dates and a per-append index. Each
test appends the same records to a checkpoint and to `_OneRowPerRecord`, a reference holding one
row per record in the layout and with the queries of a checkpoint of format 3, and requires
`read_all`, `read_recent` and `iter_source_returns` to give what the reference gives, value for
value and in the same order, with the same error when the dates interleave."""

import random
import sqlite3
from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest

from agentic_trading.agents.schemas import PerformanceRecord
from agentic_trading.backtest import checkpoint_store

_START = date(2024, 1, 2)
_ANALYSTS = ("risk_minimizing_analyst", "gbm_momentum_analyst", "shared_id", "fred_analyst")
_MANAGERS = ("risk_manager", "alt_data_manager", "shared_id")
_ASSETS = ("SPY", "QQQ", "TLT", "SPY 2024-03-15 P 450.00", "EURUSD=X")
# Values with every storage class SQLite gives a REAL: integral (stored as an integer), tiny,
# huge, negative and inexact.
_VALUES = (0.0, 1.0, -2.0, 5e-324, 1.7976931348623157e308, -0.013, 0.1 + 0.2, 3.25)


class _OneRowPerRecord:
    """The track record as a checkpoint of format 3 stored and read it: one row per record, in
    the table and with the queries of that format."""

    _COLUMNS = (
        "source_id, source_type, asset, as_of_date, attributable_return, "
        "raw_attributable_return, signal_score, realized_return"
    )

    def __init__(self) -> None:
        self.connection = sqlite3.connect(":memory:")
        self.connection.execute(
            "CREATE TABLE track_record (source_id TEXT NOT NULL, source_type TEXT NOT NULL, "
            "asset TEXT NOT NULL, as_of_date TEXT NOT NULL, attributable_return REAL NOT NULL, "
            "raw_attributable_return REAL NOT NULL, signal_score REAL NOT NULL DEFAULT 0.0, "
            "realized_return REAL NOT NULL DEFAULT 0.0)"
        )

    def append(self, records: Sequence[PerformanceRecord]) -> None:
        self.connection.executemany(
            "INSERT INTO track_record VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
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
                for entry in records
            ],
        )

    @staticmethod
    def _rows(rows: Any) -> list[dict[str, Any]]:
        names = (
            "source_id",
            "source_type",
            "asset",
            "as_of_date",
            "attributable_return",
            "raw_attributable_return",
            "signal_score",
            "realized_return",
        )
        return [dict(zip(names, row, strict=True)) for row in rows]

    def read_all(self) -> list[dict[str, Any]]:
        return self._rows(
            self.connection.execute(
                f"SELECT {self._COLUMNS} FROM track_record ORDER BY as_of_date, rowid"
            )
        )

    def read_recent(self, track_record_from: date) -> dict[str, Any]:
        cutoff = track_record_from.isoformat()
        (interleaved,) = self.connection.execute(
            "SELECT COUNT(*) FROM track_record WHERE as_of_date < ? AND rowid > "
            "(SELECT MIN(rowid) FROM track_record WHERE as_of_date >= ?)",
            (cutoff, cutoff),
        ).fetchone()
        if interleaved:
            raise ValueError(
                f"the checkpoint's track record has {interleaved} row(s) dated before {cutoff} "
                "appended after a row dated on or after it"
            )
        (count,) = self.connection.execute("SELECT COUNT(*) FROM track_record").fetchone()
        return {
            "track_record": self._rows(
                self.connection.execute(
                    f"SELECT {self._COLUMNS} FROM track_record WHERE as_of_date >= ? "
                    "ORDER BY rowid",
                    (cutoff,),
                )
            ),
            "track_record_count": count,
            "track_record_dropped_first_dates": dict(
                self.connection.execute(
                    "SELECT source_id, MIN(as_of_date) FROM track_record WHERE as_of_date < ? "
                    "GROUP BY source_id ORDER BY source_id",
                    (cutoff,),
                )
            ),
        }

    def iter_source_returns(self, start: date, end: date) -> list[tuple[str, float, float]]:
        return list(
            self.connection.execute(
                "SELECT source_id, attributable_return, raw_attributable_return FROM "
                "track_record WHERE as_of_date BETWEEN ? AND ? ORDER BY rowid",
                (start.isoformat(), end.isoformat()),
            )
        )


def _record(
    rng: random.Random, source_type: str, day: date, outcome: tuple[Any, ...] | None = None
) -> PerformanceRecord:
    asset, attributable, raw, score, realized = outcome or (
        rng.choice(_ASSETS),
        rng.choice(_VALUES),
        rng.choice(_VALUES),
        rng.choice(_VALUES),
        rng.choice(_VALUES),
    )
    return PerformanceRecord(
        source_id=rng.choice(_ANALYSTS if source_type == "analyst" else _MANAGERS),
        source_type=source_type,  # type: ignore[arg-type]
        asset=asset,
        as_of_date=day,
        attributable_return=attributable,
        raw_attributable_return=raw,
        signal_score=score,
        realized_return=realized,
    )


def _day_records(rng: random.Random, day: date, dates: Sequence[date]) -> list[PerformanceRecord]:
    """A day's records: mostly analyst-then-manager pairs of one outcome, as the engine appends
    them, mixed with lone analyst and lone manager records, a pair whose outcomes differ in one
    field (or, with several `dates`, in its date) and a manager followed by an analyst; each
    record dated one of `dates`."""
    records: list[PerformanceRecord] = []
    for _ in range(rng.randrange(0, 9)):
        record_date = rng.choice(dates)
        kind = rng.random()
        analyst = _record(rng, "analyst", record_date)
        outcome = (
            analyst.asset,
            analyst.attributable_return,
            analyst.raw_attributable_return,
            analyst.signal_score,
            analyst.realized_return,
        )
        if kind < 0.6:
            records += [analyst, _record(rng, "manager", record_date, outcome)]
        elif kind < 0.7:
            records.append(analyst)
        elif kind < 0.8:
            records.append(_record(rng, "manager", record_date))
        elif kind < 0.9:
            changed = list(outcome)
            changed[rng.randrange(1, 5)] = 0.5
            records += [analyst, _record(rng, "manager", record_date, tuple(changed))]
            if len(dates) > 1 and rng.random() < 0.5:
                records[-1] = records[-1].model_copy(
                    update={"as_of_date": record_date + timedelta(1)}
                )
        else:
            records += [_record(rng, "manager", record_date, outcome), analyst]
    return records


def _append(
    connection: sqlite3.Connection, day: date, index: int, records: list[PerformanceRecord]
) -> None:
    checkpoint_store.append_day(
        connection,
        as_of_date=day,
        portfolio_value=1.0,
        new_track_record=records,
        decisions=[],
        new_trades=[],
        next_day_index=index + 1,
        start_date=_START,
        end_date=_START + timedelta(days=60),
        assets=list(_ASSETS),
        portfolio={"cash": 1.0},
        signal_health={},
    )


def _build(
    tmp_path: Path, seed: int, *, interleave: bool
) -> tuple[sqlite3.Connection, _OneRowPerRecord, list[date]]:
    """A checkpoint and its reference holding the same 25 days of records; with `interleave`,
    a day's records may be dated up to three days before it or one after it."""
    rng = random.Random(seed)
    connection = checkpoint_store.connect(tmp_path / f"checkpoint_{seed}.db")
    reference = _OneRowPerRecord()
    days = [_START + timedelta(days=offset) for offset in range(25)]
    for index, day in enumerate(days):
        dates = [day + timedelta(days=shift) for shift in (-3, -1, 0, 1)] if interleave else [day]
        records = _day_records(rng, day, dates)
        _append(connection, day, index, records)
        reference.append(records)
    return connection, reference, days


def _cutoffs(days: list[date]) -> list[date]:
    return [days[0] + timedelta(days=offset) for offset in range(-5, len(days) + 5)]


@pytest.mark.parametrize("seed", range(6))
def test_reads_equal_one_row_per_record(tmp_path: Path, seed: int) -> None:
    """With every record dated its append day, `read_all`, `read_recent` at every cutoff and
    `iter_source_returns` over every window give what one row per record gives."""
    connection, reference, days = _build(tmp_path, seed, interleave=False)

    full = checkpoint_store.read_all(connection)
    assert full is not None
    assert full["track_record"] == reference.read_all()
    assert len(full["track_record"]) > 50
    for cutoff in _cutoffs(days):
        recent = checkpoint_store.read_recent(connection, cutoff, decision_days=2)
        assert recent is not None
        assert {key: recent[key] for key in reference.read_recent(cutoff)} == (
            reference.read_recent(cutoff)
        ), cutoff
    for start in _cutoffs(days):
        for length in (0, 1, 4, 40):
            end = start + timedelta(days=length)
            assert list(checkpoint_store.iter_source_returns(connection, start, end)) == (
                reference.iter_source_returns(start, end)
            ), (start, end)


@pytest.mark.parametrize("seed", range(6))
def test_interleaved_dates_read_and_fail_as_one_row_per_record(tmp_path: Path, seed: int) -> None:
    """With records dated off their append day, every read gives what one row per record gives,
    and `read_recent` raises the same error at every cutoff where the dates interleave."""
    connection, reference, days = _build(tmp_path, seed, interleave=True)

    full = checkpoint_store.read_all(connection)
    assert full is not None
    assert full["track_record"] == reference.read_all()
    raised = 0
    for cutoff in _cutoffs(days):
        try:
            expected: dict[str, Any] | str = reference.read_recent(cutoff)
        except ValueError as error:
            expected = str(error)
        try:
            recent = checkpoint_store.read_recent(connection, cutoff, decision_days=2)
            assert recent is not None
            actual: dict[str, Any] | str = {key: recent[key] for key in expected}
        except ValueError as error:
            actual = str(error)
        assert actual == expected, cutoff
        raised += isinstance(expected, str)
    assert raised
    for start in _cutoffs(days):
        for length in (0, 2, 40):
            end = start + timedelta(days=length)
            assert list(checkpoint_store.iter_source_returns(connection, start, end)) == (
                reference.iter_source_returns(start, end)
            ), (start, end)


def test_an_engine_pair_is_one_row(tmp_path: Path) -> None:
    """An analyst record followed by its manager's record of the same outcome takes one row
    carrying both sources; a record without such a partner takes a row of its own."""
    rng = random.Random(0)
    analyst = _record(rng, "analyst", _START)
    outcome = (
        analyst.asset,
        analyst.attributable_return,
        analyst.raw_attributable_return,
        analyst.signal_score,
        analyst.realized_return,
    )
    manager = _record(rng, "manager", _START, outcome)
    lone = _record(rng, "manager", _START)
    connection = checkpoint_store.connect(tmp_path / "checkpoint.db")

    _append(connection, _START, 0, [analyst, manager, lone])

    rows = connection.execute(
        "SELECT analyst_source IS NOT NULL, manager_source IS NOT NULL FROM track_record "
        "ORDER BY id"
    ).fetchall()
    assert rows == [(1, 1), (0, 1)]
    recent = checkpoint_store.read_recent(connection, _START, decision_days=1)
    assert recent is not None
    assert [PerformanceRecord.model_validate(entry) for entry in recent["track_record"]] == [
        analyst,
        manager,
        lone,
    ]
    assert recent["track_record_count"] == 3
