"""Unit tests for agentic_trading.backtest.checkpoint_store.

Covers the append-only day-by-day write path and, specifically, the
`resume` flag on `connect()`: a fresh (non-resumed) run must never see
rows left over in `db_path` from an earlier, unrelated run."""

import json
import math
import sqlite3
import struct
import zlib
from datetime import date
from pathlib import Path

import pytest

from agentic_trading.agents.schemas import PerformanceRecord, PortfolioDecision
from agentic_trading.backtest import checkpoint_store

DAY_ONE = date(2024, 1, 2)
DAY_TWO = date(2024, 1, 3)


def _track_record(as_of_date: date, asset: str = "AAPL") -> PerformanceRecord:
    return PerformanceRecord(
        source_id="risk_minimizing_analyst",
        source_type="analyst",
        asset=asset,
        as_of_date=as_of_date,
        attributable_return=0.01,
        raw_attributable_return=0.02,
    )


def _decision(as_of_date: date, asset: str = "AAPL") -> PortfolioDecision:
    return PortfolioDecision(
        asset=asset,
        as_of_date=as_of_date,
        target_weight=0.5,
        contributing_signals=[],
    )


def _append(connection, as_of_date: date, next_day_index: int) -> None:
    checkpoint_store.append_day(
        connection,
        as_of_date=as_of_date,
        portfolio_value=100_000.0,
        new_track_record=[_track_record(as_of_date)],
        decisions=[_decision(as_of_date)],
        new_trades=[],
        next_day_index=next_day_index,
        start_date=DAY_ONE,
        end_date=DAY_TWO,
        assets=["AAPL"],
        portfolio={"cash": 100_000.0},
        signal_health={},
        pending_decisions=[],
    )


def test_append_day_then_read_all_round_trips(tmp_path: Path) -> None:
    connection = checkpoint_store.connect(tmp_path / "checkpoint.db")
    _append(connection, DAY_ONE, next_day_index=1)

    payload = checkpoint_store.read_all(connection)

    assert payload is not None
    assert payload["next_day_index"] == 1
    assert len(payload["track_record"]) == 1
    assert payload["track_record"][0]["as_of_date"] == DAY_ONE.isoformat()
    assert DAY_ONE.isoformat() in payload["decisions_by_day"]


def test_connect_without_resume_clears_a_stale_file_from_an_earlier_run(
    tmp_path: Path,
) -> None:
    """A non-resumed run reusing a checkpoint_path left over from an
    earlier run must start from a clean slate, not append on top of the
    old run's rows — this is what `resume=False` guards against."""
    db_path = tmp_path / "checkpoint.db"
    first_run = checkpoint_store.connect(db_path)
    _append(first_run, DAY_ONE, next_day_index=1)
    first_run.close()

    second_run = checkpoint_store.connect(db_path, resume=False)

    assert checkpoint_store.read_all(second_run) is None


def test_connect_with_resume_preserves_existing_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "checkpoint.db"
    first_run = checkpoint_store.connect(db_path)
    _append(first_run, DAY_ONE, next_day_index=1)
    first_run.close()

    resumed = checkpoint_store.connect(db_path, resume=True)
    payload = checkpoint_store.read_all(resumed)

    assert payload is not None
    assert payload["next_day_index"] == 1
    assert len(payload["track_record"]) == 1


def test_append_day_only_writes_that_day_not_prior_days(tmp_path: Path) -> None:
    connection = checkpoint_store.connect(tmp_path / "checkpoint.db")
    _append(connection, DAY_ONE, next_day_index=1)
    _append(connection, DAY_TWO, next_day_index=2)

    payload = checkpoint_store.read_all(connection)

    assert payload is not None
    assert len(payload["track_record"]) == 2
    assert payload["next_day_index"] == 2
    assert set(payload["decisions_by_day"]) == {DAY_ONE.isoformat(), DAY_TWO.isoformat()}


def test_cursor_round_trips_option_positions_and_premium_total(tmp_path: Path) -> None:
    connection = checkpoint_store.connect(tmp_path / "checkpoint.db")
    option_positions = {
        "AAPL 2024-03-15 C 185.00": {
            "contract": {
                "underlying": "AAPL",
                "expiry": "2024-03-15",
                "strike": 185.0,
                "option_type": "call",
            },
            "contracts": 2,
            "entry_price": 3.1,
            "last_mid": 3.0,
        }
    }
    checkpoint_store.append_day(
        connection,
        as_of_date=DAY_ONE,
        portfolio_value=100_000.0,
        new_track_record=[],
        decisions=[],
        new_trades=[],
        next_day_index=1,
        start_date=DAY_ONE,
        end_date=DAY_TWO,
        assets=["AAPL"],
        portfolio={"cash": 99_380.0, "positions": {}, "option_positions": option_positions},
        signal_health={},
        pending_decisions=[],
        total_option_premium_paid=620.0,
    )

    payload = checkpoint_store.read_all(connection)

    assert payload is not None
    assert payload["portfolio"]["option_positions"] == option_positions
    assert payload["total_option_premium_paid"] == 620.0


# Option marks with a negative zero, a NaN, the largest and smallest positive doubles and an
# awkward-to-print value, keyed out of sorted order.
_OPTION_MARKS = {
    "SPY 2024-03-15 P 450.00": -0.0,
    "AAPL 2024-03-15 C 185.00": math.nan,
    "SPY 2024-03-15 C 600.00": 1.7976931348623157e308,
    "QQQ 2024-06-21 C 400.00": 5e-324,
    "IWM 2024-06-21 P 180.00": 0.1 + 0.2,
}


def _bits(marks: dict[str, float]) -> list[tuple[str, bytes]]:
    """`marks` as (contract id, IEEE 754 bytes) pairs in key order."""
    return [(contract_id, struct.pack("<d", mid)) for contract_id, mid in marks.items()]


@pytest.mark.parametrize("reader", ["read_all", "read_recent"])
def test_option_marks_round_trip_with_exact_bits_and_key_order(tmp_path: Path, reader: str) -> None:
    """`option_last_mid` written by `append_day` reads back from `read_all` and `read_recent` with
    every value's exact bits, including -0.0 and NaN, and the keys in the order written."""
    connection = checkpoint_store.connect(tmp_path / "checkpoint.db")
    checkpoint_store.append_day(
        connection,
        as_of_date=DAY_ONE,
        portfolio_value=100_000.0,
        new_track_record=[],
        decisions=[],
        new_trades=[],
        next_day_index=1,
        start_date=DAY_ONE,
        end_date=DAY_TWO,
        assets=["AAPL"],
        portfolio={"cash": 100_000.0},
        signal_health={},
        pending_decisions=[],
        option_last_mid=_OPTION_MARKS,
    )

    if reader == "read_all":
        payload = checkpoint_store.read_all(connection)
    else:
        payload = checkpoint_store.read_recent(connection, DAY_ONE, decision_days=1)

    assert payload is not None
    assert _bits(payload["option_last_mid"]) == _bits(_OPTION_MARKS)


def test_a_cursor_without_the_option_marks_column_reads_them_as_none(tmp_path: Path) -> None:
    """A checkpoint whose cursor table has no `option_last_mid` column gets it, NULL, when
    connected for resume, and reads back with `option_last_mid` None."""
    path = tmp_path / "checkpoint.db"
    connection = checkpoint_store.connect(path)
    _append(connection, DAY_ONE, next_day_index=1)
    with connection:
        connection.execute("ALTER TABLE cursor DROP COLUMN option_last_mid")
    connection.close()
    raw = sqlite3.connect(path)
    assert "option_last_mid" not in {row[1] for row in raw.execute("PRAGMA table_info(cursor)")}
    raw.close()

    connection = checkpoint_store.connect(path, resume=True)
    payload = checkpoint_store.read_all(connection)

    assert payload is not None
    assert payload["option_last_mid"] is None
    assert payload["next_day_index"] == 1


def test_decisions_are_stored_compressed_and_read_back_as_written(tmp_path: Path) -> None:
    """A day's decisions are stored zlib-compressed, and `iter_decision_payloads` gives back the
    exact JSON text of the decisions appended."""
    connection = checkpoint_store.connect(tmp_path / "checkpoint.db")
    _append(connection, DAY_ONE, next_day_index=1)
    written = json.dumps([_decision(DAY_ONE).model_dump(mode="json")])

    (stored,) = connection.execute("SELECT payload FROM decisions").fetchone()

    assert isinstance(stored, bytes)
    assert zlib.decompress(stored) == written.encode()
    assert list(checkpoint_store.iter_decision_payloads(connection)) == [
        (DAY_ONE.isoformat(), written)
    ]


@pytest.mark.parametrize("version", [0, checkpoint_store.CHECKPOINT_FORMAT_VERSION - 1])
def test_a_checkpoint_of_another_format_is_refused_for_resume(tmp_path: Path, version: int) -> None:
    """A resume of a checkpoint of another format version raises `CheckpointFormatError`, naming
    the file and telling to start over, and leaves the file as it was; a fresh run on the same
    path starts from an empty checkpoint of the current format."""
    path = tmp_path / "checkpoint.db"
    connection = checkpoint_store.connect(path)
    _append(connection, DAY_ONE, next_day_index=1)
    connection.execute(f"PRAGMA user_version = {version}")
    connection.close()
    before = path.read_bytes()

    with pytest.raises(checkpoint_store.CheckpointFormatError, match="start the run over") as error:
        checkpoint_store.connect(path, resume=True)

    assert str(path) in str(error.value)
    assert path.read_bytes() == before
    fresh = checkpoint_store.connect(path)
    assert checkpoint_store.read_all(fresh) is None
    assert fresh.execute("PRAGMA user_version").fetchone() == (
        checkpoint_store.CHECKPOINT_FORMAT_VERSION,
    )


def _append_with_pending(
    connection: sqlite3.Connection,
    as_of_date: date,
    next_day_index: int,
    pending: list[PortfolioDecision] | None,
    decisions: list[PortfolioDecision] | None = None,
) -> None:
    checkpoint_store.append_day(
        connection,
        as_of_date=as_of_date,
        portfolio_value=100_000.0,
        new_track_record=[],
        decisions=[_decision(as_of_date)] if decisions is None else decisions,
        new_trades=[],
        next_day_index=next_day_index,
        start_date=DAY_ONE,
        end_date=DAY_TWO,
        assets=["AAPL"],
        portfolio={"cash": 100_000.0},
        signal_health={},
        pending_decisions=pending,
    )


@pytest.mark.parametrize("pending", ["none", "same object", "equal copy"])
def test_pending_decisions_equal_to_the_days_are_read_from_the_decisions_table(
    tmp_path: Path, pending: str
) -> None:
    """Pending decisions that are the day's own decisions (not given, the same list or an equal
    one) leave the cursor's column NULL, and every reader gives back the last day's decisions in
    their place."""
    connection = checkpoint_store.connect(tmp_path / "checkpoint.db")
    _append(connection, DAY_ONE, next_day_index=1)
    decisions = [_decision(DAY_TWO, asset="MSFT")]
    given = {
        "none": None,
        "same object": decisions,
        "equal copy": [decision.model_copy() for decision in decisions],
    }[pending]
    _append_with_pending(connection, DAY_TWO, 2, given, decisions)

    assert connection.execute("SELECT pending_decisions FROM cursor").fetchone() == (None,)
    expected = [decision.model_dump(mode="json") for decision in decisions]
    for payload in (
        checkpoint_store.read_all(connection),
        checkpoint_store.read_recent(connection, DAY_ONE, decision_days=1),
    ):
        assert payload is not None
        assert payload["pending_decisions"] == expected


def test_pending_decisions_other_than_the_days_round_trip_through_the_cursor(
    tmp_path: Path,
) -> None:
    """Pending decisions other than the day's own are stored in the cursor and read back."""
    connection = checkpoint_store.connect(tmp_path / "checkpoint.db")
    pending = [_decision(DAY_ONE, asset="MSFT")]
    _append_with_pending(connection, DAY_ONE, 1, pending)

    payload = checkpoint_store.read_all(connection)

    assert payload is not None
    assert payload["pending_decisions"] == [pending[0].model_dump(mode="json")]
    assert payload["decisions_by_day"][DAY_ONE.isoformat()] == [
        _decision(DAY_ONE).model_dump(mode="json")
    ]
