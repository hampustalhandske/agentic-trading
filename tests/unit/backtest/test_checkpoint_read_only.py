# test_checkpoint_read_only.py

"""`checkpoint_store.connect_read_only`: a checkpoint opened for reading is never changed, a
write through it fails, and a file of another layout or a missing file is refused."""

import hashlib
import sqlite3
from datetime import date
from pathlib import Path

import pytest

from agentic_trading.backtest import checkpoint_store

_DAY = date(2024, 1, 2)


def _checkpoint(path: Path) -> Path:
    connection = checkpoint_store.connect(path)
    checkpoint_store.append_day(
        connection,
        as_of_date=_DAY,
        portfolio_value=100_000.0,
        new_track_record=[],
        decisions=[],
        new_trades=[],
        next_day_index=1,
        start_date=_DAY,
        end_date=_DAY,
        assets=["AAPL"],
        portfolio={"cash": 100_000.0},
        signal_health={"analyst": {"active": 1, "unavailable": 0, "error": 0}},
        pending_decisions=[],
    )
    connection.close()
    return path


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_reading_a_checkpoint_read_only_leaves_the_file_as_it_was(tmp_path: Path) -> None:
    path = _checkpoint(tmp_path / "run.checkpoint.sqlite")
    before = _digest(path)

    connection = checkpoint_store.connect_read_only(path)
    try:
        state = checkpoint_store.read_state(connection)
        list(checkpoint_store.iter_source_returns(connection, _DAY, _DAY))
    finally:
        connection.close()

    assert state is not None
    assert state["portfolio_value_history"] == [(_DAY.isoformat(), 100_000.0)]
    assert state["signal_health"] == {"analyst": {"active": 1, "unavailable": 0, "error": 0}}
    assert "track_record" not in state and "decisions_by_day" not in state
    assert _digest(path) == before
    assert sorted(child.name for child in tmp_path.iterdir()) == ["run.checkpoint.sqlite"]


def test_a_write_through_a_read_only_connection_fails(tmp_path: Path) -> None:
    path = _checkpoint(tmp_path / "run.checkpoint.sqlite")
    connection = checkpoint_store.connect_read_only(path)
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            connection.execute("DELETE FROM portfolio_value")
    finally:
        connection.close()


def test_a_missing_checkpoint_is_not_created(tmp_path: Path) -> None:
    path = tmp_path / "absent.checkpoint.sqlite"

    with pytest.raises(FileNotFoundError):
        checkpoint_store.connect_read_only(path)
    assert not path.exists()


def test_a_checkpoint_of_another_layout_is_refused_unchanged(tmp_path: Path) -> None:
    path = _checkpoint(tmp_path / "run.checkpoint.sqlite")
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA user_version = 1")
    connection.close()
    before = _digest(path)

    with pytest.raises(checkpoint_store.CheckpointFormatError, match="format 1"):
        checkpoint_store.connect_read_only(path)
    assert _digest(path) == before
