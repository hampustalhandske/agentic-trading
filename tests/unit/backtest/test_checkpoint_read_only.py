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


def _append_day_two(path: Path) -> None:
    """Appends a second day to the checkpoint at `path` through a writer that waits for no
    lock (`timeout=0`), as the engine's writer would after its busy timeout."""
    connection = checkpoint_store.connect(path, resume=True)
    connection.execute("PRAGMA busy_timeout = 0")
    checkpoint_store.append_day(
        connection,
        as_of_date=date(2024, 1, 3),
        portfolio_value=100_100.0,
        new_track_record=[],
        decisions=[],
        new_trades=[],
        next_day_index=2,
        start_date=_DAY,
        end_date=date(2024, 1, 3),
        assets=["AAPL"],
        portfolio={"cash": 100_100.0},
        signal_health={},
        pending_decisions=[],
    )
    connection.close()


def test_a_snapshot_is_read_from_a_copy_the_writer_never_waits_for(tmp_path: Path) -> None:
    """A snapshot copies the checkpoint once and is read from the copy: while it is read, the
    run's writer commits without waiting, the snapshot keeps the state it copied, and the copy is
    gone afterwards, outside the checkpoint's directory throughout."""
    store = tmp_path / "store"
    store.mkdir()
    path = _checkpoint(store / "run.checkpoint.sqlite")
    copies = tmp_path / "copies"
    copies.mkdir()

    with checkpoint_store.snapshot(path, directory=copies) as connection:
        rows = connection.execute("SELECT as_of_date FROM portfolio_value")
        assert rows.fetchone() == (_DAY.isoformat(),)
        _append_day_two(path)
        state = checkpoint_store.read_state(connection)
        assert list(copies.iterdir())
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            connection.execute("DELETE FROM portfolio_value")

    assert state is not None
    assert state["portfolio_value_history"] == [(_DAY.isoformat(), 100_000.0)]
    assert list(copies.iterdir()) == []
    assert sorted(child.name for child in store.iterdir()) == ["run.checkpoint.sqlite"]
    with checkpoint_store.snapshot(path, directory=copies) as connection:
        assert checkpoint_store.read_last_day(connection) == date(2024, 1, 3)


def test_a_snapshot_refuses_another_layout_and_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        with checkpoint_store.snapshot(tmp_path / "absent.checkpoint.sqlite"):
            pass
    path = _checkpoint(tmp_path / "run.checkpoint.sqlite")
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA user_version = 1")
    connection.close()

    with pytest.raises(checkpoint_store.CheckpointFormatError):
        with checkpoint_store.snapshot(path):
            pass


def test_a_snapshot_is_never_written_beside_the_checkpoint(tmp_path: Path) -> None:
    path = _checkpoint(tmp_path / "run.checkpoint.sqlite")
    inside = tmp_path / "trial"
    inside.mkdir()

    with pytest.raises(ValueError, match="inside the checkpoint's directory"):
        with checkpoint_store.snapshot(path, directory=inside):
            pass
    assert list(inside.iterdir()) == []


def test_a_file_that_is_not_a_database_is_refused_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "run.checkpoint.sqlite"
    path.write_bytes(b"not a database" * 512)
    before = hashlib.sha256(path.read_bytes()).hexdigest()

    with pytest.raises(sqlite3.DatabaseError):
        checkpoint_store.connect_read_only(path)
    with pytest.raises(sqlite3.DatabaseError):
        with checkpoint_store.snapshot(path):
            pass

    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert sorted(child.name for child in tmp_path.iterdir()) == [path.name]
