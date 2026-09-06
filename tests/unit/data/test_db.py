from datetime import date

import pandas as pd

from agentic_trading.data.db import read_fetch_log, record_fetch


def test_record_fetch_persists_a_row(tmp_path) -> None:
    db_path = tmp_path / "fetch_log.db"
    frame = pd.DataFrame({"value": [1.0, 2.0, 3.0]})

    record_fetch(db_path, "test_source", "AAPL", date(2024, 1, 5), frame)

    log = read_fetch_log(db_path)
    assert len(log) == 1
    row = log.iloc[0]
    assert row["source"] == "test_source"
    assert row["identifier"] == "AAPL"
    assert row["as_of_date"] == "2024-01-05"
    assert row["row_count"] == 3


def test_record_fetch_is_idempotent_for_the_same_key(tmp_path) -> None:
    """A second call for the same (source, identifier, as_of_date) must not
    duplicate or overwrite the first row — write-once, never re-fetched
    data must never silently change underfoot."""
    db_path = tmp_path / "fetch_log.db"
    record_fetch(db_path, "test_source", "AAPL", date(2024, 1, 5), pd.DataFrame({"value": [1.0]}))
    record_fetch(db_path, "test_source", "AAPL", date(2024, 1, 5), pd.DataFrame({"value": [999.0]}))

    log = read_fetch_log(db_path)
    assert len(log) == 1
    assert log.iloc[0]["row_count"] == 1


def test_different_keys_produce_separate_rows(tmp_path) -> None:
    db_path = tmp_path / "fetch_log.db"
    record_fetch(db_path, "test_source", "AAPL", date(2024, 1, 5), pd.DataFrame({"value": [1.0]}))
    record_fetch(db_path, "test_source", "MSFT", date(2024, 1, 5), pd.DataFrame({"value": [2.0]}))
    record_fetch(db_path, "test_source", "AAPL", date(2024, 1, 6), pd.DataFrame({"value": [3.0]}))

    assert len(read_fetch_log(db_path)) == 3
    assert len(read_fetch_log(db_path, source="test_source")) == 3
    assert len(read_fetch_log(db_path, source="other_source")) == 0


def test_read_fetch_log_on_missing_db_returns_empty_frame(tmp_path) -> None:
    log = read_fetch_log(tmp_path / "does_not_exist.db")
    assert log.empty
    assert list(log.columns) == ["source", "identifier", "as_of_date", "fetched_at", "row_count"]
