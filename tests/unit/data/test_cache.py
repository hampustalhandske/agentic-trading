from datetime import UTC, date, datetime, timedelta
from unittest.mock import MagicMock

import pandas as pd
import pytest

from agentic_trading.data.cache import cached_fetch
from agentic_trading.data.db import read_fetch_log


def test_cache_miss_calls_fetch_fn_and_persists(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("agentic_trading.data.cache.CACHE_ROOT", tmp_path)
    fetch_fn = MagicMock(return_value=pd.DataFrame({"value": [1.0, 2.0]}))

    result = cached_fetch("test_source", "AAPL", date(2024, 1, 5), fetch_fn)

    fetch_fn.assert_called_once()
    pd.testing.assert_frame_equal(result, pd.DataFrame({"value": [1.0, 2.0]}))
    assert (tmp_path / "test_source" / "AAPL_2024-01-05.parquet").exists()


def test_cache_hit_never_calls_fetch_fn(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("agentic_trading.data.cache.CACHE_ROOT", tmp_path)
    fetch_fn = MagicMock(return_value=pd.DataFrame({"value": [1.0]}))

    cached_fetch("test_source", "AAPL", date(2024, 1, 5), fetch_fn)
    second_fetch_fn = MagicMock(return_value=pd.DataFrame({"value": [999.0]}))
    result = cached_fetch("test_source", "AAPL", date(2024, 1, 5), second_fetch_fn)

    second_fetch_fn.assert_not_called()
    pd.testing.assert_frame_equal(result, pd.DataFrame({"value": [1.0]}))


def test_different_identifiers_are_cached_separately(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("agentic_trading.data.cache.CACHE_ROOT", tmp_path)
    fetch_fn = MagicMock(
        side_effect=[pd.DataFrame({"value": [1.0]}), pd.DataFrame({"value": [2.0]})]
    )

    first = cached_fetch("test_source", "AAPL", date(2024, 1, 5), fetch_fn)
    second = cached_fetch("test_source", "MSFT", date(2024, 1, 5), fetch_fn)

    assert fetch_fn.call_count == 2
    assert first["value"].iloc[0] == 1.0
    assert second["value"].iloc[0] == 2.0


def test_cache_miss_records_the_fetch_in_the_sqlite_log(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("agentic_trading.data.cache.CACHE_ROOT", tmp_path)
    fetch_fn = MagicMock(return_value=pd.DataFrame({"value": [1.0, 2.0]}))

    cached_fetch("test_source", "AAPL", date(2024, 1, 5), fetch_fn)

    log = read_fetch_log(tmp_path / "fetch_log.db")
    assert len(log) == 1
    assert log.iloc[0]["source"] == "test_source"
    assert log.iloc[0]["row_count"] == 2


def test_cache_hit_does_not_touch_the_sqlite_log_again(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("agentic_trading.data.cache.CACHE_ROOT", tmp_path)
    first_fetch_fn = MagicMock(return_value=pd.DataFrame({"value": [1.0]}))
    second_fetch_fn = MagicMock(return_value=pd.DataFrame({"value": [999.0]}))
    cached_fetch("test_source", "AAPL", date(2024, 1, 5), first_fetch_fn)
    cached_fetch("test_source", "AAPL", date(2024, 1, 5), second_fetch_fn)

    log = read_fetch_log(tmp_path / "fetch_log.db")
    assert len(log) == 1
    assert log.iloc[0]["row_count"] == 1


def test_a_failing_source_is_not_retried_within_the_failure_ttl(tmp_path, monkeypatch) -> None:
    """This is the FINRA-short-interest bottleneck found in the Task 0
    audit: a source that's currently down must not pay its full
    connection/timeout cost again for every simulated day in one backtest
    run — the failure is remembered until the TTL elapses."""
    monkeypatch.setattr("agentic_trading.data.cache.CACHE_ROOT", tmp_path)
    monkeypatch.setattr("agentic_trading.data.cache._recent_failures", {})
    fixed_now = datetime(2024, 1, 5, 12, 0, 0, tzinfo=UTC)
    monkeypatch.setattr("agentic_trading.data.cache._now", lambda: fixed_now)
    fetch_fn = MagicMock(side_effect=RuntimeError("endpoint is down"))

    with pytest.raises(RuntimeError, match="endpoint is down"):
        cached_fetch("flaky_source", "AAPL", date(2024, 1, 5), fetch_fn)

    with pytest.raises(RuntimeError, match="failed within the last"):
        cached_fetch("flaky_source", "AAPL", date(2024, 1, 6), fetch_fn)

    fetch_fn.assert_called_once()


def test_a_failing_source_is_retried_once_the_ttl_elapses(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("agentic_trading.data.cache.CACHE_ROOT", tmp_path)
    monkeypatch.setattr("agentic_trading.data.cache._recent_failures", {})
    clock = {"now": datetime(2024, 1, 5, 12, 0, 0, tzinfo=UTC)}
    monkeypatch.setattr("agentic_trading.data.cache._now", lambda: clock["now"])
    fetch_fn = MagicMock(
        side_effect=[RuntimeError("endpoint is down"), pd.DataFrame({"value": [1.0]})]
    )

    with pytest.raises(RuntimeError):
        cached_fetch("flaky_source", "AAPL", date(2024, 1, 5), fetch_fn)

    clock["now"] += timedelta(minutes=10)
    result = cached_fetch("flaky_source", "AAPL", date(2024, 1, 6), fetch_fn)

    assert fetch_fn.call_count == 2
    assert result["value"].iloc[0] == 1.0
