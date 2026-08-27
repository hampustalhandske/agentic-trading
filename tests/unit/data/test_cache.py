from datetime import date
from unittest.mock import MagicMock

import pandas as pd

from agentic_trading.data.cache import cached_fetch


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
