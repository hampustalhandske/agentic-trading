from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from agentic_trading.data.loaders.macro_loader import DGS3MO, load_fred_series


@patch("agentic_trading.data.loaders.macro_loader.require_env", return_value="fake-key")
@patch("agentic_trading.data.loaders.macro_loader.Fred")
def test_load_fred_series_passes_realtime_end_for_point_in_time(
    mock_fred_cls: MagicMock, mock_require_env: MagicMock, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("agentic_trading.data.cache.CACHE_ROOT", tmp_path)
    mock_fred = mock_fred_cls.return_value
    mock_fred.get_series.return_value = pd.Series([5.0, 5.1])

    as_of_date = pd.Timestamp("2024-06-01")
    load_fred_series(DGS3MO, as_of_date)

    mock_fred.get_series.assert_called_once_with(DGS3MO, realtime_end=as_of_date.date())


@patch("agentic_trading.data.loaders.macro_loader.require_env", return_value="fake-key")
@patch("agentic_trading.data.loaders.macro_loader.Fred")
def test_load_fred_series_is_cached_and_does_not_refetch(
    mock_fred_cls: MagicMock, mock_require_env: MagicMock, tmp_path, monkeypatch
) -> None:
    """A second call for the same (series, as_of_date) must hit the cache,
    not re-query FRED — this is what keeps a backtest re-run from re-hitting
    the FRED API on every simulated day."""
    monkeypatch.setattr("agentic_trading.data.cache.CACHE_ROOT", tmp_path)
    mock_fred = mock_fred_cls.return_value
    mock_fred.get_series.return_value = pd.Series([5.0, 5.1])

    as_of_date = pd.Timestamp("2024-06-01")
    load_fred_series(DGS3MO, as_of_date)
    load_fred_series(DGS3MO, as_of_date)

    mock_fred.get_series.assert_called_once()


@patch("agentic_trading.data.loaders.macro_loader.require_env", return_value="fake-key")
def test_unsupported_series_raises_value_error(mock_require_env: MagicMock) -> None:
    with pytest.raises(ValueError):
        load_fred_series("NOT_A_SERIES", pd.Timestamp("2024-06-01"))
