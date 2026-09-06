from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from agentic_trading.data.loaders.alpha_vantage_loader import load_alpha_vantage_price_history


def _series_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {"close": [100.0, 101.0, 102.0]},
        index=pd.to_datetime(["2024-06-01", "2024-06-02", "2024-06-03"]),
    )


@patch("agentic_trading.data.loaders.alpha_vantage_loader.require_env", return_value="fake-key")
@patch("agentic_trading.data.loaders.alpha_vantage_loader._fetch_alpha_vantage_series")
def test_only_rows_on_or_before_as_of_date_are_visible(
    mock_fetch: MagicMock, mock_require_env: MagicMock, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("agentic_trading.data.cache.CACHE_ROOT", tmp_path)
    mock_fetch.return_value = _series_frame()

    result = load_alpha_vantage_price_history("IBM", pd.Timestamp("2024-06-02"))

    assert list(result.index) == [pd.Timestamp("2024-06-01"), pd.Timestamp("2024-06-02")]


@patch("agentic_trading.data.loaders.alpha_vantage_loader.require_env", return_value="fake-key")
@patch("agentic_trading.data.loaders.alpha_vantage_loader._fetch_alpha_vantage_series")
def test_repeated_calls_the_same_day_do_not_refetch(
    mock_fetch: MagicMock, mock_require_env: MagicMock, tmp_path, monkeypatch
) -> None:
    """The free tier allows ~25 requests/day, so repeated calls must hit
    the cache rather than burning the daily quota."""
    monkeypatch.setattr("agentic_trading.data.cache.CACHE_ROOT", tmp_path)
    mock_fetch.return_value = _series_frame()

    load_alpha_vantage_price_history("IBM", pd.Timestamp("2024-06-02"))
    load_alpha_vantage_price_history("IBM", pd.Timestamp("2024-06-03"))

    mock_fetch.assert_called_once()


def test_missing_api_key_raises_clear_error(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("agentic_trading.data.cache.CACHE_ROOT", tmp_path)
    monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="ALPHAVANTAGE_API_KEY"):
        load_alpha_vantage_price_history("IBM", pd.Timestamp("2024-06-02"))
