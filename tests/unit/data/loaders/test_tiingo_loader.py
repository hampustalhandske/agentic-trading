from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from agentic_trading.data.loaders.tiingo_loader import load_tiingo_price_history


def _prices_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {"close": [190.0, 191.5, 189.0]},
        index=pd.to_datetime(["2024-06-01", "2024-06-02", "2024-06-03"]),
    )


@patch("agentic_trading.data.loaders.tiingo_loader.require_env", return_value="fake-token")
@patch("agentic_trading.data.loaders.tiingo_loader._fetch_tiingo_prices")
def test_only_rows_on_or_before_as_of_date_are_visible(
    mock_fetch: MagicMock, mock_require_env: MagicMock, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("agentic_trading.data.cache.CACHE_ROOT", tmp_path)
    mock_fetch.return_value = _prices_frame()

    result = load_tiingo_price_history("AAPL", pd.Timestamp("2024-06-02"))

    assert list(result.index) == [pd.Timestamp("2024-06-01"), pd.Timestamp("2024-06-02")]


@patch("agentic_trading.data.loaders.tiingo_loader.require_env", return_value="fake-token")
@patch("agentic_trading.data.loaders.tiingo_loader._fetch_tiingo_prices")
def test_second_call_for_same_as_of_date_is_cached(
    mock_fetch: MagicMock, mock_require_env: MagicMock, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("agentic_trading.data.cache.CACHE_ROOT", tmp_path)
    mock_fetch.return_value = _prices_frame()

    as_of_date = pd.Timestamp("2024-06-02")
    load_tiingo_price_history("AAPL", as_of_date)
    load_tiingo_price_history("AAPL", as_of_date)

    mock_fetch.assert_called_once()


@patch("agentic_trading.data.loaders.tiingo_loader.require_env", return_value="fake-token")
@patch("agentic_trading.data.loaders.tiingo_loader._fetch_tiingo_prices")
def test_a_later_simulated_day_the_same_real_day_reuses_the_cached_series(
    mock_fetch: MagicMock, mock_require_env: MagicMock, tmp_path, monkeypatch
) -> None:
    """A backtest calls this once per simulated day per ticker — the cache
    key is the real wall-clock date, not the simulated `as_of_date`, so a
    later simulated day within the same backtest run must not trigger a
    second network call."""
    monkeypatch.setattr("agentic_trading.data.cache.CACHE_ROOT", tmp_path)
    mock_fetch.return_value = _prices_frame()

    load_tiingo_price_history("AAPL", pd.Timestamp("2024-06-01"))
    load_tiingo_price_history("AAPL", pd.Timestamp("2024-06-02"))

    mock_fetch.assert_called_once()


def test_missing_api_key_raises_clear_error(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("agentic_trading.data.cache.CACHE_ROOT", tmp_path)
    monkeypatch.delenv("TIINGO_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="TIINGO_API_KEY"):
        load_tiingo_price_history("AAPL", pd.Timestamp("2024-06-02"))
