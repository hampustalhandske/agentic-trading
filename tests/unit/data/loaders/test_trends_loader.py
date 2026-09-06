from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd

from agentic_trading.data.loaders.trends_loader import load_search_interest


def _interest_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {"interest": [40, 55, 90]},
        index=pd.to_datetime(["2024-06-01", "2024-06-02", "2024-06-03"]),
    )


@patch("agentic_trading.data.loaders.trends_loader._fetch_trends")
def test_only_rows_on_or_before_as_of_date_are_visible(
    mock_fetch: MagicMock, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("agentic_trading.data.cache.CACHE_ROOT", tmp_path)
    mock_fetch.return_value = _interest_frame()

    result = load_search_interest("AAPL", pd.Timestamp("2024-06-02"))

    assert list(result.index) == [pd.Timestamp("2024-06-01"), pd.Timestamp("2024-06-02")]


@patch("agentic_trading.data.loaders.trends_loader.TrendReq")
def test_provisional_is_partial_rows_are_dropped(mock_trendreq_cls: MagicMock) -> None:
    """A still-provisional Trends data point must never be treated as
    final — it can still be revised."""
    from agentic_trading.data.loaders.trends_loader import _fetch_trends

    mock_pytrends = mock_trendreq_cls.return_value
    raw = pd.DataFrame(
        {"AAPL": [40, 55, 90], "isPartial": [False, False, True]},
        index=pd.to_datetime(["2024-06-01", "2024-06-02", "2024-06-03"]),
    )
    mock_pytrends.interest_over_time.return_value = raw

    result = _fetch_trends("AAPL", date(2024, 3, 5), date(2024, 6, 3))

    assert list(result.index) == [pd.Timestamp("2024-06-01"), pd.Timestamp("2024-06-02")]


@patch("agentic_trading.data.loaders.trends_loader._fetch_trends")
def test_second_call_for_same_as_of_date_is_cached(
    mock_fetch: MagicMock, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("agentic_trading.data.cache.CACHE_ROOT", tmp_path)
    mock_fetch.return_value = _interest_frame()

    as_of_date = pd.Timestamp("2024-06-02")
    load_search_interest("AAPL", as_of_date)
    load_search_interest("AAPL", as_of_date)

    mock_fetch.assert_called_once()


@patch("agentic_trading.data.loaders.trends_loader._fetch_trends")
def test_a_later_simulated_day_the_same_real_day_reuses_the_cached_series(
    mock_fetch: MagicMock, tmp_path, monkeypatch
) -> None:
    """A backtest calls this once per simulated day per ticker — the cache
    key is the real wall-clock date, not the simulated `as_of_date`, so a
    later simulated day within the same backtest run must not trigger a
    second network call."""
    monkeypatch.setattr("agentic_trading.data.cache.CACHE_ROOT", tmp_path)
    mock_fetch.return_value = _interest_frame()

    load_search_interest("AAPL", pd.Timestamp("2024-06-01"))
    load_search_interest("AAPL", pd.Timestamp("2024-06-02"))
    load_search_interest("AAPL", pd.Timestamp("2024-06-03"))

    mock_fetch.assert_called_once()
