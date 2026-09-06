from unittest.mock import MagicMock, patch

import pandas as pd

from agentic_trading.data.loaders.news_loader import load_news_sentiment


@patch("agentic_trading.data.loaders.news_loader.require_env", return_value="fake-token")
@patch("agentic_trading.data.loaders.news_loader.requests.get")
def test_query_is_bounded_by_as_of_date(
    mock_get: MagicMock, mock_require_env: MagicMock, tmp_path, monkeypatch
) -> None:
    """Point-in-time is enforced by the bounded query itself: `to` must
    equal as_of_date so nothing published later can come back."""
    monkeypatch.setattr("agentic_trading.data.cache.CACHE_ROOT", tmp_path)
    news_response = MagicMock()
    news_response.json.return_value = [
        {"datetime": 1717200000, "headline": "Some headline"},
    ]
    sentiment_response = MagicMock()
    sentiment_response.json.return_value = {"companyNewsScore": 0.42}
    mock_get.side_effect = [news_response, sentiment_response]

    as_of_date = pd.Timestamp("2024-06-01")
    result = load_news_sentiment("AAPL", as_of_date)

    news_call_kwargs = mock_get.call_args_list[0].kwargs
    assert news_call_kwargs["params"]["to"] == "2024-06-01"
    assert result.iloc[0]["sentiment_score"] == 0.42


@patch("agentic_trading.data.loaders.news_loader.require_env", return_value="fake-token")
@patch("agentic_trading.data.loaders.news_loader.requests.get")
def test_no_articles_returns_empty_frame_with_expected_columns(
    mock_get: MagicMock, mock_require_env: MagicMock, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("agentic_trading.data.cache.CACHE_ROOT", tmp_path)
    news_response = MagicMock()
    news_response.json.return_value = []
    sentiment_response = MagicMock()
    sentiment_response.json.return_value = {"companyNewsScore": 0.0}
    mock_get.side_effect = [news_response, sentiment_response]

    result = load_news_sentiment("AAPL", pd.Timestamp("2024-06-01"))

    assert list(result.columns) == ["datetime", "headline", "sentiment_score"]
    assert result.empty


@patch("agentic_trading.data.loaders.news_loader.require_env", return_value="fake-token")
@patch("agentic_trading.data.loaders.news_loader.requests.get")
def test_second_call_for_same_as_of_date_is_cached(
    mock_get: MagicMock, mock_require_env: MagicMock, tmp_path, monkeypatch
) -> None:
    """A second call for the same (ticker, as_of_date) must hit the cache,
    not re-query Finnhub — this is what keeps a backtest re-run from
    re-hitting Finnhub's rate-limited free tier on every simulated day."""
    monkeypatch.setattr("agentic_trading.data.cache.CACHE_ROOT", tmp_path)
    news_response = MagicMock()
    news_response.json.return_value = [
        {"datetime": 1717200000, "headline": "Some headline"},
    ]
    sentiment_response = MagicMock()
    sentiment_response.json.return_value = {"companyNewsScore": 0.42}
    mock_get.side_effect = [news_response, sentiment_response]

    as_of_date = pd.Timestamp("2024-06-01")
    load_news_sentiment("AAPL", as_of_date)
    load_news_sentiment("AAPL", as_of_date)

    assert mock_get.call_count == 2
