from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from agentic_trading.data.loaders.reddit_loader import load_mention_counts


def _mentions_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "created_utc": pd.to_datetime(["2024-06-01", "2024-06-02", "2024-06-03"]),
            "score": [10, 25, 100],
            "num_comments": [2, 5, 40],
        }
    )


@patch("agentic_trading.data.loaders.reddit_loader._fetch_mentions")
def test_only_posts_created_on_or_before_as_of_date_are_visible(
    mock_fetch: MagicMock, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("agentic_trading.data.cache.CACHE_ROOT", tmp_path)
    mock_fetch.return_value = _mentions_frame()

    result = load_mention_counts("GME", pd.Timestamp("2024-06-02"), subreddits=("stocks",))

    assert list(result["created_utc"]) == [pd.Timestamp("2024-06-01"), pd.Timestamp("2024-06-02")]


@patch("agentic_trading.data.loaders.reddit_loader._fetch_mentions")
def test_second_call_for_same_as_of_date_is_cached(
    mock_fetch: MagicMock, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("agentic_trading.data.cache.CACHE_ROOT", tmp_path)
    mock_fetch.return_value = _mentions_frame()

    as_of_date = pd.Timestamp("2024-06-02")
    load_mention_counts("GME", as_of_date, subreddits=("stocks",))
    load_mention_counts("GME", as_of_date, subreddits=("stocks",))

    mock_fetch.assert_called_once()


def test_missing_client_credentials_raise_clear_error(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("agentic_trading.data.cache.CACHE_ROOT", tmp_path)
    monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)
    monkeypatch.delenv("REDDIT_CLIENT_SECRET", raising=False)

    with pytest.raises(RuntimeError, match="REDDIT_CLIENT_ID"):
        load_mention_counts("GME", pd.Timestamp("2024-06-02"), subreddits=("stocks",))
