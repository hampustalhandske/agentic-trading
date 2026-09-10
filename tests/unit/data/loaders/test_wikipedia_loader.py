from unittest.mock import MagicMock, patch

import pandas as pd

from agentic_trading.data.loaders.wikipedia_loader import load_pageviews


def _pageviews_frame() -> pd.DataFrame:
    dates = pd.to_datetime(["2024-06-01", "2024-06-02", "2024-06-03"])
    return pd.DataFrame(
        {
            "views": [1000, 1200, 5000],
            "available_date": dates + pd.Timedelta(days=1),
        },
        index=dates,
    )


@patch("agentic_trading.data.loaders.wikipedia_loader._fetch_pageviews")
def test_publication_lag_gates_visibility(mock_fetch: MagicMock, tmp_path, monkeypatch) -> None:
    """A pageview count for day D is published with a ~1 day lag — it must
    not be visible on day D itself, only from D+1 onward."""
    monkeypatch.setattr("agentic_trading.data.cache.CACHE_ROOT", tmp_path)
    mock_fetch.return_value = _pageviews_frame()

    result = load_pageviews("Apple_Inc.", pd.Timestamp("2024-06-03"))

    assert list(result["views"]) == [1000, 1200]


@patch("agentic_trading.data.loaders.wikipedia_loader._fetch_pageviews")
def test_later_as_of_date_sees_the_later_spike_too(
    mock_fetch: MagicMock, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("agentic_trading.data.cache.CACHE_ROOT", tmp_path)
    mock_fetch.return_value = _pageviews_frame()

    result = load_pageviews("Apple_Inc.", pd.Timestamp("2024-06-04"))

    assert list(result["views"]) == [1000, 1200, 5000]


@patch("agentic_trading.data.loaders.wikipedia_loader._fetch_pageviews")
def test_a_later_simulated_day_the_same_real_day_reuses_the_cached_series(
    mock_fetch: MagicMock, tmp_path, monkeypatch
) -> None:
    """Different historical windows use distinct cache entries so an
    earlier simulation cannot reuse a narrower later window."""
    monkeypatch.setattr("agentic_trading.data.cache.CACHE_ROOT", tmp_path)
    mock_fetch.return_value = _pageviews_frame()

    load_pageviews("Apple_Inc.", pd.Timestamp("2024-06-03"))
    load_pageviews("Apple_Inc.", pd.Timestamp("2024-06-04"))

    assert mock_fetch.call_count == 2
