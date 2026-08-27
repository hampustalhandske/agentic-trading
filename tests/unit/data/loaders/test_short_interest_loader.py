from unittest.mock import patch

import pandas as pd

from agentic_trading.data.loaders.short_interest_loader import load_short_interest


def _history_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "settlement_date": pd.to_datetime(["2024-05-15", "2024-05-31"]),
            "dissemination_date": pd.to_datetime(["2024-05-22", "2024-06-07"]),
            "short_interest_shares": [1_000_000, 1_500_000],
            "days_to_cover": [2.0, 3.5],
            "avg_daily_volume": [500_000, 430_000],
        }
    )


@patch("agentic_trading.data.loaders.short_interest_loader._fetch_short_interest_history")
def test_only_disseminated_settlements_are_visible(mock_fetch, tmp_path, monkeypatch) -> None:
    """Gates on dissemination_date, not settlement_date — a settlement that
    hasn't been disseminated yet must not be visible even though its
    settlement date has already passed."""
    monkeypatch.setattr("agentic_trading.data.cache.CACHE_ROOT", tmp_path)
    mock_fetch.return_value = _history_frame()

    result = load_short_interest("GME", pd.Timestamp("2024-05-25"))

    assert len(result) == 1
    assert result.iloc[0]["settlement_date"] == pd.Timestamp("2024-05-15")


@patch("agentic_trading.data.loaders.short_interest_loader._fetch_short_interest_history")
def test_later_as_of_date_sees_both_settlements(mock_fetch, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("agentic_trading.data.cache.CACHE_ROOT", tmp_path)
    mock_fetch.return_value = _history_frame()

    result = load_short_interest("GME", pd.Timestamp("2024-06-10"))

    assert len(result) == 2
