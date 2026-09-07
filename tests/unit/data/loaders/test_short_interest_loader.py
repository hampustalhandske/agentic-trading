from unittest.mock import MagicMock, patch

import pandas as pd

from agentic_trading.data.loaders.short_interest_loader import (
    _fetch_short_interest_history,
    load_short_interest,
)


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


@patch("agentic_trading.data.loaders.short_interest_loader.requests.post")
def test_fetch_parses_csv_response_and_derives_dissemination_date(mock_post) -> None:
    """FINRA's API returns CSV text (Content-Type: text/plain) regardless
    of the request body, not JSON — and carries no dissemination-date
    field, only settlementDate. This must parse as CSV and derive
    dissemination_date as settlement_date + 8 business days rather than
    raising a JSON decode error or leaving the column missing."""
    mock_response = MagicMock()
    mock_response.text = (
        '"settlementDate","currentShortPositionQuantity","daysToCoverQuantity",'
        '"averageDailyVolumeQuantity"\n'
        '"2024-05-15","1000000","2.0","500000"\n'
    )
    mock_post.return_value = mock_response

    result = _fetch_short_interest_history("GME")

    assert list(result.columns) == [
        "settlement_date",
        "dissemination_date",
        "short_interest_shares",
        "days_to_cover",
        "avg_daily_volume",
    ]
    assert result.iloc[0]["settlement_date"] == pd.Timestamp("2024-05-15")
    assert result.iloc[0]["dissemination_date"] > result.iloc[0]["settlement_date"]


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
