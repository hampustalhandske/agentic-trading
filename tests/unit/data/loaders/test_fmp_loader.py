from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from agentic_trading.data.loaders.fmp_loader import load_fmp_fundamentals


def _fundamentals_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "concept": ["revenue", "revenue", "totalAssets"],
            "value": [100.0, 110.0, 500.0],
            "fiscal_period": ["2024Q1", "2024Q2", "2024Q2"],
            "filed": pd.to_datetime(["2024-05-01", "2024-08-01", "2024-08-01"]),
        }
    )


@patch("agentic_trading.data.loaders.fmp_loader.require_env", return_value="fake-key")
@patch("agentic_trading.data.loaders.fmp_loader._fetch_fmp_fundamentals")
def test_only_facts_filed_on_or_before_as_of_date_are_returned(
    mock_fetch: MagicMock, mock_require_env: MagicMock, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("agentic_trading.data.cache.CACHE_ROOT", tmp_path)
    mock_fetch.return_value = _fundamentals_frame()

    result = load_fmp_fundamentals("AAPL", pd.Timestamp("2024-06-01"))

    assert (result["filed"] <= pd.Timestamp("2024-06-01")).all()
    assert len(result) == 1
    assert result.iloc[0]["fiscal_period"] == "2024Q1"


@patch("agentic_trading.data.loaders.fmp_loader.require_env", return_value="fake-key")
@patch("agentic_trading.data.loaders.fmp_loader._fetch_fmp_fundamentals")
def test_later_as_of_date_sees_the_later_filing_too(
    mock_fetch: MagicMock, mock_require_env: MagicMock, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("agentic_trading.data.cache.CACHE_ROOT", tmp_path)
    mock_fetch.return_value = _fundamentals_frame()

    result = load_fmp_fundamentals("AAPL", pd.Timestamp("2024-09-01"))

    assert len(result) == 3


@patch("agentic_trading.data.loaders.fmp_loader.require_env", return_value="fake-key")
@patch("agentic_trading.data.loaders.fmp_loader._fetch_fmp_fundamentals")
def test_second_call_the_same_day_does_not_refetch(
    mock_fetch: MagicMock, mock_require_env: MagicMock, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("agentic_trading.data.cache.CACHE_ROOT", tmp_path)
    mock_fetch.return_value = _fundamentals_frame()

    load_fmp_fundamentals("AAPL", pd.Timestamp("2024-06-01"))
    load_fmp_fundamentals("AAPL", pd.Timestamp("2024-09-01"))

    mock_fetch.assert_called_once()


def test_missing_api_key_raises_clear_error(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("agentic_trading.data.cache.CACHE_ROOT", tmp_path)
    monkeypatch.delenv("FMP_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="FMP_API_KEY"):
        load_fmp_fundamentals("AAPL", pd.Timestamp("2024-06-01"))
