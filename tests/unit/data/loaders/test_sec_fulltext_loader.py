from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from agentic_trading.data.loaders.sec_fulltext_loader import load_filings


def _filings_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "accession_no": ["0001-24-000001", "0001-24-000002"],
            "form_type": ["4", "4"],
            "entity_name": ["Acme Corp", "Acme Corp"],
            "filed": pd.to_datetime(["2024-05-01", "2024-08-01"]),
        }
    )


@patch("agentic_trading.data.loaders.sec_fulltext_loader.require_env", return_value="fake-agent")
@patch("agentic_trading.data.loaders.sec_fulltext_loader._fetch_filings")
@patch("agentic_trading.data.loaders.sec_fulltext_loader._resolve_cik", return_value=320193)
def test_only_filings_filed_on_or_before_as_of_date_are_returned(
    mock_resolve_cik: MagicMock,
    mock_fetch: MagicMock,
    mock_require_env: MagicMock,
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("agentic_trading.data.cache.CACHE_ROOT", tmp_path)
    mock_fetch.return_value = _filings_frame()

    result = load_filings("AAPL", pd.Timestamp("2024-06-01"))

    assert len(result) == 1
    assert result.iloc[0]["accession_no"] == "0001-24-000001"


@patch("agentic_trading.data.loaders.sec_fulltext_loader.require_env", return_value="fake-agent")
@patch("agentic_trading.data.loaders.sec_fulltext_loader._fetch_filings")
@patch("agentic_trading.data.loaders.sec_fulltext_loader._resolve_cik", return_value=320193)
def test_later_as_of_date_sees_the_later_filing_too(
    mock_resolve_cik: MagicMock,
    mock_fetch: MagicMock,
    mock_require_env: MagicMock,
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("agentic_trading.data.cache.CACHE_ROOT", tmp_path)
    mock_fetch.return_value = _filings_frame()

    result = load_filings("AAPL", pd.Timestamp("2024-09-01"))

    assert len(result) == 2


@patch("agentic_trading.data.loaders.sec_fulltext_loader.require_env", return_value="fake-agent")
@patch("agentic_trading.data.loaders.sec_fulltext_loader._fetch_filings")
@patch("agentic_trading.data.loaders.sec_fulltext_loader._resolve_cik", return_value=320193)
def test_second_call_the_same_day_does_not_refetch(
    mock_resolve_cik: MagicMock,
    mock_fetch: MagicMock,
    mock_require_env: MagicMock,
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("agentic_trading.data.cache.CACHE_ROOT", tmp_path)
    mock_fetch.return_value = _filings_frame()

    load_filings("AAPL", pd.Timestamp("2024-06-01"))
    load_filings("AAPL", pd.Timestamp("2024-09-01"))

    mock_fetch.assert_called_once()


@patch("agentic_trading.data.loaders.sec_fulltext_loader._resolve_cik", return_value=320193)
def test_missing_user_agent_raises_clear_error(
    mock_resolve_cik: MagicMock, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("agentic_trading.data.cache.CACHE_ROOT", tmp_path)
    monkeypatch.delenv("SEC_EDGAR_USER_AGENT", raising=False)

    with pytest.raises(RuntimeError, match="SEC_EDGAR_USER_AGENT"):
        load_filings("AAPL", pd.Timestamp("2024-06-01"))
