from unittest.mock import patch

import pandas as pd

from agentic_trading.data.loaders.fundamentals_loader import load_company_facts


def _facts_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "concept": ["Revenues", "Revenues", "EarningsPerShareDiluted"],
            "value": [100.0, 110.0, 1.5],
            "fiscal_period": ["2024Q1", "2024Q2", "2024Q2"],
            "filed": pd.to_datetime(["2024-05-01", "2024-08-01", "2024-08-01"]),
        }
    )


@patch("agentic_trading.data.loaders.fundamentals_loader._fetch_company_facts")
@patch("agentic_trading.data.loaders.fundamentals_loader._resolve_cik", return_value=320193)
def test_only_facts_filed_on_or_before_as_of_date_are_returned(
    mock_resolve_cik, mock_fetch_facts, tmp_path, monkeypatch
) -> None:
    """The single most important test in this loader: a fact filed after
    as_of_date must never be visible, even though its fiscal period may
    have already ended by then."""
    monkeypatch.setattr("agentic_trading.data.cache.CACHE_ROOT", tmp_path)
    mock_fetch_facts.return_value = _facts_frame()

    result = load_company_facts("AAPL", pd.Timestamp("2024-06-01"))

    assert (result["filed"] <= pd.Timestamp("2024-06-01")).all()
    assert len(result) == 1
    assert result.iloc[0]["fiscal_period"] == "2024Q1"


@patch("agentic_trading.data.loaders.fundamentals_loader._fetch_company_facts")
@patch("agentic_trading.data.loaders.fundamentals_loader._resolve_cik", return_value=320193)
def test_later_as_of_date_sees_the_later_filing_too(
    mock_resolve_cik, mock_fetch_facts, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("agentic_trading.data.cache.CACHE_ROOT", tmp_path)
    mock_fetch_facts.return_value = _facts_frame()

    result = load_company_facts("AAPL", pd.Timestamp("2024-09-01"))

    assert len(result) == 3
