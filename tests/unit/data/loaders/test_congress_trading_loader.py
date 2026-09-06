from unittest.mock import MagicMock, patch

import pandas as pd

from agentic_trading.data.loaders.congress_trading_loader import load_congressional_trades


def _disclosures_frame(chamber: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["GME", "GME"],
            "filer": ["Jane Doe", "Jane Doe"],
            "transaction_date": pd.to_datetime(["2024-04-01", "2024-04-15"]),
            "disclosure_date": pd.to_datetime(["2024-05-15", "2024-06-01"]),
            "transaction_type": ["Purchase", "Sale"],
            "chamber": [chamber, chamber],
        }
    )


@patch("agentic_trading.data.loaders.congress_trading_loader._fetch_disclosures")
def test_gates_on_disclosure_date_not_transaction_date(
    mock_fetch: MagicMock, tmp_path, monkeypatch
) -> None:
    """A trade made 2024-04-15 but only disclosed 2024-06-01 must not be
    visible on 2024-05-20, even though the trade itself already happened —
    this is the STOCK Act's 30-45 day reporting-lag lookahead trap."""
    monkeypatch.setattr("agentic_trading.data.cache.CACHE_ROOT", tmp_path)
    mock_fetch.side_effect = lambda chamber: _disclosures_frame(chamber)

    result = load_congressional_trades("GME", pd.Timestamp("2024-05-20"), chambers=("senate",))

    assert len(result) == 1
    assert result.iloc[0]["transaction_date"] == pd.Timestamp("2024-04-01")


@patch("agentic_trading.data.loaders.congress_trading_loader._fetch_disclosures")
def test_later_as_of_date_sees_both_disclosures(
    mock_fetch: MagicMock, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("agentic_trading.data.cache.CACHE_ROOT", tmp_path)
    mock_fetch.side_effect = lambda chamber: _disclosures_frame(chamber)

    result = load_congressional_trades("GME", pd.Timestamp("2024-06-10"), chambers=("senate",))

    assert len(result) == 2


@patch("agentic_trading.data.loaders.congress_trading_loader._fetch_disclosures")
def test_filters_to_the_requested_ticker(mock_fetch: MagicMock, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("agentic_trading.data.cache.CACHE_ROOT", tmp_path)
    mock_fetch.side_effect = lambda chamber: _disclosures_frame(chamber)

    result = load_congressional_trades("AAPL", pd.Timestamp("2024-06-10"), chambers=("senate",))

    assert result.empty


@patch("agentic_trading.data.loaders.congress_trading_loader._fetch_disclosures")
def test_a_later_simulated_day_the_same_real_day_reuses_the_cached_dump(
    mock_fetch: MagicMock, tmp_path, monkeypatch
) -> None:
    """The bulk dump is cached under the real wall-clock date, not the
    simulated `as_of_date` — a backtest calling this once per simulated day
    must not re-download it for every day."""
    monkeypatch.setattr("agentic_trading.data.cache.CACHE_ROOT", tmp_path)
    mock_fetch.side_effect = lambda chamber: _disclosures_frame(chamber)

    load_congressional_trades("GME", pd.Timestamp("2024-05-20"), chambers=("senate",))
    load_congressional_trades("GME", pd.Timestamp("2024-06-10"), chambers=("senate",))

    mock_fetch.assert_called_once()
