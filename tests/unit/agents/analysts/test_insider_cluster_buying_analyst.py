import numpy as np
import pandas as pd

from agentic_trading.agents.analysts.alt_data_pod import insider_cluster_buying_analyst as mod
from agentic_trading.config import settings


def _market_data() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=30)
    return pd.DataFrame({"close": np.full(30, 100.0), "log_return": np.full(30, 0.0)}, index=dates)


def _filings(rows: list[tuple[int, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "owner_cik": [r[0] for r in rows],
            "transaction_code": [r[1] for r in rows],
            "filed_date": pd.to_datetime([r[2] for r in rows]),
            "transaction_date": pd.to_datetime([r[2] for r in rows]),
            "shares": 100.0,
        }
    )


def test_cluster_of_distinct_buyers_is_long_with_capped_confidence(monkeypatch) -> None:
    filings = _filings(
        [
            (1, "P", "2024-02-01"),
            (2, "P", "2024-02-05"),
            (3, "P", "2024-02-06"),
            (3, "P", "2024-02-07"),
        ]
    )
    monkeypatch.setattr(mod, "load_insider_transactions_from_lakehouse", lambda *_a: filings)
    signal = mod.InsiderClusterBuyingAnalyst().generate_signal(
        "AAPL", _market_data(), pd.Timestamp("2024-02-09")
    )
    assert signal.direction == "long"
    expected = 3 / settings.analysts.insider_full_confidence_buyers
    assert signal.confidence == expected * settings.analysts.insider_max_confidence
    assert signal.sample_count == 4
    assert signal.source_as_of == pd.Timestamp("2024-02-07").date()


def test_sales_grants_and_stale_purchases_do_not_count(monkeypatch) -> None:
    filings = _filings([(1, "S", "2024-02-01"), (2, "A", "2024-02-02"), (3, "P", "2023-11-01")])
    monkeypatch.setattr(mod, "load_insider_transactions_from_lakehouse", lambda *_a: filings)
    signal = mod.InsiderClusterBuyingAnalyst().generate_signal(
        "AAPL", _market_data(), pd.Timestamp("2024-02-09")
    )
    assert signal.direction == "flat"
    assert signal.status == "active"
    assert signal.confidence == 0.0


def test_no_coverage_and_non_equity_are_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(mod, "load_insider_transactions_from_lakehouse", lambda *_a: None)
    analyst = mod.InsiderClusterBuyingAnalyst()
    assert (
        analyst.generate_signal("AAPL", _market_data(), pd.Timestamp("2024-02-09")).status
        == "unavailable"
    )
    assert (
        analyst.generate_signal("GC=F", _market_data(), pd.Timestamp("2024-02-09")).status
        == "unavailable"
    )
