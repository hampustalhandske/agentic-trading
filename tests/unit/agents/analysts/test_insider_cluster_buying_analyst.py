from datetime import date

import numpy as np
import pandas as pd

from agentic_trading.agents.analysts.alt_data_pod import insider_cluster_buying_analyst as mod
from agentic_trading.config import settings
from agentic_trading.data.loaders import lakehouse_loader as ll
from agentic_trading.stage_a.decisions import check_source_as_of, row_source


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


def _fetched_purchases(rows: list[tuple[int, str, str]]) -> pd.DataFrame:
    """`_fetch_insider_transactions` rows of open-market purchases, one per
    (owner_cik, filed date, America/New_York acceptance time)."""
    filed = pd.to_datetime([r[1] for r in rows])
    return pd.DataFrame(
        {
            "owner_cik": [r[0] for r in rows],
            "owner_name": [f"OWNER {r[0]}" for r in rows],
            "is_officer": True,
            "is_director": False,
            "is_ten_percent_owner": False,
            "transaction_date": filed - pd.Timedelta(days=1),
            "transaction_code": "P",
            "shares": 100.0,
            "price_per_share": 50.0,
            "acquired_disposed": "A",
            "filed_date": filed,
            "accepted_at": [
                pd.Timestamp(r[2], tz="America/New_York").tz_convert("UTC") for r in rows
            ],
        }
    )


def test_signal_from_loader_selection_passes_source_as_of_audit(monkeypatch) -> None:
    """A Form 4 filed and accepted before the close on the decision day is not yet known, so
    the signal's `source_as_of` passes the after-close audit of the analyst's declared source
    on that day, and on the next session, when the filing counts."""
    fetched = _fetched_purchases(
        [
            (1, "2017-08-17", "2017-08-17 11:00"),
            (2, "2017-08-21", "2017-08-21 15:00"),
            (3, "2017-08-22", "2017-08-22 10:00"),
        ]
    )
    monkeypatch.setattr(
        mod,
        "load_insider_transactions_from_lakehouse",
        lambda _ticker, as_of: ll._select_insider_transactions(fetched, as_of),
    )
    analyst = mod.InsiderClusterBuyingAnalyst()
    source = row_source(analyst)
    assert source == "sec_insider_transactions"

    decision_day = pd.Timestamp("2017-08-22")
    signal = analyst.generate_signal("AAPL", _market_data(), decision_day)
    assert signal.status == "active"
    assert signal.source_as_of == date(2017, 8, 21)
    assert signal.sample_count == 2
    check_source_as_of(signal.source_as_of, decision_day.date(), source)

    next_session = pd.Timestamp("2017-08-23")
    later = analyst.generate_signal("AAPL", _market_data(), next_session)
    assert later.source_as_of == date(2017, 8, 22)
    assert later.sample_count == 3
    check_source_as_of(later.source_as_of, next_session.date(), source)
