# test_loader_row_order.py

"""A loader's rows come back in one order whatever order the lake stores them in: the company
facts, insider transactions and option snapshot queries sort on every column they return, so a
pinned and an unpinned read of the same rows give identical frames."""

from datetime import UTC, date, datetime
from typing import Any

import pandas as pd
import pytest

from agentic_trading.data.loaders import lakehouse_loader as ll

_T = datetime(2024, 8, 1, tzinfo=UTC)


def _fact(concept: str, value: float, form: str) -> dict[str, Any]:
    return {
        "ticker": "AAPL",
        "cik": 320193,
        "concept": concept,
        "unit": "USD",
        "period_start": "2023-01-01",
        "period_end": "2023-04-01",
        "value": value,
        "fiscal_year": 2023,
        "fiscal_period": "Q2",
        "form": form,
        "accession_number": "ACC-1",
        "filed_date": "2023-05-05",
        "scraped_at": pd.Timestamp("2024-05-01", tz="UTC"),
        "dt": "2024-05-01",
        "table": "sec_company_facts_log",
    }


def _trade(owner: str, shares: float, code: str) -> dict[str, Any]:
    return {
        "issuer_ticker": "JPM",
        "owner_cik": 1000 + len(owner),
        "owner_name": owner,
        "is_officer": True,
        "is_director": False,
        "is_ten_percent_owner": False,
        "transaction_date": "2023-05-01",
        "transaction_code": code,
        "shares": shares,
        "price_per_share": 140.0,
        "acquired_disposed": "A",
        "filed_date": "2023-05-03",
        "accession_number": "ACC-1",
        "scraped_at": pd.Timestamp("2024-05-01", tz="UTC"),
        "dt": "2024-05-01",
        "table": "sec_insider_transactions_log",
    }


def _quote(expiration: str, option_type: str, strike: float) -> dict[str, Any]:
    return {
        "underlying": "AAPL",
        "date": "2024-03-15",
        "expiration": expiration,
        "option_type": option_type,
        "strike": strike,
        "bid": 1.0 + strike / 1000,
        "ask": 1.2 + strike / 1000,
        "implied_volatility": 0.2,
        "delta": 0.5,
        "gamma": 0.01,
        "theta": -0.1,
        "vega": 0.2,
        "rho": 0.05,
    }


def _filings(lake: Any) -> None:
    lake.add(
        "sec_filings",
        pd.DataFrame(
            {
                "accession_number": ["ACC-1"],
                "accepted_at": [pd.Timestamp("2023-05-05 12:00", tz="UTC")],
            }
        ),
    )


def _facts(lake: Any, rows: list[dict[str, Any]]) -> None:
    lake.add(
        "sec_company_facts_log",
        pd.DataFrame(rows),
        date_columns=("dt", "filed_date", "period_start", "period_end"),
    )
    _filings(lake)


def _insider(lake: Any, rows: list[dict[str, Any]]) -> None:
    lake.add(
        "sec_insider_transactions_log",
        pd.DataFrame(rows),
        date_columns=("dt", "filed_date", "transaction_date"),
    )
    _filings(lake)


def _chain(lake: Any, rows: list[dict[str, Any]]) -> None:
    lake.add(
        "dolthub_option_chain",
        pd.DataFrame(rows),
        date_columns=("date", "expiration"),
        key_columns=["underlying", "date", "expiration", "option_type", "strike"],
    )
    lake.add(
        "yahoo_split_close_basis",
        pd.DataFrame(
            {
                "ticker": ["ZZUNREL"],
                "ex_date": ["2000-01-03"],
                "ratio": [0.5],
                "in_close_basis": [True],
            }
        ),
        date_columns=("ex_date",),
        scraped_at="2024-01-01T00:00:00Z",
        rebuilt_whole=True,
    )


_FACTS = [
    _fact("Revenues", 100.0, "10-Q"),
    _fact("Assets", 900.0, "10-Q"),
    _fact("Revenues", 101.0, "10-Q/A"),
    _fact("EarningsPerShareDiluted", 1.5, "10-Q"),
]
_TRADES = [
    _trade("Smith", 500.0, "P"),
    _trade("Jones", 200.0, "P"),
    _trade("Smith", 300.0, "S"),
    _trade("Adams", 100.0, "P"),
]
_QUOTES = [
    _quote("2024-04-19", "put", 170.0),
    _quote("2024-04-19", "call", 180.0),
    _quote("2024-03-22", "call", 175.0),
    _quote("2024-04-19", "call", 170.0),
]
_CASES = {
    "company_facts": (
        _facts,
        _FACTS,
        lambda: ll.load_company_facts_from_lakehouse("AAPL", date(2024, 6, 30)),
    ),
    "insider_transactions": (
        _insider,
        _TRADES,
        lambda: ll.load_insider_transactions_from_lakehouse("JPM", date(2024, 6, 30)),
    ),
    "options_snapshot": (
        _chain,
        _QUOTES,
        lambda: ll.load_options_snapshot_from_lakehouse("AAPL", date(2024, 3, 15), 10),
    ),
}


@pytest.mark.parametrize("case", list(_CASES))
@pytest.mark.parametrize("pinned", [False, True], ids=["unpinned", "pinned"])
def test_rows_come_back_in_one_order_whatever_the_stored_order(
    fake_lake: Any, case: str, pinned: bool
) -> None:
    """The same rows stored in opposite orders give identical frames, pinned and unpinned."""
    seed, rows, load = _CASES[case]

    def read(stored: list[dict[str, Any]]) -> pd.DataFrame:
        seed(fake_lake, stored)
        if pinned:
            with ll.lake_session(_T):
                frame = load()
        else:
            frame = load()
        assert frame is not None and len(frame) > 1
        return frame

    forward = read(rows)
    backward = read(list(reversed(rows)))

    pd.testing.assert_frame_equal(forward, backward)
