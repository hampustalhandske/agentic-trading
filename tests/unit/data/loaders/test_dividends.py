# test_dividends.py

"""Tests for `load_dividends_from_lakehouse` against the `equity_distributions` table.

Only `regular_cash` rows are returned, on the split-adjusted basis, indexed by ex-date.
"""

from datetime import date
from typing import Any

import pandas as pd

from agentic_trading.data.loaders import lakehouse_loader

_ROWS = [
    ("AAPL", "2020-05-08", 0.82, 0.205, "regular_cash"),
    ("AAPL", "2020-08-07", 0.82, 0.205, "regular_cash"),
    ("AAPL", "2020-11-06", 0.205, 0.205, "regular_cash"),
    ("COST", "2023-11-02", 1.02, 1.02, "regular_cash"),
    ("COST", "2023-12-27", 15.00, 15.00, "special_cash"),
    ("COST", "2024-02-01", 1.02, 1.02, "regular_cash"),
    ("XOM", "2024-02-13", 0.95, 0.95, "regular_cash"),
    ("XOM", "2024-05-14", 0.95, 0.95, "regular_cash"),
    ("XOM", "2024-08-15", 0.95, 0.95, "regular_cash"),
    ("SPIN", "2023-07-03", 5.00, 5.00, "spin_off"),
    ("BIG", "2023-07-03", 90.0, 90.0, "unknown_large"),
]


def _seed(fake_lake: Any) -> None:
    fake_lake.add(
        "equity_distributions",
        pd.DataFrame(
            _ROWS,
            columns=[
                "ticker",
                "ex_date",
                "amount_as_traded",
                "amount_split_adjusted",
                "distribution_type",
            ],
        ),
        date_columns=("ex_date",),
    )


def test_split_is_not_read_as_a_cut_and_gate_is_inclusive(fake_lake: Any) -> None:
    """AAPL's 4-for-1 split leaves equal adjusted payouts; the as-of date gates ex-dates
    inclusively."""
    _seed(fake_lake)

    result = lakehouse_loader.load_dividends_from_lakehouse("AAPL", date(2020, 11, 6))
    before = lakehouse_loader.load_dividends_from_lakehouse("AAPL", date(2020, 11, 5))

    assert result is not None
    assert list(result) == [0.205, 0.205, 0.205]
    assert result.iloc[-1] == result.iloc[-2]
    assert str(result.index.dtype).startswith("datetime64")
    assert result.index.is_monotonic_increasing
    assert result.index.is_unique
    assert before is not None
    assert list(before.index) == [pd.Timestamp("2020-05-08"), pd.Timestamp("2020-08-07")]


def test_special_cash_is_excluded(fake_lake: Any) -> None:
    """COST's special distribution never appears in the regular series."""
    _seed(fake_lake)

    result = lakehouse_loader.load_dividends_from_lakehouse("COST", date(2024, 2, 1))

    assert result is not None
    assert list(result) == [1.02, 1.02]
    assert list(result.index) == [pd.Timestamp("2023-11-02"), pd.Timestamp("2024-02-01")]


def test_regular_payments_are_all_returned(fake_lake: Any) -> None:
    """XOM's three equal payouts all appear, including the middle one."""
    _seed(fake_lake)

    result = lakehouse_loader.load_dividends_from_lakehouse("XOM", date(2024, 8, 15))

    assert result is not None
    assert list(result) == [0.95, 0.95, 0.95]
    assert pd.Timestamp("2024-05-14") in result.index


def test_no_regular_rows_returns_none(fake_lake: Any) -> None:
    """Spin-off-only, unknown_large-only and unknown tickers give None."""
    _seed(fake_lake)
    as_of = date(2024, 12, 31)

    assert lakehouse_loader.load_dividends_from_lakehouse("SPIN", as_of) is None
    assert lakehouse_loader.load_dividends_from_lakehouse("BIG", as_of) is None
    assert lakehouse_loader.load_dividends_from_lakehouse("ZZZZ", as_of) is None
    assert lakehouse_loader.load_dividends_from_lakehouse("COST", date(2023, 11, 1)) is None
