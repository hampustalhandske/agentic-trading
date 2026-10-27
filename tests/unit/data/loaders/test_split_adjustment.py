# test_split_adjustment.py
"""Unit tests for agentic_trading.data.loaders.lakehouse_loader._split_adjustment_factor."""

from datetime import date
from typing import Any

import pandas as pd
import pytest

from agentic_trading.data.loaders import lakehouse_loader

_ROWS = [
    ("AAPL", "2020-08-31", 0.25, "dolthub,yahoo", "confirmed"),
    ("AMZN", "2022-05-26", 0.05, "dolthub", "rejected_duplicate"),
    ("AMZN", "2022-06-06", 0.05, "dolthub,yahoo", "confirmed"),
    ("GOOGL", "2022-06-30", 0.05, "dolthub", "rejected_duplicate"),
    ("GOOGL", "2022-07-18", 0.05, "dolthub,yahoo", "confirmed"),
    ("NVDA", "2021-07-20", 0.25, "dolthub,yahoo", "confirmed"),
    ("NVDA", "2024-06-10", 0.10, "dolthub,yahoo", "confirmed"),
    ("TSLA", "2020-08-31", 0.20, "yahoo", "before_first_bar"),
    ("TSLA", "2020-08-31", 0.25, "dolthub", "before_first_bar"),
    ("TSLA", "2022-08-25", 1 / 3, "dolthub,yahoo", "confirmed"),
    ("GE", "2023-01-04", 0.6, "dolthub", "reclassified_spin_off"),
    ("XYZ", "2023-03-01", 0.5, "dolthub", "rejected_no_move"),
    ("INF", "2023-03-01", 0.5, "dolthub", "inferred"),
]


def _load_splits(fake_lake: Any, rows: list[tuple]) -> None:
    frame = pd.DataFrame(rows, columns=["ticker", "ex_date", "ratio", "sources", "status"])
    frame = frame.astype(
        {"ticker": "string", "ratio": float, "sources": "string", "status": "string"}
    )
    fake_lake.add("equity_splits", frame, date_columns=("ex_date",))


@pytest.mark.parametrize(
    ("ticker", "quote", "expected"),
    [
        ("AAPL", date(2020, 8, 28), 4.0),
        ("AAPL", date(2020, 8, 31), 1.0),
        ("AAPL", date(2020, 9, 1), 1.0),
        ("NVDA", date(2024, 6, 7), 10.0),
        ("NVDA", date(2021, 7, 1), 40.0),
        ("MSFT", date(2020, 1, 2), 1.0),
    ],
)
def test_factor_is_product_of_inverse_ratios_after_quote_date(
    fake_lake: Any, ticker: str, quote: date, expected: float
) -> None:
    """Only splits with ex_date strictly after the quote date contribute 1/ratio."""
    _load_splits(fake_lake, _ROWS)
    result = lakehouse_loader._split_adjustment_factor(ticker, quote)
    assert result == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize(
    ("ticker", "quote", "expected"),
    [
        ("TSLA", date(2022, 8, 1), 3.0),
        ("TSLA", date(2020, 8, 28), 15.0),
    ],
)
def test_duplicate_ex_date_prefers_yahoo_row(
    fake_lake: Any, ticker: str, quote: date, expected: float
) -> None:
    """When rows share (ticker, ex_date), the row sourced from yahoo is the one counted."""
    _load_splits(fake_lake, _ROWS)
    result = lakehouse_loader._split_adjustment_factor(ticker, quote)
    assert result == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize(
    ("ticker", "quote", "expected"),
    [
        ("AMZN", date(2022, 5, 1), 20.0),
        ("AMZN", date(2022, 5, 27), 20.0),
        ("GOOGL", date(2022, 6, 1), 20.0),
        ("GE", date(2023, 1, 3), 1.0),
        ("XYZ", date(2023, 2, 1), 1.0),
    ],
)
def test_rejected_and_spin_off_rows_are_ignored(
    fake_lake: Any, ticker: str, quote: date, expected: float
) -> None:
    """Rows with rejected_* or reclassified_spin_off status never contribute to the factor."""
    _load_splits(fake_lake, _ROWS)
    result = lakehouse_loader._split_adjustment_factor(ticker, quote)
    assert result == pytest.approx(expected, rel=1e-6)


def test_inferred_status_counts(fake_lake: Any) -> None:
    """Rows with status inferred are counted as real splits."""
    _load_splits(fake_lake, _ROWS)
    result = lakehouse_loader._split_adjustment_factor("INF", date(2023, 2, 1))
    assert result == pytest.approx(2.0, rel=1e-6)


def test_empty_table_returns_one(fake_lake: Any) -> None:
    """An equity_splits table with no rows yields a factor of 1.0."""
    _load_splits(fake_lake, [])
    result = lakehouse_loader._split_adjustment_factor("AAPL", date(2020, 8, 28))
    assert result == 1.0
