# test_split_adjustment.py
"""Unit tests for agentic_trading.data.loaders.lakehouse_loader.split_adjustment_factor."""

from datetime import date
from typing import Any

import pandas as pd
import pytest

from agentic_trading.data.loaders import lakehouse_loader

_COLUMNS = ["ticker", "ex_date", "ratio", "in_close_basis", "c0", "c1", "observed_move", "evidence"]

_ROWS = [
    ("AAPL", "2014-06-09", 0.142857, True),
    ("AAPL", "2020-08-31", 0.25, True),
    ("NVDA", "2021-07-20", 0.25, True),
    ("NVDA", "2024-06-10", 0.10, True),
    ("TSLA", "2020-08-31", 0.2, True),
    ("TSLA", "2022-08-25", 1 / 3, True),
    ("AVB", "2026-08-17", 0.358038, False),
    ("SSP", "2015-04-01", 0.887311, True),
    ("SSP", "2025-12-08", 0.751880, False),
    ("XYZ", "2023-03-01", 0.5, None),
]


def _load_splits(fake_lake: Any, rows: list[tuple]) -> None:
    """Registers yahoo_split_close_basis with a nullable boolean in_close_basis column."""
    frame = pd.DataFrame(
        [(t, d, r, b, None, None, None, None) for t, d, r, b in rows], columns=_COLUMNS
    )
    frame = frame.astype(
        {
            "ticker": "string",
            "ratio": float,
            "in_close_basis": "boolean",
            "c0": float,
            "c1": float,
            "observed_move": float,
            "evidence": "string",
        }
    )
    fake_lake.add("yahoo_split_close_basis", frame, date_columns=("ex_date",))


@pytest.mark.parametrize(
    ("ticker", "quote", "expected"),
    [
        ("AAPL", date(2020, 8, 28), 4.0),
        ("AAPL", date(2020, 8, 31), 1.0),
        ("AAPL", date(2014, 6, 6), 4 / 0.142857),
        ("NVDA", date(2024, 6, 7), 10.0),
        ("NVDA", date(2021, 7, 1), 40.0),
        ("TSLA", date(2022, 8, 1), 3.0),
        ("TSLA", date(2020, 8, 28), 15.0),
    ],
)
def test_factor_is_product_of_inverse_ratios_after_quote_date(
    fake_lake: Any, ticker: str, quote: date, expected: float
) -> None:
    """Only splits with ex_date strictly after the quote date contribute 1/ratio."""
    _load_splits(fake_lake, _ROWS)
    result = lakehouse_loader.split_adjustment_factor(ticker, quote)
    assert result == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize(
    ("ticker", "quote", "expected"),
    [
        ("AVB", date(2026, 8, 14), 1.0),
        ("SSP", date(2015, 3, 31), 1 / 0.887311),
        ("SSP", date(2025, 12, 5), 1.0),
        ("XYZ", date(2023, 2, 1), 2.0),
    ],
)
def test_in_close_basis_false_never_counts_and_null_counts(
    fake_lake: Any, ticker: str, quote: date, expected: float
) -> None:
    """Rows with in_close_basis false are excluded; true and null rows are included."""
    _load_splits(fake_lake, _ROWS)
    result = lakehouse_loader.split_adjustment_factor(ticker, quote)
    assert result == pytest.approx(expected, rel=1e-6)


def test_unknown_ticker_and_empty_table_return_one(fake_lake: Any) -> None:
    """A ticker with no rows, or an empty table, yields a factor of 1.0."""
    _load_splits(fake_lake, _ROWS)
    assert lakehouse_loader.split_adjustment_factor("MSFT", date(2020, 1, 2)) == 1.0
    _load_splits(fake_lake, [])
    assert lakehouse_loader.split_adjustment_factor("AAPL", date(2020, 8, 28)) == 1.0
