# test_log_dedup_ties.py

"""A pinned read of a "current" table keeps one row per key whatever order the lake stores its
log rows in: among rows of one key with the newest `scraped_at`, the row first in the table's
other columns, in column order, ascending with nulls last, wins."""

from datetime import UTC, date, datetime
from typing import Any

import pandas as pd
import pytest

from agentic_trading.data.loaders import lakehouse_loader as ll

_T = datetime(2024, 8, 1, tzinfo=UTC)
_OLDER = pd.Timestamp("2024-04-01", tz="UTC")
_TIED = pd.Timestamp("2024-05-01", tz="UTC")
_KEYS = ["ticker", "date"]


def _toy_row(
    ticker: str, close: float | None, note: str | None, dt: str, scraped_at: Any
) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "date": "2024-01-02",
        "dt": dt,
        "close": close,
        "note": note,
        "scraped_at": scraped_at,
    }


def _toy_rows() -> list[dict[str, Any]]:
    """Log rows of table `toy` keyed on (ticker, date): AAA has an older version and two
    versions written at one `scraped_at` that differ in `close` and in the bookkeeping column
    `dt`, whose order is the other way round; BBB has two tied versions, one with a null
    `close`; CCC has two identical tied rows."""
    return [
        _toy_row("AAA", 99.0, "x", "2024-04-01", _OLDER),
        _toy_row("AAA", 101.0, "a", "2024-05-01", _TIED),
        _toy_row("AAA", 100.0, "z", "2024-05-02", _TIED),
        _toy_row("BBB", None, "a", "2024-05-01", _TIED),
        _toy_row("BBB", 7.0, "b", "2024-05-01", _TIED),
        _toy_row("CCC", 5.0, "c", "2024-05-01", _TIED),
        _toy_row("CCC", 5.0, "c", "2024-05-01", _TIED),
    ]


def _read_toy(
    lake: Any, rows: list[dict[str, Any]], tie_order: list[str] | None = None
) -> pd.DataFrame:
    """The pinned read of `toy` at `_T`, with the lake's `tie_order`, with its log holding
    `rows` in that order."""
    lake.add("toy", pd.DataFrame(rows), ("date",), key_columns=_KEYS)
    relation = ll._deduplicated_log_sql("toy", _KEYS, _T, tie_order)
    return lake.sql(f"SELECT ticker, date, close, note FROM ({relation}) ORDER BY ticker, date", [])


def test_tied_log_rows_give_one_answer_in_every_insertion_order(fake_lake: Any) -> None:
    """Without a tie order from the lake, each rotation of the log's rows, forwards and
    reversed, gives the same rows: one per key, the tied version first in its non-key columns
    other than the bookkeeping ones (AAA's lower close, BBB's non-null close)."""
    rows = _toy_rows()
    orders = [rows[i:] + rows[:i] for i in range(len(rows))]
    orders += [list(reversed(order)) for order in orders]
    results = [_read_toy(fake_lake, order) for order in orders]

    expected = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC"],
            "date": pd.to_datetime(["2024-01-02"] * 3),
            "close": [100.0, 7.0, 5.0],
            "note": ["z", "b", "c"],
        }
    )
    for result in results[1:]:
        pd.testing.assert_frame_equal(result, results[0])
    for result in results:
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected, check_dtype=False)


def test_a_tie_order_from_the_lake_decides_in_its_own_order(fake_lake: Any) -> None:
    """With the lake's tie order (`note`, then `close`), AAA's tied version with the first
    note wins, in either insertion order."""
    rows = _toy_rows()
    forwards = _read_toy(fake_lake, rows, ["note", "close"])
    backwards = _read_toy(fake_lake, list(reversed(rows)), ["note", "close"])

    pd.testing.assert_frame_equal(forwards, backwards)
    assert forwards["close"].tolist()[0] == 101.0


def test_the_fallback_tie_order_is_the_non_key_value_columns_in_column_order(
    fake_lake: Any,
) -> None:
    """For an `instrument_prices`-shaped log keyed on (ticker, date), the fallback tie order is
    asset_class, open, high, low, close, adj_close, volume: every non-key column in the
    table's column order, without `scraped_at`, `dt` and `table`."""
    prices = pd.DataFrame(
        {
            "ticker": ["SPY"],
            "date": ["2024-01-02"],
            "asset_class": ["etf"],
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "adj_close": [1.0],
            "volume": [1.0],
            "scraped_at": [_TIED],
            "dt": ["2024-05-01"],
            "table": ["instrument_prices"],
        }
    )
    fake_lake.add("instrument_prices", prices, ("date",), key_columns=_KEYS)
    selector = ll._fallback_tie_columns_sql(_KEYS)
    columns = fake_lake.sql(f"SELECT {selector} FROM instrument_prices_log LIMIT 0", []).columns

    assert list(columns) == ["asset_class", "open", "high", "low", "close", "adj_close", "volume"]


def _bis_rows(first_value: float, second_value: float) -> pd.DataFrame:
    """Two `bis_observations` log rows of one US WS_CBPOL observation, written at one
    `scraped_at` and differing only in `value`."""
    return pd.DataFrame(
        {
            "dataflow": ["WS_CBPOL", "WS_CBPOL"],
            "series_key": ["D.US", "D.US"],
            "period": ["2024-01-03", "2024-01-03"],
            "date": ["2024-01-03", "2024-01-03"],
            "ref_area": ["US", "US"],
            "freq": ["D", "D"],
            "value": [first_value, second_value],
            "obs_status": ["A", "A"],
            "pre_break_value": pd.array([None, None], dtype="Float64"),
            "scraped_at": [_TIED, _TIED],
        }
    )


@pytest.mark.parametrize("values", [(5.25, 5.5), (5.5, 5.25)])
def test_a_loader_reads_the_same_tied_version_in_either_insertion_order(
    fake_lake: Any, values: tuple[float, float]
) -> None:
    """A policy-rate read inside a lake session takes the lower of two tied values, whichever
    the lake stored first."""
    fake_lake.add(
        "bis_observations",
        _bis_rows(*values),
        ("date",),
        key_columns=["dataflow", "series_key", "period"],
    )

    with ll.lake_session(_T):
        rates = ll.load_policy_rates_from_lakehouse(["USD"], date(2024, 1, 2), date(2024, 1, 8))

    assert rates is not None
    assert rates.loc["2024-01-03", "USD"] == pytest.approx(0.0525)


def test_the_fallback_picks_what_the_lakes_per_column_rule_picks_with_nulls(
    fake_lake: Any,
) -> None:
    """Tied rows that differ only by a null in one tie column: the fallback order (a struct of
    the non-key columns) keeps the same row as the lake's rule written out column by column,
    each ascending with nulls last. DDD's tied rows differ only by a null `note`, so the row
    with a note wins; EEE's pair a null `close` with a note against a close with a null note,
    so the row with a close wins."""
    rows = [
        _toy_row("DDD", 5.0, None, "2024-05-01", _TIED),
        _toy_row("DDD", 5.0, "a", "2024-05-01", _TIED),
        _toy_row("EEE", None, "a", "2024-05-01", _TIED),
        _toy_row("EEE", 5.0, None, "2024-05-01", _TIED),
    ]
    expected = pd.DataFrame(
        {
            "ticker": ["DDD", "EEE"],
            "date": pd.to_datetime(["2024-01-02"] * 2),
            "close": [5.0, 5.0],
            "note": ["a", None],
        }
    )
    for order in (rows, list(reversed(rows))):
        fallback = _read_toy(fake_lake, order).reset_index(drop=True)
        per_column = _read_toy(fake_lake, order, ["close", "note"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(fallback, per_column)
        pd.testing.assert_frame_equal(fallback, expected, check_dtype=False)
