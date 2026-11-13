# test_log_dedup_ties.py

"""A pinned read of a "current" table keeps one row per key whatever order the lake stores its
log rows in: among rows of one key with the newest `scraped_at`, the row first in the table's
`tie_order` (as `/v1/tables/{table}` serves it), each column ascending with nulls last, wins. A
current table whose metadata carries no `tie_order` is refused, naming the table."""

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
    """The pinned read of `toy` at `_T`, with its log holding `rows` in that order, deduplicated
    on `tie_order`, or on the tie order the lake serves for `toy` when `tie_order` is `None`."""
    lake.add("toy", pd.DataFrame(rows), ("date",), key_columns=_KEYS)
    order = lake.describe("toy")["tie_order"] if tie_order is None else tie_order
    relation = ll._deduplicated_log_sql("toy", _KEYS, _T, order)
    return lake.sql(f"SELECT ticker, date, close, note FROM ({relation}) ORDER BY ticker, date", [])


def test_tied_log_rows_give_one_answer_in_every_insertion_order(fake_lake: Any) -> None:
    """With the tie order the lake serves (`close`, then `note`), each rotation of the log's
    rows, forwards and reversed, gives the same rows: one per key, the tied version first in
    that order (AAA's lower close, BBB's non-null close)."""
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


def test_an_empty_tie_order_orders_on_scraped_at_alone(fake_lake: Any) -> None:
    """A table whose every column is a key has an empty tie order: the dedupe orders on
    `scraped_at` alone and keeps the newest row of each key."""
    rows = pd.DataFrame(
        {"ticker": ["AAA", "AAA"], "scraped_at": [_OLDER, _TIED]},
    )
    fake_lake.add("keys_only", rows, key_columns=["ticker"])
    relation = ll._deduplicated_log_sql("keys_only", ["ticker"], _T, [])
    frame = fake_lake.sql(f"SELECT ticker, scraped_at FROM ({relation})", [])

    assert frame["ticker"].tolist() == ["AAA"]
    assert pd.Timestamp(frame["scraped_at"].iloc[0]) == _TIED


def test_the_fake_lake_serves_the_non_key_value_columns_as_the_tie_order(
    fake_lake: Any,
) -> None:
    """The fake lake serves, as the lake does, an `instrument_prices`-shaped table keyed on
    (ticker, date) with the tie order asset_class, open, high, low, close, adj_close, volume:
    every non-key column in the table's column order, without `scraped_at`, `dt` and `table`,
    and its key columns as `keys` beside `key_columns`."""
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
    metadata = fake_lake.describe("instrument_prices")

    assert metadata["tie_order"] == [
        "asset_class",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    ]
    assert metadata["keys"] == metadata["key_columns"] == _KEYS


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


def test_a_current_table_without_a_tie_order_is_refused_naming_it(fake_lake: Any) -> None:
    """A current table whose metadata carries no `tie_order` cannot be deduplicated as of T: a
    read of it inside a lake session raises `MissingTieOrderError` naming the table, before
    any query of the table is sent."""
    fake_lake.add(
        "bis_observations",
        _bis_rows(5.25, 5.5),
        ("date",),
        key_columns=["dataflow", "series_key", "period"],
    )
    fake_lake.tie_order_overrides["bis_observations"] = None

    with ll.lake_session(_T):
        sent = len(fake_lake.calls)
        with pytest.raises(ll.MissingTieOrderError, match="bis_observations"):
            ll.load_policy_rates_from_lakehouse(["USD"], date(2024, 1, 2), date(2024, 1, 8))
        assert not any("bis_observations" in query for query, _ in fake_lake.calls[sent:])


def test_tie_columns_compare_one_by_one_with_nulls_last(fake_lake: Any) -> None:
    """Tied rows that differ only by a null in one tie column, with the tie order (`close`,
    `note`): DDD's tied rows differ only by a null `note`, so the row with a note wins; EEE's
    pair a null `close` with a note against a close with a null note, so the row with a close
    wins. Either insertion order gives the same rows."""
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
        result = _read_toy(fake_lake, order, ["close", "note"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(result, expected, check_dtype=False)
