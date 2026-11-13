# test_security_daily_bars.py

"""Adjusted and raw daily bars by security id, and the daily S&P 500 member count."""

import importlib
import math
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from agentic_trading.data.loaders.lakehouse_loader import (
    READ_TABLES,
    SECURITY_RAW_PRICE_SQL,
    AdjustmentFactorCollisionError,
    EmptyAdjustmentTableError,
    load_security_daily_bars,
    load_security_log_returns,
    load_sp500_member_count,
)

pytestmark = pytest.mark.unpinned_lake_reads

_BARS = {
    "ticker": "VARCHAR",
    "date": "DATE",
    "open": "DOUBLE",
    "close": "DOUBLE",
    "volume": "DOUBLE",
    "official_close": "DOUBLE",
    "security_id": "BIGINT",
}
_FACTORS = {
    "ticker": "VARCHAR",
    "effective_date": "DATE",
    "split_factor": "DOUBLE",
    "cum_factor": "DOUBLE",
    "security_id": "BIGINT",
}
_MEMBER_COUNT = {
    "date": "DATE",
    "member_lines": "INTEGER",
    "nominal_members": "INTEGER",
    "shortfall": "INTEGER",
}
_FIELDS = {"close", "raw_close", "open", "volume", "log_return"}
_SCRIPTS = Path(__file__).resolve().parents[4] / "scripts"

_MON, _TUE, _WED, _THU, _FRI = (date(2024, 3, d) for d in (4, 5, 6, 7, 8))
_NEXT_MON, _NEXT_TUE = date(2024, 3, 11), date(2024, 3, 12)


def _bar(
    lake: Any,
    security_id: int,
    day: date,
    close: float | None,
    official: float | None = None,
    open_: float | None = None,
    volume: float = 1000.0,
    ticker: str = "AAA",
) -> None:
    """One bar; its open is the close unless `open_` is given."""
    lake._connection.execute(
        "INSERT INTO equity_daily_from_bars "
        "(ticker, date, open, close, volume, official_close, security_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [ticker, day, close if open_ is None else open_, close, volume, official, security_id],
    )


def _factor(
    lake: Any,
    security_id: int,
    effective: date,
    cum_factor: float | None,
    split: float = 1.0,
    ticker: str = "AAA",
) -> None:
    lake._connection.execute(
        "INSERT INTO equity_adjustment_factors (ticker, effective_date, split_factor, cum_factor, "
        "security_id) VALUES (?, ?, ?, ?, ?)",
        [ticker, effective, split, cum_factor, security_id],
    )


def _count(lake: Any, day: date, lines: int, nominal: int, shortfall: int) -> None:
    lake._connection.execute(
        "INSERT INTO sp500_member_count VALUES (?, ?, ?, ?)", [day, lines, nominal, shortfall]
    )


def _days(*days: date) -> list[pd.Timestamp]:
    return [pd.Timestamp(day) for day in days]


@pytest.fixture
def lake(fake_lake: Any) -> Any:
    fake_lake.add_empty("equity_daily_from_bars", _BARS)
    fake_lake.add_empty("equity_adjustment_factors", _FACTORS)
    fake_lake.add_empty("sp500_member_count", _MEMBER_COUNT)
    fake_lake._connection.execute(
        "INSERT INTO equity_adjustment_factors VALUES ('ZZUNREL', DATE '2000-01-03', 2.0, 0.5, 999)"
    )
    return fake_lake


def test_daily_bars_layout_and_split_adjustment(lake: Any) -> None:
    """A 2:1 split keeps close and open continuous, halves raw_close and divides volume by the
    same factor."""
    _bar(lake, 1, _TUE, 200.0, open_=196.0, volume=1000.0)
    _bar(lake, 1, _WED, 101.0, open_=100.0, volume=2400.0)
    _bar(lake, 1, _THU, 102.0, open_=101.0, volume=2000.0)
    _factor(lake, 1, _WED, 0.5, split=2.0)

    frame = load_security_daily_bars([1], _TUE, _THU)

    assert isinstance(frame.index, pd.DatetimeIndex)
    assert list(frame.index) == _days(_TUE, _WED, _THU)
    assert isinstance(frame.columns, pd.MultiIndex)
    assert list(frame.columns.names) == ["security_id", "field"]
    assert set(frame.columns.get_level_values(0)) == {1}
    assert set(frame.columns.get_level_values(1)) == _FIELDS
    assert frame[(1, "close")].tolist() == pytest.approx([100.0, 101.0, 102.0])
    assert frame[(1, "raw_close")].tolist() == pytest.approx([200.0, 101.0, 102.0])
    assert frame[(1, "open")].tolist() == pytest.approx([98.0, 100.0, 101.0])
    assert frame[(1, "volume")].tolist() == pytest.approx([2000.0, 2400.0, 2000.0])
    log_return = frame[(1, "log_return")]
    assert math.isnan(log_return.iloc[0])
    assert log_return.iloc[1:].tolist() == pytest.approx([math.log(1.01), math.log(102 / 101)])


def test_daily_bars_dividend_factor_and_factor_row_rule(lake: Any) -> None:
    """Each bar takes the cum_factor of the first factor row strictly after it; a dividend
    factor lowers earlier levels; official_close wins over close."""
    _bar(lake, 1, _TUE, 100.0)
    _bar(lake, 1, _WED, 99.0)
    _bar(lake, 1, _THU, 999.0, official=99.5)
    _bar(lake, 1, _FRI, 50.0)
    _factor(lake, 1, _MON, 0.245, split=2.0)
    _factor(lake, 1, _WED, 0.49)
    _factor(lake, 1, _FRI, 0.5, split=2.0)

    frame = load_security_daily_bars([1], _TUE, _FRI)

    assert frame[(1, "close")].tolist() == pytest.approx([49.0, 49.5, 49.75, 50.0])
    assert frame[(1, "raw_close")].tolist() == pytest.approx([100.0, 99.0, 99.5, 50.0])
    assert frame[(1, "open")].tolist() == pytest.approx([49.0, 49.5, 499.5, 50.0])
    assert frame[(1, "log_return")].iloc[1:].tolist() == pytest.approx(
        [math.log(99 / 98), math.log(99.5 / 99), math.log(50 / 49.75)]
    )


def test_daily_bars_log_return_matches_load_security_log_returns(lake: Any) -> None:
    """log_return is the existing loader's series where it has one and NaN on every other day,
    across a missing bar, a non-positive price and a session gap."""
    bars = [
        (_MON, 100.0),
        (_TUE, 102.0),
        (_WED, None),
        (_THU, 105.0),
        (_NEXT_MON, 107.0),
        (_NEXT_TUE, 108.0),
    ]
    for day, close in bars:
        _bar(lake, 1, day, close)
    for day, close in [(_TUE, 0.0), (_WED, 80.0), (_THU, 41.0), (_FRI, 40.0), (_NEXT_MON, 41.0)]:
        _bar(lake, 2, day, close, ticker="BBB")
    _factor(lake, 2, _THU, 0.5, split=2.0, ticker="BBB")

    frame = load_security_daily_bars([1, 2], _MON, _NEXT_TUE)
    expected = load_security_log_returns([1, 2], _MON, _NEXT_TUE)

    assert set(expected) == {1, 2}
    for security_id, series in expected.items():
        present = frame[(security_id, "log_return")].dropna()
        assert list(present.index) == list(series.index)
        assert present.tolist() == pytest.approx(series.tolist())
    assert frame[(1, "log_return")].dropna().tolist() == pytest.approx(
        [math.log(1.02), math.log(108 / 107)]
    )
    assert frame[(2, "log_return")].dropna().tolist() == pytest.approx(
        [math.log(41 / 40), math.log(40 / 41), math.log(41 / 40)]
    )


def test_daily_bars_keep_securities_sharing_a_ticker_apart(lake: Any) -> None:
    """Two securities under one ticker stay two column groups, joined to factors by id only."""
    for day, close in [(_MON, 100.0), (_TUE, 102.0), (_WED, 104.0)]:
        _bar(lake, 10, day, close, ticker="TKR")
    for day, close in [(_WED, 30.0), (_THU, 31.0)]:
        _bar(lake, 11, day, close, ticker="TKR")
    _factor(lake, 11, _WED, 0.5, split=2.0, ticker="TKR")

    frame = load_security_daily_bars([10, 11, 12], _MON, _THU)
    old, new = frame[10], frame[11]

    assert set(frame.columns.get_level_values(0)) == {10, 11}
    assert old["close"].dropna().tolist() == pytest.approx([100.0, 102.0, 104.0])
    assert new["close"].dropna().tolist() == pytest.approx([30.0, 31.0])
    assert old.loc[pd.Timestamp(_THU)].isna().all()
    assert new.loc[_days(_MON, _TUE)].isna().all().all()
    assert math.isnan(new.loc[pd.Timestamp(_WED), "log_return"])
    assert new.loc[pd.Timestamp(_THU), "log_return"] == pytest.approx(math.log(31 / 30))


def test_daily_bars_range_is_inclusive_and_back_adjusted(lake: Any) -> None:
    """Bars run start to end inclusive; a factor after end still scales close but not raw_close."""
    for day, close in [(_MON, 10.0), (_TUE, 20.0), (_WED, 22.0), (_THU, 24.0), (_FRI, 13.0)]:
        _bar(lake, 1, day, close)
    _factor(lake, 1, _FRI, 0.5, split=2.0)

    frame = load_security_daily_bars([1], _TUE, _THU)

    assert list(frame.index) == _days(_TUE, _WED, _THU)
    assert frame.index.is_monotonic_increasing
    assert frame[(1, "close")].tolist() == pytest.approx([10.0, 11.0, 12.0])
    assert frame[(1, "raw_close")].tolist() == pytest.approx([20.0, 22.0, 24.0])
    assert math.isnan(frame[(1, "log_return")].iloc[0])
    assert frame[(1, "log_return")].iloc[1] == pytest.approx(math.log(1.1))


def test_daily_bars_missing_prices_and_non_positive_open_and_volume(lake: Any) -> None:
    """A null or non-positive price leaves every field NaN; a non-positive open or volume is NaN."""
    _bar(lake, 1, _TUE, 100.0, open_=0.0, volume=0.0)
    _bar(lake, 1, _WED, 101.0, open_=-1.0, volume=-5.0)
    _bar(lake, 1, _THU, None, open_=100.0)
    _bar(lake, 1, _FRI, 0.0, open_=100.0)
    _bar(lake, 1, _NEXT_MON, 100.0, official=0.0, open_=100.0)
    for day in (_TUE, _WED, _THU, _FRI, _NEXT_MON):
        _bar(lake, 2, day, 50.0, ticker="BBB")

    frame = load_security_daily_bars([1, 2], _TUE, _NEXT_MON)
    first = frame[1]

    assert first.loc[pd.Timestamp(_TUE), "close"] == pytest.approx(100.0)
    assert first.loc[pd.Timestamp(_WED), "log_return"] == pytest.approx(math.log(1.01))
    assert first.loc[_days(_TUE, _WED), ["open", "volume"]].isna().all().all()
    assert first.loc[_days(_THU, _FRI, _NEXT_MON)].isna().all().all()
    assert frame[(2, "close")].notna().all()


def test_daily_bars_none_without_ids_or_data(lake: Any) -> None:
    """No ids, unknown ids, or no bars inside the range give None."""
    _bar(lake, 1, _MON, 100.0)

    assert load_security_daily_bars([], _MON, _FRI) is None
    assert load_security_daily_bars([7], _MON, _FRI) is None
    assert load_security_daily_bars([1], _TUE, _FRI) is None


def test_daily_bars_raise_on_empty_factor_table(fake_lake: Any) -> None:
    """A read against an `equity_adjustment_factors` table with no rows raises."""
    fake_lake.add_empty("equity_daily_from_bars", _BARS)
    fake_lake.add_empty("equity_adjustment_factors", _FACTORS)
    _bar(fake_lake, 1, _MON, 100.0)
    _bar(fake_lake, 1, _TUE, 101.0)

    with pytest.raises(EmptyAdjustmentTableError):
        load_security_daily_bars([1], _MON, _TUE)


def test_member_count_range_columns_and_dtypes(lake: Any) -> None:
    """Rows dated start to end inclusive, ascending by date, with the three integer columns."""
    for day, lines in [(_FRI, 501), (_MON, 499), (_WED, 500), (_TUE, 498), (_THU, 502)]:
        _count(lake, day, lines, 503, 503 - lines)

    frame = load_sp500_member_count(_TUE, _THU)

    assert isinstance(frame.index, pd.DatetimeIndex)
    assert list(frame.index) == _days(_TUE, _WED, _THU)
    assert set(frame.columns) == {"member_lines", "nominal_members", "shortfall"}
    assert all(pd.api.types.is_integer_dtype(dtype) for dtype in frame.dtypes)
    assert frame["member_lines"].tolist() == [498, 500, 502]
    assert frame["nominal_members"].tolist() == [503, 503, 503]
    assert frame["shortfall"].tolist() == [5, 3, 1]


def test_member_count_none_when_no_rows_match(lake: Any) -> None:
    """An empty table or a range with no rows gives None."""
    assert load_sp500_member_count(_MON, _FRI) is None

    _count(lake, _MON, 500, 503, 3)

    assert load_sp500_member_count(_TUE, _FRI) is None


def test_member_count_table_is_a_read_table() -> None:
    """`sp500_member_count` is one of the tables the loader reads."""
    assert "sp500_member_count" in READ_TABLES


def test_daily_bars_volume_is_continuous_across_a_split_and_participation_is_as_traded(
    lake: Any,
) -> None:
    """Across a 2:1 split the adjusted volume is continuous, close times volume is the traded
    dollar volume, and an order's share of volume at the adjusted close is its share of the
    traded volume at the traded price."""
    _bar(lake, 1, _TUE, 200.0, volume=1000.0)
    _bar(lake, 1, _WED, 100.0, volume=2000.0)
    _factor(lake, 1, _WED, 0.5, split=2.0)

    frame = load_security_daily_bars([1], _TUE, _WED)[1]
    notional = 50_000.0

    assert frame["volume"].tolist() == pytest.approx([2000.0, 2000.0])
    assert (frame["close"] * frame["volume"]).tolist() == pytest.approx([200_000.0, 200_000.0])
    adjusted = (notional / frame["close"]) / frame["volume"]
    traded = notional / pd.Series([200.0 * 1000.0, 100.0 * 2000.0], index=frame.index)
    assert adjusted.tolist() == pytest.approx(traded.tolist())


def test_security_bar_queries_price_bars_with_the_one_raw_price_expression(lake: Any) -> None:
    """Both stored security-bar sources and the frozen reference copy in
    `scripts/at126_legacy.py` select and filter a bar's raw price with `SECURITY_RAW_PRICE_SQL`,
    which the reference copy holds unchanged, and divide volume by the price factor."""
    _bar(lake, 1, _TUE, 100.0, official=50.0)
    _bar(lake, 1, _WED, 110.0, official=60.0)
    sys.path.insert(0, str(_SCRIPTS))
    try:
        legacy = importlib.import_module("at126_legacy")
    finally:
        sys.path.remove(str(_SCRIPTS))

    load_security_log_returns([1], _TUE, _WED)
    load_security_daily_bars([1], _TUE, _WED)
    legacy.legacy_security_log_returns([1], _TUE, _WED)
    legacy.legacy_security_daily_bars([1], _TUE, _WED)

    assert legacy.SECURITY_RAW_PRICE_SQL == SECURITY_RAW_PRICE_SQL
    queries = [query for query, _ in lake.calls if "equity_daily_from_bars" in query]
    assert len(queries) == 4
    for query in queries:
        assert f"{SECURITY_RAW_PRICE_SQL} AS raw_close" in query
        assert f"AND {SECURITY_RAW_PRICE_SQL} > 0" in query
    for query in (queries[1], queries[3]):
        assert "bars.volume / COALESCE(factors.cum_factor, 1.0)" in query


def test_daily_bars_identical_factor_copies_act_as_one_row(lake: Any) -> None:
    """Identical factor rows on one effective date (one per ticker that held the security)
    give exactly the bars one such row gives."""
    for security_id, copies in ((1, ("AAA",)), (2, ("AAA", "AAB", "AAC"))):
        _bar(lake, security_id, _TUE, 200.0, volume=1000.0)
        _bar(lake, security_id, _WED, 101.0, volume=2400.0)
        _bar(lake, security_id, _THU, 102.0, volume=2000.0)
        for ticker in copies:
            _factor(lake, security_id, _WED, 0.5, split=2.0, ticker=ticker)

    frame = load_security_daily_bars([1, 2], _TUE, _THU)

    pd.testing.assert_frame_equal(frame[1], frame[2], check_exact=True)
    assert list(frame.index) == _days(_TUE, _WED, _THU)


def test_daily_bars_collapse_factor_copies_differing_outside_cum_factor(lake: Any) -> None:
    """Factor rows of one security and effective date that share their cum_factor, or are all
    null, count as one row whatever their ticker and split_factor."""
    for security_id, copies in ((1, (("AAA", 2.0),)), (2, (("AAA", 2.0), ("AAB", 3.0)))):
        _bar(lake, security_id, _TUE, 200.0)
        _bar(lake, security_id, _WED, 101.0)
        _bar(lake, security_id, _THU, 102.0)
        for ticker, split in copies:
            _factor(lake, security_id, _WED, 0.5, split=split, ticker=ticker)
            _factor(lake, security_id, _NEXT_MON, None, ticker=ticker)

    frame = load_security_daily_bars([1, 2], _TUE, _THU)

    pd.testing.assert_frame_equal(frame[1], frame[2], check_exact=True)
    assert frame[(2, "close")].tolist() == pytest.approx([100.0, 101.0, 102.0])


@pytest.mark.parametrize("other", [0.25, None])
def test_daily_bars_refuse_two_cum_factors_on_one_effective_date(
    lake: Any, other: float | None
) -> None:
    """Two factor rows of one security and effective date with different cum_factors (a null
    one included) are refused, naming the table, the security and the date."""
    _bar(lake, 7, _TUE, 200.0)
    _bar(lake, 7, _WED, 101.0)
    _factor(lake, 7, _WED, 0.5, ticker="AAA")
    _factor(lake, 7, _WED, other, ticker="AAB")

    with pytest.raises(AdjustmentFactorCollisionError) as raised:
        load_security_daily_bars([7], _TUE, _WED)

    message = str(raised.value)
    assert "equity_adjustment_factors" in message
    assert "security_id 7" in message
    assert _WED.isoformat() in message
