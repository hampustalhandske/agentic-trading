# test_sp500_member_prices.py

"""S&P 500 members and their split-adjusted log returns are resolved by security id."""

import math
from datetime import date
from typing import Any

import pandas as pd
import pytest

from agentic_trading.data.loaders.lakehouse_loader import (
    IndexMember,
    load_security_log_returns,
    load_sp500_member_securities_asof,
)

_COVERAGE = {
    "ticker": "VARCHAR",
    "start_date": "DATE",
    "end_date": "DATE",
    "security_id": "BIGINT",
    "first_bar_date": "DATE",
    "last_bar_date": "DATE",
    "n_bar_days": "INTEGER",
}
_BARS = {
    "ticker": "VARCHAR",
    "date": "DATE",
    "close": "DOUBLE",
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

_MON, _TUE, _WED, _THU, _FRI = (date(2024, 3, d) for d in (4, 5, 6, 7, 8))
_NEXT_MON = date(2024, 3, 11)


def _span(lake: Any, ticker: str, start: str, end: str | None, security_id: int) -> None:
    lake._connection.execute(
        "INSERT INTO sp500_price_coverage (ticker, start_date, end_date, security_id) "
        "VALUES (?, ?, ?, ?)",
        [ticker, start, end, security_id],
    )


def _bar(
    lake: Any,
    security_id: int,
    day: date,
    close: float,
    official: float | None = None,
    ticker="AAA",
) -> None:
    lake._connection.execute(
        "INSERT INTO equity_daily_from_bars (ticker, date, close, official_close, security_id) "
        "VALUES (?, ?, ?, ?, ?)",
        [ticker, day, close, official, security_id],
    )


@pytest.fixture
def lake(fake_lake: Any) -> Any:
    fake_lake.add_empty("sp500_price_coverage", _COVERAGE)
    fake_lake.add_empty("equity_daily_from_bars", _BARS)
    fake_lake.add_empty("equity_adjustment_factors", _FACTORS)
    return fake_lake


def test_member_securities_follow_the_security_across_a_rename(lake: Any) -> None:
    """FB and META are one security; the ticker is the one whose span covers the date."""
    _span(lake, "FB", "2013-12-23", "2022-06-09", 21100)
    _span(lake, "META", "2022-06-09", None, 21100)

    assert load_sp500_member_securities_asof(date(2022, 6, 8)) == [
        IndexMember(security_id=21100, ticker="FB")
    ]
    assert load_sp500_member_securities_asof(date(2022, 6, 9)) == [
        IndexMember(security_id=21100, ticker="META")
    ]


def test_member_securities_span_bounds_ordering_and_empty(lake: Any) -> None:
    """Start is inclusive, end exclusive, results sorted by ticker, None when nothing counts."""
    _span(lake, "ZZZ", "2015-01-02", None, 3)
    _span(lake, "AAA", "2010-01-04", None, 1)
    _span(lake, "GONE", "2012-01-03", "2020-01-02", 2)

    on_start = load_sp500_member_securities_asof(date(2012, 1, 3))
    on_end = load_sp500_member_securities_asof(date(2020, 1, 2))

    assert on_start == [
        IndexMember(security_id=1, ticker="AAA"),
        IndexMember(security_id=2, ticker="GONE"),
    ]
    assert on_end == [
        IndexMember(security_id=1, ticker="AAA"),
        IndexMember(security_id=3, ticker="ZZZ"),
    ]
    assert load_sp500_member_securities_asof(date(2009, 1, 1)) is None


def test_member_securities_reused_ticker_and_duplicate_security(lake: Any) -> None:
    """A reused ticker resolves to the covering security; one security yields one entry."""
    _span(lake, "TKR", "2010-01-04", "2020-01-02", 10)
    _span(lake, "TKR", "2020-01-02", None, 11)
    _span(lake, "OLD", "2010-01-04", None, 20)
    _span(lake, "NEW", "2015-01-02", None, 20)

    before = load_sp500_member_securities_asof(date(2019, 6, 3))
    after = load_sp500_member_securities_asof(date(2021, 6, 3))

    assert before == [
        IndexMember(security_id=20, ticker="NEW"),
        IndexMember(security_id=10, ticker="TKR"),
    ]
    assert after == [
        IndexMember(security_id=20, ticker="NEW"),
        IndexMember(security_id=11, ticker="TKR"),
    ]


def test_log_returns_split_adjusted_across_the_effective_date(lake: Any) -> None:
    """A 7:1 split leaves the return across it at log(101/100), not log(101/700)."""
    _bar(lake, 1, _TUE, 700.0)
    _bar(lake, 1, _WED, 101.0)
    lake._connection.execute(
        "INSERT INTO equity_adjustment_factors (ticker, effective_date, split_factor, cum_factor, "
        "security_id) VALUES ('AAA', ?, 7.0, ?, 1)",
        [_WED, 1 / 7],
    )

    series = load_security_log_returns([1], _TUE, _WED)[1]

    assert isinstance(series.index, pd.DatetimeIndex)
    assert list(series.index) == [pd.Timestamp(_WED)]
    assert series.iloc[0] == pytest.approx(math.log(101 / 100))


def test_log_returns_prefer_official_close_and_fall_back_to_close(lake: Any) -> None:
    """official_close is used when present and close when it is null."""
    _bar(lake, 1, _TUE, 100.0, official=50.0)
    _bar(lake, 1, _WED, 999.0, official=55.0)
    _bar(lake, 2, _TUE, 100.0)
    _bar(lake, 2, _WED, 110.0)

    result = load_security_log_returns([1, 2], _TUE, _WED)

    assert result[1].iloc[0] == pytest.approx(math.log(55 / 50))
    assert result[2].iloc[0] == pytest.approx(math.log(110 / 100))


def test_log_returns_never_span_a_missing_session(lake: Any) -> None:
    """A return needs bars on the day and the session before; a gap yields no multi-day return."""
    for day, close in [(_TUE, 100.0), (_WED, 102.0), (_FRI, 106.0), (_NEXT_MON, 107.0)]:
        _bar(lake, 1, day, close)

    series = load_security_log_returns([1], _TUE, _NEXT_MON)[1]

    assert list(series.index) == [pd.Timestamp(_WED), pd.Timestamp(_NEXT_MON)]
    assert series.iloc[0] == pytest.approx(math.log(102 / 100))
    assert series.iloc[1] == pytest.approx(math.log(107 / 106))
    assert series.index.is_monotonic_increasing
    assert list(load_security_log_returns([1], _TUE, _WED)[1].index) == [pd.Timestamp(_WED)]


def test_log_returns_join_on_security_id_only(lake: Any) -> None:
    """Two securities sharing a ticker string keep their own prices; bar-less ones are absent."""
    for day, close in [(_TUE, 100.0), (_WED, 110.0)]:
        _bar(lake, 1, day, close, ticker="SAME")
    for day, close in [(_TUE, 50.0), (_WED, 40.0)]:
        _bar(lake, 2, day, close, ticker="SAME")

    only_first = load_security_log_returns([1], _TUE, _WED)
    both = load_security_log_returns([1, 2, 3], _TUE, _WED)

    assert set(only_first) == {1}
    assert only_first[1].iloc[0] == pytest.approx(math.log(1.1))
    assert set(both) == {1, 2}
    assert both[2].iloc[0] == pytest.approx(math.log(0.8))
