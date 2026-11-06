# test_loader_security_routing.py

"""`load_price_history_with_fallback(security_ids=...)` reads the listed tickers by lake
`security_id` from `equity_daily_from_bars` and every other ticker as before, giving each routed
ticker the frame layout a ticker read gives; and `check_price_coverage` refuses a firm equity
missing a window session that is not a known lake gap and a fixed instrument starting late or
ending early, and reports a short warm-up without failing it."""

from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import pytest

from agentic_trading.backtest.preflight import PriceCoverageError, check_price_coverage
from agentic_trading.data.loader import load_price_history_with_fallback
from agentic_trading.data.trading_calendar import sessions_between

_START, _END = date(2024, 3, 4), date(2024, 3, 8)
_DAYS = sessions_between(_START, _END)
_AAA_ID, _MON_ID = 7, 38786
_FIELDS = ["open", "close", "raw_close", "volume", "log_return"]


def _yahoo_row(lake: Any, ticker: str, day: date, close: float, volume: int) -> None:
    """One `instrument_prices` row whose adjusted and raw prices are equal."""
    lake._connection.execute(
        "INSERT INTO instrument_prices (ticker, date, open, high, low, close, adj_close, volume) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [ticker, day, close - 1.0, close + 1.0, close - 2.0, close, close, volume],
    )


def _bar(lake: Any, security_id: int, ticker: str, day: date, close: float, volume: float) -> None:
    """One `equity_daily_from_bars` bar whose official close is its close."""
    lake._connection.execute(
        "INSERT INTO equity_daily_from_bars "
        "(ticker, date, open, high, low, close, official_close, volume, security_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [ticker, day, close - 1.0, close + 1.0, close - 2.0, close, close, volume, security_id],
    )


@pytest.fixture
def lake(preflight_lake: Any) -> Any:
    """AAA in both price tables with the same prices, MON only by security id (no
    instrument_prices rows) and without a bar on the middle session, SPY only by ticker."""
    for index, day in enumerate(_DAYS):
        close = 100.0 + 2.0 * index
        _yahoo_row(preflight_lake, "AAA", day, close, 1000 + index)
        _bar(preflight_lake, _AAA_ID, "AAA", day, close, 1000.0 + index)
        _yahoo_row(preflight_lake, "SPY", day, 400.0 + index, 5000)
        if index != 2:
            _bar(preflight_lake, _MON_ID, "MON", day, 80.0 - index, 300.0)
    return preflight_lake


def test_a_security_routed_ticker_has_the_ticker_read_layout(lake: Any) -> None:
    """Routed by id or read by ticker, AAA has the same fields, order, dtypes, index and column
    level names, and on equal prices the same values, under both alignments."""
    for align in ("sessions", "intersection"):
        by_ticker, _ = load_price_history_with_fallback(["AAA", "SPY"], _START, _END, align=align)
        by_id, _ = load_price_history_with_fallback(
            ["AAA", "SPY"], _START, _END, align=align, security_ids={"AAA": _AAA_ID}
        )

        assert list(by_id["AAA"].columns) == list(by_ticker["AAA"].columns) == _FIELDS
        assert by_id.columns.names == by_ticker.columns.names
        assert sorted(by_id.columns) == sorted(by_ticker.columns)
        assert by_id.index.equals(by_ticker.index) and by_id.index.name == by_ticker.index.name
        assert (by_id["AAA"].dtypes == np.float64).all()
        pd.testing.assert_frame_equal(by_id["AAA"], by_ticker["AAA"], check_exact=False)
        pd.testing.assert_frame_equal(by_id["SPY"], by_ticker["SPY"])


def test_a_ticker_absent_from_instrument_prices_loads_by_security_id(lake: Any) -> None:
    """MON has no instrument_prices rows, so a ticker read raises; by its id it loads, NaN on
    the session it has no bar, its log return spanning that session as a ticker's would."""
    with pytest.raises(ValueError, match="MON"):
        load_price_history_with_fallback(["MON", "SPY"], _START, _END)

    frame, _ = load_price_history_with_fallback(
        ["MON", "SPY"], _START, _END, security_ids={"MON": _MON_ID}
    )

    mon = frame["MON"]
    assert list(mon.columns) == _FIELDS
    assert list(frame.index.date) == _DAYS
    assert mon.loc[pd.Timestamp(_DAYS[2])].isna().all()
    assert mon["close"].dropna().tolist() == [80.0, 79.0, 77.0, 76.0]
    assert mon.loc[pd.Timestamp(_DAYS[3]), "log_return"] == pytest.approx(np.log(77.0 / 79.0))
    assert mon["volume"].dropna().tolist() == [300.0] * 4


_YEAR = sessions_between(date(2022, 1, 3), date(2023, 12, 29))
_WINDOW = [day for day in _YEAR if day.year == 2023]


def _coverage_frame(bars: dict[str, list[date]]) -> pd.DataFrame:
    """A `(ticker, close)` frame on every XNYS session of 2022-2023, each ticker closed on its
    `bars`."""
    index = pd.DatetimeIndex(_YEAR)
    return pd.concat(
        {
            ticker: pd.DataFrame(
                {"close": [1.0 if day.date() in set(days) else np.nan for day in index]},
                index=index,
            )
            for ticker, days in bars.items()
        },
        axis=1,
    )


def _check(frame: pd.DataFrame, members: list[str], **options: Any) -> Any:
    """`check_price_coverage` over a 2022 warm-up and a 2023 window."""
    return check_price_coverage(
        frame, members, date(2022, 1, 1), date(2023, 1, 1), date(2023, 12, 31), **options
    )


def test_a_firm_equity_missing_one_window_session_fails_unless_it_is_a_known_gap() -> None:
    """A complete member with an interior one-day gap fails, the error naming the day; the same
    gap listed as a known lake gap of its security passes."""
    hole = _WINDOW[100]
    frame = _coverage_frame({"AAA": [day for day in _YEAR if day != hole]})

    with pytest.raises(PriceCoverageError, match=f"AAA missing 1 session\\(s\\): {hole}"):
        _check(frame, ["AAA"], complete={"AAA": 7})

    passing = _check(frame, ["AAA"], complete={"AAA": 7}, known_gaps=[(7, hole)])
    assert passing[0].covers_run and passing[0].missing_sessions == []
    assert passing[0].longest_gap_sessions == 1
    with pytest.raises(PriceCoverageError):
        _check(frame, ["AAA"], complete={"AAA": 7}, known_gaps=[(8, hole)])


def test_missing_sessions_listed_up_to_ten_with_their_count() -> None:
    """An equity missing many sessions is listed with its first ten and the count of the rest,
    and every failing member appears in the one error."""
    frame = _coverage_frame({"AAA": _YEAR[:-30], "BBB": _YEAR[:-1]})

    with pytest.raises(PriceCoverageError) as raised:
        _check(frame, ["AAA", "BBB"], complete={"AAA": 7, "BBB": 8})

    message = str(raised.value)
    assert "AAA missing 30 session(s): " + ", ".join(d.isoformat() for d in _YEAR[-30:-20]) in (
        message
    )
    assert "and 20 more" in message
    assert f"BBB missing 1 session(s): {_YEAR[-1].isoformat()}" in message


def test_a_short_warmup_passes_and_is_reported() -> None:
    """A member listing inside the warm-up (as V did in 2008) passes with its first bar and
    the warm-up sessions it has; a crypto pair starting inside the window is exempt at the
    start, not at the end."""
    listing = _YEAR[120]
    frame = _coverage_frame(
        {
            "VVV": [day for day in _YEAR if day >= listing],
            "FIX": _YEAR,
            "BTC-USD": _WINDOW[200:],
        }
    )

    coverage = _check(
        frame, ["VVV", "FIX", "BTC-USD"], complete={"VVV": 9}, start_exempt=["BTC-USD"]
    )

    short = coverage[0].short_warmup
    assert all(entry.covers_run for entry in coverage)
    assert short is not None and short.instrument == "VVV" and short.first_bar == listing
    assert short.warmup_sessions == len(_YEAR) - len(_WINDOW) - 120
    assert coverage[1].short_warmup is None and coverage[2].short_warmup is None

    ended = _coverage_frame({"BTC-USD": _WINDOW[200:-6]})
    with pytest.raises(PriceCoverageError, match="BTC-USD first bar"):
        _check(ended, ["BTC-USD"], start_exempt=["BTC-USD"])


def test_a_fixed_instrument_fails_when_it_starts_late_or_ends_early() -> None:
    """A fixed instrument needs a bar within five sessions of the window's first and last
    sessions; an interior gap never fails it, and every failing member is listed."""
    gap = set(_WINDOW[50:120])
    frame = _coverage_frame(
        {
            "GAPPY": [day for day in _YEAR if day not in gap],
            "SLACK": [day for day in _YEAR if _WINDOW[5] <= day <= _WINDOW[-6]],
            "LATE": [day for day in _YEAR if day >= _WINDOW[6]],
            "EARLY": [day for day in _YEAR if day <= _WINDOW[-7]],
        }
    )

    with pytest.raises(PriceCoverageError) as raised:
        _check(frame, ["GAPPY", "SLACK", "LATE", "EARLY", "GONE"])

    message = str(raised.value)
    assert f"LATE first bar {_WINDOW[6].isoformat()}, last bar {_WINDOW[-1].isoformat()}" in (
        message
    )
    assert f"EARLY first bar {_YEAR[0].isoformat()}, last bar {_WINDOW[-7].isoformat()}" in (
        message
    )
    assert "GONE first bar none, last bar none" in message
    assert "GAPPY" not in message and "SLACK" not in message
    passing = _check(frame, ["GAPPY", "SLACK"])
    assert passing[0].longest_gap_sessions == 70
    assert passing[1].short_warmup is not None
