# test_loader_security_routing.py

"""`load_price_history_with_fallback(security_ids=...)` reads the listed tickers by lake
`security_id` from `equity_daily_from_bars` and every other ticker as before, giving each routed
ticker the frame layout a ticker read gives; and `check_price_coverage` refuses a universe member
whose bars start late or end early, listing every such member."""

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
    """One `equity_daily_from_bars` bar; its official_close is never read."""
    lake._connection.execute(
        "INSERT INTO equity_daily_from_bars "
        "(ticker, date, open, high, low, close, official_close, volume, security_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [ticker, day, close - 1.0, close + 1.0, close - 2.0, close, 1.0, volume, security_id],
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


def _coverage_frame(bars: dict[str, list[date]]) -> pd.DataFrame:
    """A `(ticker, close)` frame on every XNYS session of 2023, each ticker closed on `bars`."""
    index = pd.DatetimeIndex(sessions_between(date(2023, 1, 3), date(2023, 12, 29)))
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


def test_price_coverage_lists_every_member_starting_late_or_ending_early() -> None:
    """A first bar more than five sessions after the first session, a last bar more than five
    before the last, or no bar at all fails, all members listed in one error; an interior gap
    and bars within the slack pass."""
    sessions = sessions_between(date(2023, 1, 3), date(2023, 12, 29))
    gap = set(sessions[100:150])
    frame = _coverage_frame(
        {
            "FULL": [day for day in sessions if day not in gap],
            "SLACK": sessions[5:-5],
            "LATE": sessions[6:],
            "EARLY": sessions[:-6],
        }
    )

    with pytest.raises(PriceCoverageError) as raised:
        check_price_coverage(
            frame, ["FULL", "SLACK", "LATE", "EARLY", "GONE"], date(2023, 1, 1), date(2023, 12, 31)
        )

    message = str(raised.value)
    assert f"LATE first bar {sessions[6].isoformat()}, last bar {sessions[-1].isoformat()}" in (
        message
    )
    assert f"EARLY first bar {sessions[0].isoformat()}, last bar {sessions[-7].isoformat()}" in (
        message
    )
    assert "GONE first bar none, last bar none" in message
    assert "FULL" not in message and "SLACK" not in message

    passing = check_price_coverage(frame, ["FULL", "SLACK"], date(2023, 1, 1), date(2023, 12, 31))
    assert [entry.covers_run for entry in passing] == [True, True]
    assert passing[0].longest_gap_sessions == 50
    assert (passing[1].first_bar, passing[1].last_bar) == (sessions[5], sessions[-6])
