# test_lake_pin_during_writes.py

"""A lake session at lake time T reads the lake as it stood at T while the lake keeps writing:
a rebuilt-whole table that gets a new build after T, and `crypto_klines_1m` minutes that arrive
after T (a late minute whose `open_time` is before T among them), leave every read of the
session as it was, and the session closes without an error. A session at a later T' sees the
new build and the late minutes."""

from datetime import UTC, date, datetime
from typing import Any

import pandas as pd
import pytest

from agentic_trading.data.lakehouse_client import LakehouseError
from agentic_trading.data.loaders import lakehouse_loader as ll

_T = datetime(2026, 11, 10, 19, 55, 57, tzinfo=UTC)
_LATER_T = datetime(2026, 11, 11, 12, tzinfo=UTC)
_BUILD_AT_T = "2026-11-10T03:00:00Z"
_AFTER_T = "2026-11-11T03:00:00Z"
_SPLIT_COLUMNS = ["ticker", "ex_date", "ratio", "in_close_basis"]
_AS_OF = date(2024, 12, 31)


def _splits(rows: list[tuple[str, str, float, bool]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=_SPLIT_COLUMNS)


def _seed_splits(lake: Any) -> None:
    """`yahoo_split_close_basis` with one build, written before T: one AAPL 4-for-1 split."""
    lake.add(
        "yahoo_split_close_basis",
        _splits([("AAPL", "2024-08-01", 0.25, True)]),
        ("ex_date",),
        scraped_at=_BUILD_AT_T,
        rebuilt_whole=True,
    )


def _rebuild_splits(lake: Any) -> None:
    """A new build of `yahoo_split_close_basis`, written after T, with other rows: the AAPL
    split restated as 2-for-1 and a second split added."""
    lake.add_build(
        "yahoo_split_close_basis",
        _splits([("AAPL", "2024-08-01", 0.5, True), ("AAPL", "2024-10-01", 0.5, True)]),
        _AFTER_T,
    )


def _split_reads() -> tuple[float, list[date], int]:
    return (
        ll.split_adjustment_factor("AAPL", date(2024, 1, 2)),
        ll.load_split_ex_dates("AAPL", _AS_OF),
        ll.adjustment_table_row_count("yahoo_split_close_basis"),
    )


def test_a_rebuild_during_the_session_leaves_its_reads_at_the_build_of_t(fake_lake: Any) -> None:
    """A build written after T, in the middle of an open session, changes nothing the session
    reads, and the session closes without an error."""
    _seed_splits(fake_lake)

    with ll.lake_session(_T):
        before = _split_reads()
        _rebuild_splits(fake_lake)
        sent = len(fake_lake.calls)
        after = _split_reads()
        assert len(fake_lake.calls) > sent
    assert ll.current_lake_time() is None

    assert before == after == (4.0, [date(2024, 8, 1)], 1)


def test_a_session_at_a_later_lake_time_sees_the_new_build(fake_lake: Any) -> None:
    """Negative control: a session at a T' after the rebuild reads the new build, and a
    session at T still reads the old one."""
    _seed_splits(fake_lake)
    _rebuild_splits(fake_lake)

    with ll.lake_session(_LATER_T):
        later = _split_reads()
    with ll.lake_session(_T):
        at_t = _split_reads()

    assert later == (4.0, [date(2024, 8, 1), date(2024, 10, 1)], 2)
    assert at_t == (4.0, [date(2024, 8, 1)], 1)


def test_a_lake_time_before_every_build_is_refused_by_the_lake(fake_lake: Any) -> None:
    """A T before the oldest build kept is answered with the lake's HTTP 400, never with an
    empty table."""
    _seed_splits(fake_lake)

    with (
        pytest.raises(LakehouseError, match=r"HTTP 400.*yahoo_split_close_basis"),
        ll.lake_session(datetime(2026, 11, 9, tzinfo=UTC)),
    ):
        ll.load_split_ex_dates("AAPL", _AS_OF)


def _minute(open_time: str, close: float, volume: float) -> dict[str, Any]:
    """One BTCUSDT minute starting at `open_time` (naive UTC)."""
    stamp = pd.Timestamp(open_time)
    return {
        "symbol": "BTCUSDT",
        "open_time": stamp,
        "open": close - 1.0,
        "high": close + 1.0,
        "low": close - 2.0,
        "close": close,
        "volume": volume,
        "quote_volume": close * volume,
        "n_trades": 10,
        "taker_buy_base_volume": volume / 2,
        "taker_buy_quote_volume": close * volume / 2,
        "dt": stamp.normalize(),
        "table": "crypto_klines_1m",
    }


def _seed_klines(lake: Any) -> None:
    """The session's first and last minute of 2026-11-09 (09:30 and 15:59 New York time),
    written before T."""
    minutes = [_minute("2026-11-09 14:30", 100.0, 1.0), _minute("2026-11-09 20:59", 101.0, 2.0)]
    lake.add(
        "crypto_klines_1m",
        pd.DataFrame(minutes),
        ("dt",),
        scraped_at=_BUILD_AT_T,
        kind="append-only",
    )


def _append_klines(lake: Any) -> None:
    """Minutes written after T: a late minute of 2026-11-09 (13:00 New York time, `open_time`
    a day before T) and the first and last minute of 2026-11-10, whose first minute also opens
    before T."""
    minutes = [
        _minute("2026-11-09 18:00", 100.5, 40.0),
        _minute("2026-11-10 14:30", 102.0, 3.0),
        _minute("2026-11-10 20:59", 103.0, 4.0),
    ]
    lake.add_log_rows("crypto_klines_1m", pd.DataFrame(minutes), _AFTER_T)


def _crypto_bars() -> list[tuple[str, float, float, float]]:
    frame = ll.load_price_history_from_lakehouse(["BTC-USD"], date(2026, 11, 9), date(2026, 11, 10))
    assert frame is not None
    bars = frame["BTC-USD"]
    return [
        (stamp.date().isoformat(), float(row["open"]), float(row["close"]), float(row["volume"]))
        for stamp, row in bars.iterrows()
    ]


def test_minutes_arriving_during_the_session_are_invisible_to_it(fake_lake: Any) -> None:
    """Minutes written after T, a late one with `open_time` before T among them, change
    nothing the session reads, and the session closes without an error."""
    _seed_klines(fake_lake)

    with ll.lake_session(_T):
        before = _crypto_bars()
        _append_klines(fake_lake)
        after = _crypto_bars()
    assert ll.current_lake_time() is None

    assert before == after == [("2026-11-09", 99.0, 101.0, 3.0)]


def test_a_session_at_a_later_lake_time_sees_the_late_minutes(fake_lake: Any) -> None:
    """Negative control: a session at a T' after the appended minutes counts the late minute
    in 2026-11-09's volume and has a bar for 2026-11-10."""
    _seed_klines(fake_lake)
    _append_klines(fake_lake)

    with ll.lake_session(_LATER_T):
        later = _crypto_bars()
    with ll.lake_session(_T):
        at_t = _crypto_bars()

    assert later == [("2026-11-09", 99.0, 101.0, 43.0), ("2026-11-10", 101.0, 103.0, 7.0)]
    assert at_t == [("2026-11-09", 99.0, 101.0, 3.0)]
