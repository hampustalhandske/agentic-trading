# test_crypto_volume_determinism.py

"""`_fetch_crypto_prices` returns bit-identical daily volumes on every read of the same minutes,
with DuckDB running the query on several threads.

The fake lake holds a year of every UTC minute for both crypto symbols, inserted in a seeded
random order so each session's minutes are spread over many row groups, and therefore over
several threads' partial sums. Volumes are seeded random floats spanning six orders of
magnitude, with all their significant digits. On this data a plain `sum(volume)` over the
DOUBLE column gives daily volumes that differ in their last bits between reads (DuckDB 1.5.6,
8 threads).

`_fetch_crypto_prices` also refuses a session holding a minute volume of magnitude 1e24 or
more, and leaves every other session's volume exactly as the DECIMAL sum gives it."""

from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import pytest

from agentic_trading.data.loaders import lakehouse_loader

_TICKERS = ["BTC-USD", "ETH-USD"]
_START = date(2023, 1, 2)
_DAYS = 365
_READS = 20
_THREADS = 8


def _klines() -> pd.DataFrame:
    """Every UTC minute of `_DAYS` days from `_START` for both symbols, in a seeded random row
    order, with log-uniform volumes from 1e-3 to 1e3."""
    rng = np.random.default_rng(147)
    minutes = pd.date_range(pd.Timestamp(_START), periods=_DAYS * 1440, freq="min")
    frames = []
    for symbol in ("BTCUSDT", "ETHUSDT"):
        price = rng.uniform(1_000.0, 30_000.0, minutes.size)
        frames.append(
            pd.DataFrame(
                {
                    "symbol": symbol,
                    "open_time": minutes,
                    "open": price,
                    "close": price * rng.uniform(0.999, 1.001, minutes.size),
                    "volume": 10.0 ** rng.uniform(-3.0, 3.0, minutes.size),
                }
            )
        )
    klines = pd.concat(frames, ignore_index=True)
    return klines.iloc[rng.permutation(len(klines))].reset_index(drop=True)


def _read(fake_lake: Any) -> dict[str, pd.DataFrame]:
    """One `_fetch_crypto_prices` read of both tickers over the whole year."""
    end = _START + pd.Timedelta(days=_DAYS - 1)
    return lakehouse_loader._fetch_crypto_prices(_TICKERS, _START, end)


def _volume_bits(frames: dict[str, pd.DataFrame]) -> dict[str, bytes]:
    """Each ticker's daily volumes as raw float64 bytes."""
    return {
        ticker: frame["volume"].to_numpy(dtype=np.float64).tobytes()
        for ticker, frame in frames.items()
    }


def test_crypto_daily_volume_is_bit_identical_across_multithreaded_reads(fake_lake: Any) -> None:
    """`_READS` reads of the same minutes on `_THREADS` DuckDB threads give the same bytes for
    every ticker's daily volumes, and each volume is its session minutes' sum to within 1e-12."""
    klines = _klines()
    fake_lake.add("crypto_klines_1m", klines, pin_column="open_time", kind="append-only")
    fake_lake._connection.execute(f"SET threads TO {_THREADS}")

    first = _read(fake_lake)
    assert all(len(frame) >= _DAYS - 1 for frame in first.values())
    first_bits = _volume_bits(first)
    for _ in range(_READS - 1):
        assert _volume_bits(_read(fake_lake)) == first_bits

    clock = klines["open_time"].dt.tz_localize("UTC").dt.tz_convert("America/New_York")
    minute = clock.dt.strftime("%H:%M")
    session = (minute >= "09:30") & (minute <= "15:59") & (klines["symbol"] == "BTCUSDT")
    expected = klines.loc[session, "volume"].groupby(clock[session].dt.date).sum()
    btc = first["BTC-USD"]
    days = [pd.Timestamp(day).date() for day in btc["date"]]
    np.testing.assert_allclose(
        btc["volume"].to_numpy(dtype=np.float64), expected.loc[days].to_numpy(), rtol=1e-12
    )


# Five UTC days of minutes for the guard tests, from a Monday.
_GUARD_START = date(2023, 6, 12)
_GUARD_DAYS = 5


def _guard_klines() -> pd.DataFrame:
    """Every UTC minute of `_GUARD_DAYS` days from `_GUARD_START` for both symbols, with seeded
    log-uniform volumes from 1e-3 to 1e3, and one BTCUSDT session minute of volume 9.99e23."""
    rng = np.random.default_rng(1024)
    minutes = pd.date_range(pd.Timestamp(_GUARD_START), periods=_GUARD_DAYS * 1440, freq="min")
    frames = [
        pd.DataFrame(
            {
                "symbol": symbol,
                "open_time": minutes,
                "open": 20_000.0,
                "close": 20_000.0,
                "volume": 10.0 ** rng.uniform(-3.0, 3.0, minutes.size),
            }
        )
        for symbol in ("BTCUSDT", "ETHUSDT")
    ]
    klines = pd.concat(frames, ignore_index=True)
    klines.loc[_minute(klines, "BTCUSDT", "2023-06-13 15:00"), "volume"] = 9.99e23
    return klines


def _minute(klines: pd.DataFrame, symbol: str, open_time: str) -> pd.Series:
    """The mask of `symbol`'s kline opening at UTC `open_time`."""
    return (klines["symbol"] == symbol) & (klines["open_time"] == pd.Timestamp(open_time))


def _read_guard_days() -> dict[str, pd.DataFrame]:
    """One `_fetch_crypto_prices` read of both tickers over the guard days."""
    end = _GUARD_START + pd.Timedelta(days=_GUARD_DAYS - 1)
    return lakehouse_loader._fetch_crypto_prices(_TICKERS, _GUARD_START, end)


def _plain_decimal_volumes(fake_lake: Any) -> dict[str, bytes]:
    """Each symbol's daily session volumes from a plain DECIMAL(38, 10) sum of the lake's
    minutes, with no guard, as raw float64 bytes in date order."""
    frame = fake_lake._connection.execute(
        """
        SELECT symbol, CAST(local_time AS DATE) AS date,
               CAST(sum(CAST(volume AS DECIMAL(38, 10))) AS DOUBLE) AS volume
        FROM (
            SELECT symbol, volume,
                   timezone('America/New_York', open_time AT TIME ZONE 'UTC') AS local_time
            FROM crypto_klines_1m
        )
        WHERE CAST(local_time AS TIME) BETWEEN TIME '09:30:00' AND TIME '15:59:00'
        GROUP BY symbol, CAST(local_time AS DATE)
        ORDER BY symbol, date
        """
    ).df()
    return {
        symbol: rows["volume"].to_numpy(dtype=np.float64).tobytes()
        for symbol, rows in frame.groupby("symbol")
    }


def test_volumes_below_the_minute_limit_are_the_plain_decimal_sum(fake_lake: Any) -> None:
    """With every session minute volume below 1e24, including one of 9.99e23, each daily volume
    has the same bytes as a plain DECIMAL(38, 10) sum of its session minutes."""
    fake_lake.add("crypto_klines_1m", _guard_klines(), pin_column="open_time", kind="append-only")

    frames = _read_guard_days()

    assert {
        lakehouse_loader._CRYPTO_KLINE_SYMBOLS[ticker]: frame["volume"]
        .to_numpy(dtype=np.float64)
        .tobytes()
        for ticker, frame in frames.items()
    } == _plain_decimal_volumes(fake_lake)
    assert len(frames["BTC-USD"]) == _GUARD_DAYS


@pytest.mark.parametrize("volume", [1e24, -1e24, 5e27])
def test_a_session_minute_volume_of_1e24_or_more_raises(fake_lake: Any, volume: float) -> None:
    """A session minute whose volume has magnitude 1e24 or more makes the read raise, naming the
    symbol and the date."""
    klines = _guard_klines()
    klines.loc[_minute(klines, "ETHUSDT", "2023-06-14 18:00"), "volume"] = volume
    fake_lake.add("crypto_klines_1m", klines, pin_column="open_time", kind="append-only")

    with pytest.raises(ValueError, match=r"ETHUSDT on 2023-06-14 .* not below 1e\+24"):
        _read_guard_days()


def test_a_minute_volume_of_1e24_outside_the_session_is_ignored(fake_lake: Any) -> None:
    """A minute volume of 1e24 outside every session's minutes leaves the read and its volumes
    unchanged."""
    klines = _guard_klines()
    klines.loc[_minute(klines, "ETHUSDT", "2023-06-14 02:00"), "volume"] = 1e24
    fake_lake.add("crypto_klines_1m", klines, pin_column="open_time", kind="append-only")

    frames = _read_guard_days()

    assert (
        frames["ETH-USD"]["volume"].to_numpy(dtype=np.float64).tobytes()
        == _plain_decimal_volumes(fake_lake)["ETHUSDT"]
    )
