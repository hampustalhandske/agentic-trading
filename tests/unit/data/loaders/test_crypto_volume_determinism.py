# test_crypto_volume_determinism.py

"""`_fetch_crypto_prices` returns bit-identical daily volumes on every read of the same minutes,
with DuckDB running the query on several threads.

The fake lake holds a year of every UTC minute for both crypto symbols, inserted in a seeded
random order so each session's minutes are spread over many row groups, and therefore over
several threads' partial sums. Volumes are seeded random floats spanning six orders of
magnitude, with all their significant digits. On this data a plain `sum(volume)` over the
DOUBLE column gives daily volumes that differ in their last bits between reads (DuckDB 1.5.6,
8 threads)."""

from datetime import date
from typing import Any

import numpy as np
import pandas as pd

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
