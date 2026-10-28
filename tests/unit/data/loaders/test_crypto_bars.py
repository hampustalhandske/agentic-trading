# test_crypto_bars.py

"""Crypto price bars built from `crypto_klines_1m` on the NYSE session clock (America/New_York)."""

from datetime import date
from typing import Any

import pandas as pd
import pytest

from agentic_trading.data.loaders import lakehouse_loader

_START = date(2023, 1, 17)
_END = date(2023, 6, 17)


def _klines(rows: list[tuple[str, str, float, float, float]]) -> pd.DataFrame:
    """Rows of (symbol, UTC minute start, open, close, volume)."""
    return pd.DataFrame(
        [
            {
                "symbol": symbol,
                "open_time": pd.Timestamp(minute),
                "open": o,
                "high": max(o, c),
                "low": min(o, c),
                "close": c,
                "volume": v,
                "quote_volume": v * c,
                "n_trades": 10,
                "taker_buy_base_volume": v / 2,
                "taker_buy_quote_volume": v * c / 2,
                "dt": pd.Timestamp(minute).normalize(),
                "table": "crypto_klines_1m",
            }
            for symbol, minute, o, c, v in rows
        ]
    )


def _close_dates(result: pd.DataFrame | None, ticker: str) -> set[date]:
    """Dates on which `ticker` has a non-NaN close."""
    if result is None or ticker not in result.columns.get_level_values(0):
        return set()
    return {stamp.date() for stamp in result[ticker]["close"].dropna().index}


def _seed(fake_lake: Any, rows: list[tuple[str, str, float, float, float]]) -> None:
    fake_lake.add("crypto_klines_1m", _klines(rows), date_columns=("dt",))


def test_crypto_bar_uses_new_york_session_minutes_across_dst(fake_lake: Any) -> None:
    """Open is the 09:30 ET minute, close the 15:59 ET minute and volume the
    09:30-15:59 ET sum, in both EDT and EST."""
    _seed(
        fake_lake,
        [
            ("BTCUSDT", "2023-06-14 13:29", 1.0, 1.0, 100.0),
            ("BTCUSDT", "2023-06-14 13:30", 25_900.0, 25_890.0, 1.0),
            ("BTCUSDT", "2023-06-14 15:00", 25_890.0, 25_880.0, 2.0),
            ("BTCUSDT", "2023-06-14 19:59", 25_875.0, 25_874.59, 3.0),
            ("BTCUSDT", "2023-06-14 20:00", 25_874.0, 25_877.73, 100.0),
            ("ETHUSDT", "2023-06-14 13:30", 1_700.0, 1_699.0, 5.0),
            ("ETHUSDT", "2023-06-14 19:59", 1_698.0, 1_701.5, 5.0),
            ("BTCUSDT", "2023-01-17 14:30", 21_000.0, 21_010.0, 4.0),
            ("BTCUSDT", "2023-01-17 19:59", 20_000.0, 20_001.0, 50.0),
            ("BTCUSDT", "2023-01-17 20:59", 21_100.0, 21_123.45, 6.0),
            ("BTCUSDT", "2023-01-17 21:00", 21_200.0, 21_200.0, 70.0),
        ],
    )
    fake_lake.add(
        "instrument_prices",
        pd.DataFrame(
            [("BTC-USD", "2023-06-14", 1.0, 1.0, 1.0, 1.0)],
            columns=["ticker", "date", "open", "close", "adj_close", "volume"],
        ),
        date_columns=("date",),
    )

    result = lakehouse_loader.load_price_history_from_lakehouse(
        ["BTC-USD", "ETH-USD"], _START, _END
    )

    assert result is not None
    summer, winter = pd.Timestamp("2023-06-14"), pd.Timestamp("2023-01-17")
    btc = result["BTC-USD"]
    assert btc.loc[summer, "close"] == pytest.approx(25_874.6, rel=5e-4)
    assert btc.loc[summer, "close"] == pytest.approx(25_874.59)
    assert btc.loc[summer, "raw_close"] == pytest.approx(25_874.59)
    assert btc.loc[summer, "open"] == pytest.approx(25_900.0)
    assert btc.loc[summer, "volume"] == pytest.approx(6.0)
    assert btc.loc[winter, "close"] == pytest.approx(21_123.45)
    assert btc.loc[winter, "open"] == pytest.approx(21_000.0)
    assert btc.loc[winter, "volume"] == pytest.approx(60.0)
    assert result["ETH-USD"].loc[summer, "close"] == pytest.approx(1_701.5)
    assert result["ETH-USD"].loc[summer, "open"] == pytest.approx(1_700.0)


def test_crypto_has_no_bar_on_weekends_or_without_a_close_minute(fake_lake: Any) -> None:
    """Weekend days and sessions lacking a 15:59 ET minute produce no bar."""
    _seed(
        fake_lake,
        [
            ("BTCUSDT", "2023-06-14 13:30", 25_900.0, 25_890.0, 1.0),
            ("BTCUSDT", "2023-06-14 19:59", 25_875.0, 25_874.59, 3.0),
            ("BTCUSDT", "2023-06-15 13:30", 26_000.0, 26_010.0, 1.0),
            ("BTCUSDT", "2023-06-17 13:30", 26_100.0, 26_110.0, 1.0),
            ("BTCUSDT", "2023-06-17 19:59", 26_120.0, 26_130.0, 1.0),
        ],
    )

    result = lakehouse_loader.load_price_history_from_lakehouse(["BTC-USD"], _START, _END)

    assert _close_dates(result, "BTC-USD") == {date(2023, 6, 14)}


def test_non_crypto_ticker_still_reads_instrument_prices(fake_lake: Any) -> None:
    """An equity requested alongside crypto is served from `instrument_prices`."""
    _seed(
        fake_lake,
        [
            ("BTCUSDT", "2023-06-14 13:30", 25_900.0, 25_890.0, 1.0),
            ("BTCUSDT", "2023-06-14 19:59", 25_875.0, 25_874.59, 3.0),
        ],
    )
    fake_lake.add(
        "instrument_prices",
        pd.DataFrame(
            [("AAPL", "2023-06-14", 183.0, 183.5, 183.4, 1_000_000)],
            columns=["ticker", "date", "open", "close", "adj_close", "volume"],
        ),
        date_columns=("date",),
    )

    result = lakehouse_loader.load_price_history_from_lakehouse(["AAPL", "BTC-USD"], _START, _END)

    assert result is not None
    day = pd.Timestamp("2023-06-14")
    assert result["AAPL"].loc[day, "close"] == pytest.approx(183.4)
    assert result["AAPL"].loc[day, "raw_close"] == pytest.approx(183.5)
    assert result["BTC-USD"].loc[day, "close"] == pytest.approx(25_874.59)
