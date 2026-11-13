# test_instruments.py

"""Instrument conventions read from a ticker: a future's lake root, an FX
pair's base and quote currency, and the USD value of one unit of a marked
position and of one unit of its quote currency."""

import pytest

from agentic_trading.data.instruments import (
    CurrencyPair,
    currency_pair,
    futures_root,
    is_marked,
    quote_unit_value_usd,
    unit_value_usd,
)


@pytest.mark.parametrize(
    ("ticker", "root"),
    [("GC=F", "GC"), ("CL=F", "CL"), ("SI=F", "SI"), ("NG=F", "NG"), ("HG=F", "HG")],
)
def test_futures_root_strips_the_yahoo_suffix(ticker: str, root: str) -> None:
    assert futures_root(ticker) == root
    assert is_marked(ticker)


@pytest.mark.parametrize("ticker", ["AAPL", "SPY", "EURUSD=X", "BTC-USD", "^VIX"])
def test_futures_root_is_none_for_other_tickers(ticker: str) -> None:
    assert futures_root(ticker) is None


@pytest.mark.parametrize(
    ("ticker", "base", "quote"),
    [
        ("EURUSD=X", "EUR", "USD"),
        ("GBPUSD=X", "GBP", "USD"),
        ("USDJPY=X", "USD", "JPY"),
        ("AUDUSD=X", "AUD", "USD"),
    ],
)
def test_currency_pair_reads_base_and_quote_from_the_ticker(
    ticker: str, base: str, quote: str
) -> None:
    assert currency_pair(ticker) == CurrencyPair(base=base, quote=quote)
    assert is_marked(ticker)


@pytest.mark.parametrize("ticker", ["AAPL", "GC=F", "BTC-USD", "EURUSD"])
def test_currency_pair_is_none_for_other_tickers(ticker: str) -> None:
    assert currency_pair(ticker) is None


@pytest.mark.parametrize("ticker", ["AAPL", "SPY", "BTC-USD", "^VIX"])
def test_cash_securities_are_not_marked(ticker: str) -> None:
    assert not is_marked(ticker)


def test_unit_values_for_a_usd_quoted_pair_and_a_future() -> None:
    """One unit of EUR (EURUSD) or of a futures level is worth the price in USD, and the
    quote currency is USD itself."""
    for ticker in ("EURUSD=X", "CL=F"):
        assert unit_value_usd(ticker, 1.25) == 1.25
        assert quote_unit_value_usd(ticker, 1.25) == 1.0


def test_unit_values_for_a_usd_base_pair() -> None:
    """One unit of a USDJPY position is one USD, and one JPY is worth 1 / S USD."""
    assert unit_value_usd("USDJPY=X", 150.0) == 1.0
    assert quote_unit_value_usd("USDJPY=X", 150.0) == pytest.approx(1.0 / 150.0)


def test_a_pair_without_a_usd_leg_is_refused() -> None:
    with pytest.raises(ValueError, match="EURJPY"):
        unit_value_usd("EURJPY=X", 160.0)
    with pytest.raises(ValueError, match="EURJPY"):
        quote_unit_value_usd("EURJPY=X", 160.0)
