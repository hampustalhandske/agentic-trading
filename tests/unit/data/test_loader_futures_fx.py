# test_loader_futures_fx.py

"""`load_price_history_with_fallback` reads futures tickers from the
continuous futures loader and every other ticker from the price loader, and
keeps a roll on a dropped row by counting it on the next kept row; FX pairs
leave the price frame as it is. `load_fx_policy_rates` reads the policy
rates of a run's FX currencies. The lake loaders are mocked at the module
boundary."""

from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from agentic_trading.data.loader import load_price_history_with_fallback
from agentic_trading.data.loaders.macro_loader import load_fx_policy_rates

_DAYS = pd.DatetimeIndex(pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]))


def _priced(
    tickers: list[str], index: pd.DatetimeIndex, *, roll: list[float] | None = None
) -> pd.DataFrame:
    """A `(ticker, field)` frame of constant closes on `index`, with a `roll` column."""
    frames = {}
    for ticker in tickers:
        close = pd.Series(100.0, index=index)
        fields = {
            "open": close,
            "close": close,
            "raw_close": close,
            "volume": np.nan,
            "log_return": np.log(close / close.shift(1)),
        }
        if roll is not None:
            fields["roll"] = pd.Series(roll, index=index)
        frames[ticker] = pd.DataFrame(fields, index=index)
    return pd.concat(frames, axis=1)


@patch("agentic_trading.data.loader.load_futures_price_history_from_lakehouse")
@patch("agentic_trading.data.loader.load_price_history_from_lakehouse")
def test_futures_tickers_read_the_continuous_series(prices: MagicMock, futures: MagicMock) -> None:
    prices.return_value = _priced(["AAPL", "EURUSD=X"], _DAYS)
    futures.return_value = _priced(["CL=F"], _DAYS, roll=[0.0, 1.0, 0.0, 0.0])

    frame, _ = load_price_history_with_fallback(
        ["AAPL", "CL=F", "EURUSD=X"], date(2024, 1, 2), date(2024, 1, 5), align="intersection"
    )

    assert prices.call_args.args[0] == ["AAPL", "EURUSD=X"]
    assert futures.call_args.args[0] == ["CL=F"]
    assert frame[("CL=F", "roll")].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert ("AAPL", "roll") not in frame.columns
    assert set(frame["EURUSD=X"].columns) == {"open", "close", "raw_close", "volume", "log_return"}


@patch("agentic_trading.data.loader.load_futures_price_history_from_lakehouse")
@patch("agentic_trading.data.loader.load_price_history_from_lakehouse")
def test_only_futures_skip_the_price_loader(prices: MagicMock, futures: MagicMock) -> None:
    futures.return_value = _priced(["GC=F"], _DAYS, roll=[0.0] * 4)

    frame, _ = load_price_history_with_fallback(
        ["GC=F"], date(2024, 1, 2), date(2024, 1, 5), align="intersection"
    )

    prices.assert_not_called()
    assert list(frame.index) == list(_DAYS)


@patch("agentic_trading.data.loader.load_futures_price_history_from_lakehouse")
@patch("agentic_trading.data.loader.load_price_history_from_lakehouse")
def test_a_roll_on_a_dropped_row_counts_on_the_next_kept_row(
    prices: MagicMock, futures: MagicMock
) -> None:
    prices.return_value = _priced(["AAPL"], _DAYS.delete(1))
    futures.return_value = _priced(["CL=F"], _DAYS, roll=[0.0, 1.0, 1.0, 0.0])

    frame, _ = load_price_history_with_fallback(
        ["AAPL", "CL=F"], date(2024, 1, 2), date(2024, 1, 5), align="intersection"
    )

    assert list(frame.index) == list(_DAYS.delete(1))
    assert frame[("CL=F", "roll")].tolist() == [0.0, 2.0, 0.0]


@patch("agentic_trading.data.loaders.macro_loader.load_policy_rates_from_lakehouse")
def test_fx_policy_rates_cover_every_currency_of_the_pairs(rates: MagicMock) -> None:
    rates.return_value = pd.DataFrame(
        {"EUR": [0.04], "JPY": [-0.001], "USD": [0.05]}, index=pd.to_datetime(["2024-01-02"])
    )

    result = load_fx_policy_rates(
        ["EURUSD=X", "USDJPY=X", "AAPL", "CL=F"], date(2024, 1, 2), date(2024, 1, 5)
    )

    assert rates.call_args.args == (["EUR", "JPY", "USD"], date(2024, 1, 2), date(2024, 1, 5))
    assert result is rates.return_value


@patch("agentic_trading.data.loaders.macro_loader.load_policy_rates_from_lakehouse")
def test_fx_policy_rates_are_none_without_fx_pairs(rates: MagicMock) -> None:
    assert load_fx_policy_rates(["AAPL", "CL=F"], date(2024, 1, 2), date(2024, 1, 5)) is None
    rates.assert_not_called()


@patch("agentic_trading.data.loaders.macro_loader.load_policy_rates_from_lakehouse")
def test_a_currency_without_policy_rates_raises(rates: MagicMock) -> None:
    rates.return_value = pd.DataFrame(
        {"AUD": [np.nan], "USD": [0.05]}, index=pd.to_datetime(["2024-01-02"])
    )

    with pytest.raises(ValueError, match="AUD"):
        load_fx_policy_rates(["AUDUSD=X"], date(2024, 1, 2), date(2024, 1, 5))

    rates.return_value = None
    with pytest.raises(ValueError, match="USD"):
        load_fx_policy_rates(["AUDUSD=X"], date(2024, 1, 2), date(2024, 1, 5))
