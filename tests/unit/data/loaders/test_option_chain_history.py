# test_option_chain_history.py

"""Unit tests for the `option_chain_history` fallback in the option-chain loaders.

`dolthub_option_chain` starts 2019-02-09 for SPY and has no QQQ or IWM rows;
`option_chain_history` holds SPY, QQQ and IWM end-of-day chains from 2008.
Both tables are seeded with different bids so each result identifies its source.
"""

from datetime import date, timedelta
from typing import Any

import pandas as pd
import pytest

from agentic_trading.data.loaders import lakehouse_loader

_COLUMNS = [
    "date",
    "expiration",
    "option_type",
    "strike",
    "bid",
    "ask",
    "implied_volatility",
    "delta",
    "gamma",
    "theta",
    "vega",
    "rho",
]
_GREEKS = {
    "implied_volatility": 0.2,
    "delta": 0.5,
    "gamma": 0.01,
    "theta": -0.1,
    "vega": 0.2,
    "rho": 0.05,
}


def _row(underlying: str, day: str, bid: float) -> dict[str, Any]:
    expiration = (date.fromisoformat(day) + timedelta(days=30)).isoformat()
    return {
        "underlying": underlying,
        "expiration": expiration,
        "option_type": "call",
        "strike": 100.0,
        "bid": bid,
        "ask": bid + 0.2,
        **_GREEKS,
    }


def _dolthub(underlying: str, day: str, bid: float) -> dict[str, Any]:
    return {**_row(underlying, day, bid), "date": day}


def _history(underlying: str, day: str, bid: float) -> dict[str, Any]:
    return {
        **_row(underlying, day, bid),
        "option_symbol": f"{underlying}-{day}",
        "snapshot_date": day,
        "mark": 999.0,
        "last_trade_price": 0.01,
        "open_interest": 1234,
    }


@pytest.fixture
def lake(fake_lake: Any) -> Any:
    """A fake lake with distinguishable bids in both chain tables and no splits."""
    dolthub = [
        _dolthub("SPY", "2019-02-09", 1.0),
        _dolthub("SPY", "2019-03-01", 2.0),
        _dolthub("AAPL", "2015-06-01", 3.0),
    ]
    history = [
        _history("SPY", "2012-06-15", 7.0),
        _history("SPY", "2019-03-01", 9.0),
        _history("QQQ", "2020-06-01", 4.0),
        _history("IWM", "2020-05-20", 5.0),
        _history("AAPL", "2015-06-01", 8.0),
    ]
    fake_lake.add(
        "dolthub_option_chain", pd.DataFrame(dolthub), date_columns=("date", "expiration")
    )
    fake_lake.add(
        "option_chain_history",
        pd.DataFrame(history),
        date_columns=("snapshot_date", "expiration"),
    )
    fake_lake.add_empty(
        "yahoo_split_close_basis",
        {"ticker": "VARCHAR", "ex_date": "DATE", "ratio": "DOUBLE", "in_close_basis": "BOOLEAN"},
    )
    return fake_lake


@pytest.mark.parametrize(
    ("ticker", "as_of", "lookback", "bid"),
    [
        ("SPY", date(2012, 6, 15), 10, 7.0),
        ("SPY", date(2019, 3, 1), 10, 2.0),
        ("QQQ", date(2020, 6, 1), 10, 4.0),
        ("AAPL", date(2015, 6, 1), 10, 3.0),
        ("IWM", date(2020, 5, 25), 10, 5.0),
        ("IWM", date(2020, 6, 1), 10, None),
    ],
)
def test_snapshot_source_selection(
    lake: Any, ticker: str, as_of: date, lookback: int, bid: float | None
) -> None:
    """The snapshot loader reads history only for pre-dolthub SPY and for QQQ/IWM."""
    frame = lakehouse_loader.load_options_snapshot_from_lakehouse(ticker, as_of, lookback)
    if bid is None:
        assert frame is None
    else:
        assert frame is not None
        assert frame["bid"].tolist() == [bid]


def test_history_snapshot_shape_and_prices(lake: Any) -> None:
    """History results match the dolthub columns and types and expose bid/ask, not mark."""
    history = lakehouse_loader.load_options_snapshot_from_lakehouse("SPY", date(2012, 6, 15), 10)
    dolthub = lakehouse_loader.load_options_snapshot_from_lakehouse("SPY", date(2019, 3, 1), 10)
    assert history is not None and dolthub is not None
    assert list(history.columns) == _COLUMNS
    assert history.dtypes.to_dict() == dolthub.dtypes.to_dict()
    assert history["date"].iloc[0] == pd.Timestamp("2012-06-15")
    assert history["bid"].iloc[0] == 7.0
    assert history["ask"].iloc[0] == pytest.approx(7.2)
    assert not {"mark", "last_trade_price"} & set(history.columns)


@pytest.mark.parametrize(
    ("underlying", "as_of", "bid"),
    [
        ("SPY", date(2012, 6, 15), 7.0),
        ("SPY", date(2019, 3, 1), 2.0),
        ("QQQ", date(2020, 6, 1), 4.0),
        ("AAPL", date(2015, 6, 1), 3.0),
    ],
)
def test_quote_source_selection(lake: Any, underlying: str, as_of: date, bid: float) -> None:
    """The quote loader follows the same source rule and returns bid/ask only."""
    expiration = as_of + timedelta(days=30)
    frame = lakehouse_loader.load_option_quote_from_lakehouse(
        underlying, expiration, "call", 100.0, as_of, 10
    )
    assert frame is not None
    assert frame["bid"].tolist() == [bid]
    assert list(frame.columns) == _COLUMNS
    assert not {"mark", "last_trade_price"} & set(frame.columns)


def test_quote_history_staleness(lake: Any) -> None:
    """A history quote older than the lookback window yields None."""
    expiration = date(2020, 6, 19)
    stale = lakehouse_loader.load_option_quote_from_lakehouse(
        "IWM", expiration, "call", 100.0, date(2020, 6, 1), 10
    )
    assert stale is None
