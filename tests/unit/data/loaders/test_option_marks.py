"""Unit tests for agentic_trading.data.loaders.option_marks.

The lakehouse chain query is replaced at the module boundary
(`option_marks._query_chain`) with a synthetic chain — two snapshot
dates, four expiries (one already passed by the second snapshot), a
strike ladder with one zero-bid deep-OTM row — and the DGS3MO rate
lookup (`option_marks._query_rate`) with a constant. No DuckDB access.
"""

from collections.abc import Iterator
from datetime import date
from typing import Any

import pandas as pd
import pytest
from quantcore.pricing.black_scholes import black_scholes_call

from agentic_trading.agents.schemas import OptionContract
from agentic_trading.config import temporary_overrides
from agentic_trading.data.loaders import option_marks
from agentic_trading.data.loaders.option_marks import (
    list_expiries,
    load_option_mark,
    nearest_strike,
    select_expiry,
)

UNDERLYING = "AAPL"
SNAPSHOT_DATES = [date(2024, 3, 1), date(2024, 3, 5)]
EXPIRIES = [date(2024, 3, 1), date(2024, 3, 15), date(2024, 4, 5), date(2024, 5, 17)]
# strike -> (call bid, call ask); the 200 call is the zero-bid deep-OTM row.
CALL_QUOTES = {170.0: (11.0, 11.4), 180.0: (3.9, 4.1), 190.0: (0.9, 1.1), 200.0: (0.0, 0.05)}
PUT_QUOTES = {170.0: (0.8, 1.0), 180.0: (3.5, 3.7), 190.0: (10.5, 10.9), 200.0: (20.0, 20.6)}
IMPLIED_VOL = 0.25
RATE = 0.05


def _chain() -> pd.DataFrame:
    rows = []
    for snapshot_date in SNAPSHOT_DATES:
        for expiry in EXPIRIES:
            for option_type, quotes in (("call", CALL_QUOTES), ("put", PUT_QUOTES)):
                for strike, (bid, ask) in quotes.items():
                    rows.append(
                        {
                            "date": pd.Timestamp(snapshot_date),
                            "expiration": pd.Timestamp(expiry),
                            "option_type": option_type,
                            "strike": strike,
                            "bid": bid,
                            "ask": ask,
                            "implied_volatility": IMPLIED_VOL,
                            "delta": 0.5,
                            "gamma": 0.02,
                            "theta": -0.05,
                            "vega": 0.1,
                            "rho": 0.01,
                        }
                    )
    return pd.DataFrame(rows)


class _ChainQueries:
    """Stand-in for `_query_chain`: the latest synthetic snapshot dated on
    or before `as_of_date`, counting every call."""

    def __init__(self) -> None:
        self.frame = _chain()
        self.calls: list[tuple[str, date]] = []

    def __call__(self, underlying: str, as_of_date: date) -> pd.DataFrame | None:
        self.calls.append((underlying, as_of_date))
        if underlying != UNDERLYING:
            return None
        eligible = self.frame[self.frame["date"] <= pd.Timestamp(as_of_date)]
        if eligible.empty:
            return None
        return eligible[eligible["date"] == eligible["date"].max()].reset_index(drop=True)


@pytest.fixture
def chain(monkeypatch: pytest.MonkeyPatch, fake_lake: Any) -> Iterator[_ChainQueries]:
    """Synthetic snapshots through `_query_chain`; the exact-contract
    lookup falls through to a lake whose option chain holds no rows, and
    expiry settlement to a lake with no underlying closes."""
    fake_lake.add(
        "dolthub_option_chain",
        pd.DataFrame(
            {
                "underlying": pd.Series([], dtype="string"),
                "date": pd.Series([], dtype="datetime64[ns]"),
                "expiration": pd.Series([], dtype="datetime64[ns]"),
                "option_type": pd.Series([], dtype="string"),
                **{
                    column: pd.Series([], dtype=float)
                    for column in (
                        "strike",
                        "bid",
                        "ask",
                        "implied_volatility",
                        "delta",
                        "gamma",
                        "theta",
                        "vega",
                        "rho",
                    )
                },
            }
        ),
    )
    fake_lake.add_empty(
        "instrument_prices",
        {"ticker": "VARCHAR", "date": "DATE", "close": "DOUBLE", "adj_close": "DOUBLE"},
    )
    queries = _ChainQueries()
    option_marks.clear_cache()
    monkeypatch.setattr(option_marks, "_query_chain", queries)
    monkeypatch.setattr(option_marks, "_query_rate", lambda as_of_date: RATE)
    yield queries
    option_marks.clear_cache()


def _call(strike: float = 180.0, expiry: date = date(2024, 4, 5)) -> OptionContract:
    return OptionContract(underlying=UNDERLYING, expiry=expiry, strike=strike, option_type="call")


def _put(strike: float = 180.0, expiry: date = date(2024, 4, 5)) -> OptionContract:
    return OptionContract(underlying=UNDERLYING, expiry=expiry, strike=strike, option_type="put")


def test_snapshot_day_mark_uses_the_quoted_bid_ask_as_is(chain: _ChainQueries) -> None:
    mark = load_option_mark(_call(), date(2024, 3, 5), spot=180.0)

    assert mark is not None
    assert mark.snapshot_date == date(2024, 3, 5)
    assert mark.bid == pytest.approx(3.9)
    assert mark.ask == pytest.approx(4.1)
    assert mark.mid == pytest.approx(4.0)
    assert mark.implied_volatility == pytest.approx(IMPLIED_VOL)
    assert mark.delta == pytest.approx(0.5)
    assert not mark.is_model_mark
    assert not mark.is_expired
    assert mark.tradeable


def test_between_snapshots_the_mark_is_black_scholes_at_todays_spot(chain: _ChainQueries) -> None:
    """Two days after the latest snapshot the contract is re-marked at the
    snapshot's implied vol and the given spot: the mid is the quantcore
    Black-Scholes price (so it moves with spot), the quoted relative
    bid/ask spread is preserved around it, and the mark is flagged as a
    model mark."""
    as_of = date(2024, 3, 7)
    low = load_option_mark(_call(), as_of, spot=175.0)
    high = load_option_mark(_call(), as_of, spot=185.0)

    assert low is not None and high is not None
    assert low.is_model_mark and high.is_model_mark
    assert low.snapshot_date == date(2024, 3, 5)
    expected_high = black_scholes_call(185.0, 180.0, RATE, IMPLIED_VOL, 29 / 365)
    assert high.mid == pytest.approx(expected_high)
    assert high.mid > low.mid
    quoted_relative_spread = (4.1 - 3.9) / 4.0
    assert (high.ask - high.bid) / high.mid == pytest.approx(quoted_relative_spread)
    assert high.tradeable
    assert 0.0 < high.delta < 1.0
    assert high.gamma > 0.0
    assert high.vega > 0.0
    assert high.theta < 0.0


def test_model_mark_greeks_follow_the_option_type(chain: _ChainQueries) -> None:
    put = load_option_mark(_put(), date(2024, 3, 7), spot=180.0)

    assert put is not None
    assert put.is_model_mark
    assert -1.0 < put.delta < 0.0


def test_on_expiry_the_mark_is_intrinsic_with_zero_greeks(chain: _ChainQueries) -> None:
    expiry = date(2024, 3, 15)
    call = load_option_mark(_call(expiry=expiry), expiry, spot=190.0)
    put = load_option_mark(_put(expiry=expiry), expiry, spot=190.0)
    after = load_option_mark(_call(expiry=expiry), date(2024, 3, 18), spot=190.0)

    assert call is not None and put is not None and after is not None
    assert call.is_expired and put.is_expired and after.is_expired
    assert call.mid == call.bid == call.ask == pytest.approx(10.0)
    assert put.mid == pytest.approx(0.0)
    assert after.mid == pytest.approx(10.0)
    assert (call.delta, call.gamma, call.vega, call.theta) == (0.0, 0.0, 0.0, 0.0)
    assert not call.tradeable


def test_zero_bid_row_is_marked_but_untradeable(chain: _ChainQueries) -> None:
    on_snapshot = load_option_mark(_call(strike=200.0), date(2024, 3, 5), spot=180.0)
    between = load_option_mark(_call(strike=200.0), date(2024, 3, 7), spot=180.0)

    assert on_snapshot is not None and between is not None
    assert not on_snapshot.tradeable
    assert on_snapshot.mid == pytest.approx(0.025)
    assert between.is_model_mark
    assert between.bid == 0.0
    assert not between.tradeable


def test_never_quoted_contract_has_no_mark(chain: _ChainQueries) -> None:
    assert load_option_mark(_call(strike=999.0), date(2024, 3, 5), spot=180.0) is None
    assert load_option_mark(_call(expiry=date(2024, 6, 21)), date(2024, 3, 5), spot=180.0) is None


def test_snapshot_older_than_the_age_window_marks_but_is_untradeable(
    chain: _ChainQueries, monkeypatch
) -> None:
    """Past the trading window the contract is still re-marked from its
    last quote (so a held position keeps a value) but cannot be traded;
    past the marking window there is no mark at all."""
    from agentic_trading.data.loaders import option_marks as mod

    monkeypatch.setattr(mod, "load_option_quote_from_lakehouse", lambda *_a, **_k: None)
    with temporary_overrides({"options.max_snapshot_age_days": 10}):
        assert load_option_mark(_call(), date(2024, 3, 20), spot=180.0) is None
        assert load_option_mark(_call(), date(2024, 3, 15), spot=180.0) is not None


def test_no_mark_before_the_first_snapshot(chain: _ChainQueries) -> None:
    assert load_option_mark(_call(), date(2024, 2, 20), spot=180.0) is None


def test_list_expiries_excludes_expiries_already_passed(chain: _ChainQueries) -> None:
    assert list_expiries(UNDERLYING, date(2024, 3, 1)) == EXPIRIES[1:]
    assert list_expiries(UNDERLYING, date(2024, 3, 5)) == EXPIRIES[1:]
    assert list_expiries("ZZZZ", date(2024, 3, 5)) == []


def test_select_expiry_picks_the_listed_expiry_nearest_the_target_tenor(
    chain: _ChainQueries,
) -> None:
    assert select_expiry(UNDERLYING, date(2024, 3, 5), 30) == date(2024, 4, 5)
    assert select_expiry(UNDERLYING, date(2024, 3, 5), 7) == date(2024, 3, 15)
    assert select_expiry("ZZZZ", date(2024, 3, 5), 30) is None


def test_nearest_strike_skips_zero_bid_rows(chain: _ChainQueries) -> None:
    as_of = date(2024, 3, 5)
    assert nearest_strike(UNDERLYING, date(2024, 4, 5), "call", 183.0, as_of) == 180.0
    assert nearest_strike(UNDERLYING, date(2024, 4, 5), "call", 199.0, as_of) == 190.0
    assert nearest_strike(UNDERLYING, date(2024, 4, 5), "put", 199.0, as_of) == 200.0
    assert nearest_strike(UNDERLYING, date(2024, 6, 21), "call", 180.0, as_of) is None


def test_chain_is_queried_once_per_underlying_and_day(chain: _ChainQueries) -> None:
    as_of = date(2024, 3, 5)
    load_option_mark(_call(), as_of, spot=180.0)
    load_option_mark(_put(190.0), as_of, spot=180.0)
    list_expiries(UNDERLYING, as_of)
    nearest_strike(UNDERLYING, date(2024, 4, 5), "call", 180.0, as_of)

    assert chain.calls == [(UNDERLYING, as_of)]

    load_option_mark(_call(), date(2024, 3, 6), spot=180.0)

    assert chain.calls == [(UNDERLYING, as_of), (UNDERLYING, date(2024, 3, 6))]
