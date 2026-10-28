# test_option_settlement.py

"""Unit tests for expiry settlement in `option_marks.load_option_mark`.

On or after expiry a contract settles at intrinsic value (per share, reported as
bid = ask = mid) against the underlying's close on the last session on or before
the expiry, falling back to the passed spot when the lake has no such close.
"""

from collections.abc import Iterator
from datetime import date
from typing import Any

import pandas as pd
import pytest

from agentic_trading.agents.schemas import OptionContract
from agentic_trading.data.loaders import option_marks
from agentic_trading.data.loaders.option_marks import load_option_mark

FRIDAY_EXPIRY = date(2020, 1, 17)
HOLIDAY_EXPIRY = date(2020, 1, 20)
AFTER_HOLIDAY = date(2020, 1, 21)


def _call(expiry: date) -> OptionContract:
    return OptionContract(underlying="XOM", expiry=expiry, strike=70.0, option_type="call")


def _seed_prices(lake: Any, closes: list[tuple[str, str, float]]) -> None:
    lake.add(
        "instrument_prices",
        pd.DataFrame(
            [(ticker, day, close, close, close, 1_000_000) for ticker, day, close in closes],
            columns=["ticker", "date", "open", "close", "adj_close", "volume"],
        ),
        date_columns=("date",),
    )


@pytest.fixture
def xom_closes(fake_lake: Any) -> Iterator[Any]:
    """Lake holding XOM closes around MLK Day 2020 and clean option caches."""
    _seed_prices(
        fake_lake,
        [
            ("XOM", "2020-01-16", 70.5),
            ("XOM", "2020-01-17", 71.0),
            ("XOM", "2020-01-21", 74.0),
        ],
    )
    option_marks.clear_cache()
    yield fake_lake
    option_marks.clear_cache()


def test_settles_at_the_expiry_session_close_not_the_later_spot(xom_closes: Any) -> None:
    """A mark after expiry uses the expiry-day close, not the later day's spot."""
    mark = load_option_mark(_call(FRIDAY_EXPIRY), AFTER_HOLIDAY, spot=74.0)

    assert mark is not None
    assert mark.is_expired
    assert mark.mid == mark.bid == mark.ask == pytest.approx(1.0)


def test_mark_on_the_expiry_day_is_intrinsic_at_that_close(xom_closes: Any) -> None:
    mark = load_option_mark(_call(FRIDAY_EXPIRY), FRIDAY_EXPIRY, spot=71.0)

    assert mark is not None
    assert mark.mid == pytest.approx(1.0)


def test_holiday_expiry_settles_at_the_prior_session_close(xom_closes: Any) -> None:
    """An expiry on a market holiday uses the last close before it."""
    mark = load_option_mark(_call(HOLIDAY_EXPIRY), AFTER_HOLIDAY, spot=74.0)

    assert mark is not None
    assert mark.mid == pytest.approx(1.0)


def test_falls_back_to_the_passed_spot_without_a_close(fake_lake: Any) -> None:
    """No close for the ticker, or none within 10 days of expiry, uses `spot`."""
    _seed_prices(fake_lake, [("AAPL", "2020-01-17", 300.0), ("XOM", "2019-12-01", 60.0)])
    option_marks.clear_cache()

    stale = load_option_mark(_call(FRIDAY_EXPIRY), AFTER_HOLIDAY, spot=74.0)
    other_ticker = load_option_mark(
        OptionContract(underlying="CVX", expiry=FRIDAY_EXPIRY, strike=70.0, option_type="call"),
        AFTER_HOLIDAY,
        spot=74.0,
    )

    assert stale is not None and other_ticker is not None
    assert stale.mid == pytest.approx(4.0)
    assert other_ticker.mid == pytest.approx(4.0)
    option_marks.clear_cache()
