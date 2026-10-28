# test_option_publication_lag.py

"""Unit tests for the option-chain publication lag (`settings.options.publication_lag_days`).

Agents may only see a chain snapshot dated on or before
`as_of_date - publication_lag_days`; the engine's marks and fills are not lagged.
"""

from collections.abc import Iterator
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
import pytest

from agentic_trading.agents.analysts.cvar_pod._options_data import (
    real_atm_options_by_expiration,
)
from agentic_trading.agents.managers import cvar_manager
from agentic_trading.agents.managers.tail_hedge_manager import TailHedgeManager
from agentic_trading.agents.portfolio_construction import PortfolioConstructor
from agentic_trading.agents.schemas import OptionContract
from agentic_trading.config import settings, temporary_overrides
from agentic_trading.data.loaders import option_marks
from agentic_trading.scoring.weighting import EqualWeight

AS_OF = pd.Timestamp("2020-01-02")
SPOT = 70.0
EXPIRATION = "2020-03-20"


def _chain(snapshots: dict[str, list[float]]) -> pd.DataFrame:
    rows = [
        {
            "underlying": "XOM",
            "date": snapshot,
            "expiration": EXPIRATION,
            "option_type": option_type,
            "strike": strike,
            "bid": 1.0,
            "ask": 1.2,
            "implied_volatility": 0.25,
            "delta": 0.5 if option_type == "call" else -0.5,
            "gamma": 0.02,
            "theta": -0.05,
            "vega": 0.1,
            "rho": 0.01,
        }
        for snapshot, strikes in snapshots.items()
        for strike in strikes
        for option_type in ("call", "put")
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def xom_chain(fake_lake: Any) -> Any:
    """Lake with two XOM snapshots (2020-01-02, 2019-12-31) and no splits."""
    fake_lake.add(
        "dolthub_option_chain",
        _chain({"2020-01-02": [70.0, 72.5], "2019-12-31": [65.0, 67.5]}),
        date_columns=("date", "expiration"),
    )
    fake_lake.add_empty(
        "yahoo_split_close_basis",
        {"ticker": "VARCHAR", "ex_date": "DATE", "ratio": "DOUBLE", "in_close_basis": "BOOLEAN"},
    )
    return fake_lake


@pytest.fixture
def recorded_as_of(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[date]]:
    """Replaces the chain loader seen by `option_marks` with a recorder returning no chain."""
    seen: list[date] = []

    def _loader(ticker: str, as_of_date: date, lookback_days: int = 10) -> None:
        seen.append(as_of_date)

    option_marks.clear_cache()
    monkeypatch.setattr(option_marks, "load_options_snapshot_from_lakehouse", _loader)
    monkeypatch.setattr(option_marks, "load_option_quote_from_lakehouse", lambda *_a, **_k: None)
    yield seen
    option_marks.clear_cache()


def test_default_publication_lag_is_one_day() -> None:
    """The shipped default hides a chain for exactly one calendar day."""
    assert settings.options.publication_lag_days == 1


def test_atm_helper_only_sees_snapshots_older_than_the_lag(xom_chain: Any) -> None:
    """The default lag hides the same-day snapshot; a zero or two-day lag moves the cutoff."""
    lagged = real_atm_options_by_expiration("XOM", AS_OF, SPOT)
    assert lagged is not None
    assert lagged["strike"].tolist() == [67.5]
    assert lagged["days_to_expiration"].tolist() == [(date(2020, 3, 20) - date(2019, 12, 31)).days]

    with temporary_overrides({"options.publication_lag_days": 0}):
        unlagged = real_atm_options_by_expiration("XOM", AS_OF, SPOT)
    assert unlagged is not None
    assert unlagged["strike"].tolist() == [70.0]
    assert unlagged["days_to_expiration"].tolist() == [(date(2020, 3, 20) - date(2020, 1, 2)).days]

    with temporary_overrides({"options.publication_lag_days": 2}):
        two_day = real_atm_options_by_expiration("XOM", AS_OF, SPOT)
    assert two_day is not None
    assert two_day["strike"].tolist() == [67.5]

    assert settings.options.publication_lag_days == 1


def test_cvar_manager_contract_selection_reads_the_chain_as_of_the_lag(
    recorded_as_of: list[date],
) -> None:
    """`_select_contract` queries the chain as of `day - lag`, following the setting."""
    day = date(2024, 6, 20)

    cvar_manager._select_contract("AAPL", "call", 190.0, 190.0, day)
    assert recorded_as_of
    assert set(recorded_as_of) == {day - timedelta(days=1)}

    recorded_as_of.clear()
    option_marks.clear_cache()
    with temporary_overrides({"options.publication_lag_days": 3}):
        cvar_manager._select_contract("AAPL", "call", 190.0, 190.0, day)
    assert set(recorded_as_of) == {day - timedelta(days=3)}


def test_tail_hedge_put_selection_reads_the_chain_as_of_the_lag(
    recorded_as_of: list[date],
) -> None:
    """`TailHedgeManager._select_put` queries the chain as of `day - lag`."""
    as_of = pd.Timestamp("2024-06-20")
    dates = pd.bdate_range(end=as_of, periods=3)
    market_data = pd.concat(
        {"SPY": pd.DataFrame({"close": np.full(3, 500.0), "log_return": np.zeros(3)}, index=dates)},
        axis=1,
    )
    manager = TailHedgeManager(
        managed_analysts=[],
        weighting_strategy=EqualWeight(),
        portfolio_constructor=PortfolioConstructor(),
        assets=("SPY",),
    )

    manager._select_put(market_data, as_of, 0.01)

    assert recorded_as_of
    assert set(recorded_as_of) == {date(2024, 6, 19)}


def test_engine_marks_are_not_lagged(recorded_as_of: list[date]) -> None:
    """`load_option_mark` queries the chain as of the mark date itself."""
    day = date(2024, 6, 20)
    contract = OptionContract(
        underlying="AAPL", expiry=date(2024, 7, 19), strike=190.0, option_type="call"
    )

    option_marks.load_option_mark(contract, day, spot=190.0)

    assert recorded_as_of
    assert set(recorded_as_of) == {day}
