"""Unit tests for TailHedgeManager's protective-put expression: the
sleeve buys a near-tenor OTM SPY put sized by the hedge budget times
the analyst's confidence, rolls it inside the roll window, and falls
back to the short-SPY equity weight when no usable chain exists."""

from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
import pytest

from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.managers.tail_hedge_manager import TailHedgeManager
from agentic_trading.agents.portfolio_construction import PortfolioConstructor
from agentic_trading.agents.schemas import AgentSignal, OptionContract, PortfolioDecision
from agentic_trading.config import settings
from agentic_trading.data.loaders import option_marks
from agentic_trading.data.loaders.option_marks import OptionMark
from agentic_trading.scoring.scoreboard import PerformanceScoreboard
from agentic_trading.scoring.track_record import TrackRecord
from agentic_trading.scoring.weighting import EqualWeight

AS_OF = pd.Timestamp("2024-06-20")
SPOT = 500.0
CONFIDENCE = 0.5
MARK_MID = 8.0


def _expected_weight(budget: float) -> float:
    """The sleeve spends min(budget, notional fraction x mid / spot)."""
    return min(budget, settings.options.tail_hedge_notional_fraction * MARK_MID / SPOT)


def _market_data(end: pd.Timestamp = AS_OF, periods: int = 3) -> pd.DataFrame:
    dates = pd.bdate_range(end=end, periods=periods)
    return pd.concat(
        {
            "SPY": pd.DataFrame(
                {"close": np.full(periods, SPOT), "log_return": np.zeros(periods)}, index=dates
            )
        },
        axis=1,
    )


class _ShortAnalyst(AnalystAgent):
    agent_id: str = "vix_tail_hedge_analyst"

    def generate_signal(
        self, asset: str, market_data: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> AgentSignal:
        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction="short",
            confidence=CONFIDENCE,
            rationale="VIX elevated",
        )


def _manager() -> TailHedgeManager:
    return TailHedgeManager(
        managed_analysts=[_ShortAnalyst()],
        weighting_strategy=EqualWeight(),
        portfolio_constructor=PortfolioConstructor(),
        assets=("SPY",),
    )


def _scoreboard() -> PerformanceScoreboard:
    return PerformanceScoreboard(TrackRecord())


def _mark(contract: OptionContract, as_of_date: date, tradeable: bool = True) -> OptionMark:
    return OptionMark(
        contract=contract,
        as_of_date=as_of_date,
        snapshot_date=as_of_date,
        mid=MARK_MID,
        bid=7.8 if tradeable else 0.0,
        ask=8.2,
        implied_volatility=0.2,
        delta=-0.3,
        gamma=0.01,
        vega=0.5,
        theta=-0.1,
        is_model_mark=False,
        is_expired=False,
        tradeable=tradeable,
    )


@pytest.fixture
def chain(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """A stub chain: one expiry `target_tenor_days` out from the query
    date, the strike the manager asks for, and a tradeable mark. Records
    the strike target the manager requested."""
    requested: dict[str, object] = {}

    def select_expiry(
        underlying: str, as_of_date: date, target_tenor_days: int, after: date | None = None
    ) -> date:
        return as_of_date + timedelta(days=target_tenor_days)

    def nearest_strike(underlying, expiry, option_type, target_strike, as_of_date) -> float:
        requested["option_type"] = option_type
        requested["target_strike"] = target_strike
        return round(target_strike)

    def load_option_mark(contract: OptionContract, as_of_date: date, spot: float) -> OptionMark:
        return _mark(contract, as_of_date)

    monkeypatch.setattr(option_marks, "select_expiry", select_expiry)
    monkeypatch.setattr(option_marks, "nearest_strike", nearest_strike)
    monkeypatch.setattr(option_marks, "load_option_mark", load_option_mark)
    return requested


def _by_asset(decisions: list[PortfolioDecision]) -> dict[str, PortfolioDecision]:
    return {decision.asset: decision for decision in decisions}


def test_emits_a_long_put_at_the_budget_weight_and_zero_spy_equity(chain) -> None:
    decisions = _manager().construct_portfolio(["SPY"], _market_data(), AS_OF, _scoreboard())

    by_asset = _by_asset(decisions)
    puts = [decision for decision in decisions if decision.instrument is not None]
    assert len(puts) == 1
    put = puts[0]
    assert put.instrument is not None
    assert put.instrument.option_type == "put"
    assert put.instrument.underlying == "SPY"
    assert put.instrument.expiry == AS_OF.date() + timedelta(
        days=settings.options.tail_hedge_tenor_days
    )
    assert put.target_weight == pytest.approx(
        _expected_weight(settings.commander.tail_hedge_budget_fraction * CONFIDENCE)
    )
    assert put.contributing_signals and put.contributing_signals[0].direction == "short"
    assert chain["option_type"] == "put"
    assert chain["target_strike"] == pytest.approx(
        SPOT * settings.options.tail_hedge_strike_moneyness
    )
    assert by_asset["SPY"].target_weight == 0.0
    assert by_asset["SPY"].contributing_signals


def test_floors_the_sleeve_at_the_base_confidence(chain, monkeypatch) -> None:
    monkeypatch.setattr(
        _ShortAnalyst,
        "generate_signal",
        lambda self, asset, market_data, as_of_date: AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction="short",
            confidence=0.0,
            rationale="calm",
        ),
    )

    decisions = _manager().construct_portfolio(["SPY"], _market_data(), AS_OF, _scoreboard())

    put = next(decision for decision in decisions if decision.instrument is not None)
    assert put.target_weight == pytest.approx(
        _expected_weight(
            settings.commander.tail_hedge_budget_fraction
            * settings.analysts.tail_hedge_base_confidence
        )
    )


def test_holds_the_contract_until_the_roll_window_then_rolls(chain) -> None:
    manager = _manager()
    first = _by_asset(manager.construct_portfolio(["SPY"], _market_data(), AS_OF, _scoreboard()))
    held_id = next(asset for asset, decision in first.items() if decision.instrument is not None)
    held_expiry = OptionContract.parse(held_id).expiry

    # Well before the roll window: the same contract is re-emitted.
    hold_day = pd.Timestamp(held_expiry - timedelta(days=settings.options.roll_min_dte + 5))
    held_again = manager.construct_portfolio(
        ["SPY"], _market_data(end=hold_day), hold_day, _scoreboard()
    )
    option_rows = [decision for decision in held_again if decision.instrument is not None]
    assert [decision.asset for decision in option_rows] == [held_id]
    assert option_rows[0].target_weight > 0.0

    # Inside the roll window: the old contract is closed, a new one bought.
    roll_day = pd.Timestamp(held_expiry - timedelta(days=settings.options.roll_min_dte - 1))
    rolled = _by_asset(
        manager.construct_portfolio(["SPY"], _market_data(end=roll_day), roll_day, _scoreboard())
    )
    assert rolled[held_id].target_weight == 0.0
    assert rolled[held_id].instrument is not None
    new_ids = [
        asset
        for asset, decision in rolled.items()
        if decision.instrument is not None and asset != held_id
    ]
    assert len(new_ids) == 1
    assert OptionContract.parse(new_ids[0]).expiry == roll_day.date() + timedelta(
        days=settings.options.tail_hedge_tenor_days
    )
    assert rolled[new_ids[0]].target_weight == pytest.approx(
        _expected_weight(settings.commander.tail_hedge_budget_fraction * CONFIDENCE)
    )


def test_falls_back_to_the_short_spy_equity_weight_without_a_chain(monkeypatch) -> None:
    monkeypatch.setattr(option_marks, "select_expiry", lambda *_, **__: None)

    decisions = _manager().construct_portfolio(["SPY"], _market_data(), AS_OF, _scoreboard())

    assert all(decision.instrument is None for decision in decisions)
    assert _by_asset(decisions)["SPY"].target_weight < 0.0


def test_falls_back_when_the_selected_put_is_untradeable(chain, monkeypatch) -> None:
    monkeypatch.setattr(
        option_marks,
        "load_option_mark",
        lambda contract, as_of_date, spot: _mark(contract, as_of_date, tradeable=False),
    )

    decisions = _manager().construct_portfolio(["SPY"], _market_data(), AS_OF, _scoreboard())

    assert all(decision.instrument is None for decision in decisions)
    assert _by_asset(decisions)["SPY"].target_weight < 0.0


def _lake_chain_expiring_on(fake_lake: Any, underlying: str, day: date, strike: float) -> None:
    """Seeds `fake_lake` with one quoted snapshot dated `day - 1` whose only
    expiry is `day`, and no splits."""
    fake_lake.add(
        "dolthub_option_chain",
        pd.DataFrame(
            [
                {
                    "underlying": underlying,
                    "date": (day - timedelta(days=1)).isoformat(),
                    "expiration": day.isoformat(),
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
                for option_type in ("call", "put")
            ]
        ),
        date_columns=("date", "expiration"),
    )
    fake_lake.add_empty(
        "yahoo_split_close_basis",
        {"ticker": "VARCHAR", "ex_date": "DATE", "ratio": "DOUBLE", "in_close_basis": "BOOLEAN"},
    )


def test_lagged_chain_listing_only_todays_expiry_selects_no_put(
    fake_lake: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The chain read at `day - 1` lists only `day`'s expiry; no put is selected even
    though any contract chosen would mark as tradeable."""
    day = AS_OF.date()
    _lake_chain_expiring_on(
        fake_lake, "SPY", day, SPOT * settings.options.tail_hedge_strike_moneyness
    )
    monkeypatch.setattr(
        option_marks,
        "load_option_mark",
        lambda contract, as_of_date, spot: _mark(contract, as_of_date),
    )
    option_marks.clear_cache()
    try:
        assert option_marks.list_expiries("SPY", day - timedelta(days=1)) == [day]
        assert _manager()._select_put(_market_data(), AS_OF, 0.01) is None
    finally:
        option_marks.clear_cache()
