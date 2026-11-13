# test_tail_hedge_premium_cap.py

"""`settings.options.tail_hedge_annual_premium_cap` bounds the premium the tail-hedge pod commits
to new puts over any trailing 365 days; `TailHedgeManager.premium_committed` reports it."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.managers.tail_hedge_manager import TailHedgeManager
from agentic_trading.agents.portfolio_construction import PortfolioConstructor
from agentic_trading.agents.schemas import AgentSignal, OptionContract, PortfolioDecision
from agentic_trading.config import settings, temporary_overrides
from agentic_trading.data.loaders import option_marks
from agentic_trading.data.loaders.option_marks import OptionMark
from agentic_trading.scoring.scoreboard import PerformanceScoreboard
from agentic_trading.scoring.track_record import TrackRecord
from agentic_trading.scoring.weighting import EqualWeight

START = pd.Timestamp("2024-06-20")
SPOT = 500.0
CONFIDENCE = 0.5
MARK_MID = 8.0
DAYS = 420


def _put_weight() -> float:
    """The premium weight of one uncapped put."""
    budget = settings.commander.tail_hedge_budget_fraction * CONFIDENCE
    return min(budget, settings.options.tail_hedge_notional_fraction * MARK_MID / SPOT)


def _market_data(end: pd.Timestamp, periods: int = 3) -> pd.DataFrame:
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


def _mark(contract: OptionContract, as_of_date: date) -> OptionMark:
    return OptionMark(
        contract=contract,
        as_of_date=as_of_date,
        snapshot_date=as_of_date,
        mid=MARK_MID,
        bid=7.8,
        ask=8.2,
        implied_volatility=0.2,
        delta=-0.3,
        gamma=0.01,
        vega=0.5,
        theta=-0.1,
        is_model_mark=False,
        is_expired=False,
        tradeable=True,
    )


@pytest.fixture(autouse=True)
def _chain(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stub chain with one expiry `target_tenor_days` out and a tradeable mark."""

    def select_expiry(underlying, as_of_date, target_tenor_days, after=None) -> date:
        return as_of_date + timedelta(days=target_tenor_days)

    def load_option_mark(contract, as_of_date, spot) -> OptionMark:
        return _mark(contract, as_of_date)

    monkeypatch.setattr(option_marks, "select_expiry", select_expiry)
    monkeypatch.setattr(option_marks, "nearest_strike", lambda _u, _e, _t, tgt, _d: round(tgt))
    monkeypatch.setattr(option_marks, "load_option_mark", load_option_mark)


def _run(
    manager: TailHedgeManager,
) -> tuple[list[tuple[date, list[PortfolioDecision]]], dict[str, float], list[float]]:
    """Daily decisions over `DAYS` days, each contract's first positive weight, and the
    committed premium after every day."""
    scoreboard = PerformanceScoreboard(TrackRecord())
    days: list[tuple[date, list[PortfolioDecision]]] = []
    opened: dict[str, float] = {}
    committed: list[float] = []
    for offset in range(DAYS):
        day = START + pd.Timedelta(days=offset)
        decisions = manager.construct_portfolio(["SPY"], _market_data(day), day, scoreboard)
        days.append((day.date(), decisions))
        for decision in decisions:
            if decision.instrument is not None and decision.target_weight > 0.0:
                opened.setdefault(decision.asset, decision.target_weight)
        committed.append(manager.premium_committed(day.date()))
    return days, opened, committed


def test_default_cap_is_two_percent_of_nav_within_bounds() -> None:
    """The setting defaults to 0.02 and rejects values outside [0, 1]."""
    assert settings.options.tail_hedge_annual_premium_cap == 0.02
    for bad in (-0.01, 1.01):
        with pytest.raises(ValueError):
            with temporary_overrides({"options.tail_hedge_annual_premium_cap": bad}):
                pass


def test_committed_premium_never_exceeds_the_cap_over_repeated_rolls() -> None:
    """With room for 1.5 puts a year, the second put is cut to the headroom and later rolls wait."""
    full = _put_weight()
    cap = 1.5 * full
    with temporary_overrides({"options.tail_hedge_annual_premium_cap": cap}):
        _, opened, committed = _run(_manager())

    assert max(committed) <= cap + 1e-12
    assert max(committed) == pytest.approx(cap)
    weights = list(opened.values())
    assert len(weights) >= 3
    assert weights[0] == pytest.approx(full)
    assert weights[1] == pytest.approx(cap - full)
    assert all(weight > 0.0 for weight in weights)


def test_zero_cap_opens_no_put() -> None:
    """A cap of 0 leaves no option row with a positive weight and nothing committed."""
    with temporary_overrides({"options.tail_hedge_annual_premium_cap": 0.0}):
        days, opened, committed = _run(_manager())

    assert opened == {}
    assert set(committed) == {0.0}
    assert all(
        decision.target_weight <= 0.0
        for _, decisions in days
        for decision in decisions
        if decision.instrument is not None
    )


def test_cap_above_the_yearly_spend_leaves_every_put_at_full_weight() -> None:
    """With a cap of 1.0 every opened put carries the uncapped weight and rolls recur."""
    with temporary_overrides({"options.tail_hedge_annual_premium_cap": 1.0}):
        _, opened, committed = _run(_manager())

    assert len(opened) >= 3
    assert list(opened.values()) == pytest.approx([_put_weight()] * len(opened))
    assert max(committed) > _put_weight()
    assert max(committed) <= 1.0
