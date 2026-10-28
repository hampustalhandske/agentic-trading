# test_tail_hedge_resume.py

"""A backtest whose Commander runs only the tail-hedge pod, interrupted after any day and resumed
from its checkpoint, reproduces the uninterrupted run: the premium cap's trailing commitments, the
held put and its scale travel in `CommanderRiskState.manager_states`."""

from collections.abc import Callable, Iterator
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.commander import Commander
from agentic_trading.agents.managers.tail_hedge_manager import TailHedgeManager
from agentic_trading.agents.portfolio_construction import PortfolioConstructor
from agentic_trading.agents.schemas import AgentSignal, OptionContract
from agentic_trading.backtest.engine import BacktestEngine
from agentic_trading.backtest.portfolio import Portfolio
from agentic_trading.config import settings, temporary_overrides
from agentic_trading.data.loaders import option_marks
from agentic_trading.data.loaders.option_marks import OptionMark
from agentic_trading.scoring.track_record import TrackRecord
from agentic_trading.scoring.weighting import EqualWeight

# Monday 2024-01-08 to Monday 2024-01-15: six sessions.
_INDEX = pd.bdate_range("2024-01-08", periods=6)
_DAYS = [day.date() for day in _INDEX]
SPOT = 500.0
CONFIDENCE = 0.5
# Every put the stub chain offers expires this many calendar days out, inside
# `settings.options.roll_min_dte`: the first put is due to roll the next day.
_EXPIRY_DAYS = 3


class _InterruptError(Exception):
    pass


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


def _mid(contract: OptionContract, as_of_date: date) -> float:
    """Time value decaying with days to expiry; the put stays out of the money at `SPOT`."""
    return 6.0 + 0.5 * (contract.expiry - as_of_date).days


def _mark(contract: OptionContract, as_of_date: date, spot: float) -> OptionMark:
    """A tradeable quote 0.10 either side of `_mid` before expiry; intrinsic on or after it."""
    if as_of_date >= contract.expiry:
        intrinsic = max(contract.strike - spot, 0.0)
        return OptionMark(
            contract=contract,
            as_of_date=as_of_date,
            snapshot_date=None,
            mid=intrinsic,
            bid=intrinsic,
            ask=intrinsic,
            implied_volatility=0.0,
            delta=0.0,
            gamma=0.0,
            vega=0.0,
            theta=0.0,
            is_model_mark=False,
            is_expired=True,
            tradeable=False,
        )
    mid = _mid(contract, as_of_date)
    return OptionMark(
        contract=contract,
        as_of_date=as_of_date,
        snapshot_date=as_of_date,
        mid=mid,
        bid=mid - 0.1,
        ask=mid + 0.1,
        implied_volatility=0.2,
        delta=-0.3,
        gamma=0.01,
        vega=0.5,
        theta=-0.1,
        is_model_mark=False,
        is_expired=False,
        tradeable=True,
    )


def _put_weight() -> float:
    """The premium weight of one uncapped put on the day it opens."""
    budget = settings.commander.tail_hedge_budget_fraction * CONFIDENCE
    opening_mid = 6.0 + 0.5 * _EXPIRY_DAYS
    return min(budget, settings.options.tail_hedge_notional_fraction * opening_mid / SPOT)


@pytest.fixture(autouse=True)
def _chain(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub SPY chain and marks for both the manager and the engine; no split adjustment."""

    def select_expiry(underlying, as_of_date, target_tenor_days, after=None) -> date:
        return (after or as_of_date) + timedelta(days=_EXPIRY_DAYS)

    monkeypatch.setattr(option_marks, "select_expiry", select_expiry)
    monkeypatch.setattr(option_marks, "nearest_strike", lambda _u, _e, _t, tgt, _d: round(tgt))
    monkeypatch.setattr(option_marks, "load_option_mark", _mark)
    monkeypatch.setattr("agentic_trading.backtest.engine.load_option_mark", _mark)
    monkeypatch.setattr(
        "agentic_trading.backtest.engine.lakehouse_loader.split_adjustment_factor",
        lambda *_args: 1.0,
    )


@pytest.fixture
def cap() -> Iterator[float]:
    """A premium cap with room for 1.5 puts: the first roll is cut and every later open waits."""
    value = 1.5 * _put_weight()
    with temporary_overrides({"options.tail_hedge_annual_premium_cap": value}):
        yield value


def _frame() -> pd.DataFrame:
    flat = np.full(len(_INDEX), SPOT)
    return pd.DataFrame(
        {
            ("SPY", "open"): flat,
            ("SPY", "close"): flat,
            ("SPY", "log_return"): np.zeros(len(_INDEX)),
        },
        index=_INDEX,
    )


def _commander() -> tuple[Commander, TailHedgeManager]:
    manager = TailHedgeManager(
        managed_analysts=[_ShortAnalyst()],
        weighting_strategy=EqualWeight(),
        portfolio_constructor=PortfolioConstructor(),
        assets=("SPY",),
    )
    return Commander(managers=[manager], weighting_strategy=EqualWeight()), manager


def _engine(
    commander: Commander, checkpoint_path: Path | None = None, resume: bool = False
) -> BacktestEngine:
    return BacktestEngine(
        commander=commander,
        portfolio=Portfolio(cash=1_000_000.0, positions={}, commission_rate=0.0, slippage_bps=0.0),
        track_record=TrackRecord(),
        data_loader=lambda _assets, _start, _end: _frame(),
        start_date=_DAYS[0],
        end_date=_DAYS[-1],
        assets=["SPY"],
        checkpoint_path=checkpoint_path,
        resume=resume,
    )


def _stop_after(day: int) -> Callable[..., None]:
    def stop(day_index: int, *_args: object) -> None:
        if day_index == day:
            raise _InterruptError

    return stop


def test_resume_after_any_day_reproduces_the_uninterrupted_tail_hedge_run(
    cap: float, tmp_path: Path
) -> None:
    """Interrupted after day k and resumed, every k gives the same pod costs, NAV and decisions."""
    full = _put_weight()
    assert _EXPIRY_DAYS < settings.options.roll_min_dte
    expected = _engine(_commander()[0]).run()

    opened: dict[str, tuple[date, float]] = {}
    for day in _DAYS:
        for decision in expected.decisions_by_day[day]:
            if decision.instrument is not None and decision.target_weight > 0.0:
                opened.setdefault(decision.asset, (day, decision.target_weight))
    assert [day for day, _ in opened.values()] == _DAYS[:2]
    assert [weight for _, weight in opened.values()] == pytest.approx([full, cap - full])
    assert cap - full < full
    assert not any(
        decision.instrument is not None and decision.target_weight > 0.0
        for day in _DAYS[-2:]
        for decision in expected.decisions_by_day[day]
    )
    bought = {
        trade.asset
        for trade in expected.trade_log
        if trade.instrument is not None and trade.action == "buy"
    }
    assert bought == set(opened)
    assert expected.pod_costs["tail_hedge_manager"].option_premium_paid > 0.0

    for k in range(len(_DAYS)):
        checkpoint_path = tmp_path / f"checkpoint_{k}.db"
        interrupted, manager = _commander()
        with pytest.raises(_InterruptError):
            _engine(interrupted, checkpoint_path).run(on_day_complete=_stop_after(k))
        assert manager.premium_committed(_DAYS[k]) >= full - 1e-12

        result = _engine(_commander()[0], checkpoint_path, resume=True).run()

        assert result.pod_costs == expected.pod_costs, f"interrupted after day {k}"
        assert result.portfolio_value_history == expected.portfolio_value_history, k
        assert result.decisions_by_day == expected.decisions_by_day, k
        assert result.trade_log == expected.trade_log, k
