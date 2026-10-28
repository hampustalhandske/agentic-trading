# test_manager_state.py

"""Checkpointable manager state: the `ManagerState` / `TailHedgeState` schemas,
`ManagerAgent.manager_state` / `restore_manager_state`, the `TailHedgeManager` override and
`CommanderRiskState.manager_states` filled and restored by `Commander`."""

from collections.abc import Iterator
from datetime import date, timedelta
from typing import Literal

import numpy as np
import pandas as pd
import pytest

from agentic_trading.agents.base import AnalystAgent, ManagerAgent
from agentic_trading.agents.commander import Commander, CommanderRiskState
from agentic_trading.agents.managers.tail_hedge_manager import TailHedgeManager
from agentic_trading.agents.portfolio_construction import PortfolioConstructor
from agentic_trading.agents.schemas import (
    AgentSignal,
    ManagerState,
    OptionContract,
    PortfolioDecision,
    TailHedgeState,
)
from agentic_trading.config import settings, temporary_overrides
from agentic_trading.data.loaders import option_marks
from agentic_trading.data.loaders.option_marks import OptionMark
from agentic_trading.scoring.scoreboard import PerformanceScoreboard
from agentic_trading.scoring.track_record import TrackRecord
from agentic_trading.scoring.weighting import EqualWeight

_INDEX = pd.bdate_range("2024-01-02", periods=6)
_ASSETS = ["AAPL", "SPY"]
SPOT = 500.0
CONFIDENCE = 0.5
MARK_MID = 8.0
# Every put the stub chain offers expires this many calendar days out, inside
# `settings.options.roll_min_dte`, so a held put is due to roll every day.
_EXPIRY_DAYS = 3


class _Analyst(AnalystAgent):
    """Emits `direction` at `CONFIDENCE` for every asset and day."""

    direction: Literal["long", "short", "flat"] = "short"

    def generate_signal(
        self, asset: str, market_data: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> AgentSignal:
        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction=self.direction,
            confidence=CONFIDENCE,
            rationale="stub",
        )


class _RecordingConstructor(PortfolioConstructor):
    """Sizes every view at its previous weight plus 0.1; records the previous weights it gets."""

    def __init__(self) -> None:
        super().__init__()
        self.received: list[dict[str, float]] = []

    def construct(self, views, returns_history, as_of_date, previous_weights=None):
        previous = dict(previous_weights or {})
        self.received.append(previous)
        return {asset: previous.get(asset, 0.0) + 0.1 for asset in views}


class _Manager(ManagerAgent):
    def scale_factor(self, signal, market_data, as_of_date) -> float:
        return 1.0


def _base_manager(
    agent_id: str = "pod_a", constructor: _RecordingConstructor | None = None
) -> _Manager:
    return _Manager(
        agent_id=agent_id,
        managed_analysts=[_Analyst(agent_id=f"{agent_id}_analyst", direction="long")],
        weighting_strategy=EqualWeight(),
        portfolio_constructor=constructor or _RecordingConstructor(),
        assets=("AAPL",),
    )


def _tail_hedge_manager() -> TailHedgeManager:
    return TailHedgeManager(
        managed_analysts=[_Analyst(agent_id="vix_tail_hedge_analyst")],
        weighting_strategy=EqualWeight(),
        portfolio_constructor=PortfolioConstructor(),
        assets=("SPY",),
    )


def _frame() -> pd.DataFrame:
    """AAPL on a seeded random walk, SPY flat at `SPOT`."""
    rng = np.random.default_rng(7)
    aapl = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, len(_INDEX)))), _INDEX)
    spy = pd.Series(SPOT, _INDEX)
    return pd.DataFrame(
        {
            ("AAPL", "close"): aapl,
            ("AAPL", "log_return"): np.log(aapl / aapl.shift(1)),
            ("SPY", "close"): spy,
            ("SPY", "log_return"): np.zeros(len(_INDEX)),
        }
    )


def _scoreboard() -> PerformanceScoreboard:
    return PerformanceScoreboard(TrackRecord())


def _put_weight() -> float:
    """The premium weight of one uncapped put."""
    budget = settings.commander.tail_hedge_budget_fraction * CONFIDENCE
    return min(budget, settings.options.tail_hedge_notional_fraction * MARK_MID / SPOT)


def _mark(contract: OptionContract, as_of_date: date, spot: float) -> OptionMark:
    return OptionMark(
        contract=contract,
        as_of_date=as_of_date,
        snapshot_date=as_of_date,
        mid=MARK_MID,
        bid=MARK_MID - 0.2,
        ask=MARK_MID + 0.2,
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
    """A stub SPY chain: one expiry `_EXPIRY_DAYS` after the day, any strike, a tradeable mark."""

    def select_expiry(underlying, as_of_date, target_tenor_days, after=None) -> date:
        return (after or as_of_date) + timedelta(days=_EXPIRY_DAYS)

    monkeypatch.setattr(option_marks, "select_expiry", select_expiry)
    monkeypatch.setattr(option_marks, "nearest_strike", lambda _u, _e, _t, tgt, _d: round(tgt))
    monkeypatch.setattr(option_marks, "load_option_mark", _mark)


@pytest.fixture(autouse=True)
def _limits_and_cap() -> Iterator[None]:
    """Firm limits lifted so pod weights reach the book; the premium cap fits 1.5 puts."""
    with temporary_overrides(
        {
            "firm.max_single_name_weight": 100.0,
            "firm.max_gross_exposure": 1000.0,
            "firm.max_daily_var": 1000.0,
            "options.tail_hedge_annual_premium_cap": 1.5 * _put_weight(),
        }
    ):
        yield


def _run(manager: ManagerAgent, days: pd.DatetimeIndex) -> list[list[PortfolioDecision]]:
    data = _frame()
    return [
        manager.construct_portfolio(_ASSETS, data.loc[:day], day, _scoreboard()) for day in days
    ]


def test_state_schemas_default_and_round_trip_through_json() -> None:
    """Defaults are empty; the schemas survive JSON; a payload without manager_states validates."""
    assert TailHedgeState() == TailHedgeState(held_contract_id=None, held_scale=1.0, commitments=[])
    assert ManagerState() == ManagerState(previous_weights={}, tail_hedge=None)
    first = ManagerState()
    first.previous_weights["AAPL"] = 1.0
    assert ManagerState().previous_weights == {}

    put = OptionContract(
        underlying="SPY", expiry=date(2024, 3, 15), strike=475.0, option_type="put"
    )
    tail = TailHedgeState(
        held_contract_id=put.contract_id,
        held_scale=0.5,
        commitments=[(date(2024, 1, 2), 0.01), (date(2024, 1, 3), 0.005)],
    )
    state = ManagerState(previous_weights={"SPY": -0.2, "AAPL": 0.1}, tail_hedge=tail)
    restored = ManagerState.model_validate(state.model_dump(mode="json"))
    assert restored == state
    assert restored.tail_hedge is not None
    assert restored.tail_hedge.commitments[0] == (date(2024, 1, 2), 0.01)
    assert type(restored.tail_hedge.commitments[0][0]) is date
    assert TailHedgeState.model_validate(tail.model_dump(mode="json")) == tail
    assert ManagerState.model_validate(ManagerState().model_dump(mode="json")) == ManagerState()

    risk = CommanderRiskState(
        tripped_pods=["pod_a"],
        manager_states={"tail_hedge_manager": state, "pod_a": ManagerState()},
    )
    assert CommanderRiskState.model_validate(risk.model_dump(mode="json")) == risk
    assert CommanderRiskState().manager_states == {}
    legacy = {
        "tripped_pods": ["pod_a"],
        "pod_return_history": {"pod_a": [["2024-01-03", 0.01]]},
        "last_pod_weights": {"pod_a": {"AAPL": 0.3}},
    }
    assert CommanderRiskState.model_validate(legacy).manager_states == {}


def test_base_manager_state_is_a_copy_that_restore_replaces() -> None:
    """`manager_state` copies yesterday's sized weights; restore replaces them and feeds sizing."""
    manager = _base_manager()
    assert manager.manager_state() == ManagerState()
    decisions = _run(manager, _INDEX[:3])[-1]

    state = manager.manager_state()
    assert isinstance(state, ManagerState)
    assert state.tail_hedge is None
    assert decisions[0].asset == "AAPL"
    assert state.previous_weights == {"AAPL": decisions[0].target_weight}
    assert state.previous_weights["AAPL"] == pytest.approx(0.3)

    reference = manager.manager_state().model_dump(mode="json")
    state.previous_weights["AAPL"] = 9.0
    state.previous_weights["MSFT"] = 1.0
    assert manager.manager_state().model_dump(mode="json") == reference

    snapshot = manager.manager_state()
    constructor = _RecordingConstructor()
    fresh = _base_manager(constructor=constructor)
    fresh.restore_manager_state(snapshot)
    assert fresh.manager_state() == snapshot
    snapshot.previous_weights.clear()
    assert fresh.manager_state().model_dump(mode="json") == reference

    _run(fresh, _INDEX[3:4])
    assert constructor.received == [{"AAPL": pytest.approx(0.3)}]
    assert fresh.manager_state().previous_weights["AAPL"] == pytest.approx(0.4)

    fresh.restore_manager_state(ManagerState(previous_weights={"MSFT": 0.4}))
    assert fresh.manager_state() == ManagerState(previous_weights={"MSFT": 0.4})
    fresh.restore_manager_state(ManagerState())
    assert fresh.manager_state() == ManagerState()
    _run(fresh, _INDEX[4:5])
    assert constructor.received[-1] == {}


def test_tail_hedge_manager_state_round_trips_the_held_put_and_commitments() -> None:
    """A restored tail-hedge manager has the same commitments, held put and scale; decides alike."""
    full = _put_weight()
    cap = settings.options.tail_hedge_annual_premium_cap
    assert _EXPIRY_DAYS < settings.options.roll_min_dte
    original = _tail_hedge_manager()
    assert original.manager_state().tail_hedge == TailHedgeState()

    decisions = _run(original, _INDEX[:3])
    state = original.manager_state()
    tail = state.tail_hedge
    assert tail is not None
    held_rows = [row for row in decisions[-1] if row.instrument is not None]
    assert [row.asset for row in held_rows] == [tail.held_contract_id]
    assert tail.held_scale == pytest.approx((cap - full) / full)
    assert [day for day, _ in tail.commitments] == [day.date() for day in _INDEX[:2]]
    assert [weight for _, weight in tail.commitments] == pytest.approx([full, cap - full])
    assert "SPY" in state.previous_weights
    assert original.premium_committed(_INDEX[2].date()) == pytest.approx(cap)

    restored = _tail_hedge_manager()
    restored.restore_manager_state(ManagerState.model_validate(state.model_dump(mode="json")))
    assert restored.manager_state() == state
    first = _INDEX[0].date()
    probes = [
        first - timedelta(days=1),
        *(day.date() for day in _INDEX),
        first + timedelta(days=364),
        first + timedelta(days=365),
        _INDEX[1].date() + timedelta(days=365),
    ]
    for day in probes:
        assert restored.premium_committed(day) == original.premium_committed(day)

    reference = original.manager_state().model_dump(mode="json")
    tail.commitments.clear()
    tail.held_contract_id = None
    tail.held_scale = 1.0
    assert original.manager_state().model_dump(mode="json") == reference
    assert original.premium_committed(_INDEX[2].date()) == pytest.approx(cap)

    snapshot = original.manager_state()
    aliased = _tail_hedge_manager()
    aliased.restore_manager_state(snapshot)
    assert snapshot.tail_hedge is not None
    snapshot.tail_hedge.commitments.append((_INDEX[2].date(), 1.0))
    snapshot.tail_hedge.held_contract_id = None
    assert aliased.manager_state().model_dump(mode="json") == reference

    assert _run(restored, _INDEX[3:]) == _run(original, _INDEX[3:])

    original.restore_manager_state(ManagerState(tail_hedge=TailHedgeState()))
    assert original.manager_state() == ManagerState(tail_hedge=TailHedgeState())
    assert original.premium_committed(_INDEX[2].date()) == 0.0


def test_commander_risk_state_carries_and_restores_every_manager_state() -> None:
    """`risk_state` keys every manager's state by agent_id; a restored Commander decides alike."""

    def build() -> Commander:
        return Commander(
            managers=[_base_manager("pod_a"), _base_manager("pod_b"), _tail_hedge_manager()],
            weighting_strategy=EqualWeight(),
        )

    data = _frame()
    commander = build()
    for day in _INDEX[:3]:
        commander.decide(data.loc[:day], day, _scoreboard())

    state = commander.risk_state()
    assert set(state.manager_states) == {"pod_a", "pod_b", "tail_hedge_manager"}
    for manager in commander.managers:
        assert state.manager_states[manager.agent_id] == manager.manager_state()
    hedge_state = state.manager_states["tail_hedge_manager"].tail_hedge
    assert hedge_state is not None and len(hedge_state.commitments) == 2
    assert state.manager_states["pod_a"].tail_hedge is None

    fresh = build()
    fresh.restore_risk_state(CommanderRiskState.model_validate(state.model_dump(mode="json")))
    assert fresh.risk_state() == state
    for manager, restored in zip(commander.managers, fresh.managers, strict=True):
        assert restored.manager_state() == manager.manager_state()
    original_hedge, restored_hedge = commander.managers[2], fresh.managers[2]
    assert isinstance(original_hedge, TailHedgeManager)
    assert isinstance(restored_hedge, TailHedgeManager)
    assert restored_hedge.premium_committed(_INDEX[2].date()) == pytest.approx(
        original_hedge.premium_committed(_INDEX[2].date())
    )

    for day in _INDEX[3:]:
        expected = commander.decide(data.loc[:day], day, _scoreboard())
        assert fresh.decide(data.loc[:day], day, _scoreboard()) == expected


def test_commander_restore_leaves_absent_managers_and_rejects_unknown_ids() -> None:
    """An absent manager keeps its state, an unknown id raises, an old payload changes none."""
    pod_a, pod_b = _base_manager("pod_a"), _base_manager("pod_b")
    commander = Commander(managers=[pod_a, pod_b], weighting_strategy=EqualWeight())
    data = _frame()
    for day in _INDEX[:2]:
        commander.decide(data.loc[:day], day, _scoreboard())
    before_b = pod_b.manager_state()
    assert before_b.previous_weights

    replacement = ManagerState(previous_weights={"AAPL": -0.4})
    commander.restore_risk_state(CommanderRiskState(manager_states={"pod_a": replacement}))
    assert pod_a.manager_state() == replacement
    assert pod_b.manager_state() == before_b

    with pytest.raises(ValueError):
        commander.restore_risk_state(
            CommanderRiskState(manager_states={"not_a_pod": ManagerState()})
        )

    before = {manager.agent_id: manager.manager_state() for manager in commander.managers}
    legacy = {"tripped_pods": [], "pod_return_history": {}, "last_pod_weights": {}}
    commander.restore_risk_state(CommanderRiskState.model_validate(legacy))
    assert {manager.agent_id: manager.manager_state() for manager in commander.managers} == before


_SIZING_INDEX = pd.bdate_range("2024-01-02", periods=40)
_SIZING_ASSETS = ["AAPL", "MSFT", "XOM"]


class _MomentumAnalyst(AnalystAgent):
    """Long after an up day and short otherwise, at `CONFIDENCE`."""

    def generate_signal(
        self, asset: str, market_data: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> AgentSignal:
        last = float(market_data[(asset, "log_return")].iloc[-1])
        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction="long" if last >= 0.0 else "short",
            confidence=CONFIDENCE,
            rationale="stub",
        )


def _sizing_manager() -> _Manager:
    """A pod sized by a real `PortfolioConstructor` with the Kelly overlay on."""
    return _Manager(
        agent_id="pod_sized",
        managed_analysts=[_MomentumAnalyst(agent_id="momentum_analyst")],
        weighting_strategy=EqualWeight(),
        portfolio_constructor=PortfolioConstructor(
            kelly_fraction_weight=0.25, kelly_lookback_days=10
        ),
        assets=tuple(_SIZING_ASSETS),
    )


def _sizing_frame() -> pd.DataFrame:
    """Three seeded random walks over `_SIZING_INDEX`."""
    rng = np.random.default_rng(11)
    columns: dict[tuple[str, str], pd.Series] = {}
    for asset in _SIZING_ASSETS:
        close = pd.Series(
            100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.015, len(_SIZING_INDEX)))), _SIZING_INDEX
        )
        columns[(asset, "close")] = close
        columns[(asset, "log_return")] = np.log(close / close.shift(1))
    return pd.DataFrame(columns)


def _size(
    manager: ManagerAgent, data: pd.DataFrame, days: pd.DatetimeIndex
) -> list[list[PortfolioDecision]]:
    return [
        manager.construct_portfolio(_SIZING_ASSETS, data.loc[:day], day, _scoreboard())
        for day in days
    ]


def test_constructor_state_travels_so_a_restored_pod_sizes_like_the_original() -> None:
    """A pod restored after day k sizes every later day as the uninterrupted pod; without the
    constructor's return history and rebalance clock it does not."""
    data = _sizing_frame()
    expected = _size(_sizing_manager(), data, _SIZING_INDEX)

    stale_differs = False
    for k in (12, 18, 23):
        prefix = _sizing_manager()
        _size(prefix, data, _SIZING_INDEX[: k + 1])
        state = ManagerState.model_validate(prefix.manager_state().model_dump(mode="json"))
        assert state.constructor.own_return_history
        assert state.constructor.last_rebalance_date is not None

        restored = _sizing_manager()
        restored.restore_manager_state(state)
        assert _size(restored, data, _SIZING_INDEX[k + 1 :]) == expected[k + 1 :], k

        stale = _sizing_manager()
        stale.restore_manager_state(ManagerState(previous_weights=state.previous_weights))
        stale_differs = (
            stale_differs or _size(stale, data, _SIZING_INDEX[k + 1 :]) != expected[k + 1 :]
        )
    assert stale_differs
