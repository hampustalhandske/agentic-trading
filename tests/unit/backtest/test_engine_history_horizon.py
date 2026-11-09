# test_engine_history_horizon.py

"""AT-140: a backtest engine given `history_horizon_days` drops the commander's track-record rows
older than the horizon after each day and keeps only the decisions of the current and pending day
in memory, while its `BacktestResult` (but for `decisions_by_day`), its `source_performance` and
its checkpoint equal an unbounded run's, also when the run is stopped and resumed from its
checkpoint. The firm's manager weights its analysts by a trailing track-record window, so a
record dropped too early, or a read past the horizon, changes or stops the run."""

import gc
import sqlite3
import types
from collections.abc import Callable, Iterator
from contextlib import closing
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import PrivateAttr

from agentic_trading.agents.base import AnalystAgent, ManagerAgent
from agentic_trading.agents.commander import Commander
from agentic_trading.agents.portfolio_construction import PortfolioConstructor
from agentic_trading.agents.schemas import (
    AgentSignal,
    BacktestResult,
    PerformanceRecord,
    PortfolioDecision,
)
from agentic_trading.backtest.engine import (
    HELD_DECISION_DAYS,
    BacktestEngine,
    DecisionWindow,
    longest_track_record_lookback,
)
from agentic_trading.backtest.portfolio import Portfolio
from agentic_trading.config import temporary_overrides
from agentic_trading.scoring.track_record import HistoryHorizonError, TrackRecord
from agentic_trading.scoring.weighting import (
    DrawdownAwareWeight,
    EqualWeight,
    SharpeWeight,
    ShrinkageWeight,
)

# The smallest horizon the measured-IC window (at least 20 calendar days) fits inside.
_HORIZON_DAYS = 20
_WARMUP = 10
_INDEX = pd.bdate_range("2024-01-01", periods=_WARMUP + 60)
_DAYS = [day.date() for day in _INDEX[_WARMUP:]]
_SOURCES = ("momentum_analyst", "contrarian_analyst", "drawdown_manager")
# Containers that lead out of the object graph under test into shared interpreter state.
_OPAQUE = (
    types.ModuleType,
    type,
    types.FunctionType,
    types.BuiltinFunctionType,
    types.CodeType,
    types.FrameType,
)


class _StopError(Exception):
    pass


class _MomentumAnalyst(AnalystAgent):
    """Long after an up close and short after a down one (the reverse when `contrarian`), at a
    confidence that cycles with the number of signals it has given."""

    agent_id: str = "momentum_analyst"
    base_confidence: float = 0.2
    contrarian: bool = False
    _days_seen: int = PrivateAttr(default=0)

    def generate_signal(
        self, asset: str, market_data: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> AgentSignal:
        self._days_seen += 1
        up = float(market_data[(asset, "log_return")].iloc[-1]) >= 0.0
        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction="long" if up != self.contrarian else "short",
            confidence=self.base_confidence + 0.1 * (self._days_seen % 4),
            rationale="stub",
        )


class _ViewConstructor(PortfolioConstructor):
    """Sizes each asset at half its pod view's net signal."""

    def construct(self, views, returns_history, as_of_date, previous_weights=None):
        return {asset: 0.5 * view.net_signal for asset, view in views.items()}


class _PassThroughManager(ManagerAgent):
    def scale_factor(self, signal, market_data, as_of_date) -> float:
        return 1.0


def _commander() -> Commander:
    """One pod weighting two analysts by their drawdown over the horizon's trailing window."""
    manager = _PassThroughManager(
        agent_id="drawdown_manager",
        managed_analysts=[
            _MomentumAnalyst(),
            _MomentumAnalyst(agent_id="contrarian_analyst", base_confidence=0.35, contrarian=True),
        ],
        weighting_strategy=DrawdownAwareWeight(lookback_days=_HORIZON_DAYS),
        portfolio_constructor=_ViewConstructor(),
    )
    return Commander(managers=[manager], weighting_strategy=EqualWeight())


def _frame() -> pd.DataFrame:
    """AAPL on a seeded random walk: open at the previous close, zero return on the first bar."""
    rng = np.random.default_rng(17)
    close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.012, len(_INDEX)))), _INDEX)
    return pd.DataFrame(
        {
            ("AAPL", "open"): close.shift(1).fillna(close.iloc[0]),
            ("AAPL", "close"): close,
            ("AAPL", "log_return"): np.log(close / close.shift(1)).fillna(0.0),
        }
    )


def _engine(checkpoint_path: Path, horizon: int | None, resume: bool = False) -> BacktestEngine:
    return BacktestEngine(
        commander=_commander(),
        portfolio=Portfolio(cash=1_000_000.0, positions={}, commission_rate=5e-4, slippage_bps=2.0),
        track_record=TrackRecord(horizon_days=horizon),
        data_loader=lambda _assets, _start, _end: _frame(),
        start_date=_DAYS[0],
        end_date=_DAYS[-1],
        assets=["AAPL"],
        execution_lag_days=1,
        checkpoint_path=checkpoint_path,
        resume=resume,
        history_horizon_days=horizon,
    )


def _run(checkpoint_path: Path, horizon: int | None) -> tuple[BacktestEngine, BacktestResult]:
    engine = _engine(checkpoint_path, horizon)
    return engine, engine.run()


def _stop_at(day_index: int) -> Callable[..., None]:
    def stop(index: int, *_args: object) -> None:
        if index == day_index:
            raise _StopError

    return stop


def _checkpoint(path: Path) -> dict[str, list[tuple[object, ...]]]:
    """Every row of every checkpoint table in insertion order; carried states by agent id only."""
    with closing(sqlite3.connect(path)) as connection:
        tables = [
            name
            for (name,) in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
            )
        ]
        return {
            table: connection.execute(
                "SELECT agent_id FROM carried_state ORDER BY agent_id"
                if table == "carried_state"
                else f"SELECT * FROM {table} ORDER BY rowid"
            ).fetchall()
            for table in tables
        }


def _reachable(root: object, kind: type) -> int:
    """The number of distinct `kind` instances reachable from `root` outside shared state."""
    seen: set[int] = set()
    stack = [root]
    count = 0
    while stack:
        item = stack.pop()
        if id(item) in seen or isinstance(item, _OPAQUE):
            continue
        seen.add(id(item))
        count += isinstance(item, kind)
        stack.extend(gc.get_referents(item))
    return count


def _without_decisions(result: BacktestResult) -> str:
    return result.model_dump_json(exclude={"decisions_by_day", "history_horizon_days"})


@pytest.fixture(autouse=True)
def _ic_lookback_inside_the_horizon() -> Iterator[None]:
    """Shortens the measured-IC window to the horizon for every run of a test alike."""
    with temporary_overrides({"portfolio.ic_lookback_days": _HORIZON_DAYS}):
        yield


def test_horizon_result_matches_the_unbounded_run(tmp_path: Path) -> None:
    """Result but decisions, source_performance, checkpoint and len equal the unbounded run's."""
    plain_engine, plain = _run(tmp_path / "plain.sqlite", None)
    engine, bounded = _run(tmp_path / "bounded.sqlite", _HORIZON_DAYS)

    assert (_DAYS[-1] - _DAYS[0]).days > 3 * _HORIZON_DAYS
    assert plain.trade_log and set(_SOURCES) <= set(plain.source_performance)
    assert _without_decisions(bounded) == _without_decisions(plain)
    assert bounded.source_performance == plain.source_performance
    assert _DAYS[-1] in bounded.decisions_by_day
    assert set(bounded.decisions_by_day) <= set(_DAYS[-2:])
    for day, decisions in bounded.decisions_by_day.items():
        assert decisions == plain.decisions_by_day[day]
    assert _checkpoint(tmp_path / "bounded.sqlite") == _checkpoint(tmp_path / "plain.sqlite")
    assert len(engine.track_record) == len(plain_engine.track_record)


def test_horizon_bounds_what_the_engine_holds(tmp_path: Path) -> None:
    """Old decisions and windows raise, and held records never exceed those inside the horizon."""
    plain_engine, plain = _run(tmp_path / "plain.sqlite", None)
    records = plain_engine.track_record.entries
    per_day = max(len(decisions) for decisions in plain.decisions_by_day.values())
    engine = _engine(tmp_path / "bounded.sqlite", _HORIZON_DAYS)
    measured: list[int] = []

    def check(index: int, *_args: object) -> None:
        assert engine.decisions_by_day[_DAYS[index]] == plain.decisions_by_day[_DAYS[index]]
        if index >= 3:
            with pytest.raises(HistoryHorizonError):
                _ = engine.decisions_by_day[_DAYS[index - 3]]
        if index % 10 == 9:
            floor = _DAYS[index - 1] - timedelta(days=_HORIZON_DAYS)
            inside = sum(floor <= record.as_of_date <= _DAYS[index] for record in records)
            assert _reachable(engine, PerformanceRecord) <= inside
            assert _reachable(engine, PortfolioDecision) <= 5 * per_day
            measured.append(inside)

    engine.run(on_day_complete=check)

    assert measured and max(measured) * 3 < len(records)
    last = _DAYS[-1]
    for source_id in _SOURCES:
        assert engine.track_record.history_for(
            source_id, last, _HORIZON_DAYS
        ) == plain_engine.track_record.history_for(source_id, last, _HORIZON_DAYS)
        with pytest.raises(HistoryHorizonError):
            engine.track_record.history_for(source_id, last, (last - _DAYS[0]).days)
    with pytest.raises(HistoryHorizonError):
        engine.track_record.entries_for(_SOURCES[0])


@pytest.mark.parametrize("stop", [1, 17, 41])
def test_horizon_resume_matches_the_unbroken_run(tmp_path: Path, stop: int) -> None:
    """A stopped horizon run restores within the horizon and resumes to the unbroken results."""
    plain_engine, plain = _run(tmp_path / "plain.sqlite", None)
    _, unbroken = _run(tmp_path / "unbroken.sqlite", _HORIZON_DAYS)
    records = plain_engine.track_record.entries
    path = tmp_path / "stopped.sqlite"
    with pytest.raises(_StopError):
        _engine(path, _HORIZON_DAYS).run(before_checkpoint=_stop_at(stop))

    probe = _engine(path, _HORIZON_DAYS, resume=True)
    with pytest.raises(_StopError):
        probe.run(before_checkpoint=_stop_at(stop))
    floor = _DAYS[stop - 1] - timedelta(days=_HORIZON_DAYS)
    inside = sum(floor <= record.as_of_date <= _DAYS[stop] for record in records)
    assert _reachable(probe, PerformanceRecord) <= inside
    for day in _DAYS[stop - 1 : stop + 1]:
        assert probe.decisions_by_day[day] == plain.decisions_by_day[day]
    if stop >= 3:
        with pytest.raises(HistoryHorizonError):
            _ = probe.decisions_by_day[_DAYS[stop - 3]]

    resumed = _engine(path, _HORIZON_DAYS, resume=True).run()

    assert resumed.model_dump_json() == unbroken.model_dump_json()
    assert _without_decisions(resumed) == _without_decisions(plain)
    assert _checkpoint(path) == _checkpoint(tmp_path / "plain.sqlite")


def _decisions(day: date) -> list[PortfolioDecision]:
    return [
        PortfolioDecision(asset="AAPL", as_of_date=day, target_weight=0.1, contributing_signals=[])
    ]


def test_decision_window_raises_on_whole_history_reads() -> None:
    """Once a day was dropped, iteration, len and the views raise; `held()` gives the held days."""
    window = DecisionWindow(HELD_DECISION_DAYS)
    window[_DAYS[0]] = _decisions(_DAYS[0])
    window[_DAYS[1]] = _decisions(_DAYS[1])
    assert list(window) == _DAYS[:2]
    assert len(window) == 2

    window[_DAYS[2]] = _decisions(_DAYS[2])

    assert window.held() == {day: _decisions(day) for day in _DAYS[1:3]}
    assert window[_DAYS[2]] == _decisions(_DAYS[2])
    for read in (list, len, lambda w: list(w.keys()), lambda w: list(w.items()), dict):
        with pytest.raises(HistoryHorizonError):
            read(window)
    with pytest.raises(HistoryHorizonError):
        _ = window[_DAYS[0]]
    with pytest.raises(KeyError):
        _ = window[_DAYS[3]]


def test_decision_window_restored_past_dropped_days() -> None:
    """A window restored with earlier days left in the checkpoint refuses whole-history reads."""
    window = DecisionWindow(HELD_DECISION_DAYS)
    window.restore({day: _decisions(day) for day in _DAYS[3:5]}, earlier_days_dropped=True)

    assert window.held() == {day: _decisions(day) for day in _DAYS[3:5]}
    with pytest.raises(HistoryHorizonError):
        list(window)
    with pytest.raises(HistoryHorizonError):
        _ = window[_DAYS[2]]


def test_result_carries_the_horizon(tmp_path: Path) -> None:
    """A horizon run's result records its horizon and the held days; an unbounded one records
    None and every day."""
    _, plain = _run(tmp_path / "plain.sqlite", None)
    _, bounded = _run(tmp_path / "bounded.sqlite", _HORIZON_DAYS)

    assert plain.history_horizon_days is None
    assert list(plain.decisions_by_day) == _DAYS
    assert bounded.history_horizon_days == _HORIZON_DAYS
    assert list(bounded.decisions_by_day) == _DAYS[-HELD_DECISION_DAYS:]


def test_horizon_shorter_than_a_lookback_is_refused(tmp_path: Path) -> None:
    """The engine refuses a horizon shorter than the IC window or any weighting window."""
    assert longest_track_record_lookback(_commander()) == _HORIZON_DAYS
    with pytest.raises(ValueError, match="track-record window"):
        _engine(tmp_path / "short.sqlite", _HORIZON_DAYS - 1)

    with temporary_overrides({"portfolio.ic_lookback_days": _HORIZON_DAYS + 5}):
        with pytest.raises(ValueError, match="track-record window"):
            _engine(tmp_path / "ic.sqlite", _HORIZON_DAYS)

    manager = _PassThroughManager(
        agent_id="sharpe_manager",
        managed_analysts=[_MomentumAnalyst()],
        weighting_strategy=ShrinkageWeight(EqualWeight()),
        portfolio_constructor=_ViewConstructor(),
    )
    commander = Commander(
        managers=[manager], weighting_strategy=ShrinkageWeight(SharpeWeight(lookback_days=45))
    )
    assert longest_track_record_lookback(commander) == 45
    commander = Commander(managers=[manager], weighting_strategy=EqualWeight())
    # ShrinkageWeight counts observations over 30 days around a strategy without a window.
    assert longest_track_record_lookback(commander) == 30
