# test_resume_state_equality.py

"""After a resume, the engine holds exactly the state an unbroken run holds.

For each day k, one run goes unbroken to the end of day k; another stops after day k - 1 with a
checkpoint, and a fresh engine, commander, roster and track record (as a new process builds
them) resumes it and runs day k. Everything reachable from the two engines is then compared
generically, without a list of attributes: instance `__dict__`s and `__slots__`, pydantic fields
and private attributes, dicts (with their key order), lists, tuples, sets, DataFrames and Series
(exact, dtypes included), arrays (by bytes) and floats (by bits). The two checkpoints are
compared the same way, which covers the run's local state the checkpoint carries. Only the
attributes and types in `_EXCLUDED_ATTRIBUTES` and `_EXCLUDED_TYPES` are skipped, each for the
reason given there.

The book holds an equity (SPY), a future (CL=F), a spot FX pair (USDJPY=X) and a crypto asset
(BTC-USD), and a tail-hedge pod buys SPY puts that roll every few days, so on some day k a put
has been decided and closed: it has a recorded mark but no open position."""

import math
import struct
import types
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from enum import Enum
from pathlib import Path
from sqlite3 import Connection
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel, PrivateAttr

from agentic_trading.agents._market_math import RefitCache
from agentic_trading.agents.base import AnalystAgent, ManagerAgent
from agentic_trading.agents.commander import Commander
from agentic_trading.agents.managers.tail_hedge_manager import TailHedgeManager
from agentic_trading.agents.portfolio_construction import PortfolioConstructor
from agentic_trading.agents.schemas import AgentSignal, OptionContract
from agentic_trading.backtest import checkpoint_store
from agentic_trading.backtest.engine import BacktestEngine
from agentic_trading.backtest.portfolio import Portfolio
from agentic_trading.data.loaders import option_marks
from agentic_trading.data.loaders.option_marks import OptionMark
from agentic_trading.scoring.track_record import TrackRecord
from agentic_trading.scoring.weighting import EqualWeight

_INDEX = pd.bdate_range("2024-01-02", periods=24)
_SESSIONS = _INDEX[12:]
_DAYS = [day.date() for day in _SESSIONS]
_ASSETS = ["SPY", "CL=F", "USDJPY=X", "BTC-USD"]
SPOT = 500.0
# Every put the stub chain offers expires this many calendar days out, inside
# `settings.options.roll_min_dte`, so each put is rolled soon after it opens.
_EXPIRY_DAYS = 3

# "<Class>.<attribute>": why the attribute may differ between an unbroken and a resumed run.
_EXCLUDED_ATTRIBUTES: dict[str, str] = {
    "BacktestEngine.checkpoint_path": "where this run's checkpoint lives; each run has its own",
    "BacktestEngine.resume": "whether this run was started as a resume; how it began, not state",
    "BacktestEngine._unpriced_underlyings_warned": (
        "dedupes a warning log line; never read by a decision"
    ),
}
# Types never compared, with the reason.
_EXCLUDED_TYPES: dict[type, str] = {
    Connection: "the open checkpoint connection; each run opens its own file",
    ThreadPoolExecutor: "the commander's worker threads; they hold no value between days",
}


# Code, not state: classes, functions (the engine's data loader among them), methods.
_CODE_TYPES = (
    type,
    types.FunctionType,
    types.BuiltinFunctionType,
    types.MethodType,
    types.BuiltinMethodType,
)


class _InterruptError(Exception):
    pass


class _StatefulAnalyst(AnalystAgent):
    """Long on odd and short on even days it has signalled on, at a confidence read from a fit
    refit every three calendar days."""

    agent_id: str = "stateful_analyst"
    _refit_cache: RefitCache = PrivateAttr(default_factory=lambda: RefitCache(3))
    _days_seen: int = PrivateAttr(default=0)

    def generate_signal(
        self, asset: str, market_data: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> AgentSignal:
        self._days_seen += 1
        fit = self._refit_cache.get_or_refit(asset, as_of_date, lambda: as_of_date.dayofyear)
        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction="long" if self._days_seen % 2 else "short",
            confidence=0.2 + 0.1 * (fit % 4),
            rationale="stub",
        )


class _TailAnalyst(AnalystAgent):
    agent_id: str = "vix_tail_hedge_analyst"

    def generate_signal(
        self, asset: str, market_data: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> AgentSignal:
        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction="short",
            confidence=0.5,
            rationale="VIX elevated",
        )


class _CountingManager(ManagerAgent):
    """Scales every signal down as the days it has built a portfolio on add up."""

    agent_id: str = "counting_manager"
    _days_seen: int = PrivateAttr(default=0)

    def construct_portfolio(self, assets, market_data, as_of_date, scoreboard):
        self._days_seen += 1
        return super().construct_portfolio(assets, market_data, as_of_date, scoreboard)

    def scale_factor(self, signal, market_data, as_of_date) -> float:
        return 1.0 / (1.0 + 0.1 * self._days_seen)


class _ViewConstructor(PortfolioConstructor):
    """Sizes each asset at a fifth of its pod view's net signal."""

    def construct(self, views, returns_history, as_of_date, previous_weights=None):
        return {asset: 0.2 * view.net_signal for asset, view in views.items()}


def _mark(contract: OptionContract, as_of_date: date, spot: float) -> OptionMark:
    """A tradeable quote whose mid decays with days to expiry; intrinsic on or after expiry."""
    expired = as_of_date >= contract.expiry
    mid = (
        max(contract.strike - spot, 0.0)
        if expired
        else 6.0 + 0.5 * (contract.expiry - as_of_date).days
    )
    return OptionMark(
        contract=contract,
        as_of_date=as_of_date,
        snapshot_date=None if expired else as_of_date,
        mid=mid,
        bid=mid if expired else mid - 0.1,
        ask=mid if expired else mid + 0.1,
        implied_volatility=0.0 if expired else 0.2,
        delta=0.0 if expired else -0.3,
        gamma=0.0 if expired else 0.01,
        vega=0.0 if expired else 0.5,
        theta=0.0 if expired else -0.1,
        is_model_mark=False,
        is_expired=expired,
        tradeable=not expired,
    )


@pytest.fixture(autouse=True)
def _chain(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub SPY chain and marks for the tail-hedge manager and the engine; no split adjustment."""

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


def _frame() -> pd.DataFrame:
    """Seeded random walks for every asset; SPY stays near `SPOT`, CL=F rolls twice."""
    rng = np.random.default_rng(148)
    columns: dict[tuple[str, str], Any] = {}
    for asset, level, vol in (
        ("SPY", SPOT, 0.005),
        ("CL=F", 70.0, 0.02),
        ("USDJPY=X", 150.0, 0.005),
        ("BTC-USD", 40_000.0, 0.03),
    ):
        close = pd.Series(level * np.exp(np.cumsum(rng.normal(0.0, vol, len(_INDEX)))), _INDEX)
        columns[(asset, "open")] = close.shift(1).fillna(close.iloc[0])
        columns[(asset, "close")] = close
        columns[(asset, "log_return")] = np.log(close / close.shift(1))
        columns[(asset, "volume")] = pd.Series(rng.uniform(1e5, 1e6, len(_INDEX)), _INDEX)
    roll = np.zeros(len(_INDEX))
    roll[[15, 19]] = 1.0
    columns[("CL=F", "roll")] = pd.Series(roll, _INDEX)
    return pd.DataFrame(columns)


def _engine(checkpoint_path: Path, resume: bool = False) -> BacktestEngine:
    """A freshly built engine, commander, roster and track record over `_DAYS`."""
    stateful = _CountingManager(
        managed_analysts=[_StatefulAnalyst()],
        weighting_strategy=EqualWeight(),
        portfolio_constructor=_ViewConstructor(),
        assets=tuple(_ASSETS),
    )
    tail = TailHedgeManager(
        managed_analysts=[_TailAnalyst()],
        weighting_strategy=EqualWeight(),
        portfolio_constructor=PortfolioConstructor(),
        assets=("SPY",),
    )
    return BacktestEngine(
        commander=Commander(
            managers=[stateful, tail], weighting_strategy=EqualWeight(), parallel_workers=2
        ),
        portfolio=Portfolio(cash=1_000_000.0, positions={}, commission_rate=5e-4, slippage_bps=2.0),
        track_record=TrackRecord(),
        data_loader=lambda _assets, _start, _end: _frame(),
        start_date=_DAYS[0],
        end_date=_DAYS[-1],
        assets=list(_ASSETS),
        execution_lag_days=1,
        checkpoint_path=checkpoint_path,
        resume=resume,
        policy_rates=pd.DataFrame(
            {"USD": [0.05, 0.0525], "JPY": [-0.001, 0.001]},
            index=pd.to_datetime(["2023-12-29", "2024-01-05"]),
        ),
    )


def _run_through(engine: BacktestEngine, day: int) -> BacktestEngine:
    """Runs `engine` and stops it once day index `day` is complete and checkpointed."""

    def stop(day_index: int, *_args: object) -> None:
        if day_index == day:
            raise _InterruptError

    with pytest.raises(_InterruptError):
        engine.run(on_day_complete=stop)
    return engine


def _children(value: Any) -> Iterator[tuple[str, Any]]:
    """(label, value) for every object `value` holds: pydantic fields and private attributes,
    instance `__dict__` and `__slots__` entries."""
    owner = type(value).__name__
    if isinstance(value, BaseModel):
        for name in type(value).model_fields:
            yield f"{owner}.{name}", getattr(value, name)
        for name, item in (value.__pydantic_private__ or {}).items():
            yield f"{owner}.{name}", item
        return
    for name, item in getattr(value, "__dict__", {}).items():
        yield f"{owner}.{name}", item
    for cls in type(value).__mro__:
        for name in getattr(cls, "__slots__", ()):
            if name not in {"__dict__", "__weakref__"} and hasattr(value, name):
                yield f"{owner}.{name}", getattr(value, name)


def _differences(left: Any, right: Any, path: str, seen: set[tuple[int, int]]) -> list[str]:
    """Every path at which the object graphs under `left` and `right` differ."""
    if type(left) is not type(right):
        return [f"{path}: {type(left).__name__} != {type(right).__name__}"]
    if isinstance(left, tuple(_EXCLUDED_TYPES)):
        return []
    if isinstance(left, float):
        same = struct.pack("<d", left) == struct.pack("<d", right) or (
            math.isnan(left) and math.isnan(right)
        )
        return [] if same else [f"{path}: {left!r} != {right!r}"]
    if left is None or isinstance(left, bool | int | str | bytes | date | pd.Timestamp | Enum):
        return [] if left == right else [f"{path}: {left!r} != {right!r}"]
    if isinstance(left, np.ndarray):
        same = left.dtype == right.dtype and left.shape == right.shape
        return [] if same and left.tobytes() == right.tobytes() else [f"{path}: arrays differ"]
    if isinstance(left, pd.DataFrame):
        same = left.equals(right) and left.dtypes.equals(right.dtypes)
        return [] if same else [f"{path}: DataFrame differs"]
    if isinstance(left, pd.Series):
        same = left.equals(right) and left.dtype == right.dtype
        return [] if same else [f"{path}: Series differs"]
    if isinstance(left, pd.Index):
        return [] if left.equals(right) and left.dtype == right.dtype else [f"{path}: index"]
    if isinstance(left, _CODE_TYPES):
        return []
    key = (id(left), id(right))
    if key in seen:
        return []
    seen.add(key)
    if isinstance(left, dict):
        if list(left) != list(right):
            return [f"{path}: keys {list(left)} != {list(right)}"]
        found: list[str] = []
        for item_key in left:
            found += _differences(left[item_key], right[item_key], f"{path}[{item_key!r}]", seen)
        return found
    if isinstance(left, list | tuple):
        if len(left) != len(right):
            return [f"{path}: length {len(left)} != {len(right)}"]
        found = []
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            found += _differences(a, b, f"{path}[{index}]", seen)
        return found
    if isinstance(left, set | frozenset):
        return [] if left == right else [f"{path}: {sorted(map(repr, left ^ right))}"]
    found = []
    left_children = dict(_children(left))
    right_children = dict(_children(right))
    if list(left_children) != list(right_children):
        return [f"{path}: attributes {list(left_children)} != {list(right_children)}"]
    for label, item in left_children.items():
        if label in _EXCLUDED_ATTRIBUTES:
            continue
        found += _differences(item, right_children[label], f"{path}.{label.split('.')[1]}", seen)
    return found


def _checkpoint(path: Path) -> dict[str, Any] | None:
    connection = checkpoint_store.connect(path, resume=True)
    try:
        return checkpoint_store.read_all(connection)
    finally:
        connection.close()


def _decided_without_position(engine: BacktestEngine, day: date) -> set[str]:
    """Option contracts decided on or before `day` that the book does not hold after it."""
    decided = {
        decision.asset
        for decisions in engine.decisions_by_day.values()
        for decision in decisions
        if decision.instrument is not None
    }
    return decided - set(engine.portfolio.option_positions)


def test_resumed_engine_state_equals_the_unbroken_state(tmp_path: Path) -> None:
    """For every day k after the first, the engine resumed after day k - 1 and run through day k
    holds the same state, and writes the same checkpoint, as the engine run unbroken through
    day k; on some k an option contract has been decided but is not held."""
    closed_puts_seen = False
    for k in range(1, len(_DAYS)):
        unbroken = _run_through(_engine(tmp_path / f"unbroken_{k}.db"), k)
        _run_through(_engine(tmp_path / f"resumed_{k}.db"), k - 1)
        resumed = _run_through(_engine(tmp_path / f"resumed_{k}.db", resume=True), k)
        closed_puts_seen |= bool(_decided_without_position(unbroken, _DAYS[k]))

        engine_differences = _differences(unbroken, resumed, "engine", set())
        assert engine_differences == [], f"day {k}: " + "; ".join(engine_differences)
        assert (
            _differences(
                _checkpoint(tmp_path / f"unbroken_{k}.db"),
                _checkpoint(tmp_path / f"resumed_{k}.db"),
                "checkpoint",
                set(),
            )
            == []
        ), f"day {k}"

    assert closed_puts_seen


def test_the_walk_reaches_agent_caches_and_engine_marks(tmp_path: Path) -> None:
    """The comparison sees a change to an analyst's refit cache, a manager's private counter and
    the engine's option marks, each named by its path."""
    left = _run_through(_engine(tmp_path / "left.db"), 3)
    right = _run_through(_engine(tmp_path / "right.db"), 3)
    assert _differences(left, right, "engine", set()) == []

    manager = right.commander.managers[0]
    analyst = manager.managed_analysts[0]
    analyst._refit_cache._cache["SPY"] = (pd.Timestamp(_DAYS[0]), 1)
    manager._days_seen += 1
    right._option_last_mid["SPY 2030-01-18 P 1.00"] = 1.0

    differences = _differences(left, right, "engine", set())
    assert any("._refit_cache._cache" in difference for difference in differences)
    assert any("._days_seen" in difference for difference in differences)
    assert any("._option_last_mid" in difference for difference in differences)


def _without_option_marks(path: Path) -> None:
    """Rewrites the checkpoint at `path` as one written before the engine stored option marks."""
    connection = checkpoint_store.connect(path, resume=True)
    with connection:
        connection.execute("UPDATE cursor SET option_last_mid = NULL")
    connection.close()


def test_a_checkpoint_without_option_marks_after_an_option_decision_is_refused(
    tmp_path: Path,
) -> None:
    """A checkpoint that stores no option marks, of a run that has decided an option, refuses to
    resume; the same checkpoint with its marks resumes."""
    path = tmp_path / "checkpoint.db"
    engine = _run_through(_engine(path), 3)
    assert any(
        decision.instrument is not None
        for decisions in engine.decisions_by_day.values()
        for decision in decisions
    )
    _run_through(_engine(path, resume=True), 4)
    _without_option_marks(path)

    with pytest.raises(ValueError, match="predates option_last_mid"):
        _engine(path, resume=True).run()
