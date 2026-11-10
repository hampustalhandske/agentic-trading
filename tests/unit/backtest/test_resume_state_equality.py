# test_resume_state_equality.py

"""After a resume, the engine holds exactly the state an unbroken run holds.

For each day k, one run goes unbroken to the end of day k; another stops after day k - 1 with a
checkpoint, and a fresh engine, commander, roster and track record (as a new process builds
them) resumes it and runs day k. The instance state reachable from the two engines is then
compared generically, without a list of attributes (`_Walk`): pydantic fields, private and extra
attributes, instance `__dict__`s and `__slots__`, containers, pandas and numpy values by their
bytes, floats by their bits, functions by their closures and defaults, and whether the two
graphs share objects the same way. A value of a type the walk has no rule for fails the test
unless `_STATELESS_TYPES` lists it. The two checkpoints are compared the same way, which covers
the run's local state the checkpoint carries; module and class state is covered by
tests/unit/test_process_state_census.py. Only the attributes and types the `_EXCLUDED_*` and
`_UNSHARED_RECORD_TYPES` tables name are skipped, each for the reason given there.

The book holds an equity (SPY), a future (CL=F), a spot FX pair (USDJPY=X) and a crypto asset
(BTC-USD), and a tail-hedge pod buys SPY puts that roll every few days, so on some day k a put
has been decided and closed: it has a recorded mark but no open position."""

import collections
import functools
import struct
import types
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import date, time, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import Path
from sqlite3 import Connection
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel, PrivateAttr

from agentic_trading.agents._market_math import RefitCache
from agentic_trading.agents.analysts.cvar_pod import _options_data
from agentic_trading.agents.analysts.cvar_pod.black_scholes_valuation_analyst import (
    BlackScholesValuationAnalyst,
)
from agentic_trading.agents.analysts.cvar_pod.heston_vol_surface_analyst import (
    HestonVolSurfaceAnalyst,
)
from agentic_trading.agents.base import AnalystAgent, ManagerAgent
from agentic_trading.agents.commander import Commander
from agentic_trading.agents.managers.tail_hedge_manager import TailHedgeManager
from agentic_trading.agents.portfolio_construction import PortfolioConstructor
from agentic_trading.agents.schemas import (
    AgentSignal,
    ManagerSignal,
    OptionContract,
    PortfolioDecision,
)
from agentic_trading.backtest import checkpoint_store
from agentic_trading.backtest.engine import BacktestEngine
from agentic_trading.backtest.portfolio import Portfolio
from agentic_trading.config import temporary_overrides
from agentic_trading.data.loaders import option_marks
from agentic_trading.data.loaders.option_marks import OptionMark
from agentic_trading.scoring.track_record import TrackRecord
from agentic_trading.scoring.weighting import EqualWeight

# 46 warm-up bars, enough for the GARCH-based analysts, then 24 sessions (32 calendar days).
_INDEX = pd.bdate_range("2023-10-30", periods=70)
_SESSIONS = _INDEX[46:]
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
# Attributes of any class never compared, with the reason.
_EXCLUDED_ATTRIBUTE_NAMES: dict[str, str] = {
    "__pydantic_fields_set__": (
        "which fields a model was given rather than defaulted; it changes only "
        "`model_dump(exclude_unset=True)`, which no code under src/ calls, and a model rebuilt "
        "from checkpoint JSON has every field set"
    ),
}
# Record types whose sharing is not compared, with the reason; their contents still are.
_UNSHARED_RECORD_TYPES: dict[type, str] = {
    PortfolioDecision: (
        "rebuilt from checkpoint JSON as separate copies; no code changes one after it is built"
    ),
    ManagerSignal: (
        "rebuilt from checkpoint JSON as separate copies; no code changes one after it is built"
    ),
}
# Types never compared, with the reason.
_EXCLUDED_TYPES: dict[type, str] = {
    Connection: "the open checkpoint connection; each run opens its own file",
    ThreadPoolExecutor: "the commander's worker threads; they hold no value between days",
}


# Leaf types holding no state of their own, with the reason.
_STATELESS_TYPES: dict[type, str] = {
    type: "a class; class-level state is covered by the process-state census",
    types.ModuleType: "a module; module-level state is covered by the process-state census",
    types.BuiltinFunctionType: "a builtin function or a builtin bound method of a module",
    EqualWeight: "a weighting strategy with no attributes",
}
# Immutable values compared with `==`; floats are compared by their bits instead.
_EQUALITY_TYPES = (
    bool,
    int,
    str,
    bytes,
    date,
    time,
    timedelta,
    Decimal,
    complex,
    Enum,
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
    monkeypatch.setattr(_options_data, "load_options_snapshot_from_lakehouse", _snapshot)
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
    roll[[52, 60]] = 1.0
    columns[("CL=F", "roll")] = pd.Series(roll, _INDEX)
    return pd.DataFrame(columns)


# The firm record's history horizon, in calendar days: inside the sessions' 32 days, so the
# track record and decisions are dropped while the run goes on. It is the shortest the
# commander allows with `portfolio.ic_lookback_days` at its minimum, 20.
_HORIZON_DAYS = 20
# The days k a run is resumed at (after day k - 1): every third session, from the second.
_RESUME_DAYS = range(1, len(_DAYS), 3)


@pytest.fixture(autouse=True)
def _short_ic_window() -> Iterator[None]:
    with temporary_overrides({"portfolio.ic_lookback_days": _HORIZON_DAYS}):
        yield


def _snapshot(ticker: str, as_of_date: date, lookback_days: int = 10) -> pd.DataFrame:
    """A SPY chain snapshot dated `as_of_date`: a call and a put at every fifth strike from 400
    to 600 for three expiries 30, 65 and 128 days out, quoted at 20, 21 and 22 vols."""
    stamp = pd.Timestamp(as_of_date)
    return pd.DataFrame(
        [
            {
                "date": stamp,
                "expiration": stamp + pd.Timedelta(days=days_out),
                "option_type": kind,
                "strike": float(strike),
                "implied_volatility": iv,
                "delta": 0.5 if kind == "call" else -0.5,
                "gamma": 0.02,
            }
            for days_out, iv in ((30, 0.20), (65, 0.21), (128, 0.22))
            for strike in range(400, 601, 5)
            for kind in ("call", "put")
        ]
    )


def _engine(
    checkpoint_path: Path,
    resume: bool = False,
    horizon: int | None = None,
    garch_analysts: bool = False,
) -> BacktestEngine:
    """A freshly built engine, commander, roster and track record over `_DAYS`, with the history
    horizon `horizon`; `garch_analysts` adds the roster's Heston and Black-Scholes analysts,
    each holding a GARCH `RefitCache`, on SPY."""
    analysts: list[AnalystAgent] = [_StatefulAnalyst()]
    if garch_analysts:
        analysts += [HestonVolSurfaceAnalyst(), BlackScholesValuationAnalyst()]
    stateful = _CountingManager(
        managed_analysts=analysts,
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
        track_record=TrackRecord(horizon_days=horizon),
        data_loader=lambda _assets, _start, _end: _frame(),
        start_date=_DAYS[0],
        end_date=_DAYS[-1],
        assets=list(_ASSETS),
        execution_lag_days=1,
        checkpoint_path=checkpoint_path,
        resume=resume,
        history_horizon_days=horizon,
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


def _children(value: Any) -> list[tuple[str, Any]]:
    """(label, value) for every object `value` holds: pydantic fields, private attributes,
    extra fields and fields set, instance `__dict__` and `__slots__` entries."""
    owner = type(value).__name__
    if isinstance(value, BaseModel):
        found = [(f"{owner}.{name}", getattr(value, name)) for name in type(value).model_fields]
        found += [
            (f"{owner}.{name}", item) for name, item in (value.__pydantic_private__ or {}).items()
        ]
        found.append((f"{owner}.__pydantic_extra__", value.__pydantic_extra__))
        found.append((f"{owner}.__pydantic_fields_set__", value.__pydantic_fields_set__))
        return found
    found = [(f"{owner}.{name}", item) for name, item in getattr(value, "__dict__", {}).items()]
    for cls in type(value).__mro__:
        for name in getattr(cls, "__slots__", ()):
            if name not in {"__dict__", "__weakref__"} and hasattr(value, name):
                found.append((f"{owner}.{name}", getattr(value, name)))
    return found


class _Walk:
    """Compares two object graphs: `differences(left, right, path)` lists every path at which
    they differ, and raises `TypeError` on a value it has no rule for. A mutable object is
    compared once, at the first path it is reached by; reaching it again requires the other
    graph to reach its counterpart first at the same path, so both graphs share objects the
    same way."""

    def __init__(self) -> None:
        self._left_first: dict[int, str] = {}
        self._right_first: dict[int, str] = {}
        self._held: list[Any] = []

    def differences(self, left: Any, right: Any, path: str) -> list[str]:
        if type(left) is not type(right):
            return [f"{path}: {type(left).__name__} != {type(right).__name__}"]
        if isinstance(left, tuple(_EXCLUDED_TYPES) + tuple(_STATELESS_TYPES)):
            return []
        if isinstance(left, float):
            same = struct.pack("<d", left) == struct.pack("<d", right)
            return [] if same else [f"{path}: {left!r} != {right!r}"]
        if left is None or left is pd.NaT:
            return []
        if isinstance(left, _EQUALITY_TYPES):
            return [] if left == right else [f"{path}: {left!r} != {right!r}"]
        if isinstance(left, np.generic):
            same = left.dtype == right.dtype and left.tobytes() == right.tobytes()
            return [] if same else [f"{path}: {left!r} != {right!r}"]
        if isinstance(left, tuple | frozenset):
            return self._contents(left, right, path)
        if (
            isinstance(left, types.FunctionType)
            and left.__closure__ is None
            and "<" not in (left.__qualname__)
        ):
            return [] if left is right else [f"{path}: function {left!r} != {right!r}"]
        if isinstance(left, tuple(_UNSHARED_RECORD_TYPES)) or (
            isinstance(left, BaseModel) and left.model_config.get("frozen")
        ):
            return self._contents(left, right, path)
        aliasing = self._aliasing(left, right, path)
        if aliasing is not None:
            return aliasing
        return self._contents(left, right, path)

    def _aliasing(self, left: Any, right: Any, path: str) -> list[str] | None:
        """`None` the first time this pair is reached; on a later visit, [] when both sides
        were first reached at the same path, and a difference otherwise."""
        left_first = self._left_first.get(id(left))
        right_first = self._right_first.get(id(right))
        if left_first is None and right_first is None:
            self._left_first[id(left)] = path
            self._right_first[id(right)] = path
            self._held += [left, right]
            return None
        if left_first == right_first:
            return []
        return [f"{path}: shared with {left_first} on one side, {right_first} on the other"]

    def _contents(self, left: Any, right: Any, path: str) -> list[str]:
        if isinstance(left, np.ndarray):
            if left.dtype != right.dtype or left.shape != right.shape:
                return [f"{path}: array {left.dtype}{left.shape} != {right.dtype}{right.shape}"]
            if left.dtype == object:
                return self._sequence(list(left.ravel()), list(right.ravel()), path)
            return [] if left.tobytes() == right.tobytes() else [f"{path}: array values differ"]
        if isinstance(left, pd.DataFrame):
            found = self.differences(list(left.columns), list(right.columns), f"{path}.columns")
            found += self.differences(left.index, right.index, f"{path}.index")
            if found:
                return found
            for position, column in enumerate(left.columns):
                found += self.differences(
                    left.iloc[:, position], right.iloc[:, position], f"{path}[{column!r}]"
                )
            return found
        if isinstance(left, pd.Series):
            found = self.differences(str(left.dtype), str(right.dtype), f"{path}.dtype")
            found += self.differences(left.name, right.name, f"{path}.name")
            found += self.differences(left.index, right.index, f"{path}.index")
            return found + self.differences(
                left.to_numpy(copy=True), right.to_numpy(copy=True), f"{path}.values"
            )
        if isinstance(left, pd.Index):
            found = self.differences(str(left.dtype), str(right.dtype), f"{path}.dtype")
            found += self.differences(list(left.names), list(right.names), f"{path}.names")
            found += self.differences(
                str(getattr(left, "freq", None)), str(getattr(right, "freq", None)), f"{path}.freq"
            )
            values = left.to_numpy() if not isinstance(left, pd.MultiIndex) else left.to_list()
            other = right.to_numpy() if not isinstance(right, pd.MultiIndex) else right.to_list()
            return found + self.differences(values, other, f"{path}.values")
        if isinstance(left, np.random.Generator):
            return self.differences(
                left.bit_generator.state, right.bit_generator.state, f"{path}.state"
            )
        if isinstance(left, functools.partial):
            return self.differences(
                (left.func, left.args, left.keywords),
                (right.func, right.args, right.keywords),
                f"{path}.partial",
            )
        if isinstance(left, types.FunctionType):
            if left.__code__ is not right.__code__:
                return [f"{path}: function {left.__qualname__} != {right.__qualname__}"]
            cells = [cell.cell_contents for cell in left.__closure__ or ()]
            other_cells = [cell.cell_contents for cell in right.__closure__ or ()]
            found = self.differences(cells, other_cells, f"{path}.__closure__")
            found += self.differences(left.__defaults__, right.__defaults__, f"{path}.defaults")
            return found + self.differences(
                left.__kwdefaults__, right.__kwdefaults__, f"{path}.kwdefaults"
            )
        if isinstance(left, types.MethodType):
            found = self.differences(left.__func__, right.__func__, f"{path}.__func__")
            return found + self.differences(left.__self__, right.__self__, f"{path}.__self__")
        if isinstance(left, collections.defaultdict):
            found = self.differences(
                left.default_factory, right.default_factory, f"{path}.default_factory"
            )
            return found + self._mapping(left, right, path)
        if isinstance(left, dict):
            return self._mapping(left, right, path)
        if isinstance(left, collections.deque):
            found = self.differences(left.maxlen, right.maxlen, f"{path}.maxlen")
            return found + self._sequence(list(left), list(right), path)
        if isinstance(left, list | tuple):
            return self._sequence(left, right, path)
        if isinstance(left, set | frozenset):
            same = sorted(map(repr, left)) == sorted(map(repr, right))
            return (
                [] if same else [f"{path}: {sorted(map(repr, left))} != {sorted(map(repr, right))}"]
            )
        left_children = _children(left)
        if not left_children:
            raise TypeError(f"unhandled type {type(left).__qualname__} at {path}")
        right_children = _children(right)
        if [label for label, _ in left_children] != [label for label, _ in right_children]:
            return [f"{path}: attributes differ"]
        found = []
        for (label, item), (_, other) in zip(left_children, right_children, strict=True):
            if label in _EXCLUDED_ATTRIBUTES or label.split(".", 1)[1] in _EXCLUDED_ATTRIBUTE_NAMES:
                continue
            found += self.differences(item, other, f"{path}.{label.split('.', 1)[1]}")
        return found

    def _mapping(self, left: dict[Any, Any], right: dict[Any, Any], path: str) -> list[str]:
        found = self.differences(list(left), list(right), f"{path}.keys()")
        if found:
            return found
        for key, other_key in zip(left, right, strict=True):
            found += self.differences(left[key], right[other_key], f"{path}[{key!r}]")
        return found

    def _sequence(self, left: Any, right: Any, path: str) -> list[str]:
        if len(left) != len(right):
            return [f"{path}: length {len(left)} != {len(right)}"]
        found: list[str] = []
        for index, (item, other) in enumerate(zip(left, right, strict=True)):
            found += self.differences(item, other, f"{path}[{index}]")
        return found


def _differences(left: Any, right: Any, path: str) -> list[str]:
    """Every path at which the object graphs under `left` and `right` differ."""
    return _Walk().differences(left, right, path)


def _checkpoint(path: Path) -> dict[str, Any] | None:
    connection = checkpoint_store.connect(path, resume=True)
    try:
        return checkpoint_store.read_all(connection)
    finally:
        connection.close()


def _marked_without_position(engine: BacktestEngine) -> set[str]:
    """Option contracts the engine holds a recorded mark for but no position in."""
    return set(engine._option_last_mid) - set(engine.portfolio.option_positions)


def _result_json(engine: BacktestEngine) -> str:
    """`engine.run()`'s result JSON, `unfactored_split_moves` aside (a resumed run reports it
    as `None`)."""
    return engine.run().model_dump_json(exclude={"unfactored_split_moves"})


@pytest.mark.parametrize("garch_analysts", [False, True], ids=["stub_analysts", "garch_analysts"])
@pytest.mark.parametrize("horizon", [None, _HORIZON_DAYS], ids=["no_horizon", "horizon"])
def test_resumed_engine_state_equals_the_unbroken_state(
    tmp_path: Path, horizon: int | None, garch_analysts: bool
) -> None:
    """For every day k in `_RESUME_DAYS`, the engine resumed after day k - 1 and run through day k
    holds the same state, and writes the same checkpoint, as the engine run unbroken through
    day k; resumed after day k - 1 and run to the end, it holds the same final state and
    returns the same result as the unbroken run. On some k an option contract has been decided
    but is not held."""

    def engine(name: str, resume: bool = False) -> BacktestEngine:
        return _engine(tmp_path / f"{name}.db", resume, horizon, garch_analysts)

    full = engine("full")
    full_json = _result_json(full)
    for analyst in full.commander.managers[0].managed_analysts:
        assert analyst._refit_cache._cache, f"{analyst.agent_id} never fitted"
    closed_puts_seen = False
    for k in _RESUME_DAYS:
        unbroken = _run_through(engine(f"unbroken_{k}"), k)
        _run_through(engine(f"resumed_{k}"), k - 1)
        resumed = _run_through(engine(f"resumed_{k}", resume=True), k)
        closed_puts_seen |= bool(_marked_without_position(unbroken))

        engine_differences = _differences(unbroken, resumed, "engine")
        assert engine_differences == [], f"day {k}: " + "; ".join(engine_differences)
        checkpoint_differences = _differences(
            _checkpoint(tmp_path / f"unbroken_{k}.db"),
            _checkpoint(tmp_path / f"resumed_{k}.db"),
            "checkpoint",
        )
        assert checkpoint_differences == [], f"day {k}: " + "; ".join(checkpoint_differences)

        _run_through(engine(f"to_end_{k}"), k - 1)
        to_end = engine(f"to_end_{k}", resume=True)
        assert _result_json(to_end) == full_json, f"resumed after day {k - 1}"
        final_differences = _differences(full, to_end, "engine")
        assert final_differences == [], f"to the end after {k - 1}: " + "; ".join(final_differences)

    assert closed_puts_seen


def _closure(value: float) -> Any:
    def read() -> float:
        return value

    return read


def _plant(engine: BacktestEngine, counter: int, last: float, cell: float, zero: float) -> None:
    """Hangs an np.int64 counter, a bounded deque, a closure and a DataFrame on `engine`."""
    engine._probe_counter = np.int64(counter)  # type: ignore[attr-defined]
    engine._probe_deque = collections.deque([1.0, last], maxlen=3)  # type: ignore[attr-defined]
    engine._probe_closure = _closure(cell)  # type: ignore[attr-defined]
    engine._probe_frame = pd.DataFrame({"a": [1.0, zero]})  # type: ignore[attr-defined]


def test_the_walk_sees_every_kind_of_state(tmp_path: Path) -> None:
    """The comparison sees a change to an analyst's refit cache, a manager's private counter,
    the engine's option marks, an np.int64, a deque, a closure cell and a DataFrame's negative
    zero, each named by its path, and refuses a value of a type it has no rule for."""
    left = _run_through(_engine(tmp_path / "left.db"), 3)
    right = _run_through(_engine(tmp_path / "right.db"), 3)
    _plant(left, 1, 2.0, 3.0, 0.0)
    _plant(right, 1, 2.0, 3.0, 0.0)
    assert _differences(left, right, "engine") == []

    manager = right.commander.managers[0]
    analyst = manager.managed_analysts[0]
    analyst._refit_cache._cache["SPY"] = (pd.Timestamp(_DAYS[0]), 1)
    manager._days_seen += 1
    right._option_last_mid["SPY 2030-01-18 P 1.00"] = 1.0
    _plant(right, 2, 2.5, 3.5, -0.0)

    differences = _differences(left, right, "engine")
    for fragment in (
        "._refit_cache._cache",
        "._days_seen",
        "._option_last_mid",
        "._probe_counter",
        "._probe_deque[1]",
        "._probe_closure.__closure__[0]",
        "._probe_frame['a'].values",
    ):
        assert any(fragment in difference for difference in differences), fragment

    right._probe_frame = memoryview(b"x")  # type: ignore[attr-defined]
    left._probe_frame = memoryview(b"x")  # type: ignore[attr-defined]
    with pytest.raises(TypeError, match=r"unhandled type memoryview at engine\._probe_frame"):
        _differences(left, right, "engine")


def _without_option_marks(path: Path, schema: str) -> None:
    """Leaves the checkpoint at `path` with no stored option marks: for `schema` "null" its
    `option_last_mid` is set to NULL, and for "no column" the column is dropped from the cursor
    table, so the next `checkpoint_store.connect` adds it back as NULL."""
    connection = checkpoint_store.connect(path, resume=True)
    with connection:
        if schema == "null":
            connection.execute("UPDATE cursor SET option_last_mid = NULL")
        else:
            connection.execute("ALTER TABLE cursor DROP COLUMN option_last_mid")
    connection.close()


@pytest.mark.parametrize("schema", ["null", "no column"])
def test_a_checkpoint_without_option_marks_after_an_option_decision_is_refused(
    tmp_path: Path, schema: str
) -> None:
    """A checkpoint that stores no option marks, as a NULL or with no `option_last_mid` column at
    all, of a run that has decided an option, refuses to resume; the same checkpoint with its
    marks resumes."""
    path = tmp_path / "checkpoint.db"
    engine = _run_through(_engine(path), 3)
    assert any(
        decision.instrument is not None
        for decisions in engine.decisions_by_day.values()
        for decision in decisions
    )
    _run_through(_engine(path, resume=True), 4)
    _without_option_marks(path, schema)

    with pytest.raises(ValueError, match="has no stored option marks; its option realized returns"):
        _engine(path, resume=True).run()
