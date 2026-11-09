# test_process_state_census.py

"""Census of process-level mutable state in the packages a firm record runs through.

A firm record checkpoints each agent's pydantic private attributes (`carried_state`) and resumes
in a fresh process, where every module-level object starts from what the module builds at
import. An object whose value depends on which sessions this process has already run therefore
makes a resumed record differ from an unbroken one. Such state belongs on the agent as a private
attribute, where the checkpoint carries it.

The census imports every module under `_PACKAGES` and finds each object a module defines (a
top-level assignment or `def`, read with `ast`, so imported names are skipped) that is process
state: a mutable container, an instance of a class that is not known to be immutable (a
`RefitCache`, a lock, a non-frozen pydantic model), a `functools.cache`/`lru_cache` function,
or any name a function of the module rebinds with `global`. The objects found must be exactly
those in `_ALLOWED`, each with the reason its value is the same after a resume, and those in
`KNOWN_NOT_RESUME_SAFE`, each naming the issue that tracks it; no object is on both lists."""

import ast
import dataclasses
import datetime
import decimal
import enum
import importlib
import logging
import pathlib
import pkgutil
import re
import types
from types import MappingProxyType, ModuleType
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel

_PACKAGES = (
    "agentic_trading.agents",
    "agentic_trading.firm_record",
    "agentic_trading.backtest",
    "agentic_trading.scoring",
    "agentic_trading.data",
)

# Values that cannot change after they are built.
_IMMUTABLE_TYPES: tuple[type, ...] = (
    bool,
    int,
    float,
    complex,
    str,
    bytes,
    tuple,
    frozenset,
    range,
    decimal.Decimal,
    datetime.date,
    datetime.time,
    datetime.timedelta,
    pd.Timedelta,
    pathlib.PurePath,
    np.generic,
    MappingProxyType,
    re.Pattern,
    enum.Enum,
    types.GenericAlias,
    type,
    ModuleType,
    types.FunctionType,
    types.BuiltinFunctionType,
    types.MethodType,
    logging.Logger,
)

_CONSTANT = "constant table built at import and never mutated"
_LOCK = "lock serialising access to a module-level object; held only inside one call"
_ROSTER_PAIRS = (
    "derived from the roster's _COINTEGRATED_PAIRS at import, rebuilt identically on start; "
    "each analyst gets its own copy"
)
_STORE_SOURCE = (
    "window-store source definition (name, fetch function, ranged flag) set at import; the "
    "data it fetches lives in the session's window store"
)
_LAKE_TIME_MEMO = (
    "lake read memoized per lake client and lake time; the run's lake_session pins one lake "
    "time, so a resumed run re-reads the same value"
)
_OPTION_MARK_CACHE = (
    "memoized lake read keyed by its own arguments and emptied when the lake time changes; "
    "the run's lake_session pins one lake time, so a miss after a resume re-reads the same value"
)

# "<module>:<name>": why the object's value is the same after a resume as in an unbroken run.
_ALLOWED: dict[str, str] = {
    "agentic_trading.agents.analysts:__all__": _CONSTANT,
    "agentic_trading.agents.managers:__all__": _CONSTANT,
    "agentic_trading.agents.base:_DIRECTION_SIGN": _CONSTANT,
    "agentic_trading.agents.roster:_PARTNER_TICKERS_BY_ASSET": _ROSTER_PAIRS,
    "agentic_trading.agents.roster:_PARTNER_TICKER_BY_ASSET": _ROSTER_PAIRS,
    "agentic_trading.backtest.run_summary:EXECUTION_COST_KEYS": _CONSTANT,
    "agentic_trading.data.loader:DEFAULT_BASKET": _CONSTANT,
    "agentic_trading.data.loader:_PRICE_FIELDS": _CONSTANT,
    "agentic_trading.data.ticker_catalog:_BY_SYMBOL": (
        "index of the static ticker catalog built at import, never mutated"
    ),
    "agentic_trading.firm_record.availability:_COUNTED_DAY": (
        "pydantic TypeAdapter for a fixed schema, built at import; holds no data"
    ),
    "agentic_trading.firm_record.availability:_SPEC": "pydantic ConfigDict, a constant",
    "agentic_trading.firm_record.submission:_SPEC": "pydantic ConfigDict, a constant",
    "agentic_trading.data.loaders.lakehouse_loader:_lake": (
        "the process's lake HTTP client, built once from the environment; every read it serves "
        "is pinned by the run's lake_session"
    ),
    "agentic_trading.data.loaders.lakehouse_loader:_SESSION": (
        "the open lake_session; a resumed run reopens it at the lake time and data_end it was "
        "started with"
    ),
    "agentic_trading.data.loaders.lakehouse_loader:_SESSION_LOCK": _LOCK,
    "agentic_trading.data.loaders.lakehouse_loader:_STORE": (
        "the session's window store; a call's answer is the same whether it is served from the "
        "store or fetched directly, whatever the store already holds"
    ),
    "agentic_trading.data.loaders.lakehouse_loader:_STORE_LOCK": _LOCK,
    "agentic_trading.data.loaders.lakehouse_loader:_LOADER_CALL": (
        "context variable naming the public loader call in progress for query counting; reset "
        "when the call returns, never read by a decision"
    ),
    "agentic_trading.data.loaders.lakehouse_loader:_QUERY_COUNTS": (
        "diagnostic per-loader query counter echoed to stderr, never read by a decision or "
        "written to a result"
    ),
    "agentic_trading.data.loaders.lakehouse_loader:_QUERY_COUNTS_LOCK": _LOCK,
    "agentic_trading.data.loaders.lakehouse_loader:_UNFACTORED_SPLIT_MOVES_LOCK": _LOCK,
    "agentic_trading.data.loaders.lakehouse_loader:_ROW_TIME_PIN_COLUMNS": _CONSTANT,
    "agentic_trading.data.loaders.lakehouse_loader:_COT_DISAGGREGATED_CODES": _CONSTANT,
    "agentic_trading.data.loaders.lakehouse_loader:_COT_TFF_CODES": _CONSTANT,
    "agentic_trading.data.loaders.lakehouse_loader:DEFAULT_COT_TICKER_BY_ASSET": _CONSTANT,
    "agentic_trading.data.loaders.lakehouse_loader:_COT_NET_POSITION": _CONSTANT,
    "agentic_trading.data.loaders.lakehouse_loader:_CRYPTO_KLINE_SYMBOLS": _CONSTANT,
    "agentic_trading.data.loaders.lakehouse_loader:_KEN_FRENCH_DATASET_BY_MODEL": _CONSTANT,
    "agentic_trading.data.loaders.lakehouse_loader:_KEN_FRENCH_FACTORS_BY_MODEL": _CONSTANT,
    "agentic_trading.data.loaders.lakehouse_loader:_BIS_POLICY_RATE_AREAS": _CONSTANT,
    "agentic_trading.data.loaders.lakehouse_loader:_COMPANY_FACT_PERIOD_KEY": _CONSTANT,
    "agentic_trading.data.loaders.lakehouse_loader:ISSUER_CIKS_BY_TICKER": _CONSTANT,
    "agentic_trading.data.loaders.lakehouse_loader:_ADJUSTMENT_TABLE_ROWS": _LAKE_TIME_MEMO,
    "agentic_trading.data.loaders.lakehouse_loader:_FIRST_DOLTHUB_CHAIN_DATES": _LAKE_TIME_MEMO,
    "agentic_trading.data.loaders.lakehouse_loader:_SP500_COVERAGE": _STORE_SOURCE,
    "agentic_trading.data.loaders.lakehouse_loader:_SECURITY_CLOSES": _STORE_SOURCE,
    "agentic_trading.data.loaders.lakehouse_loader:_SECURITY_BARS": _STORE_SOURCE,
    "agentic_trading.data.loaders.lakehouse_loader:_SPLIT_RATIOS": _STORE_SOURCE,
    "agentic_trading.data.loaders.lakehouse_loader:_CHAIN_MONTHS": _STORE_SOURCE,
    "agentic_trading.data.loaders.lakehouse_loader:_COT_REPORTS": _STORE_SOURCE,
    "agentic_trading.data.loaders.lakehouse_loader:_DIVIDENDS": _STORE_SOURCE,
    "agentic_trading.data.loaders.lakehouse_loader:_EARNINGS": _STORE_SOURCE,
    "agentic_trading.data.loaders.lakehouse_loader:_SIC_CODES": _STORE_SOURCE,
    "agentic_trading.data.loaders.lakehouse_loader:_EQUITY_PRICES": _STORE_SOURCE,
    "agentic_trading.data.loaders.lakehouse_loader:_CRYPTO_PRICES": _STORE_SOURCE,
    "agentic_trading.data.loaders.lakehouse_loader:_FUTURES_DAILY": _STORE_SOURCE,
    "agentic_trading.data.loaders.lakehouse_loader:_RAW_CLOSES": _STORE_SOURCE,
    "agentic_trading.data.loaders.lakehouse_loader:_FRED_VERSIONS": _STORE_SOURCE,
    "agentic_trading.data.loaders.lakehouse_loader:_POLICY_RATES": _STORE_SOURCE,
    "agentic_trading.data.loaders.lakehouse_loader:_KEN_FRENCH_ROWS": _STORE_SOURCE,
    "agentic_trading.data.loaders.lakehouse_loader:_COMPANY_FACTS": _STORE_SOURCE,
    "agentic_trading.data.loaders.lakehouse_loader:_SHORT_INTEREST": _STORE_SOURCE,
    "agentic_trading.data.loaders.lakehouse_loader:_INSIDER_TRANSACTIONS": _STORE_SOURCE,
    "agentic_trading.data.loaders.option_marks:_CHAIN_CACHE": _OPTION_MARK_CACHE,
    "agentic_trading.data.loaders.option_marks:_RATE_CACHE": _OPTION_MARK_CACHE,
    "agentic_trading.data.loaders.option_marks:_SETTLEMENT_CLOSE_CACHE": _OPTION_MARK_CACHE,
    "agentic_trading.data.loaders.option_marks:_QUOTE_CACHE": _OPTION_MARK_CACHE,
    "agentic_trading.data.loaders.option_marks:_cache_lake_time": (
        "the lake time the option-mark caches were read at, used only to empty them when the "
        "lake time changes"
    ),
    "agentic_trading.data.trading_calendar:_sessions": (
        "exchange-calendar sessions over a fixed range, memoized per calendar name; a pure "
        "function of its argument"
    ),
}

# "<module>:<name>": the issue tracking process state whose value after a resume differs from an
# unbroken run's.
KNOWN_NOT_RESUME_SAFE: dict[str, str] = {
    "agentic_trading.data.loaders.lakehouse_loader:_UNFACTORED_SPLIT_MOVES": "AT-145: diagnostic split-move registry; a resumed process returns only the moves its own reads flagged",  # noqa: E501
}


def _module_names() -> list[str]:
    """Every module under `_PACKAGES`, the packages themselves included."""
    names: list[str] = []
    for package_name in _PACKAGES:
        package = importlib.import_module(package_name)
        names.append(package_name)
        names.extend(
            info.name for info in pkgutil.walk_packages(package.__path__, f"{package_name}.")
        )
    return names


def _assigned_names(target: ast.expr) -> list[str]:
    """The names an assignment target binds: a name, or the names inside a tuple/list unpack."""
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, ast.Starred):
        return _assigned_names(target.value)
    if isinstance(target, ast.Tuple | ast.List):
        return [name for element in target.elts for name in _assigned_names(element)]
    return []


def _defined_names(body: list[ast.stmt]) -> set[str]:
    """Names bound at module level by an assignment or a `def`, including inside top-level
    `if`/`try`/`with` blocks. Imports and classes are not included."""
    names: set[str] = set()
    for node in body:
        if isinstance(node, ast.Assign):
            names.update(name for target in node.targets for name in _assigned_names(target))
        elif isinstance(node, ast.AnnAssign | ast.AugAssign):
            names.update(_assigned_names(node.target))
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            names.add(node.name)
        elif isinstance(node, ast.If):
            names |= _defined_names(node.body) | _defined_names(node.orelse)
        elif isinstance(node, ast.Try | ast.TryStar):
            names |= _defined_names(node.body) | _defined_names(node.orelse)
            names |= _defined_names(node.finalbody)
            for handler in node.handlers:
                names |= _defined_names(handler.body)
        elif isinstance(node, ast.With):
            names |= _defined_names(node.body)
    return names


def _is_cache_decorator(decorator: ast.expr) -> bool:
    """Whether `decorator` is `cache` or `lru_cache`, bare, called or as a `functools.`
    attribute."""
    if isinstance(decorator, ast.Call):
        decorator = decorator.func
    if isinstance(decorator, ast.Attribute):
        return decorator.attr in {"cache", "lru_cache"}
    return isinstance(decorator, ast.Name) and decorator.id in {"cache", "lru_cache"}


def _cached_functions(body: list[ast.stmt]) -> set[str]:
    """Names of the module-level functions decorated with `cache` or `lru_cache`, read from the
    source, so a cached function a test fixture has replaced is still found."""
    return {
        node.name
        for node in body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and any(_is_cache_decorator(decorator) for decorator in node.decorator_list)
    }


def _rebound_globals(tree: ast.Module) -> set[str]:
    """Names any function in the module declares `global`, so it can rebind them."""
    return {name for node in ast.walk(tree) if isinstance(node, ast.Global) for name in node.names}


def _is_process_state(value: Any) -> bool:
    """Whether `value` can change after import: a `functools.cache`/`lru_cache` function, or any
    value but `None`, an `_IMMUTABLE_TYPES` value, a typing construct, a bare `object()`
    sentinel, a frozen pydantic model or a frozen dataclass."""
    if hasattr(value, "cache_info") and hasattr(value, "cache_clear"):
        return True
    if value is None or isinstance(value, _IMMUTABLE_TYPES):
        return False
    if type(value).__module__ in {"typing", "typing_extensions"} or type(value) is object:
        return False
    if isinstance(value, BaseModel) and value.model_config.get("frozen"):
        return False
    return not (
        dataclasses.is_dataclass(value) and type(value).__dataclass_params__.frozen  # type: ignore[attr-defined]
    )


def _process_state(module: ModuleType, source: str) -> list[str]:
    """The names of the process state `module`, whose source is `source`, defines: every defined
    name whose value `_is_process_state`, every function it decorates with `cache` or
    `lru_cache`, and every name it rebinds with `global`."""
    tree = ast.parse(source)
    defined = {
        name
        for name in _defined_names(tree.body)
        if hasattr(module, name) and _is_process_state(getattr(module, name))
    }
    return sorted(defined | _cached_functions(tree.body) | _rebound_globals(tree))


def _census() -> set[str]:
    """`<module>:<name>` for the process state every module under `_PACKAGES` defines."""
    found: set[str] = set()
    for name in _module_names():
        module = importlib.import_module(name)
        assert module.__file__ is not None
        source = pathlib.Path(module.__file__).read_text(encoding="utf-8")
        found.update(f"{name}:{state}" for state in _process_state(module, source))
    return found


def test_census_is_exactly_the_two_lists() -> None:
    """The process state the packages define is exactly `_ALLOWED` and `KNOWN_NOT_RESUME_SAFE`
    together: new state must be listed, and a listed object that is gone must be removed."""
    found = _census()
    listed = set(_ALLOWED) | set(KNOWN_NOT_RESUME_SAFE)
    unlisted = sorted(found - listed)
    assert not unlisted, (
        "module-level process state is not carried across a resume; move it onto the agent as a "
        "pydantic private attribute, or add it to _ALLOWED with the reason it is resume-safe: "
        + ", ".join(unlisted)
    )
    stale_allowed = sorted(set(_ALLOWED) - found)
    assert not stale_allowed, (
        f"_ALLOWED names objects the census no longer finds: {', '.join(stale_allowed)}"
    )
    stale_known = sorted(set(KNOWN_NOT_RESUME_SAFE) - found)
    assert not stale_known, (
        "KNOWN_NOT_RESUME_SAFE names objects the census no longer finds: " + ", ".join(stale_known)
    )


def test_allowed_and_known_not_resume_safe_are_disjoint() -> None:
    """No object is both allowed as resume-safe and known not to be."""
    assert not set(_ALLOWED) & set(KNOWN_NOT_RESUME_SAFE)


_PROBE_SOURCE = """
import functools
import logging
import re
from enum import Enum

from pydantic import BaseModel

from agentic_trading.agents._market_math import RefitCache
from agentic_trading.data.loaders.option_marks import _RATE_CACHE


class _Frozen(BaseModel):
    model_config = {"frozen": True}


class _Colour(Enum):
    RED = 1


_GARCH_REFIT_CACHE = RefitCache()
_TABLE = {"a": 1}
_COUNTS: list[int] = []
_SEEN, _ROWS = set(), bytearray()
try:
    _IN_TRY = []
except ImportError:
    pass
_state = None
_FROZEN_SET = frozenset({1})
_TUPLE = (1, 2)
_PATTERN = re.compile("x")
_LOGGER = logging.getLogger("probe")
_FROZEN_MODEL = _Frozen()
_RED = _Colour.RED
_SENTINEL = object()
_TYPE_ALIAS = dict[str, int]


@functools.cache
def _cached() -> int:
    return 1


def _set_state() -> None:
    global _state
    _state = 1


def _plain() -> int:
    return 1
"""


def test_census_finds_each_kind_of_process_state() -> None:
    """Mutable containers, a `RefitCache`, a cached function and a `global`-rebound name defined
    by a module are found; imported, immutable and frozen objects are not."""
    module = ModuleType("process_state_probe")
    exec(compile(_PROBE_SOURCE, "process_state_probe.py", "exec"), module.__dict__)

    assert _process_state(module, _PROBE_SOURCE) == sorted(
        [
            "_COUNTS",
            "_GARCH_REFIT_CACHE",
            "_IN_TRY",
            "_ROWS",
            "_SEEN",
            "_TABLE",
            "_cached",
            "_state",
        ]
    )
