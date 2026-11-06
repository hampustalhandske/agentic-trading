# at126_ab.py

"""A/B check of the loaders' window store against the per-call loaders.

`loaders` mode: inside one `lake_session` pinned to one lake time, calls
every stored public loader for each decision day (the XNYS sessions from
`--start` to `--end`, every `--every`th) and each ticker, security, FRED
series, option contract, futures ticker and policy-rate currency, in four
passes: the frozen per-call reference loaders (`at126_legacy.py`) twice
("legacy", "legacy_repeat"), the current loaders with no window declared
("direct") and the current loaders with the window declared ("store").
A direct or store answer matches when it equals the reference exactly:
frames and series by `assert_frame_equal`/`assert_series_equal` with
`check_exact=True`, floats bit for bit (NaN equal to NaN), `None` only
where the reference gives `None`, and a raised error by type and message.

Acceptance: a mismatch is allowed only when the reference's two runs of
that call differ as well and the difference traces to a tie
(`trace_tie`): at every row where the answers differ, both rows hold
the same values in every column of the reference query's ORDER BY
(`TIE_COLUMNS`). Any other mismatch fails its loader, as does a
judged path's unfactored split-move registry differing from the
reference's. Prints JSON with, per loader, the calls, the reference's
self-mismatches, the mismatches and allowed ones per path, whether any
is float-only (equal within `FLOAT_ONLY_RTOL`, as a reordered split
product is), the `/v1/sql` queries and wall seconds per path and the
verdict; the allowed ties per loader, path and key with their count; and
the first mismatches in full. Then prints `PASS` or `FAIL` per loader and
`OVERALL PASS|FAIL`; exits 1 on an overall FAIL.

`backtest` mode: runs the single-phase CLI backtest twice at one lake
time (the given `--lake-time`, or the default T of a session opened now),
once with `--no-loader-store` and once with `--loader-store`, and compares
the two runs' `result` (the BacktestResult JSON) byte for byte. Prints
JSON with each run's wall seconds, total queries and per-loader query
counts, and the first differing line; exits 1 when the results differ.

Usage:
  uv run python scripts/at126_ab.py loaders --start 2019-01-02 --end 2019-06-28 \\
      --ticker AAPL --ticker MSFT --ticker SPY [--lake-time T] [--every 5] [--out PATH]
  uv run python scripts/at126_ab.py backtest SPY TLT GLD --start-date 2019-01-02 \\
      --end-date 2019-06-28 [--lake-time T] [--out PATH] [<extra CLI options, e.g. --set g.f=v>]
"""

import argparse
import json
import math
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from at126_legacy import LEGACY_LOADERS
from pydantic import BaseModel, ConfigDict

from agentic_trading.cli import LOADER_QUERY_COUNTS_PREFIX
from agentic_trading.config import settings
from agentic_trading.data.instruments import currency_pair, futures_root, is_marked
from agentic_trading.data.loaders import lakehouse_loader as ll
from agentic_trading.data.trading_calendar import sessions_between

RunPath = Literal["legacy", "legacy_repeat", "direct", "store"]
PATHS: tuple[RunPath, ...] = ("legacy", "legacy_repeat", "direct", "store")
# The paths judged against the "legacy" pass.
JUDGED: tuple[RunPath, ...] = ("direct", "store")

# FRED series the roster's analysts and the option marks read.
DEFAULT_FRED_SERIES = ("DGS10", "DGS3MO", "BAMLH0A0HYM2")
# Calendar days of history each ranged call asks for: the longest lookback
# an analyst passes to a price or bar loader.
DEFAULT_PRICE_LOOKBACK_DAYS = 400
# Calendar days the option marks' settlement read looks back for a close.
DEFAULT_RAW_CLOSE_LOOKBACK_DAYS = 10
# Mismatches printed in full after the summary.
MISMATCHES_SHOWN = 20
# Relative tolerance under which a mismatch counts as float-only (the
# values agree to within rounding, as a reordered product does).
FLOAT_ONLY_RTOL = 1e-12

# Per loader, the columns of `tie_rows(answer)` that carry the reference
# query's ORDER BY: two answers that differ only in rows equal on these
# columns differ by how the lake ordered rows that tie on that ORDER BY.
# `@index0` is the answer's first index level and `@key` a dict answer's
# key. An empty tuple marks a query with no ORDER BY at all (every row
# ties); `None` marks a loader whose answer does not carry its ORDER BY
# columns, so no difference in it can be traced to a tie.
TIE_COLUMNS: dict[str, tuple[str, ...] | None] = {
    "load_sp500_member_securities_asof": None,
    "load_security_log_returns": ("@key", "@index0"),
    "load_security_daily_bars": ("@index0",),
    "split_adjustment_factor": (),
    "load_options_snapshot_from_lakehouse": (
        "date", "expiration", "option_type", "strike", "bid", "ask", "implied_volatility",
        "delta", "gamma", "theta", "vega", "rho",
    ),
    "load_option_quote_from_lakehouse": ("date",),
    "load_cot_reports_from_lakehouse": ("@index0",),
    "load_cot_positioning_from_lakehouse": ("@index0",),
    "load_dividends_from_lakehouse": ("@index0",),
    "load_earnings_surprise_from_lakehouse": ("item2",),
    "load_sic_code_from_lakehouse": (),
    "load_price_history_from_lakehouse": ("@index0",),
    "load_futures_price_history_from_lakehouse": ("@index0",),
    "load_raw_close_from_lakehouse": None,
    "load_fred_vintages_from_lakehouse": ("@index0", "realtime_start"),
    "load_fred_series_from_lakehouse": ("@index0",),
    "load_fama_french_factors_from_lakehouse": ("@index0",),
    "load_policy_rates_from_lakehouse": ("@index0",),
    "load_company_facts_from_lakehouse": (
        "filed", "concept", "unit", "period_start", "period_end", "fiscal_year",
        "fiscal_period", "form", "value",
    ),
    "load_short_interest_from_lakehouse": ("settlement_date",),
    "load_insider_transactions_from_lakehouse": (
        "filed_date", "transaction_date", "owner_cik", "owner_name", "transaction_code",
        "acquired_disposed", "shares", "price_per_share", "is_officer", "is_director",
        "is_ten_percent_owner",
    ),
}  # fmt: skip


# ---------------------------------------------------------------- comparison


def _float_difference(expected: float, actual: float, rtol: float | None) -> str | None:
    if math.isnan(expected) and math.isnan(actual):
        return None
    if np.float64(expected).tobytes() == np.float64(actual).tobytes():
        return None
    if rtol is not None and math.isclose(expected, actual, rel_tol=rtol, abs_tol=0.0):
        return None
    return f"{expected!r} != {actual!r}"


def compare(expected: Any, actual: Any, path: str = "", rtol: float | None = None) -> str | None:
    """`None` when `actual` is exactly `expected`, else a description of
    the first difference found, prefixed with its `path` inside the value.

    Two values match when they have the same type and: for DataFrames and
    Series, `pandas.testing` finds them equal with `check_exact=True`
    (values, dtypes, index, columns and names); for floats, the same bits
    or both NaN; for dicts, the same keys in the same order and matching
    values; for lists and tuples, the same length and matching items; for
    Pydantic models, matching `model_dump()`s; for Timestamps, equal and of
    the same unit; anything else, `==`. With `rtol`, floats (in frames and
    series too) need only agree to that relative tolerance."""
    where = path or "value"
    if expected is None or actual is None:
        if expected is None and actual is None:
            return None
        return f"{where}: expected {_short(expected)}, got {_short(actual)}"
    if type(expected) is not type(actual):
        return f"{where}: expected type {type(expected).__name__}, got {type(actual).__name__}"
    exact = rtol is None
    tolerance = {} if exact else {"rtol": rtol, "atol": 0.0}
    if isinstance(expected, pd.DataFrame):
        try:
            pd.testing.assert_frame_equal(expected, actual, check_exact=exact, **tolerance)
        except AssertionError as error:
            return f"{where}: {error}"
        return None
    if isinstance(expected, pd.Series):
        try:
            pd.testing.assert_series_equal(expected, actual, check_exact=exact, **tolerance)
        except AssertionError as error:
            return f"{where}: {error}"
        return None
    if isinstance(expected, float):
        difference = _float_difference(expected, actual, rtol)
        return None if difference is None else f"{where}: {difference}"
    if isinstance(expected, dict):
        if list(expected) != list(actual):
            return f"{where}: keys {list(expected)!r} != {list(actual)!r}"
        for key in expected:
            difference = compare(expected[key], actual[key], f"{where}[{key!r}]", rtol)
            if difference is not None:
                return difference
        return None
    if isinstance(expected, list | tuple):
        if len(expected) != len(actual):
            return f"{where}: length {len(expected)} != {len(actual)}"
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            difference = compare(left, right, f"{where}[{index}]", rtol)
            if difference is not None:
                return difference
        return None
    if isinstance(expected, BaseModel):
        return compare(expected.model_dump(), actual.model_dump(), where, rtol)
    if isinstance(expected, pd.Timestamp):
        if expected != actual or expected.unit != actual.unit:
            return f"{where}: {expected!r} != {actual!r}"
        return None
    if expected != actual:
        return f"{where}: {expected!r} != {actual!r}"
    return None


def _short(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, pd.DataFrame | pd.Series):
        return f"{type(value).__name__} of {len(value)} rows"
    return repr(value)[:200]


# ---------------------------------------------------------------- tie tracing


def tie_rows(answer: Any) -> pd.DataFrame | None:
    """A loader answer as rows: a frame with its index levels as columns
    `@index0`, `@index1`, ... (and MultiIndex columns joined with `|`); a
    series likewise, its values in `value`; a dict of those stacked with
    the key in `@key`; a tuple as one row of `item0`, `item1`, ...; a list
    of models as one row per `model_dump()`; any other value as one row
    of `value`. `None` for `None` and a raised error."""
    if answer is None or isinstance(answer, RaisedError):
        return None
    if isinstance(answer, pd.Series):
        return tie_rows(answer.to_frame("value"))
    if isinstance(answer, pd.DataFrame):
        frame = answer.copy()
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = ["|".join(str(level) for level in column) for column in frame.columns]
        names = [f"@index{level}" for level in range(frame.index.nlevels)]
        return frame.reset_index(names=names)
    if isinstance(answer, dict):
        parts = []
        for key, value in answer.items():
            rows = tie_rows(value)
            if rows is None:
                return None
            parts.append(rows.assign(**{"@key": key}))
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame({"@key": []})
    if isinstance(answer, tuple):
        return pd.DataFrame([{f"item{i}": item for i, item in enumerate(answer)}])
    if isinstance(answer, list):
        return pd.DataFrame(
            [item.model_dump() if isinstance(item, BaseModel) else {"value": item}
             for item in answer]
        )  # fmt: skip
    return pd.DataFrame({"value": [answer]})


def _cells_equal(left: pd.Series, right: pd.Series) -> np.ndarray:
    """Per cell, whether two aligned columns hold the same value (NaN equal to NaN)."""
    same = (left == right).fillna(False).to_numpy(dtype=bool)
    both_missing = (left.isna() & right.isna()).to_numpy(dtype=bool)
    return same | both_missing


def trace_tie(expected: Any, actual: Any, tie_columns: tuple[str, ...] | None) -> str | None:
    """`None` when `actual` differs from `expected` only in rows that tie
    on the reference query's ORDER BY: both answers have the same rows and
    columns with the same dtypes, and at every row position where they
    differ, the two rows hold the same values in every one of
    `tie_columns` (any rows when the query has no ORDER BY, `()`).
    Otherwise the reason the difference cannot be traced to a tie."""
    if tie_columns is None:
        return "the answer does not carry the ORDER BY columns"
    left, right = tie_rows(expected), tie_rows(actual)
    if left is None or right is None:
        return "one answer is None or an error"
    if list(left.columns) != list(right.columns):
        return f"columns differ: {list(left.columns)} != {list(right.columns)}"
    if len(left) != len(right):
        return f"row count differs: {len(left)} != {len(right)}"
    if list(left.dtypes) != list(right.dtypes):
        return "dtypes differ"
    missing = [column for column in tie_columns if column not in left.columns]
    if missing:
        return f"ORDER BY columns {missing} are not in the answer"
    equal = {column: _cells_equal(left[column], right[column]) for column in left.columns}
    differing = ~np.logical_and.reduce([equal[column] for column in left.columns])
    for position in np.flatnonzero(differing):
        for column in tie_columns:
            if not equal[column][position]:
                return f"row {position} differs on ORDER BY column {column!r}"
    return None


# ---------------------------------------------------------------- loaders mode


class LoaderCall(BaseModel):
    """One public loader call: the loader's name in `lakehouse_loader`
    (and `LEGACY_LOADERS`), its positional `args`, and the decision `day`
    it belongs to (calls run in day order)."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    loader: str
    args: tuple[Any, ...]
    day: date

    @property
    def key(self) -> str:
        """The call's arguments other than its dates, as text."""
        return repr(tuple(arg for arg in self.args if not isinstance(arg, date)))


class RaisedError(BaseModel):
    """An exception a loader call raised, kept as its type name and message."""

    type: str
    message: str


class PassResult(BaseModel):
    """What one path returned for each call, in call order, with the
    split-move registry it left, its query counts and wall seconds per
    loader."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    path: str
    outputs: list[Any]
    registry: list[Any]
    counts: list[ll.LoaderQueryCount]
    seconds: dict[str, float]


class Mismatch(BaseModel):
    """A call whose `path` answer differs from the reference's.

    `legacy_varies` says whether the reference's two runs of the call
    differ too; `tie` is `None` when the difference traces to rows tying
    on the reference ORDER BY, else why it does not; `float_only` says
    whether the answers agree within `FLOAT_ONLY_RTOL`. The mismatch is
    `allowed` only when the reference varies and the tie traces."""

    loader: str
    key: str
    day: date
    path: str
    difference: str
    legacy_varies: bool
    tie: str | None
    float_only: bool

    @property
    def allowed(self) -> bool:
        return self.legacy_varies and self.tie is None


class AllowedTie(BaseModel):
    """Allowed differences of one loader, path and key, and how many calls had one."""

    loader: str
    path: str
    key: str
    count: int


class LoaderSummary(BaseModel):
    """Per loader: calls made, the reference's self-mismatches, the
    mismatches per judged path (all, and allowed ones), whether any is
    float-only, the `/v1/sql` queries and wall seconds per path, and the
    verdict (`PASS` when every mismatch is allowed)."""

    loader: str
    calls: int
    legacy_self_mismatches: int
    mismatches: dict[str, int]
    allowed: dict[str, int]
    float_only: bool
    queries: dict[str, int]
    seconds: dict[str, float]
    verdict: Literal["PASS", "FAIL"]


class LoadersReport(BaseModel):
    """The `loaders` mode result: the window, every loader's summary, the
    allowed tie differences per loader and key, the registry comparison
    per path against the reference, every mismatch, and the overall
    verdict (`PASS` when every loader passes and the judged paths'
    registries match the reference's)."""

    window: ll.LoaderWindow
    days: int
    loaders: list[LoaderSummary]
    allowed_ties: list[AllowedTie]
    registry_differences: dict[str, str | None]
    mismatches: list[Mismatch]

    @property
    def verdict(self) -> Literal["PASS", "FAIL"]:
        registries_match = all(self.registry_differences.get(path) is None for path in JUDGED)
        loaders_pass = all(summary.verdict == "PASS" for summary in self.loaders)
        return "PASS" if registries_match and loaders_pass else "FAIL"

    @property
    def ok(self) -> bool:
        return self.verdict == "PASS"


def _loader(path: RunPath, name: str) -> Callable[..., Any]:
    if path in ("legacy", "legacy_repeat"):
        return ll._counted(name)(LEGACY_LOADERS[name])
    function: Callable[..., Any] = getattr(ll, name)
    return function


def run_pass(path: RunPath, calls: Sequence[LoaderCall], window: ll.LoaderWindow) -> PassResult:
    """Runs every call on `path` inside the open session: "store" with
    `window` declared, the others ("legacy" and "legacy_repeat" with the
    reference loaders) with no window. Starts from an empty split-move
    registry and zeroed query counts; leaves no window declared."""
    ll.clear_loader_window()
    if path == "store":
        ll.declare_loader_window(window.start, window.end)
    ll.clear_unfactored_split_move_registry()
    ll.reset_loader_query_counts()
    outputs: list[Any] = []
    seconds: dict[str, float] = {}
    try:
        for call in calls:
            function = _loader(path, call.loader)
            started = time.perf_counter()
            try:
                output: Any = function(*call.args)
            except Exception as error:
                output = RaisedError(type=type(error).__name__, message=str(error))
            seconds[call.loader] = seconds.get(call.loader, 0.0) + time.perf_counter() - started
            outputs.append(output)
    finally:
        ll.clear_loader_window()
    return PassResult(
        path=path,
        outputs=outputs,
        registry=ll.unfactored_split_move_registry(),
        counts=ll.loader_query_counts(),
        seconds=seconds,
    )


def judge(call: LoaderCall, path: str, expected: Any, repeat: Any, actual: Any) -> Mismatch | None:
    """The mismatch of `actual` (a judged path's answer to `call`) against
    the reference answer `expected`, judged with the reference's second
    answer `repeat`, or `None` when they are equal."""
    difference = compare(expected, actual)
    if difference is None:
        return None
    return Mismatch(
        loader=call.loader,
        key=call.key[:300],
        day=call.day,
        path=path,
        difference=difference[:2000],
        legacy_varies=compare(expected, repeat) is not None,
        tie=trace_tie(expected, actual, TIE_COLUMNS.get(call.loader)),
        float_only=compare(expected, actual, rtol=FLOAT_ONLY_RTOL) is None,
    )


def compare_passes(
    calls: Sequence[LoaderCall], passes: dict[RunPath, PassResult], window: ll.LoaderWindow
) -> LoadersReport:
    """Judges the "direct" and "store" passes against the "legacy" pass,
    call by call and on the registry, with "legacy_repeat" telling where
    the reference varies against itself, and tallies the counts per
    loader."""
    reference, repeat = passes["legacy"], passes["legacy_repeat"]
    mismatches: list[Mismatch] = []
    for path in JUDGED:
        for index, call in enumerate(calls):
            mismatch = judge(
                call,
                path,
                reference.outputs[index],
                repeat.outputs[index],
                passes[path].outputs[index],
            )
            if mismatch is not None:
                mismatches.append(mismatch)
    self_mismatches: dict[str, int] = {}
    for index, call in enumerate(calls):
        if compare(reference.outputs[index], repeat.outputs[index]) is not None:
            self_mismatches[call.loader] = self_mismatches.get(call.loader, 0) + 1
    queries: dict[str, dict[str, int]] = {}
    for path, result in passes.items():
        for count in result.counts:
            per_loader = queries.setdefault(count.loader, {})
            per_loader[path] = per_loader.get(path, 0) + count.queries
    loaders = sorted({call.loader for call in calls} | set(queries))
    summaries = []
    for name in loaders:
        own = [mismatch for mismatch in mismatches if mismatch.loader == name]
        summaries.append(
            LoaderSummary(
                loader=name,
                calls=sum(call.loader == name for call in calls),
                legacy_self_mismatches=self_mismatches.get(name, 0),
                mismatches={path: sum(m.path == path for m in own) for path in JUDGED},
                allowed={path: sum(m.path == path and m.allowed for m in own) for path in JUDGED},
                float_only=any(m.float_only for m in own),
                queries={path: queries.get(name, {}).get(path, 0) for path in PATHS},
                seconds={path: round(passes[path].seconds.get(name, 0.0), 3) for path in PATHS},
                verdict="PASS" if all(m.allowed for m in own) else "FAIL",
            )
        )
    ties: dict[tuple[str, str, str], int] = {}
    for mismatch in mismatches:
        if mismatch.allowed:
            tie_key = (mismatch.loader, mismatch.path, mismatch.key)
            ties[tie_key] = ties.get(tie_key, 0) + 1
    return LoadersReport(
        window=window,
        days=len({call.day for call in calls}),
        loaders=summaries,
        allowed_ties=[
            AllowedTie(loader=loader, path=path, key=key, count=count)
            for (loader, path, key), count in sorted(ties.items())
        ],
        registry_differences={
            path: compare(reference.registry, passes[path].registry, "registry")
            for path in ("legacy_repeat", *JUDGED)
        },
        mismatches=mismatches,
    )


class CallPlan(BaseModel):
    """What `build_calls` asks about: the decision days, the tickers (used
    for every per-ticker loader), the security ids (for the bar loaders),
    the FRED series, Ken French models, COT tickers, option underlyings
    and option contracts `(underlying, expiration, option_type, strike)`,
    the futures tickers, the policy-rate currencies, and the lookbacks."""

    days: list[date]
    tickers: list[str]
    security_ids: list[int]
    fred_series: list[str]
    fama_french_models: list[str]
    cot_tickers: list[str]
    option_underlyings: list[str]
    contracts: list[tuple[str, date, str, float]]
    futures_tickers: list[str] = []
    currencies: list[str] = []
    price_lookback_days: int = DEFAULT_PRICE_LOOKBACK_DAYS
    raw_close_lookback_days: int = DEFAULT_RAW_CLOSE_LOOKBACK_DAYS
    quote_lookback_days: int = settings.options.max_mark_age_days


def build_calls(plan: CallPlan) -> list[LoaderCall]:
    """Every call of `plan`, day by day in a fixed loader order, and the
    SIC reads (no knowledge time) on the first day."""
    calls: list[LoaderCall] = []
    first = plan.days[0]
    for ticker in plan.tickers:
        calls.append(LoaderCall(loader="load_sic_code_from_lakehouse", args=(ticker,), day=first))
    for day in plan.days:
        start = day - timedelta(days=plan.price_lookback_days)

        def add(loader: str, *args: Any, on: date = day) -> None:
            calls.append(LoaderCall(loader=loader, args=args, day=on))

        add("load_sp500_member_securities_asof", day)
        if plan.security_ids:
            add("load_security_log_returns", list(plan.security_ids), start, day)
            add("load_security_daily_bars", list(plan.security_ids), start, day)
        if plan.tickers:
            add("load_price_history_from_lakehouse", list(plan.tickers), start, day)
        if plan.futures_tickers:
            add("load_futures_price_history_from_lakehouse", list(plan.futures_tickers),
                start, day)  # fmt: skip
        if plan.currencies:
            add("load_policy_rates_from_lakehouse", list(plan.currencies), start, day)
        for ticker in plan.tickers:
            add("load_raw_close_from_lakehouse", ticker, day, plan.raw_close_lookback_days)
            add("load_dividends_from_lakehouse", ticker, day)
            add("load_earnings_surprise_from_lakehouse", ticker, day)
            add("load_company_facts_from_lakehouse", ticker, day)
            add("load_short_interest_from_lakehouse", ticker, day)
            add("load_insider_transactions_from_lakehouse", ticker, day)
        for series_id in plan.fred_series:
            add("load_fred_vintages_from_lakehouse", series_id, day)
            add("load_fred_series_from_lakehouse", series_id, day)
        for model in plan.fama_french_models:
            add("load_fama_french_factors_from_lakehouse", day, model)
        for ticker in plan.cot_tickers:
            add("load_cot_reports_from_lakehouse", ticker, day)
            add("load_cot_positioning_from_lakehouse", ticker, day)
        for underlying in plan.option_underlyings:
            add("split_adjustment_factor", underlying, day)
            add("load_options_snapshot_from_lakehouse", underlying, day)
        for underlying, expiration, option_type, strike in plan.contracts:
            add(
                "load_option_quote_from_lakehouse",
                underlying,
                expiration,
                option_type,
                strike,
                day,
                plan.quote_lookback_days,
            )
    return calls


def plan_window(plan: CallPlan) -> ll.LoaderWindow:
    """The window a run over `plan` declares: its first day minus the
    longest lookback, through its last day."""
    longest = max(
        plan.price_lookback_days + ll.POLICY_RATE_LOOKBACK_DAYS,
        plan.raw_close_lookback_days,
        plan.quote_lookback_days,
        ll.DEFAULT_COT_LOOKBACK_DAYS,
        ll.DEFAULT_OPTIONS_SNAPSHOT_LOOKBACK_DAYS,
    )
    return ll.LoaderWindow(start=plan.days[0] - timedelta(days=longest), end=plan.days[-1])


def discover_contracts(underlyings: Sequence[str], day: date) -> list[tuple[str, date, str, float]]:
    """One contract per underlying to quote: the middle row of the
    reference snapshot on `day`, plus the same expiry and type at a strike
    no chain lists (so the quote path's `None` is exercised)."""
    contracts: list[tuple[str, date, str, float]] = []
    for underlying in underlyings:
        snapshot = LEGACY_LOADERS["load_options_snapshot_from_lakehouse"](underlying, day)
        if snapshot is None:
            continue
        row = snapshot.iloc[len(snapshot) // 2]
        expiration = pd.Timestamp(row["expiration"]).date()
        contracts.append((underlying, expiration, str(row["option_type"]), float(row["strike"])))
        contracts.append((underlying, expiration, str(row["option_type"]), 1.0e9))
    return contracts


def run_loaders_ab(plan: CallPlan) -> LoadersReport:
    """Runs the four passes of `plan` inside the open session and
    judges them."""
    calls = sorted(build_calls(plan), key=lambda call: call.day)
    window = plan_window(plan)
    passes = {path: run_pass(path, calls, window) for path in PATHS}
    return compare_passes(calls, passes, window)


def _member_ids(tickers: Sequence[str], day: date) -> list[int]:
    members = LEGACY_LOADERS["load_sp500_member_securities_asof"](day) or []
    wanted = set(tickers)
    return sorted({member.security_id for member in members if member.ticker in wanted})


def _loaders_main(args: argparse.Namespace) -> int:
    lake_time = None if args.lake_time is None else datetime.fromisoformat(args.lake_time)
    days = sessions_between(date.fromisoformat(args.start), date.fromisoformat(args.end))
    days = days[:: args.every]
    if not days:
        raise SystemExit("no XNYS sessions in the range")
    tickers = list(dict.fromkeys(args.ticker))
    with ll.lake_session(lake_time) as session:
        security_ids = args.security_id or _member_ids(tickers, days[0])
        cot_tickers = [
            ticker
            for ticker in dict.fromkeys(
                [ll.DEFAULT_COT_TICKER_BY_ASSET.get(ticker, ticker) for ticker in tickers]
            )
            if ticker in ll._COT_DISAGGREGATED_CODES or ticker in ll._COT_TFF_CODES
        ]
        underlyings = args.option_underlying or [
            ticker for ticker in tickers if not is_marked(ticker)
        ]
        pairs = [pair for ticker in tickers if (pair := currency_pair(ticker)) is not None]
        plan = CallPlan(
            days=days,
            tickers=tickers,
            security_ids=security_ids,
            fred_series=args.fred_series or list(DEFAULT_FRED_SERIES),
            fama_french_models=["3factor", "5factor"],
            cot_tickers=cot_tickers,
            option_underlyings=underlyings,
            contracts=discover_contracts(underlyings, days[0]),
            futures_tickers=[ticker for ticker in tickers if futures_root(ticker)],
            currencies=sorted({code for pair in pairs for code in (pair.base, pair.quote)}),
            price_lookback_days=args.price_lookback_days,
        )
        report = run_loaders_ab(plan)
        lake_time_used = session.lake_time.isoformat()
    payload = {
        "lake_time": lake_time_used,
        "verdict": report.verdict,
        "window": report.window.model_dump(mode="json"),
        "days": report.days,
        "loaders": [summary.model_dump() for summary in report.loaders],
        "allowed_ties": [tie.model_dump() for tie in report.allowed_ties],
        "float_only_loaders": [s.loader for s in report.loaders if s.float_only],
        "registry_differences": report.registry_differences,
        "mismatch_count": len(report.mismatches),
        "mismatches": [
            {**item.model_dump(mode="json"), "allowed": item.allowed}
            for item in report.mismatches[:MISMATCHES_SHOWN]
        ],
    }
    _emit(payload, args.out)
    for summary in report.loaders:
        print(f"{summary.verdict} {summary.loader}")
    print(f"OVERALL {report.verdict}")
    return 0 if report.ok else 1


# ---------------------------------------------------------------- backtest mode


class CliRun(BaseModel):
    """One CLI backtest subprocess: its exit code, output and wall seconds."""

    store: bool
    returncode: int
    stdout: str
    stderr: str
    seconds: float


def result_text(stdout: str) -> str:
    """The `result` of the CLI's final JSON payload (`{"provenance": ...,
    "result": ...}`, printed from a line holding only `{`), serialised
    with `indent=2` as the CLI prints it. Raises `ValueError` when stdout
    holds no such payload."""
    lines = stdout.splitlines()
    for index in range(len(lines) - 1, -1, -1):
        if lines[index] != "{":
            continue
        try:
            payload = json.loads("\n".join(lines[index:]))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "result" in payload:
            return json.dumps(payload["result"], indent=2)
    raise ValueError("no {provenance, result} JSON payload in the CLI output")


def query_counts(stderr: str) -> list[dict[str, Any]]:
    """The loader query counts the CLI printed on stderr (the last line
    starting with `LOADER_QUERY_COUNTS_PREFIX`), or `[]` when it printed
    none."""
    found: list[dict[str, Any]] = []
    for line in stderr.splitlines():
        if line.startswith(LOADER_QUERY_COUNTS_PREFIX):
            found = json.loads(line[len(LOADER_QUERY_COUNTS_PREFIX) :])
    return found


def first_difference(expected: str, actual: str) -> str | None:
    """`None` when the two texts are byte-identical, else the first line
    where they differ, numbered from 1, with both versions."""
    if expected == actual:
        return None
    left, right = expected.splitlines(), actual.splitlines()
    for number, (one, other) in enumerate(zip(left, right, strict=False), start=1):
        if one != other:
            return f"line {number}: {one!r} != {other!r}"
    return f"line {min(len(left), len(right)) + 1}: one text ends ({len(left)} vs {len(right)})"


def backtest_command(
    assets: Sequence[str], start: str, end: str, lake_time: str, store: bool, extra: Sequence[str]
) -> list[str]:
    """The subprocess command running the CLI backtest at `lake_time` with
    the window store on or off."""
    return [
        sys.executable,
        "-c",
        "import sys; from agentic_trading.cli import main; main(sys.argv[1:])",
        *assets,
        "--start-date",
        start,
        "--end-date",
        end,
        "--yes",
        "--lake-time",
        lake_time,
        "--loader-store" if store else "--no-loader-store",
        *extra,
    ]


def _run_cli(command: list[str], store: bool) -> CliRun:
    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    return CliRun(
        store=store,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        seconds=round(time.perf_counter() - started, 1),
    )


def compare_backtests(direct: CliRun, stored: CliRun) -> dict[str, Any]:
    """The comparison of a `--no-loader-store` and a `--loader-store` run:
    `identical` when both exited 0 and their results are byte-identical,
    the first differing line, and each run's seconds and query counts."""
    summary: dict[str, Any] = {}
    texts: dict[str, str | None] = {}
    for label, run in (("direct", direct), ("store", stored)):
        counts = query_counts(run.stderr)
        summary[label] = {
            "returncode": run.returncode,
            "seconds": run.seconds,
            "queries": sum(int(count["queries"]) for count in counts),
            "counts": counts,
        }
        try:
            texts[label] = result_text(run.stdout) if run.returncode == 0 else None
        except ValueError:
            texts[label] = None
        if texts[label] is None:
            summary[label]["stderr_tail"] = run.stderr[-2000:]
    direct_text, store_text = texts["direct"], texts["store"]
    if direct_text is None or store_text is None:
        summary["identical"] = False
        summary["difference"] = "a run failed or printed no result"
    else:
        difference = first_difference(direct_text, store_text)
        summary["identical"] = difference is None
        summary["difference"] = difference
        summary["result_bytes"] = len(direct_text.encode())
    return summary


def _backtest_main(args: argparse.Namespace, extra: list[str]) -> int:
    lake_time = args.lake_time
    if lake_time is None:
        with ll.lake_session() as session:
            lake_time = session.lake_time.isoformat()
    runs = [
        _run_cli(
            backtest_command(args.assets, args.start_date, args.end_date, lake_time, store, extra),
            store,
        )
        for store in (False, True)
    ]
    summary = {"lake_time": lake_time, **compare_backtests(runs[0], runs[1])}
    _emit(summary, args.out)
    return 0 if summary["identical"] else 1


# ---------------------------------------------------------------- entry point


def _emit(payload: dict[str, Any], out: Path | None) -> None:
    text = json.dumps(payload, indent=2, default=str)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")
    print(text)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    modes = parser.add_subparsers(dest="mode", required=True)

    loaders = modes.add_parser("loaders", help="compare every loader on the three paths")
    loaders.add_argument("--start", required=True)
    loaders.add_argument("--end", required=True)
    loaders.add_argument("--ticker", action="append", default=[], required=True)
    loaders.add_argument("--security-id", action="append", type=int, default=[])
    loaders.add_argument("--fred-series", action="append", default=[])
    loaders.add_argument("--option-underlying", action="append", default=[])
    loaders.add_argument("--every", type=int, default=1)
    loaders.add_argument("--price-lookback-days", type=int, default=DEFAULT_PRICE_LOOKBACK_DAYS)
    loaders.add_argument("--lake-time", default=None)
    loaders.add_argument("--out", type=Path, default=None)

    backtest = modes.add_parser("backtest", help="diff a backtest with and without the store")
    backtest.add_argument("assets", nargs="+")
    backtest.add_argument("--start-date", required=True)
    backtest.add_argument("--end-date", required=True)
    backtest.add_argument("--lake-time", default=None)
    backtest.add_argument("--out", type=Path, default=None)

    args, extra = parser.parse_known_args(argv)
    if args.mode == "loaders":
        if extra:
            parser.error(f"unrecognized arguments: {' '.join(extra)}")
        return _loaders_main(args)
    return _backtest_main(args, [item for item in extra if item != "--"])


if __name__ == "__main__":
    sys.exit(main())
