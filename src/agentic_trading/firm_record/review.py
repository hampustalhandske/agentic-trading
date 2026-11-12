# review.py

"""A finished firm record read back for review: the engine's whole-run
results of the record, rebuilt from its engine checkpoint, in the compact
run-summary shape the dashboard's "Load a saved run" panel reads
(`backtest.run_summary`).

A firm record (`firm_record.runner`) keeps its engine checkpoint at
`<store>/<spec hash>.checkpoint.sqlite`, beside its trial folder
`<store>/<spec hash>/`, whose spec.json (`FirmSpecRecord`) names the
window, the universe, the roster, the lake time and the lake version the
record read at. `record_result` opens the checkpoint read-only
(`checkpoint_store.connect_read_only`) and computes the `BacktestResult`
the engine's run returns from the histories it holds, through the engine's
own functions (`backtest.engine.run_metrics`, `source_performance_of`,
`pod_performance_of`, `_pod_cost_attribution`, `execution_cost_totals`):
the same numbers, but for what only the engine's memory held (its last
decisions, its horizon setting and the split moves its own reads found).

Sharpe and Sortino are on excess returns over the DGS3MO accrual and the
information ratio is against SPY's total return; neither series is in the
checkpoint, so `load_record_benchmarks` reads both from the lake through
the loaders the engine's inputs come from
(`macro_loader.load_risk_free_rate_series`,
`engine_inputs.load_benchmark_closes`), inside one `lake_session` pinned
to the record's lake time, lake version and `stage_a.data_end`.

`firm_record_summary` puts it together: one period block keyed `record`
(`run_summary.summarize_period`), the record's `experiment` description,
its `lake` reading, and a `baselines` entry holding SPY's total-return
curve over the record's sessions, started at the firm's first value. Run
it with `agentic-trading-firm-review` (`cli.firm_review`) to write the
summary as JSON, or load the checkpoint in the dashboard."""

import json
import math
from contextlib import nullcontext
from datetime import date
from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd
from pydantic import BaseModel, ConfigDict

from agentic_trading.agents.roster import FIRM_V1_EXCLUDED_ANALYST_IDS
from agentic_trading.agents.schemas import (
    BacktestResult,
    CapitalUsage,
    PodCostDay,
    SizingDiagnostics,
    TradeRecord,
    UnfilledOrder,
)
from agentic_trading.backtest import checkpoint_store
from agentic_trading.backtest.engine import (
    _pod_cost_attribution,
    execution_cost_totals,
    pod_performance_of,
    run_metrics,
    source_performance_of,
    source_returns_of,
)
from agentic_trading.backtest.engine_inputs import benchmark_returns_of, load_benchmark_closes
from agentic_trading.backtest.run_summary import summarize_baseline, summarize_period
from agentic_trading.data.loaders.lakehouse_loader import lake_answer_cache, lake_session
from agentic_trading.data.loaders.macro_loader import load_risk_free_rate_series
from agentic_trading.firm_record.submission import META_FILE, AgenticMeta
from agentic_trading.logging_config import get_logger
from agentic_trading.stage_a.spec import FirmSpecRecord

logger = get_logger(__name__)

# The instrument whose total return the record is measured against: SPY's
# lake close is its dividend- and split-adjusted close.
BENCHMARK_TICKER = "SPY"
# The summary's period key for the record, and its baseline's name.
RECORD_PERIOD = "record"
BENCHMARK_BASELINE = "SPY total return"
# The suffix of an engine checkpoint file after its trial's spec hash.
CHECKPOINT_SUFFIX = ".checkpoint.sqlite"
# Where a firm record's price frame comes from (`data.loader`).
_MARKET_DATA_SOURCE = "lakehouse data"
# Roster entries' class prefixes, by role.
_MANAGER_MODULES = "agentic_trading.agents.managers."
_ANALYST_MODULES = "agentic_trading.agents.analysts."


class RecordBenchmarks(BaseModel):
    """The series a record's ratios are measured with: the daily decimal
    risk-free rate cash accrues (`load_risk_free_rate_series`, indexed by
    calendar date) and the benchmark's daily closes (`load_benchmark_closes`,
    indexed by bar date)."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    risk_free_rate_series: pd.Series
    benchmark_closes: pd.Series


def checkpoint_trial_dir(checkpoint_path: Path) -> Path:
    """The trial folder of the firm record whose engine checkpoint is
    `checkpoint_path`: `<store>/<spec hash>/` for a checkpoint at
    `<store>/<spec hash>.checkpoint.sqlite`. Raises `ValueError` when the
    file's name does not end in `CHECKPOINT_SUFFIX`."""
    name = checkpoint_path.name
    if not name.endswith(CHECKPOINT_SUFFIX) or name == CHECKPOINT_SUFFIX:
        raise ValueError(
            f"{checkpoint_path} is not named <spec hash>{CHECKPOINT_SUFFIX}, so its trial "
            "folder cannot be found beside it"
        )
    return checkpoint_path.parent / name.removesuffix(CHECKPOINT_SUFFIX)


def read_trial_spec(trial_dir: Path) -> FirmSpecRecord:
    """The `FirmSpecRecord` in `trial_dir`'s spec.json. Raises
    `FileNotFoundError` without one and `ValueError`
    (`pydantic.ValidationError`) when it does not hold a valid record."""
    return FirmSpecRecord.model_validate_json((trial_dir / "spec.json").read_bytes())


class _CheckpointReading(NamedTuple):
    """What a review reads from an engine checkpoint: `state`, everything
    `checkpoint_store.read_state` gives, and each source's attributable
    returns over the run's window (`source_returns_of`). Held within this
    module only, as read, without validating its millions of values."""

    state: dict[str, Any]
    source_returns: dict[str, tuple[list[float], list[float]]]


def _read_checkpoint(checkpoint_path: Path) -> _CheckpointReading:
    """Reads the checkpoint at `checkpoint_path` opened read-only. Raises
    `FileNotFoundError` when the file is absent, `CheckpointFormatError`
    when it is of another layout and `ValueError` when it holds no day."""
    connection = checkpoint_store.connect_read_only(checkpoint_path)
    try:
        state = checkpoint_store.read_state(connection)
        if state is None:
            raise ValueError(f"checkpoint {checkpoint_path} holds no day")
        source_returns = source_returns_of(
            checkpoint_store.iter_source_returns(
                connection,
                date.fromisoformat(state["config"]["start_date"]),
                date.fromisoformat(state["config"]["end_date"]),
            )
        )
    finally:
        connection.close()
    return _CheckpointReading(state=state, source_returns=source_returns)


def _result_of(
    reading: _CheckpointReading,
    *,
    risk_free_rate_series: pd.Series | None,
    benchmark_returns: pd.Series | None,
    market_data_source: str | None,
) -> BacktestResult:
    """`record_result` of the checkpoint `reading` was read from."""
    state = reading.state
    portfolio_value_history = [
        (date.fromisoformat(day), float(value)) for day, value in state["portfolio_value_history"]
    ]
    trade_log = [TradeRecord.model_validate(trade) for trade in state["trade_log"]]
    pod_return_history = {
        pod_id: [(date.fromisoformat(day), float(value)) for day, value in entries]
        for pod_id, entries in state["pod_return_history"].items()
    }
    pod_cost_days = [PodCostDay.model_validate(entry) for entry in state["pod_cost_days"]]
    metrics = run_metrics(
        portfolio_value_history,
        risk_free_rate_series=risk_free_rate_series,
        benchmark_returns=benchmark_returns,
    )
    return BacktestResult(
        start_date=date.fromisoformat(state["config"]["start_date"]),
        end_date=date.fromisoformat(state["config"]["end_date"]),
        portfolio_value_history=portfolio_value_history,
        **metrics.model_dump(),
        trade_log=trade_log,
        unfilled_orders=[UnfilledOrder.model_validate(order) for order in state["unfilled_orders"]],
        capital_usage=[CapitalUsage.model_validate(entry) for entry in state["capital_usage"]],
        sizing_diagnostics=[
            SizingDiagnostics.model_validate(entry) for entry in state["sizing_diagnostics"]
        ],
        signal_health=state["signal_health"],
        market_data_source=market_data_source,
        source_performance=source_performance_of(reading.source_returns),
        **execution_cost_totals(trade_log).model_dump(),
        total_financing_cost=state["total_financing_cost"],
        total_option_premium_paid=state["total_option_premium_paid"],
        pod_return_history=pod_return_history,
        pod_performance=pod_performance_of(pod_return_history),
        pod_costs=_pod_cost_attribution(pod_cost_days, len(portfolio_value_history)),
        unfactored_split_moves=None,
    )


def record_result(
    checkpoint_path: Path,
    *,
    risk_free_rate_series: pd.Series | None,
    benchmark_returns: pd.Series | None,
    market_data_source: str | None,
) -> BacktestResult:
    """The `BacktestResult` of the engine run that wrote the checkpoint at
    `checkpoint_path`, computed from the checkpoint opened read-only, with
    the run's `risk_free_rate_series` and `benchmark_returns` as the engine
    takes them. Every field is the engine's, from the same functions on the
    same histories, but `decisions_by_day` (empty), `history_horizon_days`
    (`None`) and `unfactored_split_moves` (`None`: the checkpoint does not
    carry the registry). Raises `FileNotFoundError` when the file is
    absent, `CheckpointFormatError` when it is of another layout and
    `ValueError` when it holds no day."""
    return _result_of(
        _read_checkpoint(checkpoint_path),
        risk_free_rate_series=risk_free_rate_series,
        benchmark_returns=benchmark_returns,
        market_data_source=market_data_source,
    )


def load_record_benchmarks(
    record: FirmSpecRecord, *, lake_cache_dir: Path | None = None
) -> RecordBenchmarks:
    """Reads `record`'s risk-free rate series and `BENCHMARK_TICKER`'s
    closes over its window, as the firm record's engine inputs are read
    (`load_risk_free_rate_series`, `load_benchmark_closes`), inside one
    `lake_session` at the record's lake time, pinned to its lake version
    and cut at the `stage_a.data_end` its spec records. With
    `lake_cache_dir`, every read goes through a lake answer cache there
    (`lake_answer_cache`). The loaders' and the session's errors
    propagate."""
    spec = record.spec
    data_end = date.fromisoformat(spec.settings["stage_a"]["data_end"])
    answers = nullcontext() if lake_cache_dir is None else lake_answer_cache(lake_cache_dir)
    with answers:
        with lake_session(record.lake_time, data_end=data_end, lake_version=record.lake_version):
            risk_free_rate_series = load_risk_free_rate_series(spec.window_start, spec.window_end)
            closes = load_benchmark_closes(BENCHMARK_TICKER, spec.window_start, spec.window_end)
    return RecordBenchmarks(risk_free_rate_series=risk_free_rate_series, benchmark_closes=closes)


def benchmark_baseline(
    portfolio_value_history: list[tuple[date, float]],
    closes: pd.Series,
    *,
    risk_free_rate_series: pd.Series | None,
) -> BacktestResult | None:
    """A buy-and-hold of the benchmark over the firm's valuation days: on
    each day of `portfolio_value_history` on which `closes` has a close,
    the firm's value on the first such day times the close over that day's
    close; days without a close are left out. Its metrics are
    `run_metrics` with `risk_free_rate_series` and no benchmark. `None`
    when `closes` has no close on any of the days."""
    on_days = closes.reindex([pd.Timestamp(day) for day, _ in portfolio_value_history])
    priced = [
        (day, value, float(close))
        for (day, value), close in zip(portfolio_value_history, on_days, strict=True)
        if not math.isnan(close)
    ]
    if not priced:
        return None
    _, start_value, start_close = priced[0]
    history = [(day, start_value * close / start_close) for day, _, close in priced]
    metrics = run_metrics(
        history, risk_free_rate_series=risk_free_rate_series, benchmark_returns=None
    )
    return BacktestResult(
        start_date=history[0][0],
        end_date=history[-1][0],
        portfolio_value_history=history,
        **metrics.model_dump(),
    )


def _read_meta(trial_dir: Path) -> AgenticMeta | None:
    """`trial_dir`'s agentic_meta.json, `None` when it has none."""
    path = trial_dir / META_FILE
    if not path.is_file():
        return None
    return AgenticMeta.model_validate_json(path.read_bytes())


def _experiment(record: FirmSpecRecord, meta: AgenticMeta | None) -> dict[str, Any]:
    """The summary's `experiment` block: the trial's spec hash, window and
    universe (`assets`), the manager and analyst ids of its roster in
    roster order, the registered firm's excluded analysts, the quantcore
    version, and from agentic_meta.json the commit the record ran from and
    whether it is counted (each `None` without the file)."""
    spec = record.spec
    roster_classes = [(entry["agent_id"], str(entry.get("class", ""))) for entry in spec.roster]
    return {
        "spec_hash": record.spec_hash,
        "window_start": spec.window_start.isoformat(),
        "window_end": spec.window_end.isoformat(),
        "assets": list(spec.universe),
        "managers": [
            agent_id for agent_id, cls in roster_classes if cls.startswith(_MANAGER_MODULES)
        ],
        "analysts": [
            agent_id for agent_id, cls in roster_classes if cls.startswith(_ANALYST_MODULES)
        ],
        "excluded_analysts": sorted(FIRM_V1_EXCLUDED_ANALYST_IDS),
        "quantcore_version": record.quantcore_version,
        "git_sha": None if meta is None else meta.git_sha,
        "counted": None if meta is None else meta.counted,
    }


def _require_matching_checkpoint(
    checkpoint_path: Path, reading: _CheckpointReading, record: FirmSpecRecord
) -> None:
    """Raises `ValueError` unless the checkpoint at `checkpoint_path`, read
    as `reading`, is the one `record`'s trial wrote: named after its spec
    hash, over its window and universe, at its lake time."""
    if checkpoint_path.name != f"{record.spec_hash}{CHECKPOINT_SUFFIX}":
        raise ValueError(
            f"checkpoint {checkpoint_path.name} is not the one of the trial with spec hash "
            f"{record.spec_hash}"
        )
    config = reading.state["config"]
    spec = record.spec
    if (
        config["start_date"] != spec.window_start.isoformat()
        or config["end_date"] != spec.window_end.isoformat()
        or config["assets"] != list(spec.universe)
    ):
        raise ValueError(
            f"checkpoint {checkpoint_path.name} runs {config['start_date']} to "
            f"{config['end_date']} over {len(config['assets'])} assets, not the trial's window "
            "and universe"
        )
    lake_time = reading.state["lake_time"]
    if lake_time != record.lake_time:
        written_at = "no lake session" if lake_time is None else lake_time.isoformat()
        raise ValueError(
            f"checkpoint {checkpoint_path.name} was written at lake time {written_at}, not the "
            f"trial's {record.lake_time.isoformat()}"
        )


def firm_record_summary(
    checkpoint_path: Path,
    trial_dir: Path | None = None,
    *,
    lake_cache_dir: Path | None = None,
    benchmarks: RecordBenchmarks | None = None,
) -> dict[str, Any]:
    """The run summary of the firm record whose engine checkpoint is
    `checkpoint_path` and whose trial folder is `trial_dir` (by default
    the one beside it, `checkpoint_trial_dir`), as JSON values.

    It holds `experiment` (`_experiment`), `market_data_source`, `lake`
    (the record's `lake_time` and `lake_version`), one period block keyed
    `RECORD_PERIOD` (`summarize_period` of `record_result`, its ratios
    measured with `benchmarks`: the risk-free accrual for Sharpe and
    Sortino, `BENCHMARK_TICKER`'s daily returns for the information ratio)
    and `baselines`, holding under `BENCHMARK_BASELINE` the benchmark's
    buy-and-hold over the firm's sessions (`benchmark_baseline`,
    `summarize_baseline`) when it has a close on any of them. Without
    `benchmarks` they are read from the lake (`load_record_benchmarks`,
    through an answer cache at `lake_cache_dir` when given). Nothing under
    the checkpoint's directory is written.

    Raises `ValueError` when `lake_cache_dir` is inside the checkpoint's
    directory (the record's store), when the checkpoint is not the trial's
    (`_require_matching_checkpoint`) or holds no day; the reading and
    loading errors of `read_trial_spec`, `record_result` and
    `load_record_benchmarks` propagate."""
    store = checkpoint_path.resolve().parent
    if lake_cache_dir is not None and lake_cache_dir.resolve().is_relative_to(store):
        raise ValueError(
            f"lake cache directory {lake_cache_dir} is inside the record's store {store}; "
            "give one outside it"
        )
    trial_dir = checkpoint_trial_dir(checkpoint_path) if trial_dir is None else trial_dir
    record = read_trial_spec(trial_dir)
    reading = _read_checkpoint(checkpoint_path)
    _require_matching_checkpoint(checkpoint_path, reading, record)
    if benchmarks is None:
        benchmarks = load_record_benchmarks(record, lake_cache_dir=lake_cache_dir)
    result = _result_of(
        reading,
        risk_free_rate_series=benchmarks.risk_free_rate_series,
        benchmark_returns=benchmark_returns_of(benchmarks.benchmark_closes),
        market_data_source=_MARKET_DATA_SOURCE,
    )
    baseline = benchmark_baseline(
        result.portfolio_value_history,
        benchmarks.benchmark_closes,
        risk_free_rate_series=benchmarks.risk_free_rate_series,
    )
    if baseline is None:
        logger.warning(
            "Firm record %s: %s has no close on any of its sessions; no baseline",
            record.spec_hash,
            BENCHMARK_TICKER,
        )
    summary = {
        "experiment": _experiment(record, _read_meta(trial_dir)),
        "market_data_source": _MARKET_DATA_SOURCE,
        "lake": {
            "lake_time": record.lake_time.isoformat(),
            "lake_version": record.lake_version,
        },
        RECORD_PERIOD: summarize_period(result.model_dump(mode="json")),
        "baselines": (
            {}
            if baseline is None
            else {BENCHMARK_BASELINE: summarize_baseline(baseline.model_dump(mode="json"))}
        ),
    }
    # Through JSON once, so the summary holds the values a saved file gives back.
    loaded: dict[str, Any] = json.loads(json.dumps(summary))
    return loaded
