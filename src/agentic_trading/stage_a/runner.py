"""The stage (a) batch runner: a list of analysts over a window, stored
day by day, resumable, under a memory and disk guard.

`run_batch` runs each analyst in turn over the sessions of
`settings.stage_a.session_calendar` in the window inside one
`lake_session` at the given lake time T, cut at
`settings.stage_a.data_end`. On each session it asks the analyst once per
asset with a close that day, at that day's midnight `pd.Timestamp`, with
the point-in-time slice of the prices restricted to the rows on which
that asset has a close; an asset without a close that day is stored as
unavailable without asking the analyst. It writes the day through the
analyst's `SignalStore`. A trial resumes after its store's last completed
day, so a rerun of a finished trial computes and writes nothing. Every
`settings.stage_a.check_every_days` days written, it calls the resource
check (`system_resources_ok` by default) and, when that fails, stops
after the day it has just written; the next run resumes from there.

Given a `TrialLedger`, the batch also records its trials there: each
analyst's stage (a) hypothesis is registered in
`settings.trials.firm_family` before its first day is written, and each
analyst that wrote a day gets one trial row (`trials.rows`), so a rerun
that writes nothing adds no row and a resumed batch adds another row
under the same spec hash."""

import re
import shutil
import subprocess
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict

from agentic_trading.agents.base import (
    SAFE_FALLBACK_CONFIDENCE,
    SAFE_FALLBACK_DIRECTION,
    AnalystAgent,
)
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.config import settings
from agentic_trading.data.loaders.lakehouse_loader import (
    HoldoutReadError,
    LakeSession,
    LakeTimeError,
    lake_session,
)
from agentic_trading.data.point_in_time import get_point_in_time_data
from agentic_trading.data.trading_calendar import sessions_between
from agentic_trading.logging_config import get_logger
from agentic_trading.stage_a.spec import SpecRecord, build_spec, spec_hash
from agentic_trading.stage_a.store import InstrumentKind, SignalStore
from agentic_trading.trials.ledger import TrialLedger
from agentic_trading.trials.rows import (
    current_trial_provenance,
    ensure_stage_a_registration,
    stage_a_hypothesis_id,
    stage_a_trial_row,
)

logger = get_logger(__name__)

# macOS's memory report; its last line gives the share of memory free.
_MEMORY_PRESSURE_COMMAND = ("memory_pressure",)
_MEMORY_PRESSURE_TIMEOUT_SECONDS = 10.0
_FREE_MEMORY_PATTERN = re.compile(r"System-wide memory free percentage:\s*(\d+(?:\.\d+)?)%")
# Bytes per GB in `settings.stage_a.min_free_disk_gb` (decimal gigabytes).
_BYTES_PER_GB = 1_000_000_000


class AnalystBatchResult(BaseModel):
    """One analyst's part of a batch: its trial's `spec_hash`, the
    `days_written` by this batch, and the store's `last_completed` day
    after it (`None` when no day is stored)."""

    model_config = ConfigDict(frozen=True)

    spec_hash: str
    days_written: int
    last_completed: date | None


class BatchResult(BaseModel):
    """What one `run_batch` call did: `analysts` maps the agent_id of each
    analyst the batch reached, in run order, to its `AnalystBatchResult`;
    `stopped_for_resources` is True when the resource check failed and the
    batch stopped before finishing every analyst."""

    model_config = ConfigDict(frozen=True)

    analysts: dict[str, AnalystBatchResult]
    stopped_for_resources: bool


def _free_memory_pct() -> float | None:
    """The system-wide free memory percentage `memory_pressure` reports,
    or `None`, with a WARNING logged, when the command is missing, fails,
    times out or prints no percentage."""
    try:
        completed = subprocess.run(
            _MEMORY_PRESSURE_COMMAND,
            capture_output=True,
            text=True,
            check=False,
            timeout=_MEMORY_PRESSURE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as error:
        logger.warning("Cannot read free memory: memory_pressure did not run (%s)", error)
        return None
    if completed.returncode != 0:
        logger.warning(
            "Cannot read free memory: memory_pressure exited with code %d", completed.returncode
        )
        return None
    match = _FREE_MEMORY_PATTERN.search(completed.stdout)
    if match is None:
        logger.warning("Cannot read free memory: memory_pressure printed no free percentage")
        return None
    return float(match.group(1))


def system_resources_ok(store_dir: Path) -> bool:
    """Whether a batch writing to `store_dir` may go on: free system
    memory (from macOS's `memory_pressure`) is at least
    `settings.stage_a.min_free_memory_pct` percent and free disk on
    `store_dir`'s volume at least `settings.stage_a.min_free_disk_gb` GB.
    False, with a WARNING logged, when either is lower, and when free
    memory cannot be read: a batch that cannot see its memory stops."""
    free_memory = _free_memory_pct()
    if free_memory is None:
        return False
    free_disk_gb = shutil.disk_usage(store_dir).free / _BYTES_PER_GB
    limits = settings.stage_a
    if free_memory < limits.min_free_memory_pct or free_disk_gb < limits.min_free_disk_gb:
        logger.warning(
            "Low resources: %.1f%% memory free (minimum %.1f%%), %.1f GB disk free on %s "
            "(minimum %.1f GB)",
            free_memory,
            limits.min_free_memory_pct,
            free_disk_gb,
            store_dir,
            limits.min_free_disk_gb,
        )
        return False
    return True


def _signal(
    analyst: AnalystAgent, asset: str, market_data: pd.DataFrame, as_of: pd.Timestamp
) -> AgentSignal:
    """`analyst`'s signal for `asset` at `as_of`. Any exception other than
    `HoldoutReadError` and `LakeTimeError`, which propagate, becomes a
    flat, zero-confidence signal with status `error`, logged as a
    WARNING."""
    try:
        return analyst.generate_signal(asset, market_data, as_of)
    except (HoldoutReadError, LakeTimeError):
        raise
    except Exception as error:
        logger.warning(
            "Analyst %s raised on %s at %s; storing an error signal",
            analyst.agent_id,
            asset,
            as_of.date().isoformat(),
            exc_info=True,
        )
        return AgentSignal(
            agent_id=analyst.agent_id,
            asset=asset,
            as_of_date=as_of.date(),
            direction=SAFE_FALLBACK_DIRECTION,
            confidence=SAFE_FALLBACK_CONFIDENCE,
            rationale=f"generate_signal raised {type(error).__name__}: {error}",
            status="error",
        )


def _day_signals(
    analyst: AnalystAgent,
    universe: Sequence[str],
    market_data: pd.DataFrame,
    day: date,
    session: LakeSession,
) -> list[AgentSignal]:
    """`analyst`'s signal for each asset of `universe` on `day`, in
    `universe` order. An asset with a close on `day` in `market_data` gets
    the analyst's signal from the point-in-time slice of `market_data`
    restricted to the rows on which that asset has a close, every column
    kept. An asset with no close on `day` (no row that day, or a NaN close)
    gets the analyst's `unavailable_signal` without the analyst being
    asked. Raises `HoldoutReadError` when a read past the session's
    `data_end` was refused while computing them, even if the analyst
    caught the refusal, so no such day is stored."""
    as_of = pd.Timestamp(day)
    visible = get_point_in_time_data(market_data, as_of)
    calendar = settings.stage_a.session_calendar
    refused_before = len(session.holdout_violations)
    signals = []
    for asset in universe:
        closes = visible[(asset, "close")] if (asset, "close") in visible.columns else None
        if closes is None or as_of not in closes.index or pd.isna(closes.loc[as_of]):
            signals.append(
                analyst.unavailable_signal(
                    asset, as_of, f"no {calendar} bar for {asset} on {day.isoformat()}"
                )
            )
        else:
            signals.append(_signal(analyst, asset, visible.loc[closes.notna()], as_of))
    refused = session.holdout_violations[refused_before:]
    if refused:
        raise HoldoutReadError(
            f"{analyst.agent_id} made {len(refused)} read(s) past data_end on "
            f"{day.isoformat()}: " + "; ".join(refused)
        )
    return signals


def run_batch(
    analysts: Sequence[AnalystAgent],
    universe: Sequence[str],
    instrument_kind: InstrumentKind,
    window_start: date,
    window_end: date,
    store_dir: Path,
    market_data: pd.DataFrame,
    *,
    lake_time: datetime,
    lake_version: str,
    quantcore_version: str,
    quantcore_sha: str | None,
    resources_ok: Callable[[Path], bool] = system_resources_ok,
    ledger: TrialLedger | None = None,
    repo_root: Path | None = None,
) -> BatchResult:
    """Run each of `analysts`, in order, over `universe` on the sessions of
    `settings.stage_a.session_calendar` (read at call time) from
    `window_start` to `window_end`, storing each trial under `store_dir`
    (see the module docstring).

    Takes `market_data` in `load_price_history_from_lakehouse`'s (asset,
    field) column shape, covering the window and its warm-up; each
    analyst sees only its point-in-time slice. Every row is keyed by its
    asset as instrument id and written as `instrument_kind`. Each trial's
    spec is built from the analyst, `universe` and the window when the
    batch starts, and recorded with `lake_time` (aware), `lake_version`,
    `quantcore_version` and `quantcore_sha`. Every read happens inside
    `lake_session(lake_time, data_end=settings.stage_a.data_end)`, which
    joins a session already open at the same T and data_end. counts.json
    is written for each analyst that wrote a day, or has none yet.
    `resources_ok(store_dir)` runs after every
    `settings.stage_a.check_every_days`-th day this batch writes, across
    analysts; on False the batch returns after that day with
    `stopped_for_resources` True and the analysts after it not started.

    With a `ledger`, `repo_root` is the repository root its registry
    files live under. Before an analyst's first pending day is computed,
    its hypothesis (`stage_a_hypothesis_id`) is registered through
    `ensure_stage_a_registration` in `settings.trials.firm_family` (read
    at call time). After an analyst's counts.json is written, when this
    call wrote at least one day for it, its `stage_a_trial_row` is
    recorded with the provenance of the code running now and `n_days` the
    sessions from `window_start` to the store's last completed day.
    Without a ledger, nothing is written outside `store_dir`.

    Raises `ValueError`, before reading or writing anything, when
    `window_start` is after `window_end`, `window_end` is after
    `settings.stage_a.data_end` (read at call time), two analysts share
    an agent_id, or a `ledger` is given without a `repo_root`. A
    `LedgerError` from registering or recording propagates. An analyst's
    exception is stored as that asset's error signal for the day, except
    `HoldoutReadError` and `LakeTimeError`, which propagate with the day
    unwritten; so does `HoldoutReadError` for a day on which a read past
    data_end was refused and caught. A store mismatch or a refused row
    propagates from `SignalStore`."""
    data_end = settings.stage_a.data_end
    if window_start > window_end:
        raise ValueError(
            f"window_start {window_start.isoformat()} is after window_end {window_end.isoformat()}"
        )
    if window_end > data_end:
        raise ValueError(
            f"window_end {window_end.isoformat()} is after stage_a.data_end "
            f"{data_end.isoformat()}; the holdout stays locked"
        )
    agent_ids = [analyst.agent_id for analyst in analysts]
    repeated = sorted({agent_id for agent_id in agent_ids if agent_ids.count(agent_id) > 1})
    if repeated:
        raise ValueError(f"analysts share an agent_id: {repeated}")
    if ledger is not None and repo_root is None:
        raise ValueError("a ledger needs the repo_root its registry files live under")
    family = settings.trials.firm_family

    records = []
    for analyst in analysts:
        spec = build_spec(analyst, list(universe), window_start, window_end)
        records.append(
            SpecRecord(
                spec=spec,
                spec_hash=spec_hash(spec),
                lake_time=lake_time,
                lake_version=lake_version,
                quantcore_version=quantcore_version,
                quantcore_sha=quantcore_sha,
            )
        )
    store_dir = Path(store_dir)
    calendar = settings.stage_a.session_calendar
    sessions = sessions_between(window_start, window_end, calendar)
    instrument_ids = {asset: asset for asset in universe}
    check_every_days = settings.stage_a.check_every_days

    results: dict[str, AnalystBatchResult] = {}
    stopped = False
    days_since_check = 0
    with lake_session(lake_time, data_end=data_end) as session:
        for analyst, record in zip(analysts, records, strict=True):
            store = SignalStore(store_dir, record)
            store.open()
            last = store.last_completed()
            pending = [day for day in sessions if last is None or day > last]
            logger.info(
                "Stage (a) %s (trial %s): %d of %d sessions to compute",
                analyst.agent_id,
                record.spec_hash,
                len(pending),
                len(sessions),
            )
            hypothesis_id = stage_a_hypothesis_id(analyst.agent_id)
            if ledger is not None and repo_root is not None and pending:
                ensure_stage_a_registration(
                    ledger=ledger,
                    repo_root=repo_root,
                    agent_id=analyst.agent_id,
                    analyst_class=record.spec.analyst_class,
                    family=family,
                    now=datetime.now(UTC),
                )
            written = 0
            for day in pending:
                signals = _day_signals(analyst, universe, market_data, day, session)
                store.write_day(day, signals, instrument_ids, instrument_kind)
                written += 1
                days_since_check += 1
                if days_since_check >= check_every_days:
                    days_since_check = 0
                    if not resources_ok(store_dir):
                        stopped = True
                        break
            if written or not (store.trial_dir / "counts.json").exists():
                store.write_counts()
            last_completed = store.last_completed()
            if ledger is not None and written and last_completed is not None:
                ledger.record_trial(
                    stage_a_trial_row(
                        record,
                        hypothesis_id,
                        family,
                        n_days=len(sessions_between(window_start, last_completed, calendar)),
                        provenance=current_trial_provenance(),
                    )
                )
            results[analyst.agent_id] = AnalystBatchResult(
                spec_hash=record.spec_hash,
                days_written=written,
                last_completed=last_completed,
            )
            if stopped:
                logger.warning(
                    "Stage (a) batch stopped for resources after %s on %s",
                    analyst.agent_id,
                    last_completed,
                )
                break
    return BatchResult(analysts=results, stopped_for_resources=stopped)
