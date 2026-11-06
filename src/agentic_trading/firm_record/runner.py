# runner.py

"""The firm record pass: a Commander run through the backtest engine on
every session of `settings.stage_a.session_calendar` in a window, its pod
targets written each day as a `decision_file.v1` trial.

`run_firm_record` builds the firm's `FirmSpec` and opens its
`FirmDecisionStore` under `store_dir / <spec hash>`, then runs the engine
inside one `lake_session` at the given lake time, cut at
`settings.stage_a.data_end`. The price frame is aligned on the window's
sessions: a session without any bar is a row of missing values, so the
firm still decides on it. An instrument held over a session without its
bar keeps each pod's last decided target, which the engine fills at the
next session with a bar; one without a bar and without a decided target
gets no decision and one unavailable, weight-0 row per pod in its scope
(`firm_record.rows`). Each day's rows are written once the engine has
accounted the day and before its checkpoint takes the day, so a store
refusal stops the pass with the checkpoint at the day before; a day on
which a read past `data_end` was refused, even one an analyst caught,
stops the pass before the day is written.

The engine checkpoints every day to `store_dir / <spec hash>.checkpoint.sqlite`,
beside the trial directory, so a pass stopped after `max_days`, or killed,
resumes with the firm's state as it stood and ends with the rows of an
unbroken pass. A pass killed after a day's rows were written but before its
checkpoint drops the store's days past the checkpoint and computes them
again. The checkpoint holds the book's value and stays outside the trial
directory, which holds only the submission: the store's spec.json,
sources.json, decisions.parquet and progress.json, and the edgelab replay
folder around them (`firm_record.submission`: submission.json, strategy/,
data/ and agentic_meta.json), written at each start of the pass for the
commit it runs from. unfilled_orders.json lists the orders the engine
could not fill, and agentic_meta.json their number; analyst_availability.json
counts each analyst's signals by status per year and over the record
(`firm_record.availability`), and agentic_meta.json lists the analysts whose
share of available signals is below `settings.firm.low_availability_threshold`
(`low_availability`), each marked `undeclared` when the registry file's
`expected_availability` does not name it, a mapping agentic_meta.json
copies. All of these are derived from the engine checkpoint at the start
and at the end of each pass, so a resumed pass reports the days its
checkpoint holds as an unbroken one does; at the end of each pass every
analyst under the threshold is logged as a warning.

No performance figure leaves the pass: the result reports only what was
written, and the engine's value and ratio log lines are held back while it
runs. With a ledger, the firm's hypothesis is registered from its
hand-approved registry file before the first day is computed, and a pass
that wrote a day records one trial row (`trials.rows.firm_trial_row`); a
counted pass needs a clean `src/` tree at a known commit, which its
spec.json records, and a registry file stating `trials.declared_n` and
`trials.prior_exposure`, which its submission.json declares. Without the
registry file the pass is uncounted, the ledger is not touched, and its
submission declares one trial; every pass needs a known commit, which its
submission.json names."""

import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict

from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.commander import Commander
from agentic_trading.agents.managers.cvar_manager import ConditionalValueAtRiskManager
from agentic_trading.agents.managers.tail_hedge_manager import TailHedgeManager
from agentic_trading.agents.schemas import PortfolioDecision, ShortWarmup, UnfilledOrder
from agentic_trading.backtest import checkpoint_store
from agentic_trading.backtest.engine import BacktestEngine
from agentic_trading.backtest.portfolio import Portfolio
from agentic_trading.backtest.resume_fingerprint import roster_agents
from agentic_trading.config import settings
from agentic_trading.data.loaders.lakehouse_loader import HoldoutReadError, lake_session
from agentic_trading.data.trading_calendar import sessions_between
from agentic_trading.firm_record.availability import analyst_availability
from agentic_trading.firm_record.rows import (
    analyst_sources,
    firm_day_rows,
    firm_sources,
    instrument_kind,
    row_provenance,
)
from agentic_trading.firm_record.spec import build_firm_spec
from agentic_trading.firm_record.submission import (
    AgenticMeta,
    submission_for,
    write_analyst_availability,
    write_submission_folder,
    write_unfilled_orders,
)
from agentic_trading.logging_config import get_logger
from agentic_trading.provenance import source_tree_state
from agentic_trading.scoring.track_record import TrackRecord
from agentic_trading.stage_a.spec import FirmSpecRecord, firm_spec_hash
from agentic_trading.stage_a.store import FirmDecisionStore
from agentic_trading.trials.ledger import TrialLedger
from agentic_trading.trials.rows import (
    current_trial_provenance,
    ensure_firm_registration,
    firm_expected_availability,
    firm_registered_trials,
    firm_trial_row,
)

logger = get_logger(__name__)

# The engine's logger: its per-day value and closing ratio lines are held
# back during a record pass.
_ENGINE_LOGGER = "agentic_trading.backtest.engine"


class FirmRecordResult(BaseModel):
    """What one `run_firm_record` call did: the trial's `spec_hash`, the
    `days_written` by this call, the store's `last_completed` day after it,
    whether every session of the window is now written (`complete`), and
    whether the pass is counted in the trial ledger (`counted`: the firm's
    registry file exists and a ledger was given). It carries no
    performance figure."""

    model_config = ConfigDict(frozen=True)

    spec_hash: str
    days_written: int
    last_completed: date | None
    complete: bool
    counted: bool


class _RecordPausedError(Exception):
    """Raised from the engine's day hook to stop a pass after `max_days`."""


@contextmanager
def _engine_quiet() -> Iterator[None]:
    """Hold the engine's INFO lines (each day's book value, the closing
    Sharpe) back for the block; its warnings still pass."""
    engine_logger = logging.getLogger(_ENGINE_LOGGER)
    previous = engine_logger.level
    engine_logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        engine_logger.setLevel(previous)


def _on_sessions(market_data: pd.DataFrame, sessions: list[date]) -> pd.DataFrame:
    """`market_data`'s rows before the first of `sessions` (the warm-up),
    then exactly one row per session: the frame's row on it, or missing
    values when it has none. Rows inside the window that are not sessions,
    and rows after it, are dropped."""
    first = pd.Timestamp(sessions[0])
    window = market_data.reindex(
        pd.DatetimeIndex([pd.Timestamp(day) for day in sessions], name=market_data.index.name)
    )
    return pd.concat([market_data.loc[market_data.index < first], window])


def _checkpoint_payload(path: Path) -> dict[str, Any] | None:
    """`checkpoint_store.read_all` of the engine checkpoint at `path`,
    `None` when the file is absent or holds no state."""
    if not path.exists():
        return None
    connection = checkpoint_store.connect(path, resume=True)
    try:
        return checkpoint_store.read_all(connection)
    finally:
        connection.close()


def _checkpoint_last_day(path: Path) -> date | None:
    """The last day the engine checkpoint at `path` holds, `None` when the
    file is absent or holds no day."""
    payload = _checkpoint_payload(path)
    if payload is None or not payload["portfolio_value_history"]:
        return None
    return date.fromisoformat(payload["portfolio_value_history"][-1][0])


def _write_checkpoint_reports(
    trial_dir: Path, checkpoint_path: Path, meta: AgenticMeta
) -> AgenticMeta:
    """Write `trial_dir`'s unfilled_orders.json and analyst_availability.json
    from the engine checkpoint at `checkpoint_path` (every unfilled order it
    records, in day order, and the analysts' signals in the decisions of
    every day it holds; neither has any when the file is absent or holds no
    state), with `meta` updated to match as agentic_meta.json; return that
    `meta`."""
    payload = _checkpoint_payload(checkpoint_path)
    orders = (
        []
        if payload is None
        else [UnfilledOrder.model_validate(order) for order in payload["unfilled_orders"]]
    )
    decisions_by_day = (
        {}
        if payload is None
        else {
            date.fromisoformat(day): [
                PortfolioDecision.model_validate(decision) for decision in decisions
            ]
            for day, decisions in payload["decisions_by_day"].items()
        }
    )
    meta = write_unfilled_orders(trial_dir, orders, meta)
    return write_analyst_availability(trial_dir, analyst_availability(decisions_by_day), meta)


def _held_contracts(portfolio: Portfolio) -> dict[str, float]:
    """Each open option position's contract id to its signed contracts."""
    return {
        contract_id: position.contracts
        for contract_id, position in portfolio.option_positions.items()
    }


def _without_bar(frame: pd.DataFrame, day: date, universe: Sequence[str]) -> list[str]:
    """The instruments of `universe` with no close on `day` in `frame`."""
    stamp = pd.Timestamp(day)
    return [
        ticker
        for ticker in universe
        if (ticker, "close") not in frame.columns or pd.isna(frame.loc[stamp, (ticker, "close")])
    ]


def run_firm_record(
    commander: Commander,
    universe: Sequence[str],
    window_start: date,
    window_end: date,
    store_dir: Path,
    market_data: pd.DataFrame,
    *,
    lake_time: datetime,
    lake_version: str,
    quantcore_version: str,
    risk_free_rate_series: pd.Series | None = None,
    max_days: int | None = None,
    ledger: TrialLedger | None = None,
    repo_root: Path | None = None,
    short_warmup: Sequence[ShortWarmup] = (),
) -> FirmRecordResult:
    """Record `commander` over `universe` on the sessions of
    `settings.stage_a.session_calendar` (read at call time) from
    `window_start` to `window_end` under `store_dir` (see the module
    docstring), writing at most `max_days` days in this call.

    Takes `market_data` in `load_price_history_from_lakehouse`'s (asset,
    field) column shape, aligned on the session calendar, covering the
    window and its warm-up. The book starts at `settings.firm.starting_cash`
    with the firm's commission and slippage, fills
    `settings.firm.execution_lag_days` after each decision and accrues
    `risk_free_rate_series` on cash. The spec is recorded with `lake_time`
    (aware), `lake_version` and `quantcore_version`. With a `ledger`,
    `repo_root` is the repository root its registry files live under.
    `short_warmup` lists the instruments whose bars start inside the
    warm-up, recorded in agentic_meta.json. With a `repo_root`, the firm's
    registry file's `expected_availability`, when it has one, is copied
    into agentic_meta.json.

    Raises `ValueError`, before writing anything, when the window is
    reversed, holds no session, or ends after `settings.stage_a.data_end`,
    when `max_days` is below 1, when a `ledger` is given without a
    `repo_root`, when HEAD is unknown, when the pass would be counted but
    the `src/` tree has changes or the registry file lacks
    `trials.declared_n` or `trials.prior_exposure`, or when the submission
    breaks edgelab's model (`pydantic.ValidationError`, a one-day window
    among them); `ValueError` when the engine checkpoint is ahead of the
    store, so the pass cannot resume. A registry file that is not a YAML
    mapping, or whose `expected_availability` is malformed, raises
    `LedgerError` before anything is written.
    `HoldoutReadError` and the store's refusals propagate with the day
    unwritten; a `LedgerError` from registering or recording propagates."""
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
    if max_days is not None and max_days < 1:
        raise ValueError(f"max_days must be at least 1, got {max_days}")
    if ledger is not None and repo_root is None:
        raise ValueError("a ledger needs the repo_root its registry files live under")
    calendar = settings.stage_a.session_calendar
    sessions = sessions_between(window_start, window_end, calendar)
    if not sessions:
        raise ValueError(
            f"no {calendar} session from {window_start.isoformat()} to {window_end.isoformat()}"
        )

    tickers = list(universe)
    family = settings.trials.firm_family
    registered_trials = None if repo_root is None else firm_registered_trials(repo_root)
    expected_availability = {} if repo_root is None else firm_expected_availability(repo_root)
    will_count = ledger is not None and registered_trials is not None
    head_sha, dirty = source_tree_state()
    if will_count and (head_sha is None or dirty is not False):
        raise ValueError(
            "a counted firm record needs a clean src/ tree at a known commit; "
            f"HEAD is {head_sha}, src/ has changes: {dirty}"
        )
    if head_sha is None or dirty is None:
        raise ValueError(
            "a firm record's submission names the commit it ran from, and HEAD is unknown"
        )
    spec = build_firm_spec(commander, tickers, window_start, window_end, lake_time)
    record = FirmSpecRecord(
        spec=spec,
        spec_hash=firm_spec_hash(spec),
        lake_time=spec.lake_time,
        lake_version=lake_version,
        quantcore_version=quantcore_version,
        git_sha=head_sha if will_count else None,
    )
    analysts = [agent for agent in roster_agents(commander) if isinstance(agent, AnalystAgent)]
    option_rows = any(
        isinstance(manager, ConditionalValueAtRiskManager | TailHedgeManager)
        for manager in commander.managers
    )
    sources = firm_sources(analysts, option_rows=option_rows)
    submission, meta = submission_for(
        record,
        sources=sources,
        git_sha=head_sha,
        src_tree_clean=not dirty,
        counted=will_count,
        registered_trials=registered_trials,
        short_warmup=short_warmup,
        expected_availability=expected_availability,
    )
    store_dir = Path(store_dir)
    store = FirmDecisionStore(store_dir, record, sources=sources)
    sources_by_analyst = analyst_sources(analysts)
    kinds = {ticker: instrument_kind(ticker) for ticker in tickers}
    scopes = {
        ticker: [
            manager.agent_id
            for manager in commander.managers
            if manager.assets is None or ticker in manager.assets
        ]
        for ticker in tickers
    }
    checkpoint_path = store_dir / f"{record.spec_hash}.checkpoint.sqlite"

    with lake_session(lake_time, data_end=data_end) as session, _engine_quiet():
        store.open()
        last = store.last_completed()
        checkpointed = _checkpoint_last_day(checkpoint_path)
        if checkpointed is not None and (last is None or checkpointed > last):
            raise ValueError(
                f"the engine checkpoint {checkpoint_path.name} ends on {checkpointed} but the "
                f"store of trial {record.spec_hash} on {last}; the record cannot resume, so "
                "remove both and record again"
            )
        if last is not None and checkpointed != last:
            logger.info(
                "Firm record: the store is ahead of the checkpoint (%s); its days after %s "
                "are computed again",
                last,
                checkpointed,
            )
            store.rewind(checkpointed)
            last = checkpointed
        counted = False
        if ledger is not None and repo_root is not None:
            counted = (
                ensure_firm_registration(ledger=ledger, repo_root=repo_root, family=family)
                is not None
            )
        write_submission_folder(store.trial_dir, record, submission, meta)
        meta = _write_checkpoint_reports(store.trial_dir, checkpoint_path, meta)
        pending = [day for day in sessions if last is None or day > last]
        logger.info(
            "Firm record (trial %s, %s): %d of %d sessions to compute",
            record.spec_hash,
            "counted" if counted else "uncounted",
            len(pending),
            len(sessions),
        )
        written = 0
        if pending:
            frame = _on_sessions(market_data, sessions)
            engine = BacktestEngine(
                commander=commander,
                portfolio=Portfolio(
                    cash=settings.firm.starting_cash,
                    positions={},
                    commission_rate=settings.firm.commission_rate,
                    slippage_bps=settings.firm.slippage_bps,
                ),
                track_record=TrackRecord(),
                data_loader=lambda _assets, _start, _end: frame,
                start_date=window_start,
                end_date=window_end,
                assets=tickers,
                market_data_source="lakehouse data",
                execution_lag_days=settings.firm.execution_lag_days,
                checkpoint_path=checkpoint_path,
                resume=checkpoint_path.exists(),
                risk_free_rate_series=risk_free_rate_series,
            )
            refused_before = len(session.holdout_violations)
            previous = row_provenance(store.day_rows(last)) if last is not None else {}

            def write_day(_index: int, _total: int, day: date, value: float) -> None:
                nonlocal written, previous
                if max_days is not None and written >= max_days:
                    raise _RecordPausedError
                refused = session.holdout_violations[refused_before:]
                if refused:
                    raise HoldoutReadError(
                        f"the firm made {len(refused)} read(s) past data_end by "
                        f"{day.isoformat()}: " + "; ".join(refused)
                    )
                absent = {ticker: scopes[ticker] for ticker in _without_bar(frame, day, tickers)}
                rows = firm_day_rows(
                    day,
                    commander.pod_targets(),
                    absent=absent,
                    analyst_sources=sources_by_analyst,
                    instrument_kinds=kinds,
                    calendar=calendar,
                    previous=previous,
                    held_contracts=_held_contracts(engine.portfolio),
                    nav=value,
                )
                store.write_day(day, rows)
                previous = row_provenance(rows)
                written += 1

            try:
                engine.run(before_checkpoint=write_day)
            except _RecordPausedError:
                commander.close()
                logger.info("Firm record paused after %d day(s)", written)
            meta = _write_checkpoint_reports(store.trial_dir, checkpoint_path, meta)
        for flagged in meta.low_availability:
            logger.warning(
                "Firm record (trial %s): analyst %s had an available signal in %.1f%% of its "
                "signals, below the %.1f%% threshold (%s)",
                record.spec_hash,
                flagged.analyst_id,
                100.0 * flagged.available_share,
                100.0 * settings.firm.low_availability_threshold,
                "undeclared in the registration"
                if flagged.undeclared
                else f"declared {meta.expected_availability[flagged.analyst_id]!r}",
            )
        last_completed = store.last_completed()
        if counted and written and ledger is not None and last_completed is not None:
            ledger.record_trial(
                firm_trial_row(
                    record,
                    family,
                    n_days=len(sessions_between(window_start, last_completed, calendar)),
                    provenance=current_trial_provenance(),
                )
            )
    return FirmRecordResult(
        spec_hash=record.spec_hash,
        days_written=written,
        last_completed=last_completed,
        complete=last_completed == sessions[-1],
        counted=counted,
    )
