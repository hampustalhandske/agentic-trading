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
directory, which holds only the submission (spec.json, sources.json,
decisions.parquet, progress.json).

No performance figure leaves the pass: the result reports only what was
written, and the engine's value and ratio log lines are held back while it
runs. With a ledger, the firm's hypothesis is registered from its
hand-approved registry file before the first day is computed, and a pass
that wrote a day records one trial row (`trials.rows.firm_trial_row`); a
counted pass needs a clean `src/` tree at a known commit, which its
spec.json records. Without the registry file the pass is uncounted and the
ledger is not touched."""

import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict

from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.commander import Commander
from agentic_trading.agents.managers.cvar_manager import ConditionalValueAtRiskManager
from agentic_trading.agents.managers.tail_hedge_manager import TailHedgeManager
from agentic_trading.backtest import checkpoint_store
from agentic_trading.backtest.engine import BacktestEngine
from agentic_trading.backtest.portfolio import Portfolio
from agentic_trading.backtest.resume_fingerprint import roster_agents
from agentic_trading.config import settings
from agentic_trading.data.loaders.lakehouse_loader import HoldoutReadError, lake_session
from agentic_trading.data.trading_calendar import sessions_between
from agentic_trading.firm_record.rows import (
    analyst_sources,
    firm_day_rows,
    firm_sources,
    instrument_kind,
    row_provenance,
)
from agentic_trading.firm_record.spec import build_firm_spec
from agentic_trading.logging_config import get_logger
from agentic_trading.provenance import source_tree_state
from agentic_trading.scoring.track_record import TrackRecord
from agentic_trading.stage_a.spec import FirmSpecRecord, firm_spec_hash
from agentic_trading.stage_a.store import FirmDecisionStore
from agentic_trading.trials.ledger import TrialLedger
from agentic_trading.trials.rows import (
    FIRM_HYPOTHESIS_ID,
    current_trial_provenance,
    ensure_firm_registration,
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


def _checkpoint_last_day(path: Path) -> date | None:
    """The last day the engine checkpoint at `path` holds, `None` when the
    file is absent or holds no day."""
    if not path.exists():
        return None
    connection = checkpoint_store.connect(path, resume=True)
    try:
        payload = checkpoint_store.read_all(connection)
    finally:
        connection.close()
    if payload is None or not payload["portfolio_value_history"]:
        return None
    return date.fromisoformat(payload["portfolio_value_history"][-1][0])


def _held_premium(portfolio: Portfolio, value: float) -> dict[str, float]:
    """Each open option position's contract id to its premium weight in
    the book of `value`: contracts x its last mid x the contract
    multiplier / `value`; empty when `value` is not positive."""
    if not value > 0.0:
        return {}
    multiplier = float(settings.options.contract_multiplier)
    return {
        contract_id: position.contracts * position.last_mid * multiplier / value
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

    Raises `ValueError`, before writing anything, when the window is
    reversed, holds no session, or ends after `settings.stage_a.data_end`,
    when `max_days` is below 1, or when a `ledger` is given without a
    `repo_root`; `ValueError` when the engine checkpoint is ahead of the
    store, so the pass cannot resume, or when the pass would be counted but
    the `src/` tree has changes or no known commit.
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
    registry_file = (
        None
        if repo_root is None
        else Path(repo_root) / settings.trials.registry_dir / f"{FIRM_HYPOTHESIS_ID}.yaml"
    )
    will_count = ledger is not None and registry_file is not None and registry_file.exists()
    git_sha: str | None = None
    if will_count:
        git_sha, dirty = source_tree_state()
        if git_sha is None or dirty is not False:
            raise ValueError(
                "a counted firm record needs a clean src/ tree at a known commit; "
                f"HEAD is {git_sha}, src/ has changes: {dirty}"
            )
    spec = build_firm_spec(commander, tickers, window_start, window_end, lake_time)
    record = FirmSpecRecord(
        spec=spec,
        spec_hash=firm_spec_hash(spec),
        lake_time=spec.lake_time,
        lake_version=lake_version,
        quantcore_version=quantcore_version,
        git_sha=git_sha,
    )
    analysts = [agent for agent in roster_agents(commander) if isinstance(agent, AnalystAgent)]
    option_rows = any(
        isinstance(manager, ConditionalValueAtRiskManager | TailHedgeManager)
        for manager in commander.managers
    )
    store_dir = Path(store_dir)
    store = FirmDecisionStore(
        store_dir, record, sources=firm_sources(analysts, option_rows=option_rows)
    )
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
                    held_premium=_held_premium(engine.portfolio, value),
                )
                store.write_day(day, rows)
                previous = row_provenance(rows)
                written += 1

            try:
                engine.run(before_checkpoint=write_day)
            except _RecordPausedError:
                commander.close()
                logger.info("Firm record paused after %d day(s)", written)
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
