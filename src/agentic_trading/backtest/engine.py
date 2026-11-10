# engine.py

import itertools
import json
import math
import sqlite3
import time
from collections.abc import Callable, Iterator, Mapping
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import numpy as np
import pandas as pd
from quantcore.performance.metrics import (
    calmar_ratio,
    hit_rate,
    information_ratio,
    maximum_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
)

from agentic_trading.agents.commander import CommanderRiskState
from agentic_trading.agents.schemas import (
    BacktestResult,
    CapitalUsage,
    OptionContract,
    PerformanceRecord,
    PodCostAttribution,
    PodCostDay,
    PodPerformance,
    PortfolioDecision,
    SizingDiagnostics,
    TradeRecord,
    UnfilledOrder,
)
from agentic_trading.backtest import checkpoint_store
from agentic_trading.backtest.portfolio import MarkedPosition, OptionPosition, Portfolio
from agentic_trading.backtest.resume_fingerprint import (
    agents_by_id,
    roster_agents,
    roster_fingerprint,
    settings_fingerprint,
)
from agentic_trading.config import settings
from agentic_trading.data.instruments import currency_pair, is_marked
from agentic_trading.data.loaders import lakehouse_loader
from agentic_trading.data.loaders.option_marks import OptionMark, load_option_mark
from agentic_trading.data.point_in_time import get_point_in_time_data
from agentic_trading.logging_config import get_logger, log_progress
from agentic_trading.scoring.scoreboard import PerformanceScoreboard
from agentic_trading.scoring.track_record import HistoryHorizonError, TrackRecord
from agentic_trading.scoring.weighting import WeightingStrategy

logger = get_logger(__name__)


def _summary_metric(name: str, compute: Callable[[], float]) -> float | None:
    """`compute()` as a float, the run-summary metric `name`; `None` when
    the metric is undefined on this run: `compute` raises `ValueError`
    (quantcore's undefined-result error among them) or
    `ZeroDivisionError`, or returns a non-finite value. Each `None` logs a
    warning naming the metric and the reason."""
    try:
        value = float(compute())
    except (ValueError, ZeroDivisionError) as exc:
        logger.warning(
            "Summary metric %s is undefined on this run (%s); reporting None.", name, exc
        )
        return None
    if not math.isfinite(value):
        logger.warning(
            "Summary metric %s is not finite on this run (%s); reporting None.", name, value
        )
        return None
    return value


# Largest absolute daily excess return still treated as zero. A book that
# only ever holds cash earns exactly the benchmark accrual, so its excess
# returns are floating-point residue (around 1e-17); Sharpe and Sortino of
# that series are undefined (zero or near-zero variance gives a division by
# zero or noise), and the run reports None for both.
ZERO_EXCESS_RETURN_TOLERANCE = 1e-12

# The per-pod cost kinds `PodCostDay` and `PodCostAttribution` carry, one
# field each: commission, then the execution costs `TradeRecord` books
# against its reference price.
_COST_KINDS = ("transaction_cost", "slippage_cost", "impact_cost", "spread_cost")

# Called after each simulated day completes, with (day_index, total_days,
# as_of_date, portfolio_value) — the hook a caller (the GUI) uses to drive a
# live progress display. Safe to have this call UI code: unlike the GUI's
# old @st.cache_data-wrapped simulation loop, BacktestEngine.run() is never
# itself cached, so there's no CacheReplayClosureError risk here.
DayCompleteCallback = Callable[[int, int, date, float], None]

# The decision days a horizon-bounded engine holds in memory: the current
# day's and the pending day's (the one whose orders fill with a one-day
# execution lag).
HELD_DECISION_DAYS = 2


class DecisionWindow(Mapping[date, list[PortfolioDecision]]):
    """The Commander's decisions of the last `days` decision days, keyed
    by day in day order: what a horizon-bounded engine holds in memory.
    Setting a day's decisions drops the oldest day once more than `days`
    are held. A day before the earliest day held raises
    `HistoryHorizonError` once an earlier day was dropped (or left in the
    checkpoint on restore); any other day not held raises `KeyError`.

    Once an earlier day was dropped the window no longer holds every day,
    so iterating it, its `len` and its `keys`/`items`/`values` raise
    `HistoryHorizonError` too; `held()` returns the days it holds."""

    def __init__(self, days: int) -> None:
        if days < 1:
            raise ValueError(f"days must be at least 1, got {days}")
        self._days = days
        self._held: dict[date, list[PortfolioDecision]] = {}
        # The earliest day held once an earlier day is gone; None before.
        self._horizon_start: date | None = None

    def restore(
        self, held: Mapping[date, list[PortfolioDecision]], earlier_days_dropped: bool
    ) -> None:
        """Holds the last `days` days of `held`, in day order; days before
        them, and with `earlier_days_dropped` any day before the first of
        `held`, count as dropped."""
        ordered = sorted(held.items())
        kept = ordered[-self._days :]
        self._held = dict(kept)
        if kept and (earlier_days_dropped or len(kept) < len(ordered)):
            self._horizon_start = kept[0][0]

    def __setitem__(self, day: date, decisions: list[PortfolioDecision]) -> None:
        self._held[day] = decisions
        while len(self._held) > self._days:
            del self._held[next(iter(self._held))]
            self._horizon_start = next(iter(self._held))

    def __getitem__(self, day: date) -> list[PortfolioDecision]:
        decisions = self._held.get(day)
        if decisions is not None:
            return decisions
        if self._horizon_start is not None and day < self._horizon_start:
            raise HistoryHorizonError(
                f"the decisions of {day} were dropped; the engine holds the days from "
                f"{self._horizon_start}"
            )
        raise KeyError(day)

    def held(self) -> dict[date, list[PortfolioDecision]]:
        """The days held and their decisions, in day order, as a new dict."""
        return dict(self._held)

    def _require_every_day(self, reader: str) -> None:
        """Raises `HistoryHorizonError` naming `reader` once an earlier
        day was dropped."""
        if self._horizon_start is not None:
            raise HistoryHorizonError(
                f"{reader} needs every day's decisions, but the days before "
                f"{self._horizon_start} were dropped; held() returns the days held"
            )

    def __iter__(self) -> Iterator[date]:
        self._require_every_day("iterating the decisions")
        return iter(self._held)

    def __len__(self) -> int:
        self._require_every_day("len() of the decisions")
        return len(self._held)


def longest_track_record_lookback(commander: object) -> int:
    """The longest trailing track-record window, in calendar days ending
    on the decision day, that `commander` reads while deciding:
    `settings.portfolio.ic_lookback_days` (read now; each manager's
    measured information coefficients) or the `track_record_lookback_days`
    of the commander's own `weighting_strategy` or of any of its
    managers', whichever is longest."""
    strategies = [
        getattr(agent, "weighting_strategy", None)
        for agent in [commander, *roster_agents(commander)]
    ]
    return max(
        [settings.portfolio.ic_lookback_days]
        + [
            strategy.track_record_lookback_days()
            for strategy in strategies
            if isinstance(strategy, WeightingStrategy)
        ]
    )


class CommanderLike(Protocol):
    """What `BacktestEngine` actually needs from a commander — satisfied by
    `Commander` and by `backtest/baselines.py`'s `BaselineCommander`/
    `SingleAnalystCommander`, which exist to run simple benchmark/diagnostic
    strategies through this same engine without becoming `Commander`
    subclasses.

    A commander may additionally expose `risk_state() -> CommanderRiskState`
    and `restore_risk_state(state)`; the engine checkpoints/restores that
    state and reports its pod return history when present, and simply
    skips it otherwise. A commander with a `managers` list has each
    manager's and each managed analyst's `carried_state()` checkpointed
    and restored as well, and their parameters recorded in the roster
    fingerprint (see `resume_fingerprint.py`). A commander may also
    expose `pod_contributions() -> dict[str, dict[str, float]]` (pod id
    to asset to the pod's contribution on the latest decision); the
    engine splits fill costs across pods by it, and charges every fill to
    `"commander"` without it."""

    def decide(
        self,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
        scoreboard: PerformanceScoreboard,
        instrument_returns: dict[str, float] | None = None,
    ) -> list[PortfolioDecision]: ...


class BacktestEngine:
    """Deterministic order fills and portfolio accounting. NOT an agent —
    this class contains NO LLM calls and NO unseeded randomness. Every
    decision it acts on comes from `commander`; this class's only job is
    to apply that decision to `portfolio` day by day and report the
    result. Must be fully testable in isolation.

    The single simulation loop for this project — both the CLI and the GUI
    call `run()` directly rather than each maintaining their own copy. If a
    caller needs something this loop doesn't produce, that belongs here (as
    a richer `BacktestResult`) or as another method on this class, not as a
    second, parallel simulation implementation elsewhere.
    """

    def __init__(
        self,
        commander: CommanderLike,
        portfolio: Portfolio,
        track_record: TrackRecord,
        data_loader: Callable[[list[str], date, date], pd.DataFrame],
        start_date: date,
        end_date: date,
        assets: list[str] | None = None,
        market_data_source: str | None = None,
        execution_lag_days: int = 0,
        checkpoint_path: str | Path | None = None,
        resume: bool = False,
        risk_free_rate_series: pd.Series | None = None,
        benchmark_returns: pd.Series | None = None,
        policy_rates: pd.DataFrame | None = None,
        history_horizon_days: int | None = None,
    ) -> None:
        if history_horizon_days is not None:
            if history_horizon_days < 1:
                raise ValueError(
                    f"history_horizon_days must be at least 1, got {history_horizon_days}"
                )
            if checkpoint_path is None:
                raise ValueError(
                    "history_horizon_days needs a checkpoint_path: the run's whole history "
                    "is read back from the checkpoint"
                )
            if track_record.horizon_days != history_horizon_days:
                raise ValueError(
                    f"history_horizon_days {history_horizon_days} needs a TrackRecord built "
                    f"with the same horizon_days, not {track_record.horizon_days}"
                )
            longest = longest_track_record_lookback(commander)
            if history_horizon_days < longest:
                raise ValueError(
                    f"history_horizon_days {history_horizon_days} is shorter than the "
                    f"{longest}-day track-record window the commander reads"
                )
        self.commander = commander
        self.portfolio = portfolio
        self.track_record = track_record
        self.data_loader = data_loader
        self.start_date = start_date
        self.end_date = end_date
        self.assets = assets or ["AAPL"]
        self.market_data_source = market_data_source
        if execution_lag_days not in {0, 1}:
            raise ValueError("execution_lag_days must be 0 or 1")
        self.execution_lag_days = execution_lag_days
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        self.resume = resume
        # Daily decimal rate (annual rate / trading_days_per_year), indexed
        # by calendar date (e.g.
        # `data/loaders/macro_loader.py::load_risk_free_rate_series`). The
        # rate in force on a valuation date is what cash earns, what
        # borrowed cash and short-option margin pay (plus the margin
        # spread), and what the Sharpe/Sortino benchmark accrues over the
        # gap to the next valuation date. A missing day (and a
        # fully-omitted series) both default to 0.0 rather than raising, so
        # a caller that doesn't care about this (e.g. a unit test building
        # a bare BacktestEngine) is unaffected.
        self.risk_free_rate_series = risk_free_rate_series
        # Daily decimal return, indexed by calendar date (e.g. an S&P 500
        # buy-and-hold's own pct_change()) — the benchmark
        # `final_information_ratio` is computed against. Left `None`
        # (rather than an arbitrary stand-in) when no caller supplies one.
        self.benchmark_returns = benchmark_returns
        # Central-bank policy rates as annual decimals, one column per
        # currency (ISO code), indexed by observation date (e.g.
        # `data/loaders/macro_loader.py::load_fx_policy_rates`). A spot FX
        # pair accrues carry over each gap between valuation dates at its
        # base minus its quote currency's rate in force on the earlier date
        # (`_carry_rates`). Omitted, or without either currency of a pair,
        # the pair accrues no carry.
        self.policy_rates = policy_rates
        # Calendar days of history held in memory, or None for all of it.
        # With a horizon, each session drops the track record's records
        # dated more than this many days before the session, the decisions
        # of all but the last `HELD_DECISION_DAYS` days are dropped, and
        # the whole-history results (`source_performance`) are read from
        # the checkpoint. The checkpoint, every read within the horizon and
        # the run's results are the same as without one, but for the
        # result's `decisions_by_day`, which holds only the days held, and
        # its `history_horizon_days`, which records the horizon.
        self.history_horizon_days = history_horizon_days
        # The decisions of each day of the current run, keyed by day: every
        # day without a horizon, the last `HELD_DECISION_DAYS` with one
        # (a `DecisionWindow`). Filled as `run()` steps through the days.
        self.decisions_by_day: Mapping[date, list[PortfolioDecision]] = {}
        self._checkpoint_connection: sqlite3.Connection | None = None
        # Previous day's close mark per option contract, for attribution
        # (log of today's mid over yesterday's); a contract seen for the
        # first time realizes 0.0. The checkpoint carries it whole.
        self._option_last_mid: dict[str, float] = {}
        self._unpriced_underlyings_warned: set[str] = set()

    def _risk_free_rate_annual(self, day: date) -> float:
        """Annualized risk-free rate in force on `day`: the daily decimal
        `risk_free_rate_series` value times `settings.firm.trading_days_per_year`,
        floored at 0.0 (the same floor `Portfolio.apply_financing` applies),
        and 0.0 when the series is absent or does not cover `day`."""
        if self.risk_free_rate_series is None:
            return 0.0
        daily_rate = float(self.risk_free_rate_series.get(pd.Timestamp(day), 0.0))
        return max(daily_rate * settings.firm.trading_days_per_year, 0.0)

    def _carry_rates(self, day: date) -> dict[str, float]:
        """Each spot FX pair among `assets` mapped to its base currency's
        policy rate minus its quote currency's in `policy_rates`, each the
        latest observation dated on or before `day`. A pair missing either
        rate has no entry; empty without `policy_rates`."""
        if self.policy_rates is None:
            return {}
        rates = self.policy_rates.sort_index()
        position = int(rates.index.searchsorted(pd.Timestamp(day), side="right"))
        in_force = rates.iloc[:position].ffill()
        if in_force.empty:
            return {}
        latest = in_force.iloc[-1]
        carry: dict[str, float] = {}
        for asset in self.assets:
            pair = currency_pair(asset)
            if pair is None or pair.base not in latest.index or pair.quote not in latest.index:
                continue
            differential = float(latest[pair.base]) - float(latest[pair.quote])
            if math.isfinite(differential):
                carry[asset] = differential
        return carry

    def _risk_free_accrual_array(self, valuation_dates: list[date]) -> np.ndarray:
        """Risk-free accrual over each gap between consecutive
        `valuation_dates`: for the return ending on date t with previous
        valuation date t-1, `rf_annual(t-1) * calendar_days(t-1, t) / 365`.
        That is simple actual/365 interest at the rate in force when the
        period starts, exactly what `Portfolio.apply_financing` credits a
        positive cash balance over the same gap, so an all-cash book's
        excess return is zero. One entry per consecutive pair (length
        `len(valuation_dates) - 1`, empty for fewer than two dates)."""
        return np.array(
            [
                self._risk_free_rate_annual(previous) * (current - previous).days / 365.0
                for previous, current in itertools.pairwise(valuation_dates)
            ],
            dtype=np.float64,
        )

    def _commander_risk_state(self) -> CommanderRiskState | None:
        """The commander's current risk state, or `None` for a commander
        that doesn't expose one."""
        risk_state = getattr(self.commander, "risk_state", None)
        if risk_state is None:
            return None
        state = risk_state()
        return state if isinstance(state, CommanderRiskState) else None

    def _restore_commander_risk_state(self, payload: dict[str, Any] | None) -> None:
        """Hands a checkpointed risk state back to the commander via
        `restore_risk_state`; a no-op when the checkpoint carries none or
        the commander can't accept one."""
        restore = getattr(self.commander, "restore_risk_state", None)
        if payload is None or restore is None:
            return
        restore(CommanderRiskState.model_validate(payload))

    def _carried_states(self) -> dict[str, bytes]:
        """Each of the commander's managers' and managed analysts'
        `carried_state()`, keyed by agent_id; empty for a commander without
        managers."""
        return {
            agent_id: agent.carried_state()
            for agent_id, agent in agents_by_id(self.commander).items()
        }

    def _restore_carried_states(self, carried_states: dict[str, bytes]) -> None:
        """Hands each checkpointed `carried_state()` snapshot back to the
        commander's agent with the same agent_id."""
        agents = agents_by_id(self.commander)
        for agent_id, state in carried_states.items():
            agents[agent_id].restore_carried_state(state)

    def _pod_contributions(self) -> dict[str, dict[str, float]] | None:
        """The commander's pod contributions on its latest decision with
        zero entries left out, or `None` for a commander that doesn't
        report them."""
        pod_contributions = getattr(self.commander, "pod_contributions", None)
        if not callable(pod_contributions):
            return None
        return {
            pod_id: {asset: float(weight) for asset, weight in weights.items() if weight != 0.0}
            for pod_id, weights in pod_contributions().items()
        }

    def _new_pod_returns(
        self,
        as_of_date: date,
        pod_return_history: dict[str, list[tuple[date, float]]],
        seen: set[tuple[str, date]],
    ) -> list[tuple[str, date, float]]:
        """Each pod's return dated `as_of_date` in the commander's risk
        state that is not yet in `seen`; the returned pairs are added to
        `seen`. A negative zero return is kept as positive zero, the value
        the checkpoint's SQLite REAL column holds, so a resumed run reads
        back what it recorded. Empty for a commander with no risk state."""
        state = self._commander_risk_state()
        if state is None:
            return []
        new_entries: list[tuple[str, date, float]] = []
        for pod_id, entries in state.pod_return_history.items():
            for entry_date, value in entries:
                if entry_date == as_of_date and (pod_id, entry_date) not in seen:
                    seen.add((pod_id, entry_date))
                    value = value + 0.0
                    pod_return_history.setdefault(pod_id, []).append((entry_date, value))
                    new_entries.append((pod_id, entry_date, value))
        return new_entries

    def _benchmark_returns_array(self, return_dates: list[date]) -> np.ndarray | None:
        """Daily decimal benchmark return for each of `return_dates`, from
        `self.benchmark_returns`. Returns `None` (skip information_ratio
        entirely rather than compute it against a partially-fabricated
        series) if any date is missing — a benchmark that doesn't cover
        this run's actual trading calendar isn't the right benchmark to
        compare against."""
        if self.benchmark_returns is None:
            return None
        aligned = self.benchmark_returns.reindex([pd.Timestamp(day) for day in return_dates])
        if aligned.isna().any():
            missing = int(aligned.isna().sum())
            logger.warning(
                "benchmark_returns is missing %d/%d date(s) in this run's window — "
                "skipping information_ratio rather than filling gaps with 0.0.",
                missing,
                len(return_dates),
            )
            return None
        return aligned.to_numpy(dtype=np.float64)

    def _checkpoint(self) -> sqlite3.Connection | None:
        """Opens (creating tables if needed) and caches the checkpoint
        connection for the lifetime of one `run()` call — one connection
        for the whole run, not one per day."""
        if self.checkpoint_path is None:
            return None
        if self._checkpoint_connection is None:
            self._checkpoint_connection = checkpoint_store.connect(
                self.checkpoint_path, resume=self.resume
            )
        return self._checkpoint_connection

    def _run_config(self) -> dict[str, Any]:
        """This backtest's config as a checkpoint records it: its dates and
        assets, the settings fingerprint and the roster fingerprint of its
        commander, read now."""
        return {
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "assets": self.assets,
            "settings_fingerprint": settings_fingerprint(),
            "roster_fingerprint": roster_fingerprint(self.commander),
        }

    def _restore_checkpoint(self, run_config: dict[str, Any]) -> dict[str, Any] | None:
        """The checkpointed state to resume from, or `None` when not
        resuming or the checkpoint holds no day yet. With a history
        horizon it holds only the track record dated on or after the last
        checkpointed day minus the horizon and the last
        `HELD_DECISION_DAYS` days of decisions (`checkpoint_store.read_recent`).
        Raises `ValueError`
        when the checkpoint's config differs from `run_config` (other
        dates, assets, strategy settings or roster; a checkpoint written
        without fingerprints never matches) or it carries state for an
        agent this commander does not run, or it has no stored option marks
        (`option_last_mid` is NULL) yet the run had an option decision or
        holds an option position, so its option realized returns cannot be
        restored exactly;
        and `LakeTimeError` when its
        stored lake time differs from the open lake session's (`None` on
        either side meaning no session), since the resumed days would read
        another lake state. Nothing is restored by this method."""
        if not self.resume:
            return None
        connection = self._checkpoint()
        if connection is None:
            return None
        if self.history_horizon_days is None:
            payload = checkpoint_store.read_all(connection)
        else:
            last_day = checkpoint_store.read_last_day(connection)
            payload = (
                None
                if last_day is None
                else checkpoint_store.read_recent(
                    connection,
                    last_day - timedelta(days=self.history_horizon_days),
                    HELD_DECISION_DAYS,
                )
            )
        if payload is None:
            return None
        if payload.get("config") != run_config:
            raise ValueError("checkpoint configuration does not match this backtest")
        unknown = sorted(set(payload.get("carried_states", {})) - set(agents_by_id(self.commander)))
        if unknown:
            raise ValueError(f"checkpoint carries state for unknown agents: {unknown}")
        stored_lake_time = payload.get("lake_time")
        lake_time = lakehouse_loader.current_lake_time()
        if stored_lake_time != lake_time:
            raise lakehouse_loader.LakeTimeError(
                f"checkpoint {self.checkpoint_path} was written at lake time "
                f"{_lake_time_text(stored_lake_time)}, but this run reads at "
                f"{_lake_time_text(lake_time)}; resume it at its own lake time"
            )
        if payload.get("option_last_mid") is None and _checkpoint_traded_options(
            connection, payload
        ):
            raise ValueError(
                f"checkpoint {self.checkpoint_path} has no stored option marks; its option "
                "realized returns cannot be restored exactly; rerun from the start"
            )
        return payload

    def _underlying_spot(
        self, raw_market_data: pd.DataFrame, underlying: str, as_of_date: pd.Timestamp, column: str
    ) -> float | None:
        """`underlying`'s `column` ("close"/"open") price on `as_of_date`
        from the run's frame; `None` (warned once per underlying) when the
        frame doesn't carry it."""
        value: float | None = None
        if isinstance(raw_market_data.columns, pd.MultiIndex):
            # Options are struck on unadjusted prices: the close mark uses
            # `raw_close`; the open fill scales the adjusted open back to
            # the raw basis by the same factor.
            if column == "close" and (underlying, "raw_close") in raw_market_data.columns:
                value = float(
                    cast(float, raw_market_data.loc[as_of_date, (underlying, "raw_close")])
                )
            elif (underlying, column) in raw_market_data.columns:
                value = float(cast(float, raw_market_data.loc[as_of_date, (underlying, column)]))
                if column == "open" and (underlying, "raw_close") in raw_market_data.columns:
                    adjusted_close = float(
                        cast(float, raw_market_data.loc[as_of_date, (underlying, "close")])
                    )
                    raw_close = float(
                        cast(float, raw_market_data.loc[as_of_date, (underlying, "raw_close")])
                    )
                    if adjusted_close > 0.0 and not math.isnan(raw_close):
                        value = value * raw_close / adjusted_close
        elif underlying == self.assets[0] and column in raw_market_data.columns:
            value = float(cast(float, raw_market_data.loc[as_of_date, column]))
        if value is None or math.isnan(value):
            if underlying not in self._unpriced_underlyings_warned:
                self._unpriced_underlyings_warned.add(underlying)
                logger.warning(
                    "Underlying %s is not priced in this run's frame — its option "
                    "positions are carried at their last known mid.",
                    underlying,
                )
            return None
        return value

    def _option_mark(
        self,
        raw_market_data: pd.DataFrame,
        contract_id: str,
        as_of_date: pd.Timestamp,
        column: str,
    ) -> OptionMark | None:
        """The mark for `contract_id` on `as_of_date` at the underlying's
        `column` price, `None` when the underlying is unpriced here or the
        contract has no quote in the snapshot window."""
        contract = OptionContract.parse(contract_id)
        spot = self._underlying_spot(raw_market_data, contract.underlying, as_of_date, column)
        if spot is None:
            return None
        return load_option_mark(contract, as_of_date.date(), spot)

    def _option_realized_return(self, contract_id: str, mark: OptionMark | None) -> float:
        """Log return of the contract's close mid over the previous day's,
        0.0 on its first day or without a usable mark."""
        previous = self._option_last_mid.get(contract_id, 0.0)
        if mark is None:
            return 0.0
        self._option_last_mid[contract_id] = mark.mid
        if previous <= 0.0 or mark.mid <= 0.0:
            return 0.0
        return math.log(mark.mid / previous)

    def _source_returns(self) -> dict[str, tuple[list[float], list[float]]]:
        """Each source_id's attributable and raw attributable returns over
        its track-record rows dated inside the run's window, in recording
        order, sources in order of first appearance. Read from the
        in-memory track record, or with a history horizon from the
        checkpoint, which holds every row in the same order."""
        returns_by_source: dict[str, tuple[list[float], list[float]]] = {}
        if self.history_horizon_days is not None:
            connection = self._checkpoint()
            assert connection is not None  # a horizon needs a checkpoint_path
            for source_id, attributable, raw_attributable in checkpoint_store.iter_source_returns(
                connection, self.start_date, self.end_date
            ):
                returns, raw_returns = returns_by_source.setdefault(source_id, ([], []))
                returns.append(attributable)
                raw_returns.append(raw_attributable)
            return returns_by_source
        entries = getattr(self.track_record, "_entries", None)
        if entries is None:
            entries = getattr(self.track_record, "entries", [])
        for entry in entries:
            if self.start_date <= entry.as_of_date <= self.end_date:
                returns, raw_returns = returns_by_source.setdefault(entry.source_id, ([], []))
                returns.append(entry.attributable_return)
                raw_returns.append(entry.raw_attributable_return)
        return returns_by_source

    def _source_performance(self) -> dict[str, dict[str, float]]:
        """Each source's attribution statistics over the run's window (see
        `_stats` below), from `_source_returns`."""

        def _stats(returns: np.ndarray) -> tuple[float, float, float, float]:
            """Returns (annualized_return, total_attributed_return, sharpe,
            max_drawdown). `total_attributed_return` is `np.sum(returns)`
            — a plain running sum across every
            asset-day this source touched, not a compounded return, kept
            under its own explicit name since it grows without bound over
            a long run and is easy to mistake for one. `annualized_return`
            (`mean(returns) * periods_per_year`) is what `cumulative_return`
            reports below instead: a rate, comparable across sources
            regardless of how many days/assets each one happened to have
            observations for, the same normalization a Sharpe ratio's own
            numerator already uses."""
            cumulative = np.cumsum(returns)
            running_peak = np.maximum.accumulate(cumulative)
            drawdown = float(np.min(cumulative - running_peak)) if cumulative.size else 0.0
            standard_deviation = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
            mean_return = float(np.mean(returns)) if returns.size else 0.0
            sharpe = (
                float(mean_return / standard_deviation * np.sqrt(252))
                if standard_deviation > 0.0
                else 0.0
            )
            return mean_return * 252.0, float(np.sum(returns)), sharpe, drawdown

        performance: dict[str, dict[str, float]] = {}
        for source_id, (attributable, raw_attributable) in self._source_returns().items():
            returns = np.asarray(attributable, dtype=float)
            raw_returns = np.asarray(raw_attributable, dtype=float)
            annualized_return, total_attributed_return, sharpe, drawdown = _stats(returns)
            raw_annualized_return, raw_total_attributed_return, raw_sharpe, raw_drawdown = _stats(
                raw_returns
            )
            performance[source_id] = {
                "cumulative_return": annualized_return,
                "total_attributed_return": total_attributed_return,
                "sharpe_ratio": sharpe,
                "max_drawdown": drawdown,
                "observations": float(len(returns)),
                "raw_cumulative_return": raw_annualized_return,
                "raw_total_attributed_return": raw_total_attributed_return,
                "raw_sharpe_ratio": raw_sharpe,
                "raw_max_drawdown": raw_drawdown,
            }
        return performance

    def run(
        self,
        on_day_complete: DayCompleteCallback | None = None,
        *,
        before_checkpoint: DayCompleteCallback | None = None,
    ) -> BacktestResult:
        """Steps through every simulated day from start_date to end_date,
        point-in-time slicing data (see data/point_in_time.py), calling
        the Commander for decisions, applying them to the Portfolio, and
        recording realized performance into track_record. Returns a
        BacktestResult summarizing the full run, including every day's
        PortfolioDecisions and the full trade log — the same data both the
        CLI and the GUI's Desks/Analysts drill-down are built from. With a
        `history_horizon_days`, the result's `decisions_by_day` holds only
        the last `HELD_DECISION_DAYS` days (the checkpoint holds every
        day's) and its `history_horizon_days` records the horizon;
        everything else in it is as without a horizon. A run that resumed
        from a checkpoint's saved days returns `unfactored_split_moves` as
        `None`: the split-move registry is not checkpointed, so this process
        has seen only its own reads.

        `on_day_complete`, if given, is called once per simulated day after
        that day's trades are applied — the hook a live progress UI hangs
        off of. Purely a notification; this method's own behavior does not
        depend on it being present.

        `before_checkpoint`, if given, is called with the same arguments
        once per simulated day after the day is fully accounted and before
        the day is appended to the checkpoint, so what it writes is never
        behind the checkpoint; an exception it raises stops the run with
        the checkpoint still at the day before.
        """
        logger.info(
            "Backtest starting: assets=%s, %s to %s", self.assets, self.start_date, self.end_date
        )
        run_started_at = time.perf_counter()
        lakehouse_loader.clear_unfactored_split_move_registry()
        raw_market_data = self.data_loader(self.assets, self.start_date, self.end_date)
        if raw_market_data.empty:
            logger.warning(
                "Backtest aborted: data_loader returned no rows for assets=%s (%s to %s) — "
                "returning an empty BacktestResult instead of a real backtest.",
                self.assets,
                self.start_date,
                self.end_date,
            )
            return BacktestResult(
                start_date=self.start_date,
                end_date=self.end_date,
                portfolio_value_history=[],
                final_sharpe_ratio=0.0,
                final_max_drawdown=0.0,
                final_sortino_ratio=0.0,
                final_calmar_ratio=0.0,
                final_hit_rate=0.0,
                final_profit_factor=0.0,
                history_horizon_days=self.history_horizon_days,
                unfactored_split_moves=lakehouse_loader.unfactored_split_move_registry(),
            )

        days = [
            pd.Timestamp(day)
            for day in raw_market_data.index
            if self.start_date <= pd.Timestamp(day).date() <= self.end_date
        ]

        run_config = self._run_config() if self.checkpoint_path is not None else {}
        checkpoint = self._restore_checkpoint(run_config)
        # Only a run that trades a future or a spot FX pair can hold a
        # marked position to settle, roll or accrue carry on.
        trades_marked = any(is_marked(asset) for asset in self.assets)
        start_day_index = int(checkpoint.get("next_day_index", 0)) if checkpoint else 0

        scoreboard = PerformanceScoreboard(self.track_record)
        portfolio_value_history: list[tuple[date, float]] = (
            [
                (date.fromisoformat(day), value)
                for day, value in checkpoint["portfolio_value_history"]
            ]
            if checkpoint
            else []
        )
        portfolio_values = [value for _, value in portfolio_value_history]
        restored_decisions: dict[date, list[PortfolioDecision]] = (
            {
                date.fromisoformat(day): [
                    PortfolioDecision.model_validate(decision) for decision in decisions
                ]
                for day, decisions in checkpoint["decisions_by_day"].items()
            }
            if checkpoint
            else {}
        )
        decisions_by_day: dict[date, list[PortfolioDecision]] | DecisionWindow
        if self.history_horizon_days is None:
            decisions_by_day = restored_decisions
        else:
            decisions_by_day = DecisionWindow(HELD_DECISION_DAYS)
            decisions_by_day.restore(
                restored_decisions,
                earlier_days_dropped=checkpoint is not None
                and int(checkpoint["next_day_index"]) > len(restored_decisions),
            )
        self.decisions_by_day = decisions_by_day
        trade_log = (
            [TradeRecord.model_validate(trade) for trade in checkpoint["trade_log"]]
            if checkpoint
            else []
        )
        unfilled_orders = (
            [UnfilledOrder.model_validate(order) for order in checkpoint.get("unfilled_orders", [])]
            if checkpoint
            else []
        )
        capital_usage = (
            [CapitalUsage.model_validate(entry) for entry in checkpoint.get("capital_usage", [])]
            if checkpoint
            else []
        )
        sizing_diagnostics = (
            [
                SizingDiagnostics.model_validate(entry)
                for entry in checkpoint.get("sizing_diagnostics", [])
            ]
            if checkpoint
            else []
        )
        pod_cost_days = (
            [PodCostDay.model_validate(entry) for entry in checkpoint.get("pod_cost_days", [])]
            if checkpoint
            else []
        )
        pod_return_history: dict[str, list[tuple[date, float]]] = {}
        seen_pod_returns: set[tuple[str, date]] = set()
        if checkpoint:
            for pod_id, entries in checkpoint.get("pod_return_history", {}).items():
                pod_return_history[pod_id] = [
                    (date.fromisoformat(day), float(value)) for day, value in entries
                ]
                seen_pod_returns.update(
                    (pod_id, entry_date) for entry_date, _ in pod_return_history[pod_id]
                )
        signal_health = checkpoint.get("signal_health", {}) if checkpoint else {}
        pending_decisions = (
            [
                PortfolioDecision.model_validate(decision)
                for decision in checkpoint["pending_decisions"]
            ]
            if checkpoint
            else []
        )
        if checkpoint:
            portfolio_state = checkpoint["portfolio"]
            self.portfolio.cash = float(portfolio_state["cash"])
            self.portfolio.positions = {
                asset: float(quantity) for asset, quantity in portfolio_state["positions"].items()
            }
            self.portfolio.marked_positions = {
                asset: MarkedPosition.model_validate(position)
                for asset, position in portfolio_state.get("marked_positions", {}).items()
            }
            self.portfolio.option_positions = {
                contract_id: OptionPosition.model_validate(position)
                for contract_id, position in portfolio_state.get("option_positions", {}).items()
            }
            # `_restore_checkpoint` refuses a checkpoint without option marks
            # unless the run never touched an option, whose marks are empty.
            self._option_last_mid = {
                contract_id: float(mid)
                for contract_id, mid in (checkpoint.get("option_last_mid") or {}).items()
            }
            restored_records = [
                PerformanceRecord.model_validate(entry) for entry in checkpoint["track_record"]
            ]
            if self.history_horizon_days is None:
                self.track_record.restore(restored_records)
            else:
                self.track_record.restore(
                    restored_records,
                    cutoff=portfolio_value_history[-1][0]
                    - timedelta(days=self.history_horizon_days),
                    recorded_count=int(checkpoint["track_record_count"]),
                    dropped_first_dates={
                        source_id: date.fromisoformat(first)
                        for source_id, first in checkpoint[
                            "track_record_dropped_first_dates"
                        ].items()
                    },
                )
            self._restore_commander_risk_state(checkpoint.get("commander_risk_state"))
            # After the risk state, which restores the Commander's own state
            # and each manager's constructor state; the carried state then
            # sets every agent's private attributes to the exact snapshot of
            # the same day, overlapping values included.
            self._restore_carried_states(checkpoint.get("carried_states", {}))
        total_financing_cost = (
            float(checkpoint.get("total_financing_cost", 0.0)) if checkpoint else 0.0
        )
        total_option_premium_paid = (
            float(checkpoint.get("total_option_premium_paid", 0.0)) if checkpoint else 0.0
        )

        for day_index, as_of_timestamp in enumerate(days[start_day_index:], start=start_day_index):
            day_started_at = time.perf_counter()
            if self.history_horizon_days is not None:
                self.track_record.drop_before(
                    as_of_timestamp.date() - timedelta(days=self.history_horizon_days)
                )
            market_data = get_point_in_time_data(raw_market_data, as_of_timestamp)
            # Today's mark-to-mark log return of every open option position,
            # so the Commander's pod-return tracker sees option P&L the way
            # it sees equity returns from `market_data`.
            instrument_returns: dict[str, float] = {}
            for contract_id in list(self.portfolio.option_positions):
                mark = self._option_mark(raw_market_data, contract_id, as_of_timestamp, "close")
                previous = self._option_last_mid.get(contract_id, 0.0)
                if mark is not None and previous > 0.0 and mark.mid > 0.0:
                    instrument_returns[contract_id] = math.log(mark.mid / previous)
            decisions = self.commander.decide(
                market_data, as_of_timestamp, scoreboard, instrument_returns=instrument_returns
            )
            logger.debug(
                "Day %s: commander.decide across %d assets took %.1fms",
                as_of_timestamp.date(),
                len(self.assets),
                (time.perf_counter() - day_started_at) * 1000,
            )
            decisions_by_day[as_of_timestamp.date()] = decisions
            contributions = self._pod_contributions()
            # The contribution changes the day's fills are split by: today's
            # decision against the previous one, or with a one-day lag the
            # previous decision (whose fills execute today) against the one
            # before it.
            if self.execution_lag_days:
                fill_deltas = (
                    _contribution_deltas(
                        pod_cost_days[-2].contributions if len(pod_cost_days) > 1 else {},
                        pod_cost_days[-1].contributions,
                    )
                    if pod_cost_days
                    else None
                )
            else:
                fill_deltas = _contribution_deltas(
                    pod_cost_days[-1].contributions if pod_cost_days else {}, contributions
                )
            day_pod_costs = PodCostDay(
                as_of_date=as_of_timestamp.date(), contributions=contributions
            )
            new_pod_returns = self._new_pod_returns(
                as_of_timestamp.date(), pod_return_history, seen_pod_returns
            )

            for decision in decisions:
                for signal in decision.contributing_signals:
                    source_health = signal_health.setdefault(
                        signal.source_agent_id,
                        {"active": 0, "unavailable": 0, "error": 0},
                    )
                    source_health[signal.status] += 1
                    manager_health = signal_health.setdefault(
                        signal.manager_id,
                        {"active": 0, "unavailable": 0, "error": 0},
                    )
                    manager_health[signal.status] += 1

            if isinstance(raw_market_data.columns, pd.MultiIndex):
                prices = {
                    asset: float(
                        cast(float, raw_market_data.loc[as_of_timestamp, (asset, "close")])
                    )
                    for asset in self.assets
                }
                log_returns = {
                    asset: float(
                        cast(float, raw_market_data.loc[as_of_timestamp, (asset, "log_return")])
                    )
                    for asset in self.assets
                }
                execution_prices = (
                    {
                        asset: float(
                            cast(float, raw_market_data.loc[as_of_timestamp, (asset, "open")])
                        )
                        for asset in self.assets
                    }
                    if self.execution_lag_days
                    else prices
                )
            else:
                prices = {
                    asset: float(cast(float, raw_market_data.loc[as_of_timestamp, "close"]))
                    for asset in self.assets
                }
                log_returns = {
                    asset: float(cast(float, raw_market_data.loc[as_of_timestamp, "log_return"]))
                    for asset in self.assets
                }
                execution_prices = (
                    {
                        self.assets[0]: float(
                            cast(float, raw_market_data.loc[as_of_timestamp, "open"])
                        )
                    }
                    if self.execution_lag_days
                    else prices
                )
            # Close marks for every open option position (and every option
            # decision executing today, for attribution); positions at or
            # past expiry settle at intrinsic before the day is valued.
            option_marks: dict[str, OptionMark | None] = {}
            option_contract_ids = set(self.portfolio.option_positions) | {
                decision.asset
                for decision in (pending_decisions if self.execution_lag_days else decisions)
                if decision.instrument is not None
            }
            for contract_id in option_contract_ids:
                option_marks[contract_id] = self._option_mark(
                    raw_market_data, contract_id, as_of_timestamp, "close"
                )
            marks = {
                contract_id: mark for contract_id, mark in option_marks.items() if mark is not None
            }
            underlying_prices: dict[str, float] = {}
            for position in self.portfolio.option_positions.values():
                underlying = position.contract.underlying
                if underlying in underlying_prices:
                    continue
                spot = self._underlying_spot(raw_market_data, underlying, as_of_timestamp, "close")
                if spot is not None:
                    underlying_prices[underlying] = spot
            trades_before = len(trade_log)
            unfilled_before = len(unfilled_orders)
            # Futures and spot FX settle the day's move into cash at the
            # close, before the day is valued.
            if trades_marked:
                self.portfolio.settle_marked_positions(prices)
            expired = self.portfolio.settle_expired(marks, as_of_timestamp.date())
            trade_log.extend(expired)
            for trade in expired:
                _charge(day_pod_costs.transaction_cost, trade.transaction_cost, None)
            # An asset without a close today (no bar on this session) is
            # valued at its last close and never filled at that price; a
            # future or FX pair has no price at all that day, so it is not
            # re-marked and its P&L for the day is zero.
            valuation_prices = _valuation_prices(
                raw_market_data,
                as_of_timestamp,
                prices,
                set(self.portfolio.marked_positions) if trades_marked else set(),
            )
            portfolio_value = self.portfolio.total_value(valuation_prices, marks)
            # A future held through a roll pays one round trip on its
            # notional, split across the pods whose contributions it holds.
            if trades_marked:
                rolled = self.portfolio.charge_rolls(
                    as_of_timestamp.date(),
                    prices,
                    _frame_field(raw_market_data, self.assets, as_of_timestamp, "roll"),
                    portfolio_value,
                )
                held = (
                    pod_cost_days[-1 - self.execution_lag_days].contributions
                    if len(pod_cost_days) > self.execution_lag_days
                    else None
                )
                for trade in rolled:
                    trade_log.append(trade)
                    _charge_fill(day_pod_costs, trade, _holding_shares(held))
                    portfolio_value -= trade.transaction_cost + trade.slippage_cost
            # Financing (net of the interest cash earns) accrues over the
            # calendar days since the previous simulated day (a
            # Friday-to-Monday gap accrues three days), on the book carried
            # across that gap, at the risk-free rate in force on the
            # previous valuation date, when the gap starts. It is booked
            # after the mark so the day's recorded value already reflects it.
            # Spot FX carry accrues over the same gap at the policy-rate
            # differential in force on that previous date.
            previous_valuation_date = (
                portfolio_value_history[-1][0] if portfolio_value_history else None
            )
            days_held = (
                (as_of_timestamp.date() - previous_valuation_date).days
                if previous_valuation_date is not None
                else 0
            )
            financing_cost = self.portfolio.apply_financing(
                valuation_prices,
                (
                    self._risk_free_rate_annual(previous_valuation_date)
                    if previous_valuation_date is not None
                    else 0.0
                ),
                days_held,
                spots=underlying_prices,
            )
            if trades_marked and previous_valuation_date is not None:
                financing_cost += self.portfolio.apply_carry(
                    self._carry_rates(previous_valuation_date), days_held
                )
            total_financing_cost += financing_cost
            # Positive zero for a negative zero, as the checkpoint's SQLite
            # REAL column stores it.
            portfolio_value = portfolio_value - financing_cost + 0.0
            portfolio_values.append(portfolio_value)
            portfolio_value_history.append((as_of_timestamp.date(), portfolio_value))
            log_progress(
                logger,
                "Day %d/%d (%s) simulated — portfolio value $%.2f",
                day_index + 1,
                len(days),
                as_of_timestamp.date(),
                portfolio_value,
            )

            track_record_count_before = len(self.track_record)
            decisions_to_execute = pending_decisions if self.execution_lag_days else decisions
            for decision in decisions_to_execute:
                if decision.instrument is not None:
                    fill_mark = (
                        self._option_mark(raw_market_data, decision.asset, as_of_timestamp, "open")
                        if self.execution_lag_days
                        else option_marks.get(decision.asset)
                    )
                    if fill_mark is None:
                        logger.warning(
                            "Day %s: no mark for %s — option decision not executed.",
                            as_of_timestamp.date(),
                            decision.asset,
                        )
                        unfilled_orders.append(
                            UnfilledOrder(
                                as_of_date=as_of_timestamp.date(),
                                asset=decision.asset,
                                target_weight=decision.target_weight,
                                reason="no_option_mark",
                                price=None,
                            )
                        )
                    else:
                        fill_spot = self._underlying_spot(
                            raw_market_data,
                            decision.instrument.underlying,
                            as_of_timestamp,
                            "open" if self.execution_lag_days else "close",
                        )
                        split_factor = lakehouse_loader.split_adjustment_factor(
                            decision.instrument.underlying, as_of_timestamp.date()
                        )
                        if (
                            not fill_mark.tradeable
                            and self.portfolio.option_target_differs_from_held(
                                decision,
                                fill_mark,
                                portfolio_value,
                                spot=fill_spot,
                                split_factor=split_factor,
                            )
                        ):
                            unfilled_orders.append(
                                UnfilledOrder(
                                    as_of_date=as_of_timestamp.date(),
                                    asset=decision.asset,
                                    target_weight=decision.target_weight,
                                    reason="untradeable_option_mark",
                                    price=fill_mark.mid,
                                )
                            )
                        option_trade = self.portfolio.apply_option_decision(
                            decision,
                            fill_mark,
                            portfolio_value,
                            spot=fill_spot,
                            split_factor=split_factor,
                        )
                        trade_log.append(option_trade)
                        total_option_premium_paid -= option_trade.premium
                        _charge_fill(day_pod_costs, option_trade, fill_deltas)
                    realized_return = self._option_realized_return(
                        decision.asset, option_marks.get(decision.asset)
                    )
                else:
                    fill_price = (
                        execution_prices[decision.asset]
                        if self.execution_lag_days
                        else prices[decision.asset]
                    )
                    # A NaN or non-positive fill price (an open the loader
                    # masked as inconsistent with its bar, or a missing
                    # bar) leaves shares and cash as they are and records
                    # no trade; the signals are still scored on the day's
                    # close-to-close return, as for an unmarked option.
                    # Spot FX trades through the night, so with a one-day
                    # lag its order fills at the previous session's close
                    # instead, when that close is usable.
                    fill_basis: Literal["open", "prev_close"] = "open"
                    if (
                        not (math.isfinite(fill_price) and fill_price > 0.0)
                        and self.execution_lag_days
                        and currency_pair(decision.asset) is not None
                    ):
                        previous_close = _previous_session_close(
                            raw_market_data, decision.asset, as_of_timestamp
                        )
                        if previous_close is not None:
                            logger.info(
                                "Day %s: %s open unusable (got %s) — filled at the "
                                "previous session close %s.",
                                as_of_timestamp.date().isoformat(),
                                decision.asset,
                                fill_price,
                                previous_close,
                            )
                            fill_price = previous_close
                            fill_basis = "prev_close"
                    if math.isfinite(fill_price) and fill_price > 0.0:
                        adv_shares, daily_vol = _liquidity_inputs(
                            raw_market_data, decision.asset, as_of_timestamp
                        )
                        equity_trade = self.portfolio.apply_decision(
                            decision,
                            fill_price,
                            portfolio_value,
                            adv_shares=adv_shares,
                            daily_vol=daily_vol,
                        )
                        if fill_basis != "open":
                            equity_trade = equity_trade.model_copy(
                                update={"fill_basis": fill_basis}
                            )
                        trade_log.append(equity_trade)
                        _charge_fill(day_pod_costs, equity_trade, fill_deltas)
                    else:
                        logger.warning(
                            "Day %s: no fillable price for %s (got %s) — decision not executed.",
                            as_of_timestamp.date().isoformat(),
                            decision.asset,
                            fill_price,
                        )
                        unfilled_orders.append(
                            UnfilledOrder(
                                as_of_date=as_of_timestamp.date(),
                                asset=decision.asset,
                                target_weight=decision.target_weight,
                                reason="no_fill_price",
                                price=fill_price,
                            )
                        )
                    realized_return = log_returns[decision.asset]
                # The first simulated day has a NaN log_return (no prior close
                # to diff against) — treat it as flat rather than letting NaN
                # poison every later day's Hedge weighting.
                if math.isnan(realized_return):
                    realized_return = 0.0
                for signal in decision.contributing_signals:
                    if signal.status != "active":
                        continue
                    direction_sign = {"long": 1.0, "short": -1.0, "flat": 0.0}[signal.direction]
                    signal_score = direction_sign * signal.raw_confidence
                    attributable_return = direction_sign * signal.confidence * realized_return
                    raw_attributable_return = signal_score * realized_return
                    # Both analyst- and manager-level attribution: the
                    # Desks tab needs the latter (keyed by manager_id) as
                    # much as the Analysts tab needs the former.
                    self.track_record.record(
                        PerformanceRecord(
                            source_id=signal.source_agent_id,
                            source_type="analyst",
                            asset=decision.asset,
                            as_of_date=as_of_timestamp.date(),
                            attributable_return=attributable_return,
                            raw_attributable_return=raw_attributable_return,
                            signal_score=signal_score,
                            realized_return=realized_return,
                        )
                    )
                    self.track_record.record(
                        PerformanceRecord(
                            source_id=signal.manager_id,
                            source_type="manager",
                            asset=decision.asset,
                            as_of_date=as_of_timestamp.date(),
                            attributable_return=attributable_return,
                            raw_attributable_return=raw_attributable_return,
                            signal_score=signal_score,
                            realized_return=realized_return,
                        )
                    )

            pending_decisions = decisions
            idle_pod_share_fn = getattr(self.commander, "idle_pod_share", None)
            unavailable_share_fn = getattr(self.commander, "unavailable_pod_share", None)
            no_position_share_fn = getattr(self.commander, "no_position_pod_share", None)
            pod_states_fn = getattr(self.commander, "pod_states", None)
            capital_usage.append(
                CapitalUsage(
                    as_of_date=as_of_timestamp.date(),
                    idle_pod_share=idle_pod_share_fn() if callable(idle_pod_share_fn) else None,
                    unavailable_pod_share=(
                        unavailable_share_fn() if callable(unavailable_share_fn) else None
                    ),
                    no_position_pod_share=(
                        no_position_share_fn() if callable(no_position_share_fn) else None
                    ),
                    pod_states=pod_states_fn() if callable(pod_states_fn) else {},
                    cash_share=(
                        self.portfolio.cash / portfolio_value
                        if math.isfinite(portfolio_value) and portfolio_value > 0.0
                        else 0.0
                    ),
                )
            )
            sizing_diagnostics_fn = getattr(self.commander, "sizing_diagnostics", None)
            day_sizing_diagnostics: list[SizingDiagnostics] = (
                sizing_diagnostics_fn() if callable(sizing_diagnostics_fn) else []
            )
            sizing_diagnostics.extend(day_sizing_diagnostics)
            pod_cost_days.append(day_pod_costs)
            if before_checkpoint is not None:
                before_checkpoint(day_index, len(days), as_of_timestamp.date(), portfolio_value)
            connection = self._checkpoint()
            if connection is not None:
                risk_state = self._commander_risk_state()
                checkpoint_store.append_day(
                    connection,
                    as_of_date=as_of_timestamp.date(),
                    portfolio_value=portfolio_value,
                    new_track_record=self.track_record.entries_since(track_record_count_before),
                    decisions=decisions,
                    new_trades=trade_log[trades_before:],
                    new_unfilled_orders=unfilled_orders[unfilled_before:],
                    capital_usage=capital_usage[-1],
                    sizing_diagnostics=day_sizing_diagnostics,
                    new_pod_returns=new_pod_returns,
                    pod_cost_day=day_pod_costs,
                    next_day_index=day_index + 1,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    assets=self.assets,
                    portfolio=self.portfolio.model_dump(mode="json"),
                    signal_health=signal_health,
                    commander_risk_state=(
                        None if risk_state is None else risk_state.model_dump(mode="json")
                    ),
                    total_financing_cost=total_financing_cost,
                    total_option_premium_paid=total_option_premium_paid,
                    lake_time=lakehouse_loader.current_lake_time(),
                    settings_fingerprint=run_config["settings_fingerprint"],
                    roster_fingerprint=run_config["roster_fingerprint"],
                    carried_states=self._carried_states(),
                    option_last_mid=self._option_last_mid,
                )

            if on_day_complete is not None:
                on_day_complete(day_index, len(days), as_of_timestamp.date(), portfolio_value)

        if len(portfolio_values) < 2:
            daily_returns: list[float] = []
        else:
            daily_returns = [
                (current - previous) / previous if previous != 0 else 0.0
                for previous, current in itertools.pairwise(portfolio_values)
            ]

        # Each daily_returns[i] is the return realized moving from
        # valuation date i to valuation date i+1, so it's attributable to
        # the later date (the one the benchmark return is compared on) and
        # its risk-free accrual covers the calendar gap between the two.
        valuation_dates = [day for day, _ in portfolio_value_history]
        return_dates = valuation_dates[1:]

        # Each summary metric is computed on its own (`_summary_metric`), so
        # one that is undefined on this run is None without touching the
        # others. Max drawdown and hit rate are defined on any returns.
        returns_array = np.array(daily_returns, dtype=np.float64)
        if daily_returns:
            final_max_drawdown = float(maximum_drawdown(returns_array))
            final_hit_rate = float(hit_rate(returns_array))
        else:
            final_max_drawdown = 0.0
            final_hit_rate = 0.0

        final_sharpe_ratio: float | None
        final_sortino_ratio: float | None
        final_calmar_ratio: float | None
        final_profit_factor: float | None
        if len(daily_returns) < 2 or float(np.std(daily_returns, ddof=1)) == 0.0:
            # A flat (zero-variance) or too-short return series has no
            # volatility to scale by: every ratio is undefined.
            logger.warning(
                "Summary ratios (sharpe, sortino, calmar, profit factor) are undefined on this "
                "run: %d daily return(s), fewer than two or with zero variance; reporting None.",
                len(daily_returns),
            )
            final_sharpe_ratio = None
            final_sortino_ratio = None
            final_calmar_ratio = None
            final_profit_factor = None
        else:
            excess_returns_array = returns_array - self._risk_free_accrual_array(valuation_dates)
            if np.all(np.abs(excess_returns_array) <= ZERO_EXCESS_RETURN_TOLERANCE):
                logger.warning(
                    "Sharpe and Sortino are undefined on this run: every daily excess return "
                    "is within %g of zero; reporting None.",
                    ZERO_EXCESS_RETURN_TOLERANCE,
                )
                final_sharpe_ratio = None
                final_sortino_ratio = None
            else:
                final_sharpe_ratio = _summary_metric(
                    "sharpe ratio", lambda: sharpe_ratio(excess_returns_array)
                )
                final_sortino_ratio = _summary_metric(
                    "sortino ratio", lambda: sortino_ratio(excess_returns_array)
                )
            # Calmar is annualized raw return over max drawdown with no
            # excess-return concept (quantcore's own definition) — no
            # risk-free adjustment applies here.
            final_calmar_ratio = _summary_metric(
                "calmar ratio", lambda: calmar_ratio(returns_array)
            )
            final_profit_factor = _summary_metric(
                "profit factor", lambda: profit_factor(returns_array)
            )

        final_information_ratio: float | None = None
        if self.benchmark_returns is not None and len(daily_returns) >= 2:
            benchmark_array = self._benchmark_returns_array(return_dates)
            if benchmark_array is not None:
                benchmark = benchmark_array
                final_information_ratio = _summary_metric(
                    "information ratio", lambda: information_ratio(returns_array, benchmark)
                )

        logger.info(
            "Backtest complete: %d days simulated in %.1fs — final Sharpe=%s, max drawdown=%.2f",
            len(days),
            time.perf_counter() - run_started_at,
            "undefined" if final_sharpe_ratio is None else f"{final_sharpe_ratio:.3f}",
            final_max_drawdown,
        )
        pod_performance = {
            pod_id: _pod_performance([value for _, value in entries])
            for pod_id, entries in pod_return_history.items()
        }
        close_commander = getattr(self.commander, "close", None)
        if close_commander is not None:
            close_commander()
        source_performance = self._source_performance()
        if self._checkpoint_connection is not None:
            self._checkpoint_connection.close()
            self._checkpoint_connection = None
        total_transaction_cost = sum(trade.transaction_cost for trade in trade_log)
        total_slippage_cost = sum(trade.slippage_cost for trade in trade_log)
        total_impact_cost = sum(trade.impact_cost for trade in trade_log)
        total_spread_cost = sum(trade.spread_cost for trade in trade_log)
        return BacktestResult(
            start_date=self.start_date,
            end_date=self.end_date,
            portfolio_value_history=portfolio_value_history,
            final_sharpe_ratio=final_sharpe_ratio,
            final_max_drawdown=final_max_drawdown,
            final_sortino_ratio=final_sortino_ratio,
            final_calmar_ratio=final_calmar_ratio,
            final_information_ratio=final_information_ratio,
            final_hit_rate=final_hit_rate,
            final_profit_factor=final_profit_factor,
            decisions_by_day=(
                decisions_by_day.held()
                if isinstance(decisions_by_day, DecisionWindow)
                else decisions_by_day
            ),
            history_horizon_days=self.history_horizon_days,
            trade_log=trade_log,
            unfilled_orders=unfilled_orders,
            capital_usage=capital_usage,
            sizing_diagnostics=sizing_diagnostics,
            signal_health=signal_health,
            market_data_source=self.market_data_source,
            source_performance=source_performance,
            total_transaction_cost=total_transaction_cost,
            total_slippage_cost=total_slippage_cost,
            total_impact_cost=total_impact_cost,
            total_spread_cost=total_spread_cost,
            total_execution_cost=(
                total_transaction_cost + total_slippage_cost + total_impact_cost + total_spread_cost
            ),
            total_financing_cost=total_financing_cost,
            total_option_premium_paid=total_option_premium_paid,
            pod_return_history=pod_return_history,
            pod_performance=pod_performance,
            pod_costs=_pod_cost_attribution(pod_cost_days, len(portfolio_value_history)),
            unfactored_split_moves=(
                None
                if checkpoint is not None
                else lakehouse_loader.unfactored_split_move_registry()
            ),
        )


def _checkpoint_traded_options(connection: sqlite3.Connection, payload: dict[str, Any]) -> bool:
    """Whether the checkpointed run holds an option position or has an
    option decision on any stored day: one with an `instrument` or on an
    option contract id."""
    portfolio = payload.get("portfolio", {})
    if portfolio.get("option_positions") or any(
        OptionContract.is_contract_id(asset) for asset in portfolio.get("positions", {})
    ):
        return True
    for _day, decisions_json in checkpoint_store.iter_decision_payloads(connection):
        for decision in json.loads(decisions_json):
            if decision.get("instrument") is not None or OptionContract.is_contract_id(
                decision["asset"]
            ):
                return True
    return False


def _frame_field(
    raw_market_data: pd.DataFrame, assets: list[str], as_of_date: pd.Timestamp, field: str
) -> dict[str, float]:
    """Each of `assets`' finite `field` value on `as_of_date` in the
    run's `(ticker, field)` frame (a future's `roll` count). An asset
    without the field, a NaN value, a date the frame does not hold and a
    frame without `(ticker, field)` columns all give no entry."""
    if not isinstance(raw_market_data.columns, pd.MultiIndex):
        return {}
    if as_of_date not in raw_market_data.index:
        return {}
    values: dict[str, float] = {}
    for asset in assets:
        if (asset, field) not in raw_market_data.columns:
            continue
        value = float(cast(float, raw_market_data.loc[as_of_date, (asset, field)]))
        if math.isfinite(value):
            values[asset] = value
    return values


def _holding_shares(
    contributions: dict[str, dict[str, float]] | None,
) -> dict[str, dict[str, float]] | None:
    """`contributions` (pod id to asset to contribution) turned into asset
    to pod id to |contribution|, the shares `_charge_fill` splits a held
    position's cost by; `None` for `None`."""
    if contributions is None:
        return None
    shares: dict[str, dict[str, float]] = {}
    for pod_id, weights in contributions.items():
        for asset, weight in weights.items():
            if weight != 0.0:
                shares.setdefault(asset, {})[pod_id] = abs(weight)
    return shares


def _previous_session_close(
    raw_market_data: pd.DataFrame, asset: str, as_of_date: pd.Timestamp
) -> float | None:
    """`asset`'s close on the session before `as_of_date` in the run's
    frame (the row just before it), when that close is finite and
    positive; `None` when it is not, or when `as_of_date` is the frame's
    first session or not in it. An older close is never used."""
    if as_of_date not in raw_market_data.index:
        return None
    position = raw_market_data.index.get_loc(as_of_date)
    if not isinstance(position, int) or position == 0:
        return None
    column: str | tuple[str, str] = (
        (asset, "close") if isinstance(raw_market_data.columns, pd.MultiIndex) else "close"
    )
    if column not in raw_market_data.columns:
        return None
    close = float(cast(float, raw_market_data[column].iloc[position - 1]))
    return close if math.isfinite(close) and close > 0.0 else None


def _valuation_prices(
    raw_market_data: pd.DataFrame,
    as_of_date: pd.Timestamp,
    prices: dict[str, float],
    marked: set[str],
) -> dict[str, float]:
    """The prices a day's book is valued at: each of `prices` (the day's
    closes) that is finite, and for each that is not (no bar that day) the
    asset's last finite close before `as_of_date` in `raw_market_data`; an
    asset with no close yet is left out, since nothing can have filled it.
    An asset of `marked` (a held future or spot FX pair) without a finite
    close is left out too: it keeps its mark, so its unsettled P&L that day
    is zero. Equal to `prices` when every close is finite, and for a
    single-asset frame."""
    if not isinstance(raw_market_data.columns, pd.MultiIndex):
        return dict(prices)
    valued: dict[str, float] = {}
    for asset, price in prices.items():
        if math.isfinite(price):
            valued[asset] = price
            continue
        if asset in marked:
            continue
        history = raw_market_data[(asset, "close")]
        earlier = history[history.index < as_of_date].dropna()
        if not earlier.empty:
            valued[asset] = float(earlier.iloc[-1])
    return valued


def _lake_time_text(lake_time: datetime | None) -> str:
    """`lake_time` in ISO 8601, or "no lake session" for `None`."""
    return "no lake session" if lake_time is None else lake_time.isoformat()


def _contribution_deltas(
    previous: dict[str, dict[str, float]] | None,
    current: dict[str, dict[str, float]] | None,
) -> dict[str, dict[str, float]] | None:
    """|current - previous| contribution per asset then pod, keeping only
    nonzero changes; a pod or asset missing from either side counts as 0.0
    there, and a `None` `previous` as no contributions. `None` when
    `current` is `None` (the commander reports no contributions)."""
    if current is None:
        return None
    previous = previous or {}
    deltas: dict[str, dict[str, float]] = {}
    for pod_id in dict.fromkeys([*previous, *current]):
        before = previous.get(pod_id, {})
        now = current.get(pod_id, {})
        for asset in dict.fromkeys([*before, *now]):
            delta = abs(now.get(asset, 0.0) - before.get(asset, 0.0))
            if delta > 0.0:
                deltas.setdefault(asset, {})[pod_id] = delta
    return deltas


def _charge(totals: dict[str, float], amount: float, shares: dict[str, float] | None) -> None:
    """Adds `amount` to `totals`, split across pod ids pro rata to `shares`
    (each pod's |contribution change|), or all to `"commander"` when
    `shares` is `None`, empty or sums to zero. A zero `amount` adds
    nothing."""
    if amount == 0.0:
        return
    total = sum(shares.values()) if shares else 0.0
    if not shares or total <= 0.0:
        totals["commander"] = totals.get("commander", 0.0) + amount
        return
    for pod_id, share in shares.items():
        totals[pod_id] = totals.get(pod_id, 0.0) + amount * share / total


def _charge_fill(
    day: PodCostDay, trade: TradeRecord, deltas: dict[str, dict[str, float]] | None
) -> None:
    """Splits `trade`'s commission, slippage, impact and spread cost and
    its net premium paid (the negated signed `premium`) into `day` by the
    contribution changes to `trade.asset` in `deltas`, the decision that
    produced the fill."""
    shares = deltas.get(trade.asset) if deltas is not None else None
    _charge(day.transaction_cost, trade.transaction_cost, shares)
    _charge(day.slippage_cost, trade.slippage_cost, shares)
    _charge(day.impact_cost, trade.impact_cost, shares)
    _charge(day.spread_cost, trade.spread_cost, shares)
    _charge(day.option_premium_paid, -trade.premium, shares)


def _pod_cost_attribution(
    days: list[PodCostDay], observed_days: int
) -> dict[str, PodCostAttribution]:
    """Each pod's `PodCostAttribution` over `days`, plus a `"commander"`
    entry when any cost was charged to it. Each cost kind is the sum of
    its daily charges in day order, and `execution_cost` the sum of the
    four kinds. Turnover, trading days and sign changes come
    from each day's contributions against the previous day's;
    `observed_days` (the number of valuation days) annualises turnover."""
    pod_ids: dict[str, None] = {}
    moved: dict[str, float] = {}
    trading_days: dict[str, int] = {}
    sign_changes: dict[str, int] = {}
    last_sign: dict[tuple[str, str], bool] = {}
    costs: dict[str, dict[str, float]] = {kind: {} for kind in _COST_KINDS}
    option_premium_paid: dict[str, float] = {}
    previous: dict[str, dict[str, float]] | None = None
    for day in days:
        if day.contributions is not None:
            pod_ids.update(dict.fromkeys(day.contributions))
            deltas = _contribution_deltas(previous, day.contributions) or {}
            pods_moved: dict[str, None] = {}
            for by_pod in deltas.values():
                for pod_id, delta in by_pod.items():
                    moved[pod_id] = moved.get(pod_id, 0.0) + delta
                    pods_moved[pod_id] = None
            for pod_id in pods_moved:
                trading_days[pod_id] = trading_days.get(pod_id, 0) + 1
            for pod_id, weights in day.contributions.items():
                for asset, weight in weights.items():
                    if weight == 0.0:
                        continue
                    positive = weight > 0.0
                    if last_sign.get((pod_id, asset), positive) != positive:
                        sign_changes[pod_id] = sign_changes.get(pod_id, 0) + 1
                    last_sign[(pod_id, asset)] = positive
        previous = day.contributions
        for kind, totals in costs.items():
            for pod_id, amount in getattr(day, kind).items():
                totals[pod_id] = totals.get(pod_id, 0.0) + amount
        for pod_id, amount in day.option_premium_paid.items():
            option_premium_paid[pod_id] = option_premium_paid.get(pod_id, 0.0) + amount
    for totals in costs.values():
        pod_ids.update(dict.fromkeys(totals))
    pod_ids.update(dict.fromkeys(option_premium_paid))
    annualise = settings.firm.trading_days_per_year / observed_days if observed_days else 0.0
    return {
        pod_id: PodCostAttribution(
            transaction_cost=costs["transaction_cost"].get(pod_id, 0.0),
            slippage_cost=costs["slippage_cost"].get(pod_id, 0.0),
            impact_cost=costs["impact_cost"].get(pod_id, 0.0),
            spread_cost=costs["spread_cost"].get(pod_id, 0.0),
            execution_cost=sum(totals.get(pod_id, 0.0) for totals in costs.values()),
            option_premium_paid=option_premium_paid.get(pod_id, 0.0),
            turnover=moved.get(pod_id, 0.0) * annualise,
            trading_days=trading_days.get(pod_id, 0),
            sign_changes=sign_changes.get(pod_id, 0),
        )
        for pod_id in pod_ids
    }


def _pod_performance(returns: list[float]) -> PodPerformance:
    """`PodPerformance` for one pod's daily return series."""
    series = np.array(returns, dtype=np.float64)
    observations = len(series)
    cumulative_return = float(np.prod(1.0 + series) - 1.0)
    if observations < 2:
        return PodPerformance(
            observations=observations,
            cumulative_return=cumulative_return,
            annualized_vol=0.0,
            sharpe_ratio=None,
            max_drawdown=0.0,
        )
    daily_vol = float(np.std(series, ddof=1))
    return PodPerformance(
        observations=observations,
        cumulative_return=cumulative_return,
        annualized_vol=daily_vol * math.sqrt(settings.firm.trading_days_per_year),
        sharpe_ratio=float(sharpe_ratio(series)) if daily_vol > 0.0 else None,
        max_drawdown=float(maximum_drawdown(series)),
    )


def _liquidity_inputs(
    raw_market_data: pd.DataFrame, asset: str, as_of_date: pd.Timestamp
) -> tuple[float | None, float | None]:
    """Trailing average daily volume (shares) and daily log-return vol for
    `asset` over the last `settings.firm.adv_window_days` bars up to and
    including `as_of_date`, for the fill's impact and participation
    logic. `(None, None)` when the frame carries no usable volume for the
    asset (spot FX, indices, single-ticker frames without a volume field)."""
    if not isinstance(raw_market_data.columns, pd.MultiIndex):
        return None, None
    if (asset, "volume") not in raw_market_data.columns:
        return None, None
    window = raw_market_data.loc[:as_of_date].tail(settings.firm.adv_window_days)
    volume = window[(asset, "volume")].dropna()
    returns = window[(asset, "log_return")].dropna()
    if volume.empty or returns.size < 2:
        return None, None
    adv = float(volume.mean())
    if not math.isfinite(adv) or adv <= 0.0:
        return None, None
    return adv, float(returns.std(ddof=1))
