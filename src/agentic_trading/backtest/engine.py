# engine.py

import itertools
import math
import sqlite3
import time
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any, Protocol, cast

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
    OptionContract,
    PerformanceRecord,
    PortfolioDecision,
    TradeRecord,
    UnfilledOrder,
)
from agentic_trading.backtest import checkpoint_store
from agentic_trading.backtest.portfolio import OptionPosition, Portfolio
from agentic_trading.config import settings
from agentic_trading.data.loaders import lakehouse_loader
from agentic_trading.data.loaders.option_marks import OptionMark, load_option_mark
from agentic_trading.data.point_in_time import get_point_in_time_data
from agentic_trading.logging_config import get_logger, log_progress
from agentic_trading.scoring.scoreboard import PerformanceScoreboard
from agentic_trading.scoring.track_record import TrackRecord

logger = get_logger(__name__)

# Largest absolute daily excess return still treated as zero. A book that
# only ever holds cash earns exactly the benchmark accrual, so its excess
# returns are floating-point residue (around 1e-17); Sharpe and Sortino of
# that series are meaningless (zero or near-zero variance gives a division
# by zero or noise), and the run reports 0.0 for both instead.
ZERO_EXCESS_RETURN_TOLERANCE = 1e-12

# Called after each simulated day completes, with (day_index, total_days,
# as_of_date, portfolio_value) — the hook a caller (the GUI) uses to drive a
# live progress display. Safe to have this call UI code: unlike the GUI's
# old @st.cache_data-wrapped simulation loop, BacktestEngine.run() is never
# itself cached, so there's no CacheReplayClosureError risk here.
DayCompleteCallback = Callable[[int, int, date, float], None]


class CommanderLike(Protocol):
    """What `BacktestEngine` actually needs from a commander — satisfied by
    `Commander` and by `backtest/baselines.py`'s `BaselineCommander`/
    `SingleAnalystCommander`, which exist to run simple benchmark/diagnostic
    strategies through this same engine without becoming `Commander`
    subclasses.

    A commander may additionally expose `risk_state() -> CommanderRiskState`
    and `restore_risk_state(state)`; the engine checkpoints/restores that
    state and reports its pod return history when present, and simply
    skips it otherwise."""

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
    ) -> None:
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
        self._checkpoint_connection: sqlite3.Connection | None = None
        # Previous day's close mark per option contract, for attribution
        # (log of today's mid over yesterday's); a contract seen for the
        # first time realizes 0.0.
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

    def _pod_return_history(self) -> dict[str, list[tuple[date, float]]]:
        state = self._commander_risk_state()
        return {} if state is None else dict(state.pod_return_history)

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

    def _restore_checkpoint(self) -> dict[str, Any] | None:
        if not self.resume:
            return None
        connection = self._checkpoint()
        if connection is None:
            return None
        payload = checkpoint_store.read_all(connection)
        if payload is None:
            return None
        expected = {
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "assets": self.assets,
        }
        if payload.get("config") != expected:
            raise ValueError("checkpoint configuration does not match this backtest")
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

    def _source_performance(self) -> dict[str, dict[str, float]]:
        entries_by_source: dict[str, list[PerformanceRecord]] = {}
        entries = getattr(self.track_record, "_entries", None)
        if entries is None:
            entries = getattr(self.track_record, "entries", [])
        for entry in entries:
            if self.start_date <= entry.as_of_date <= self.end_date:
                entries_by_source.setdefault(entry.source_id, []).append(entry)

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
        for source_id, entries in entries_by_source.items():
            returns = np.asarray([entry.attributable_return for entry in entries], dtype=float)
            raw_returns = np.asarray(
                [entry.raw_attributable_return for entry in entries], dtype=float
            )
            annualized_return, total_attributed_return, sharpe, drawdown = _stats(returns)
            raw_annualized_return, raw_total_attributed_return, raw_sharpe, raw_drawdown = _stats(
                raw_returns
            )
            performance[source_id] = {
                "cumulative_return": annualized_return,
                "total_attributed_return": total_attributed_return,
                "sharpe_ratio": sharpe,
                "max_drawdown": drawdown,
                "observations": float(len(entries)),
                "raw_cumulative_return": raw_annualized_return,
                "raw_total_attributed_return": raw_total_attributed_return,
                "raw_sharpe_ratio": raw_sharpe,
                "raw_max_drawdown": raw_drawdown,
            }
        return performance

    def run(self, on_day_complete: DayCompleteCallback | None = None) -> BacktestResult:
        """Steps through every simulated day from start_date to end_date,
        point-in-time slicing data (see data/point_in_time.py), calling
        the Commander for decisions, applying them to the Portfolio, and
        recording realized performance into track_record. Returns a
        BacktestResult summarizing the full run, including every day's
        PortfolioDecisions and the full trade log — the same data both the
        CLI and the GUI's Desks/Analysts drill-down are built from.

        `on_day_complete`, if given, is called once per simulated day after
        that day's trades are applied — the hook a live progress UI hangs
        off of. Purely a notification; this method's own behavior does not
        depend on it being present.
        """
        logger.info(
            "Backtest starting: assets=%s, %s to %s", self.assets, self.start_date, self.end_date
        )
        run_started_at = time.perf_counter()
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
            )

        days = [
            pd.Timestamp(day)
            for day in raw_market_data.index
            if self.start_date <= pd.Timestamp(day).date() <= self.end_date
        ]

        checkpoint = self._restore_checkpoint()
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
        decisions_by_day: dict[date, list[PortfolioDecision]] = (
            {
                date.fromisoformat(day): [
                    PortfolioDecision.model_validate(decision) for decision in decisions
                ]
                for day, decisions in checkpoint["decisions_by_day"].items()
            }
            if checkpoint
            else {}
        )
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
            self.portfolio.option_positions = {
                contract_id: OptionPosition.model_validate(position)
                for contract_id, position in portfolio_state.get("option_positions", {}).items()
            }
            self._option_last_mid = {
                contract_id: position.last_mid
                for contract_id, position in self.portfolio.option_positions.items()
            }
            self.track_record.restore(
                [PerformanceRecord.model_validate(entry) for entry in checkpoint["track_record"]]
            )
            self._restore_commander_risk_state(checkpoint.get("commander_risk_state"))
        total_financing_cost = (
            float(checkpoint.get("total_financing_cost", 0.0)) if checkpoint else 0.0
        )
        total_option_premium_paid = (
            float(checkpoint.get("total_option_premium_paid", 0.0)) if checkpoint else 0.0
        )

        for day_index, as_of_timestamp in enumerate(days[start_day_index:], start=start_day_index):
            day_started_at = time.perf_counter()
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
            trade_log.extend(self.portfolio.settle_expired(marks, as_of_timestamp.date()))
            portfolio_value = self.portfolio.total_value(prices, marks)
            # Financing (net of the interest cash earns) accrues over the
            # calendar days since the previous simulated day (a
            # Friday-to-Monday gap accrues three days), on the book carried
            # across that gap, at the risk-free rate in force on the
            # previous valuation date, when the gap starts. It is booked
            # after the mark so the day's recorded value already reflects it.
            previous_valuation_date = (
                portfolio_value_history[-1][0] if portfolio_value_history else None
            )
            days_held = (
                (as_of_timestamp.date() - previous_valuation_date).days
                if previous_valuation_date is not None
                else 0
            )
            financing_cost = self.portfolio.apply_financing(
                prices,
                (
                    self._risk_free_rate_annual(previous_valuation_date)
                    if previous_valuation_date is not None
                    else 0.0
                ),
                days_held,
                spots=underlying_prices,
            )
            total_financing_cost += financing_cost
            portfolio_value -= financing_cost
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
                    if math.isfinite(fill_price) and fill_price > 0.0:
                        adv_shares, daily_vol = _liquidity_inputs(
                            raw_market_data, decision.asset, as_of_timestamp
                        )
                        trade_log.append(
                            self.portfolio.apply_decision(
                                decision,
                                fill_price,
                                portfolio_value,
                                adv_shares=adv_shares,
                                daily_vol=daily_vol,
                            )
                        )
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
                    next_day_index=day_index + 1,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    assets=self.assets,
                    portfolio=self.portfolio.model_dump(mode="json"),
                    signal_health=signal_health,
                    pending_decisions=pending_decisions,
                    commander_risk_state=(
                        None if risk_state is None else risk_state.model_dump(mode="json")
                    ),
                    total_financing_cost=total_financing_cost,
                    total_option_premium_paid=total_option_premium_paid,
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

        if len(daily_returns) < 2 or float(np.std(daily_returns, ddof=1)) == 0.0:
            # A flat (zero-variance) return series makes every ratio here
            # undefined (division by zero) rather than merely large — 0.0
            # is the same "nothing to report" default used for a too-short
            # window.
            final_sharpe_ratio = 0.0
            final_sortino_ratio = 0.0
            final_calmar_ratio = 0.0
            final_max_drawdown = 0.0
            final_hit_rate = 0.0
            final_profit_factor = 0.0
        else:
            returns_array = np.array(daily_returns, dtype=np.float64)
            try:
                excess_returns_array = returns_array - self._risk_free_accrual_array(
                    valuation_dates
                )
                if np.all(np.abs(excess_returns_array) <= ZERO_EXCESS_RETURN_TOLERANCE):
                    final_sharpe_ratio = 0.0
                    final_sortino_ratio = 0.0
                else:
                    final_sharpe_ratio = float(sharpe_ratio(excess_returns_array))
                    final_sortino_ratio = float(sortino_ratio(excess_returns_array))
                # Calmar is annualized raw return over max drawdown with no
                # excess-return concept (quantcore's own definition) — no
                # risk-free adjustment applies here.
                final_calmar_ratio = float(calmar_ratio(returns_array))
                final_max_drawdown = float(maximum_drawdown(returns_array))
                final_hit_rate = float(hit_rate(returns_array))
                final_profit_factor = float(profit_factor(returns_array))
            except ZeroDivisionError:
                # A degenerate edge case (e.g. no downside deviation at
                # all) that the zero-variance guard above didn't catch —
                # same "nothing to report" default rather than crashing
                # the run over a performance-summary metric.
                logger.debug(
                    "Performance metric computation hit a zero-division edge case — "
                    "defaulting the run's summary ratios to 0.0.",
                    exc_info=True,
                )
                final_sharpe_ratio = 0.0
                final_sortino_ratio = 0.0
                final_calmar_ratio = 0.0
                final_max_drawdown = 0.0
                final_hit_rate = 0.0
                final_profit_factor = 0.0

        final_information_ratio: float | None = None
        if self.benchmark_returns is not None and len(daily_returns) >= 2:
            benchmark_array = self._benchmark_returns_array(return_dates)
            if benchmark_array is not None:
                final_information_ratio = float(
                    information_ratio(np.array(daily_returns, dtype=np.float64), benchmark_array)
                )

        logger.info(
            "Backtest complete: %d days simulated in %.1fs — final Sharpe=%.3f, max drawdown=%.2f",
            len(days),
            time.perf_counter() - run_started_at,
            final_sharpe_ratio,
            final_max_drawdown,
        )
        pod_return_history = self._pod_return_history()
        close_commander = getattr(self.commander, "close", None)
        if close_commander is not None:
            close_commander()
        if self._checkpoint_connection is not None:
            self._checkpoint_connection.close()
            self._checkpoint_connection = None
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
            decisions_by_day=decisions_by_day,
            trade_log=trade_log,
            unfilled_orders=unfilled_orders,
            signal_health=signal_health,
            market_data_source=self.market_data_source,
            source_performance=self._source_performance(),
            total_transaction_cost=sum(trade.transaction_cost for trade in trade_log),
            total_financing_cost=total_financing_cost,
            total_option_premium_paid=total_option_premium_paid,
            pod_return_history=pod_return_history,
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
