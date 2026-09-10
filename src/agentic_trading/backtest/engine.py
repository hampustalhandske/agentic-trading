# engine.py

import itertools
import json
import math
import time
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from quantcore.performance.metrics import (
    calmar_ratio,
    hit_rate,
    maximum_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
)

from agentic_trading.agents.commander import Commander
from agentic_trading.agents.portfolio_construction import PortfolioConstructor
from agentic_trading.agents.schemas import (
    BacktestResult,
    PerformanceRecord,
    PortfolioDecision,
    TradeRecord,
)
from agentic_trading.backtest.portfolio import Portfolio
from agentic_trading.data.point_in_time import get_point_in_time_data
from agentic_trading.logging_config import get_logger, log_progress
from agentic_trading.scoring.scoreboard import PerformanceScoreboard
from agentic_trading.scoring.track_record import TrackRecord

logger = get_logger(__name__)

# Called after each simulated day completes, with (day_index, total_days,
# as_of_date, portfolio_value) — the hook a caller (the GUI) uses to drive a
# live progress display. Safe to have this call UI code: unlike the GUI's
# old @st.cache_data-wrapped simulation loop, BacktestEngine.run() is never
# itself cached, so there's no CacheReplayClosureError risk here.
DayCompleteCallback = Callable[[int, int, date, float], None]


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
        commander: Commander,
        portfolio: Portfolio,
        track_record: TrackRecord,
        data_loader: Callable[[list[str], date, date], pd.DataFrame],
        start_date: date,
        end_date: date,
        assets: list[str] | None = None,
        portfolio_constructor: PortfolioConstructor | None = None,
        market_data_source: str | None = None,
        execution_lag_days: int = 0,
        checkpoint_path: str | Path | None = None,
        resume: bool = False,
    ) -> None:
        self.commander = commander
        self.portfolio = portfolio
        self.track_record = track_record
        self.data_loader = data_loader
        self.start_date = start_date
        self.end_date = end_date
        self.assets = assets or ["AAPL"]
        # Optional Layer-4 stage after the Commander — None (the default)
        # means no portfolio construction runs at all, matching every
        # existing backtest's behavior exactly.
        self.portfolio_constructor = portfolio_constructor
        self.market_data_source = market_data_source
        if execution_lag_days not in {0, 1}:
            raise ValueError("execution_lag_days must be 0 or 1")
        self.execution_lag_days = execution_lag_days
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        self.resume = resume

    def _write_checkpoint(self, payload: dict[str, object]) -> None:
        if self.checkpoint_path is None:
            return
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.checkpoint_path.with_suffix(
            f"{self.checkpoint_path.suffix}.tmp"
        )
        temporary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary_path.replace(self.checkpoint_path)

    def _restore_checkpoint(self) -> dict[str, object] | None:
        if not self.resume or self.checkpoint_path is None or not self.checkpoint_path.exists():
            return None
        payload = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        expected = {
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "assets": self.assets,
        }
        if payload.get("config") != expected:
            raise ValueError("checkpoint configuration does not match this backtest")
        return payload

    def _apply_portfolio_construction(
        self,
        decisions: list[PortfolioDecision],
        market_data: pd.DataFrame,
        as_of_timestamp: pd.Timestamp,
    ) -> list[PortfolioDecision]:
        """Builds the wide, point-in-time-sliced returns frame
        PortfolioConstructor needs from the same `market_data` the
        Commander was already given for this day, plus an equal-weighted
        market_weights proxy — this project has no market-cap data
        source, so equal weighting is the documented stand-in for the
        Black-Litterman equilibrium prior's market portfolio."""
        assets = [decision.asset for decision in decisions]
        if isinstance(market_data.columns, pd.MultiIndex):
            returns_history = pd.concat(
                {asset: market_data[asset]["log_return"] for asset in assets}, axis=1
            )
        else:
            returns_history = market_data[["log_return"]].rename(columns={"log_return": assets[0]})
        returns_history = returns_history.dropna()
        market_weights = dict.fromkeys(assets, 1.0 / len(assets)) if assets else {}
        return self.portfolio_constructor.construct(  # type: ignore[union-attr]
            decisions=decisions,
            returns_history=returns_history,
            market_weights=market_weights,
            as_of_date=as_of_timestamp,
        )

    def _source_performance(self) -> dict[str, dict[str, float]]:
        entries_by_source: dict[str, list[PerformanceRecord]] = {}
        entries = getattr(self.track_record, "_entries", None)
        if entries is None:
            entries = getattr(self.track_record, "entries", [])
        for entry in entries:
            if self.start_date <= entry.as_of_date <= self.end_date:
                entries_by_source.setdefault(entry.source_id, []).append(entry)

        performance: dict[str, dict[str, float]] = {}
        for source_id, entries in entries_by_source.items():
            returns = np.asarray([entry.attributable_return for entry in entries], dtype=float)
            cumulative = np.cumsum(returns)
            running_peak = np.maximum.accumulate(cumulative)
            drawdown = float(np.min(cumulative - running_peak)) if cumulative.size else 0.0
            standard_deviation = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
            sharpe = (
                float(np.mean(returns) / standard_deviation * np.sqrt(252))
                if standard_deviation > 0.0
                else 0.0
            )
            performance[source_id] = {
                "cumulative_return": float(np.sum(returns)),
                "sharpe_ratio": sharpe,
                "max_drawdown": drawdown,
                "observations": float(len(entries)),
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
            self.track_record.restore(
                [PerformanceRecord.model_validate(entry) for entry in checkpoint["track_record"]]
            )

        for day_index, as_of_timestamp in enumerate(days[start_day_index:], start=start_day_index):
            day_started_at = time.perf_counter()
            market_data = get_point_in_time_data(raw_market_data, as_of_timestamp)
            decisions = self.commander.decide(market_data, as_of_timestamp, scoreboard)
            logger.debug(
                "Day %s: commander.decide across %d assets took %.1fms",
                as_of_timestamp.date(),
                len(self.assets),
                (time.perf_counter() - day_started_at) * 1000,
            )
            if self.portfolio_constructor is not None:
                decisions = self._apply_portfolio_construction(
                    decisions, market_data, as_of_timestamp
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
                        asset: float(raw_market_data.loc[as_of_timestamp, (asset, "open")])
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
                    {self.assets[0]: float(raw_market_data.loc[as_of_timestamp, "open"])}
                    if self.execution_lag_days
                    else prices
                )
            portfolio_value = self.portfolio.total_value(prices)
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

            decisions_to_execute = pending_decisions if self.execution_lag_days else decisions
            for decision in decisions_to_execute:
                fill_price = (
                    execution_prices[decision.asset]
                    if self.execution_lag_days
                    else prices[decision.asset]
                )
                trade_log.append(
                    self.portfolio.apply_decision(decision, fill_price, portfolio_value)
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
                    attributable_return = direction_sign * signal.confidence * realized_return
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
                        )
                    )
                    self.track_record.record(
                        PerformanceRecord(
                            source_id=signal.manager_id,
                            source_type="manager",
                            asset=decision.asset,
                            as_of_date=as_of_timestamp.date(),
                            attributable_return=attributable_return,
                        )
                    )

            pending_decisions = decisions
            if self.checkpoint_path is not None:
                self._write_checkpoint(
                    {
                    "config": {
                        "start_date": self.start_date.isoformat(),
                        "end_date": self.end_date.isoformat(),
                        "assets": self.assets,
                    },
                    "next_day_index": day_index + 1,
                    "portfolio": self.portfolio.model_dump(),
                    "track_record": [entry.model_dump() for entry in self.track_record.entries],
                    "portfolio_value_history": portfolio_value_history,
                    "decisions_by_day": {
                        day.isoformat(): [decision.model_dump() for decision in decisions]
                        for day, decisions in decisions_by_day.items()
                    },
                    "trade_log": [trade.model_dump() for trade in trade_log],
                    "signal_health": signal_health,
                    "pending_decisions": [decision.model_dump() for decision in pending_decisions],
                    }
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
                final_sharpe_ratio = float(sharpe_ratio(returns_array))
                final_sortino_ratio = float(sortino_ratio(returns_array))
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

        # No benchmark is configured for this engine, so information_ratio
        # (which requires one) is left None rather than computed against an
        # arbitrary stand-in.
        final_information_ratio: float | None = None

        logger.info(
            "Backtest complete: %d days simulated in %.1fs — final Sharpe=%.3f, max drawdown=%.2f",
            len(days),
            time.perf_counter() - run_started_at,
            final_sharpe_ratio,
            final_max_drawdown,
        )
        close_commander = getattr(self.commander, "close", None)
        if close_commander is not None:
            close_commander()
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
            signal_health=signal_health,
            market_data_source=self.market_data_source,
            synthetic_market_data_used=self.market_data_source == "synthetic fallback data",
            source_performance=self._source_performance(),
        )
