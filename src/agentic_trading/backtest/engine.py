# engine.py

import itertools
import math
import time
from collections.abc import Callable
from datetime import date
from typing import cast

import pandas as pd

from agentic_trading.agents.commander import Commander
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
    ) -> None:
        self.commander = commander
        self.portfolio = portfolio
        self.track_record = track_record
        self.data_loader = data_loader
        self.start_date = start_date
        self.end_date = end_date
        self.assets = assets or ["AAPL"]

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
            )

        days = [
            pd.Timestamp(day)
            for day in raw_market_data.index
            if self.start_date <= pd.Timestamp(day).date() <= self.end_date
        ]

        scoreboard = PerformanceScoreboard(self.track_record)
        portfolio_value_history: list[tuple[date, float]] = []
        portfolio_values: list[float] = []
        decisions_by_day: dict[date, list[PortfolioDecision]] = {}
        trade_log: list[TradeRecord] = []

        for day_index, as_of_timestamp in enumerate(days):
            day_started_at = time.perf_counter()
            market_data = get_point_in_time_data(raw_market_data, as_of_timestamp)
            decisions = self.commander.decide(market_data, as_of_timestamp, scoreboard)
            logger.debug(
                "Day %s: commander.decide across %d assets took %.1fms",
                as_of_timestamp.date(),
                len(self.assets),
                (time.perf_counter() - day_started_at) * 1000,
            )
            decisions_by_day[as_of_timestamp.date()] = decisions

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
            else:
                prices = {
                    asset: float(cast(float, raw_market_data.loc[as_of_timestamp, "close"]))
                    for asset in self.assets
                }
                log_returns = {
                    asset: float(cast(float, raw_market_data.loc[as_of_timestamp, "log_return"]))
                    for asset in self.assets
                }
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

            for decision in decisions:
                trade_log.append(
                    self.portfolio.apply_decision(decision, prices[decision.asset], portfolio_value)
                )
                realized_return = log_returns[decision.asset]
                # The first simulated day has a NaN log_return (no prior close
                # to diff against) — treat it as flat rather than letting NaN
                # poison every later day's Hedge weighting.
                if math.isnan(realized_return):
                    realized_return = 0.0
                for signal in decision.contributing_signals:
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

            if on_day_complete is not None:
                on_day_complete(day_index, len(days), as_of_timestamp.date(), portfolio_value)

        if len(portfolio_values) < 2:
            final_sharpe_ratio = 0.0
        else:
            daily_returns = [
                (current - previous) / previous if previous != 0 else 0.0
                for previous, current in itertools.pairwise(portfolio_values)
            ]
            if len(daily_returns) < 2:
                final_sharpe_ratio = 0.0
            else:
                mean_return = sum(daily_returns) / len(daily_returns)
                variance = sum((value - mean_return) ** 2 for value in daily_returns) / (
                    len(daily_returns) - 1
                )
                if variance == 0:
                    final_sharpe_ratio = 0.0
                else:
                    final_sharpe_ratio = mean_return / math.sqrt(variance) * math.sqrt(252)

        running_peak = -math.inf
        final_max_drawdown = 0.0
        for value in portfolio_values:
            running_peak = max(running_peak, value)
            if running_peak != -math.inf:
                final_max_drawdown = min(final_max_drawdown, value - running_peak)

        logger.info(
            "Backtest complete: %d days simulated in %.1fs — final Sharpe=%.3f, max drawdown=%.2f",
            len(days),
            time.perf_counter() - run_started_at,
            final_sharpe_ratio,
            final_max_drawdown,
        )
        return BacktestResult(
            start_date=self.start_date,
            end_date=self.end_date,
            portfolio_value_history=portfolio_value_history,
            final_sharpe_ratio=final_sharpe_ratio,
            final_max_drawdown=final_max_drawdown,
            decisions_by_day=decisions_by_day,
            trade_log=trade_log,
        )
