"""Unit tests for agentic_trading.backtest.engine.BacktestEngine.

Uses fakes for commander/portfolio/track_record/data_loader — a real end-to-end run
(real Commander, real Portfolio, real quantcore/yfinance data) belongs in
tests/integration/, not here.

BacktestEngine is the single shared simulation loop both the CLI and the GUI call
directly (see its own class docstring) — this suite exercises: `data_loader` receiving
the full `assets: list[str]` (not a single asset string), attribution being recorded at
*both* the analyst and manager level (the GUI's Desks tab needs the latter), and the
richer `BacktestResult` fields (`decisions_by_day`, `trade_log`) both front ends read.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from agentic_trading.agents.schemas import (
    ManagerSignal,
    PerformanceRecord,
    PortfolioDecision,
    TradeRecord,
)
from agentic_trading.backtest.engine import BacktestEngine
from agentic_trading.scoring.scoreboard import PerformanceScoreboard

START_DATE = date(2024, 1, 2)
END_DATE = date(2024, 1, 8)

# Business days only: the loaded frame has no rows for the 2024-01-06/07 weekend, which is
# what lets test_run_iterates_only_days_present_in_loaded_data_within_range distinguish
# "iterate calendar days" from "iterate days present in the loaded data" (the chosen
# interpretation, since real market data from yfinance never has weekend rows either).
SIMULATED_DAYS = [
    date(2024, 1, 2),
    date(2024, 1, 3),
    date(2024, 1, 4),
    date(2024, 1, 5),
    date(2024, 1, 8),
]

_SIGNAL_PARAMS: dict[date, tuple[str, float]] = {
    date(2024, 1, 2): ("long", 0.8),
    date(2024, 1, 3): ("short", 0.5),
    date(2024, 1, 4): ("flat", 0.9),
    date(2024, 1, 5): ("long", 1.0),
    date(2024, 1, 8): ("short", 0.3),
}
_DIRECTION_SIGN = {"long": 1.0, "short": -1.0, "flat": 0.0}
_TOTAL_VALUES = [10_000.0, 10_150.0, 10_080.0, 10_300.0, 10_250.0]


def _price_frame() -> pd.DataFrame:
    # 2024-01-01 is a pre-start pad day, so the first simulated day (2024-01-02) has a
    # real (non-NaN) log_return, mirroring loader.py's own log-return convention.
    index = pd.to_datetime(
        ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]
    )
    close = pd.Series([100.0, 102.0, 101.0, 104.0, 103.0, 110.0], index=index)
    return pd.DataFrame({"close": close, "log_return": np.log(close / close.shift(1))})


def _decisions_by_day() -> dict[date, list[PortfolioDecision]]:
    decisions: dict[date, list[PortfolioDecision]] = {}
    for day, (direction, confidence) in _SIGNAL_PARAMS.items():
        signal = ManagerSignal(
            manager_id="manager-1",
            source_agent_id="analyst-1",
            asset="AAPL",
            as_of_date=day,
            direction=direction,
            confidence=confidence,
            scale_applied=1.0,
            rationale="test fixture",
        )
        decisions[day] = [
            PortfolioDecision(
                asset="AAPL", as_of_date=day, target_weight=0.3, contributing_signals=[signal]
            )
        ]
    return decisions


class _FakeDataLoader:
    """Records every call; returns the same fixed frame regardless of args."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame
        self.calls: list[tuple[list[str], date, date]] = []

    def __call__(self, assets: list[str], start: date, end: date) -> pd.DataFrame:
        self.calls.append((assets, start, end))
        return self._frame


class _FakeCommander:
    def __init__(self, decisions_by_day: dict[date, list[PortfolioDecision]]) -> None:
        self._decisions_by_day = decisions_by_day
        self.calls: list[tuple[pd.DataFrame, pd.Timestamp, object]] = []

    def decide(
        self, market_data: pd.DataFrame, as_of_date: pd.Timestamp, scoreboard: object
    ) -> list[PortfolioDecision]:
        self.calls.append((market_data.copy(), as_of_date, scoreboard))
        return self._decisions_by_day[pd.Timestamp(as_of_date).date()]


class _FakePortfolio:
    def __init__(self, total_values: list[float]) -> None:
        self._total_values = total_values
        self.total_value_calls: list[dict[str, float]] = []
        self.apply_decision_calls: list[tuple[PortfolioDecision, float, float]] = []

    def total_value(self, prices: dict[str, float]) -> float:
        value = self._total_values[len(self.total_value_calls)]
        self.total_value_calls.append(dict(prices))
        return value

    def apply_decision(
        self, decision: PortfolioDecision, price: float, portfolio_value: float
    ) -> TradeRecord:
        self.apply_decision_calls.append((decision, price, portfolio_value))
        return TradeRecord(
            as_of_date=decision.as_of_date,
            asset=decision.asset,
            action="hold",
            target_weight=decision.target_weight,
            previous_weight=0.0,
            target_shares=0.0,
            delta_shares=0.0,
            price=price,
            notional=0.0,
        )


class _FakeTrackRecord:
    def __init__(self) -> None:
        self.entries: list[PerformanceRecord] = []

    def record(self, entry: PerformanceRecord) -> None:
        self.entries.append(entry)


def _build_engine() -> tuple[BacktestEngine, _FakeCommander, _FakePortfolio, _FakeTrackRecord]:
    data_loader = _FakeDataLoader(_price_frame())
    commander = _FakeCommander(_decisions_by_day())
    portfolio = _FakePortfolio(list(_TOTAL_VALUES))
    track_record = _FakeTrackRecord()
    engine = BacktestEngine(
        commander=commander,  # type: ignore[arg-type]
        portfolio=portfolio,  # type: ignore[arg-type]
        track_record=track_record,  # type: ignore[arg-type]
        data_loader=data_loader,
        start_date=START_DATE,
        end_date=END_DATE,
    )
    return engine, commander, portfolio, track_record


def test_run_never_exposes_future_rows_to_commander() -> None:
    """The single most important invariant: commander.decide() must never see a row dated
    after that call's as_of_date, for every simulated day — CLAUDE.md's no-look-ahead rule
    is structural, not a suggestion, and this is the enforcement point (get_point_in_time_data)."""
    engine, commander, _, _ = _build_engine()

    engine.run()

    assert len(commander.calls) == len(SIMULATED_DAYS)
    for market_data, as_of_date, scoreboard in commander.calls:
        assert isinstance(scoreboard, PerformanceScoreboard)
        as_of_timestamp = pd.Timestamp(as_of_date)
        assert (market_data.index <= as_of_timestamp).all()
        assert as_of_timestamp in market_data.index


def test_run_iterates_only_days_present_in_loaded_data_within_range() -> None:
    """Chosen interpretation (see module docstring): iterate days present in the loaded
    data, not every calendar day — weekends have no price to act on."""
    engine, commander, _, _ = _build_engine()

    engine.run()

    called_days = [pd.Timestamp(as_of_date).date() for _, as_of_date, _ in commander.calls]
    assert called_days == SIMULATED_DAYS


def test_run_applies_decisions_using_that_days_own_portfolio_value_snapshot() -> None:
    """portfolio_value passed to apply_decision must be exactly what total_value() returned
    for that day's prices — computed before that day's own decisions are applied, not a
    figure already reflecting them."""
    engine, _, portfolio, _ = _build_engine()
    frame = _price_frame()

    engine.run()

    assert len(portfolio.total_value_calls) == len(SIMULATED_DAYS)
    assert len(portfolio.apply_decision_calls) == len(SIMULATED_DAYS)
    for day, prices, (decision, price, portfolio_value), expected_total_value in zip(
        SIMULATED_DAYS,
        portfolio.total_value_calls,
        portfolio.apply_decision_calls,
        _TOTAL_VALUES,
        strict=True,
    ):
        expected_close = frame.loc[pd.Timestamp(day), "close"]
        assert prices == pytest.approx({"AAPL": expected_close})
        assert price == pytest.approx(expected_close)
        assert decision.as_of_date == day
        assert portfolio_value == pytest.approx(expected_total_value)


def test_run_records_attributable_performance_at_both_analyst_and_manager_level() -> None:
    """attributable_return = direction_sign * confidence * realized_return (long=+1,
    short=-1, flat=0), per PerformanceRecord's docstring. Recorded twice per contributing
    signal — once as source_type="analyst" (source_id=source_agent_id), once as
    source_type="manager" (source_id=manager_id) — since the GUI's Desks tab needs
    manager-level attribution as much as the Analysts tab needs analyst-level."""
    engine, _, _, track_record = _build_engine()
    frame = _price_frame()

    engine.run()

    assert len(track_record.entries) == 2 * len(SIMULATED_DAYS)
    analyst_entries = [e for e in track_record.entries if e.source_type == "analyst"]
    manager_entries = [e for e in track_record.entries if e.source_type == "manager"]
    assert len(analyst_entries) == len(SIMULATED_DAYS)
    assert len(manager_entries) == len(SIMULATED_DAYS)

    for day, analyst_entry, manager_entry in zip(
        SIMULATED_DAYS, analyst_entries, manager_entries, strict=True
    ):
        direction, confidence = _SIGNAL_PARAMS[day]
        realized_return = frame.loc[pd.Timestamp(day), "log_return"]
        expected_return = _DIRECTION_SIGN[direction] * confidence * realized_return

        assert analyst_entry.source_id == "analyst-1"
        assert analyst_entry.asset == "AAPL"
        assert analyst_entry.as_of_date == day
        assert analyst_entry.attributable_return == pytest.approx(expected_return)

        assert manager_entry.source_id == "manager-1"
        assert manager_entry.asset == "AAPL"
        assert manager_entry.as_of_date == day
        assert manager_entry.attributable_return == pytest.approx(expected_return)


def test_run_returns_backtest_result_matching_dates_and_value_history() -> None:
    """start_date/end_date echo the engine's own, and portfolio_value_history has one
    entry per simulated day in order, matching total_value()'s returns for that day.

    final_sharpe_ratio/final_max_drawdown's exact formula is unspecified anywhere in the
    codebase (PerformanceScoreboard's own sharpe_ratio/max_drawdown are themselves
    unimplemented), so only their type is asserted here — see report."""
    engine, *_ = _build_engine()

    result = engine.run()

    assert result.start_date == START_DATE
    assert result.end_date == END_DATE
    assert [day for day, _ in result.portfolio_value_history] == SIMULATED_DAYS
    assert [value for _, value in result.portfolio_value_history] == pytest.approx(_TOTAL_VALUES)
    assert isinstance(result.final_sharpe_ratio, float)
    assert isinstance(result.final_max_drawdown, float)


def test_run_returns_decisions_by_day_matching_what_commander_produced() -> None:
    """decisions_by_day is what both the CLI and the GUI's Desks/Analysts
    drill-down are built from — must contain exactly the fake Commander's
    decisions, keyed by the same days that were simulated."""
    engine, _, _, _ = _build_engine()

    result = engine.run()

    assert set(result.decisions_by_day.keys()) == set(SIMULATED_DAYS)
    for day in SIMULATED_DAYS:
        assert len(result.decisions_by_day[day]) == 1
        assert result.decisions_by_day[day][0].asset == "AAPL"
        assert result.decisions_by_day[day][0].as_of_date == day


def test_run_returns_a_trade_log_entry_per_decision() -> None:
    """trade_log is the single source of truth for "what did the portfolio
    actually do" — see Portfolio.apply_decision()'s docstring."""
    engine, _, _, _ = _build_engine()

    result = engine.run()

    assert len(result.trade_log) == len(SIMULATED_DAYS)
    for day, trade in zip(SIMULATED_DAYS, result.trade_log, strict=True):
        assert trade.as_of_date == day
        assert trade.asset == "AAPL"


def test_run_calls_on_day_complete_once_per_simulated_day_with_correct_args() -> None:
    engine, *_ = _build_engine()
    calls: list[tuple[int, int, date, float]] = []

    engine.run(on_day_complete=lambda *args: calls.append(args))

    assert len(calls) == len(SIMULATED_DAYS)
    for expected_index, (day_index, total_days, as_of_date, portfolio_value) in enumerate(calls):
        assert day_index == expected_index
        assert total_days == len(SIMULATED_DAYS)
        assert as_of_date == SIMULATED_DAYS[expected_index]
        assert portfolio_value == pytest.approx(_TOTAL_VALUES[expected_index])


def test_run_works_without_an_on_day_complete_callback() -> None:
    """The callback is optional — omitting it must not change behavior."""
    engine, *_ = _build_engine()

    result = engine.run()

    assert len(result.portfolio_value_history) == len(SIMULATED_DAYS)


def test_run_is_deterministic_across_equivalent_fakes() -> None:
    """Same inputs in, byte-identical BacktestResult out — CLAUDE.md requires this module
    to contain no randomness and no LLM calls; two independently-built but identical fake
    setups must produce equal results."""
    engine_a, *_ = _build_engine()
    engine_b, *_ = _build_engine()

    assert engine_a.run() == engine_b.run()
