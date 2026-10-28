"""Unit tests for agentic_trading.backtest.engine.BacktestEngine.

Uses fakes for commander/portfolio/track_record/data_loader — a real end-to-end run
(real Commander, real Portfolio, real lakehouse data) belongs in
tests/integration/, not here.

BacktestEngine is the single shared simulation loop both the CLI and the GUI call
directly (see its own class docstring) — this suite exercises: `data_loader` receiving
the full `assets: list[str]` (not a single asset string), attribution being recorded at
*both* the analyst and manager level (the GUI's Desks tab needs the latter), and the
richer `BacktestResult` fields (`decisions_by_day`, `trade_log`) both front ends read.
"""

import itertools
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agentic_trading.agents.commander import CommanderRiskState
from agentic_trading.agents.schemas import (
    ManagerSignal,
    OptionContract,
    PerformanceRecord,
    PortfolioDecision,
    TradeRecord,
)
from agentic_trading.backtest.engine import BacktestEngine
from agentic_trading.backtest.portfolio import OptionPosition, Portfolio
from agentic_trading.config import settings, temporary_overrides
from agentic_trading.data.loaders.option_marks import OptionMark
from agentic_trading.scoring.scoreboard import PerformanceScoreboard
from agentic_trading.scoring.track_record import TrackRecord


@pytest.fixture(autouse=True)
def _lift_option_notional_cap():
    """These engine tests run on a $10,000 book where one contract already
    exceeds the firm's option-notional cap; lift it so fills are governed
    by the premium target the tests assert on."""
    with temporary_overrides({"options.max_notional_fraction": 1_000.0}):
        yield


START_DATE = date(2024, 1, 2)
END_DATE = date(2024, 1, 8)

# Business days only: the loaded frame has no rows for the 2024-01-06/07 weekend, which is
# what lets test_run_iterates_only_days_present_in_loaded_data_within_range distinguish
# "iterate calendar days" from "iterate days present in the loaded data" (the chosen
# interpretation, since real market data never has weekend rows either).
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
            raw_confidence=confidence,
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
        self,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
        scoreboard: object,
        instrument_returns: dict[str, float] | None = None,
    ) -> list[PortfolioDecision]:
        self.calls.append((market_data.copy(), as_of_date, scoreboard))
        return self._decisions_by_day[pd.Timestamp(as_of_date).date()]


class _FakePortfolio:
    def __init__(self, total_values: list[float]) -> None:
        self._total_values = total_values
        self.cash = 0.0
        self.total_value_calls: list[dict[str, float]] = []
        self.apply_decision_calls: list[tuple[PortfolioDecision, float, float]] = []
        self.option_positions: dict[str, OptionPosition] = {}

    def total_value(
        self, prices: dict[str, float], option_marks: dict[str, OptionMark] | None = None
    ) -> float:
        value = self._total_values[len(self.total_value_calls)]
        self.total_value_calls.append(dict(prices))
        return value

    def apply_financing(
        self,
        prices: dict[str, float],
        risk_free_rate_annual: float,
        days_held: int,
        spots: dict[str, float] | None = None,
    ) -> float:
        return 0.0

    def settle_expired(self, marks: dict[str, OptionMark], as_of_date: date) -> list[TradeRecord]:
        return []

    def apply_decision(
        self,
        decision: PortfolioDecision,
        price: float,
        portfolio_value: float,
        adv_shares: float | None = None,
        daily_vol: float | None = None,
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

    def __len__(self) -> int:
        return len(self.entries)

    def entries_since(self, count: int) -> list[PerformanceRecord]:
        return self.entries[count:]


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
        # This fixture's ManagerSignal.raw_confidence == confidence (no manager
        # scaling applied to the raw value), so raw_attributable_return matches
        # attributable_return exactly here.
        assert analyst_entry.raw_attributable_return == pytest.approx(expected_return)

        assert manager_entry.source_id == "manager-1"
        assert manager_entry.asset == "AAPL"
        assert manager_entry.as_of_date == day
        assert manager_entry.attributable_return == pytest.approx(expected_return)
        assert manager_entry.raw_attributable_return == pytest.approx(expected_return)


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


# ---------------------------------------------------------------------------
# Financing, Commander risk state, pod P&L, and PerformanceRecord provenance
# ---------------------------------------------------------------------------


class CommanderRiskStatefulCommander(_FakeCommander):
    """A fake commander that exposes/accepts risk state like `Commander`."""

    def __init__(
        self, decisions_by_day: dict[date, list[PortfolioDecision]], state: CommanderRiskState
    ) -> None:
        super().__init__(decisions_by_day)
        self.state = state
        self.restored: list[CommanderRiskState] = []

    def risk_state(self) -> CommanderRiskState:
        return self.state

    def restore_risk_state(self, state: CommanderRiskState) -> None:
        self.restored.append(state)
        self.state = state


def _short_decisions_by_day() -> dict[date, list[PortfolioDecision]]:
    return {
        day: [
            PortfolioDecision(
                asset="AAPL", as_of_date=day, target_weight=-0.5, contributing_signals=[]
            )
        ]
        for day in SIMULATED_DAYS
    }


def _real_portfolio_engine(
    decisions_by_day: dict[date, list[PortfolioDecision]],
    *,
    cash: float = 10_000.0,
    risk_free_rate_series: pd.Series | None = None,
) -> tuple[BacktestEngine, Portfolio]:
    portfolio = Portfolio(cash=cash, positions={}, commission_rate=0.0, slippage_bps=0.0)
    engine = BacktestEngine(
        commander=_FakeCommander(decisions_by_day),  # type: ignore[arg-type]
        portfolio=portfolio,
        track_record=TrackRecord(),
        data_loader=_FakeDataLoader(_price_frame()),
        start_date=START_DATE,
        end_date=END_DATE,
        risk_free_rate_series=risk_free_rate_series,
    )
    return engine, portfolio


def test_run_charges_stock_borrow_on_a_short_position_over_the_calendar_gap() -> None:
    """A short held across simulated days accrues borrow at
    `short_borrow_rate_annual` for the actual calendar-day gap (the Friday
    2024-01-05 -> Monday 2024-01-08 step is three days), on the short
    notional marked at that day's close; nothing accrues on the first day
    (no gap yet) and cash stays positive so no margin interest applies."""
    engine, portfolio = _real_portfolio_engine(_short_decisions_by_day())
    frame = _price_frame()

    with temporary_overrides({"firm.short_borrow_rate_annual": 0.01}):
        result = engine.run()

    expected = 0.0
    for previous_day, day in itertools.pairwise(SIMULATED_DAYS):
        # Shares shorted on `previous_day` are marked at `day`'s close.
        previous_close = float(frame.loc[pd.Timestamp(previous_day), "close"])
        value_before_previous_trade = dict(result.portfolio_value_history)[previous_day]
        shares = (-0.5 * value_before_previous_trade) / previous_close
        notional = -shares * float(frame.loc[pd.Timestamp(day), "close"])
        expected += notional * 0.01 * (day - previous_day).days / 365.0

    assert result.total_financing_cost > 0.0
    assert result.total_financing_cost == pytest.approx(expected)
    assert portfolio.positions["AAPL"] < 0.0


def test_run_charges_margin_interest_on_negative_cash_at_risk_free_plus_spread() -> None:
    """A 2x long borrows cash; the interest rate is the day's annualized
    risk-free rate (daily series x trading_days_per_year) plus
    `margin_spread_annual`, accrued per calendar day of the gap."""
    leveraged = {
        day: [
            PortfolioDecision(
                asset="AAPL", as_of_date=day, target_weight=2.0, contributing_signals=[]
            )
        ]
        for day in SIMULATED_DAYS
    }
    daily_rate = 0.05 / settings.firm.trading_days_per_year
    risk_free = pd.Series(daily_rate, index=pd.to_datetime(SIMULATED_DAYS))
    engine, portfolio = _real_portfolio_engine(leveraged, risk_free_rate_series=risk_free)

    with temporary_overrides({"firm.margin_spread_annual": 0.015}):
        result = engine.run()

    # Reconstruct the charge day by day from the recorded values.
    frame = _price_frame()
    expected = 0.0
    cash = 10_000.0
    shares = 0.0
    for index, day in enumerate(SIMULATED_DAYS):
        close = float(frame.loc[pd.Timestamp(day), "close"])
        if index > 0:
            gap = (day - SIMULATED_DAYS[index - 1]).days
            charge = -cash * (0.05 + 0.015) * gap / 365.0 if cash < 0 else 0.0
            expected += charge
            cash -= charge
        value = cash + shares * close
        target_shares = 2.0 * value / close
        cash -= (target_shares - shares) * close
        shares = target_shares

    assert result.total_financing_cost > 0.0
    assert result.total_financing_cost == pytest.approx(expected)
    assert portfolio.cash < 0.0


def test_run_charges_no_financing_when_flat_with_positive_cash() -> None:
    flat = {
        day: [
            PortfolioDecision(
                asset="AAPL", as_of_date=day, target_weight=0.0, contributing_signals=[]
            )
        ]
        for day in SIMULATED_DAYS
    }
    engine, portfolio = _real_portfolio_engine(flat)

    result = engine.run()

    assert result.total_financing_cost == pytest.approx(0.0)
    assert result.total_transaction_cost == pytest.approx(0.0)
    assert portfolio.cash == pytest.approx(10_000.0)


def test_run_reports_total_transaction_cost_as_sum_of_trade_log_commissions() -> None:
    long_decisions = {
        day: [
            PortfolioDecision(
                asset="AAPL", as_of_date=day, target_weight=0.5, contributing_signals=[]
            )
        ]
        for day in SIMULATED_DAYS
    }
    portfolio = Portfolio(cash=10_000.0, positions={}, commission_rate=0.001, slippage_bps=0.0)
    engine = BacktestEngine(
        commander=_FakeCommander(long_decisions),  # type: ignore[arg-type]
        portfolio=portfolio,
        track_record=TrackRecord(),
        data_loader=_FakeDataLoader(_price_frame()),
        start_date=START_DATE,
        end_date=END_DATE,
    )

    result = engine.run()

    assert result.total_transaction_cost > 0.0
    assert result.total_transaction_cost == pytest.approx(
        sum(trade.transaction_cost for trade in result.trade_log)
    )


def test_run_populates_signal_score_and_realized_return_on_every_record() -> None:
    """signal_score = direction_sign x raw_confidence and realized_return is
    the asset's own log return that day, on both the analyst- and
    manager-level records."""
    engine, _, _, track_record = _build_engine()
    frame = _price_frame()

    engine.run()

    for entry in track_record.entries:
        direction, confidence = _SIGNAL_PARAMS[entry.as_of_date]
        realized_return = float(frame.loc[pd.Timestamp(entry.as_of_date), "log_return"])
        assert entry.signal_score == pytest.approx(_DIRECTION_SIGN[direction] * confidence)
        assert entry.realized_return == pytest.approx(realized_return)
        assert entry.raw_attributable_return == pytest.approx(
            entry.signal_score * entry.realized_return
        )


def test_run_reports_pod_return_history_from_the_commanders_risk_state() -> None:
    history = {"risk_manager": [(date(2024, 1, 3), 0.01), (date(2024, 1, 4), -0.002)]}
    commander = CommanderRiskStatefulCommander(
        _decisions_by_day(), CommanderRiskState(pod_return_history=history)
    )
    engine = BacktestEngine(
        commander=commander,  # type: ignore[arg-type]
        portfolio=_FakePortfolio(list(_TOTAL_VALUES)),  # type: ignore[arg-type]
        track_record=_FakeTrackRecord(),  # type: ignore[arg-type]
        data_loader=_FakeDataLoader(_price_frame()),
        start_date=START_DATE,
        end_date=END_DATE,
    )

    result = engine.run()

    assert result.pod_return_history == history


def test_run_reports_empty_pod_return_history_for_a_commander_without_risk_state() -> None:
    engine, *_ = _build_engine()

    result = engine.run()

    assert result.pod_return_history == {}


def test_checkpoint_round_trips_commander_risk_state_on_resume(tmp_path: Path) -> None:
    """The risk state the commander reports at the last checkpointed day is
    what a resumed run hands back through `restore_risk_state`, rebuilt as
    a `CommanderRiskState`."""
    checkpoint_path = tmp_path / "checkpoint.db"
    state = CommanderRiskState(
        tripped_pods=["cvar_manager"],
        pod_return_history={"risk_manager": [(date(2024, 1, 3), 0.01)]},
        last_pod_weights={"risk_manager": {"AAPL": 0.3}},
    )

    def build(commander: CommanderRiskStatefulCommander, *, resume: bool) -> BacktestEngine:
        return BacktestEngine(
            commander=commander,  # type: ignore[arg-type]
            portfolio=Portfolio(cash=10_000.0, positions={}),
            track_record=TrackRecord(),
            data_loader=_FakeDataLoader(_price_frame()),
            start_date=START_DATE,
            end_date=END_DATE,
            checkpoint_path=checkpoint_path,
            resume=resume,
        )

    first = CommanderRiskStatefulCommander(_decisions_by_day(), state)
    build(first, resume=False).run()

    resumed_commander = CommanderRiskStatefulCommander(_decisions_by_day(), CommanderRiskState())
    resumed = build(resumed_commander, resume=True)
    resumed.run()

    assert len(resumed_commander.restored) == 1
    restored = resumed_commander.restored[0]
    assert isinstance(restored, CommanderRiskState)
    assert restored == state


def test_checkpoint_resume_leaves_a_commander_without_risk_state_untouched(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "checkpoint.db"

    def build(*, resume: bool) -> BacktestEngine:
        return BacktestEngine(
            commander=_FakeCommander(_decisions_by_day()),  # type: ignore[arg-type]
            portfolio=Portfolio(cash=10_000.0, positions={}),
            track_record=TrackRecord(),
            data_loader=_FakeDataLoader(_price_frame()),
            start_date=START_DATE,
            end_date=END_DATE,
            checkpoint_path=checkpoint_path,
            resume=resume,
        )

    first = build(resume=False).run()
    resumed = build(resume=True).run()

    assert resumed.portfolio_value_history == first.portfolio_value_history
    assert resumed.pod_return_history == {}


# ---------------------------------------------------------------------------
# Listed-option decisions end to end
# ---------------------------------------------------------------------------

OPTION_CONTRACT = OptionContract(
    underlying="AAPL", expiry=date(2024, 1, 5), strike=100.0, option_type="call"
)
_TIME_VALUE = 1.0
_HALF_SPREAD = 0.1


def _fake_load_option_mark(contract: OptionContract, as_of_date: date, spot: float) -> OptionMark:
    """A deterministic stand-in for the lakehouse mark: intrinsic plus a
    flat time value before expiry (bid/ask a fixed 0.10 either side of
    mid), intrinsic on or after expiry."""
    intrinsic = max(spot - contract.strike, 0.0)
    if as_of_date >= contract.expiry:
        return OptionMark(
            contract=contract,
            as_of_date=as_of_date,
            snapshot_date=None,
            mid=intrinsic,
            bid=intrinsic,
            ask=intrinsic,
            implied_volatility=0.0,
            delta=0.0,
            gamma=0.0,
            vega=0.0,
            theta=0.0,
            is_model_mark=False,
            is_expired=True,
            tradeable=False,
        )
    mid = intrinsic + _TIME_VALUE
    return OptionMark(
        contract=contract,
        as_of_date=as_of_date,
        snapshot_date=as_of_date,
        mid=mid,
        bid=mid - _HALF_SPREAD,
        ask=mid + _HALF_SPREAD,
        implied_volatility=0.25,
        delta=0.5,
        gamma=0.02,
        vega=0.1,
        theta=-0.05,
        is_model_mark=False,
        is_expired=False,
        tradeable=True,
    )


def _option_decisions_by_day() -> dict[date, list[PortfolioDecision]]:
    """Buy 5% of NAV in premium on day one, re-affirm the same size on day
    two (a hold), nothing afterwards; the contract expires on day four."""
    signal = ManagerSignal(
        manager_id="cvar_manager",
        source_agent_id="tail_hedge_analyst",
        asset=OPTION_CONTRACT.contract_id,
        as_of_date=date(2024, 1, 2),
        direction="long",
        confidence=0.8,
        raw_confidence=0.8,
        scale_applied=1.0,
        rationale="test fixture",
    )
    decisions: dict[date, list[PortfolioDecision]] = {day: [] for day in SIMULATED_DAYS}
    decisions[date(2024, 1, 2)] = [
        PortfolioDecision(
            asset=OPTION_CONTRACT.contract_id,
            as_of_date=date(2024, 1, 2),
            target_weight=0.05,
            contributing_signals=[signal],
            instrument=OPTION_CONTRACT,
        )
    ]
    decisions[date(2024, 1, 3)] = [
        PortfolioDecision(
            asset=OPTION_CONTRACT.contract_id,
            as_of_date=date(2024, 1, 3),
            target_weight=0.04,
            contributing_signals=[signal.model_copy(update={"as_of_date": date(2024, 1, 3)})],
            instrument=OPTION_CONTRACT,
        )
    ]
    return decisions


def _option_engine(
    *, checkpoint_path: Path | None = None, resume: bool = False
) -> tuple[BacktestEngine, Portfolio]:
    portfolio = Portfolio(cash=10_000.0, positions={}, commission_rate=0.0, slippage_bps=0.0)
    engine = BacktestEngine(
        commander=_FakeCommander(_option_decisions_by_day()),  # type: ignore[arg-type]
        portfolio=portfolio,
        track_record=TrackRecord(),
        data_loader=_FakeDataLoader(_price_frame()),
        start_date=START_DATE,
        end_date=END_DATE,
        checkpoint_path=checkpoint_path,
        resume=resume,
    )
    return engine, portfolio


@pytest.fixture
def option_marks_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agentic_trading.backtest.engine.load_option_mark", _fake_load_option_mark)
    monkeypatch.setattr(
        "agentic_trading.backtest.engine.lakehouse_loader.split_adjustment_factor",
        lambda *_args: 1.0,
    )


def test_run_fills_marks_and_settles_an_option_decision(option_marks_stub: None) -> None:
    """Day one (spot 102): 5% of 10,000 is 500 of premium at a 3.00 mid, so
    2 contracts bought at the 3.10 ask. Day two (spot 101, mid 2.00) and
    day three (spot 104, mid 5.00) mark the position. Day four is expiry
    (spot 103): settled at 3.00 intrinsic with an "expire" record. Day
    five is flat."""
    commission = settings.options.commission_per_contract
    with temporary_overrides({"options.max_snapshot_age_days": 10}):
        engine, portfolio = _option_engine()
        result = engine.run()

    premium_paid = 2 * 3.1 * 100
    cash_after_buy = 10_000.0 - premium_paid - 2 * commission
    expected_values = [
        10_000.0,
        cash_after_buy + 2 * 2.0 * 100,
        cash_after_buy + 2 * 5.0 * 100,
        cash_after_buy + 2 * 3.0 * 100,
        cash_after_buy + 2 * 3.0 * 100,
    ]
    assert [value for _, value in result.portfolio_value_history] == pytest.approx(expected_values)
    assert portfolio.option_positions == {}
    assert portfolio.cash == pytest.approx(expected_values[-1])

    actions = [(trade.as_of_date, trade.action) for trade in result.trade_log]
    assert actions == [
        (date(2024, 1, 2), "buy"),
        (date(2024, 1, 3), "hold"),
        (date(2024, 1, 5), "expire"),
    ]
    buy, hold, expire = result.trade_log
    assert buy.instrument == OPTION_CONTRACT
    assert buy.contracts == pytest.approx(2.0)
    assert buy.premium == pytest.approx(-premium_paid)
    assert buy.transaction_cost == pytest.approx(2 * commission)
    assert hold.contracts == 0.0
    assert expire.contracts == pytest.approx(-2.0)
    assert expire.premium == pytest.approx(600.0)
    assert result.total_option_premium_paid == pytest.approx(premium_paid)
    assert result.total_transaction_cost == pytest.approx(2 * commission)
    assert result.total_financing_cost == pytest.approx(0.0)


def test_run_attributes_option_decisions_to_the_contracts_own_mark_return(
    option_marks_stub: None,
) -> None:
    """Realized return on an option decision is the log change in the
    contract's close mid: 0.0 on the first day it is seen, log(2/3) on
    day two (mid 3.00 -> 2.00)."""
    engine, _ = _option_engine()

    engine.run()

    entries = engine.track_record.entries_since(0)
    analyst_entries = [entry for entry in entries if entry.source_type == "analyst"]
    assert [entry.as_of_date for entry in analyst_entries] == [
        date(2024, 1, 2),
        date(2024, 1, 3),
    ]
    assert analyst_entries[0].asset == OPTION_CONTRACT.contract_id
    assert analyst_entries[0].realized_return == pytest.approx(0.0)
    assert analyst_entries[1].realized_return == pytest.approx(np.log(2.0 / 3.0))
    assert analyst_entries[1].attributable_return == pytest.approx(0.8 * np.log(2.0 / 3.0))


def test_checkpoint_round_trips_option_positions_on_resume(
    option_marks_stub: None, tmp_path: Path
) -> None:
    """A run interrupted after day two checkpoints its open option
    position; the resumed run carries it (no second buy), marks it and
    settles it exactly as an uninterrupted run does."""
    checkpoint_path = tmp_path / "checkpoint.db"
    uninterrupted, _ = _option_engine()
    expected = uninterrupted.run()

    class _InterruptError(Exception):
        pass

    def stop_after_day_two(day_index: int, *_args: object) -> None:
        if day_index == 1:
            raise _InterruptError

    first, first_portfolio = _option_engine(checkpoint_path=checkpoint_path)
    with pytest.raises(_InterruptError):
        first.run(on_day_complete=stop_after_day_two)
    assert first_portfolio.option_positions[OPTION_CONTRACT.contract_id].contracts == 2

    resumed, resumed_portfolio = _option_engine(checkpoint_path=checkpoint_path, resume=True)
    result = resumed.run()

    assert resumed_portfolio.option_positions == {}
    assert result.portfolio_value_history == pytest.approx(expected.portfolio_value_history)
    assert [trade.action for trade in result.trade_log] == ["buy", "hold", "expire"]
    assert result.total_option_premium_paid == pytest.approx(expected.total_option_premium_paid)


def test_run_skips_an_option_decision_with_no_mark(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agentic_trading.backtest.engine.load_option_mark", lambda *_args: None)
    engine, portfolio = _option_engine()

    result = engine.run()

    assert result.trade_log == []
    assert portfolio.option_positions == {}
    assert [value for _, value in result.portfolio_value_history] == pytest.approx(
        [10_000.0] * len(SIMULATED_DAYS)
    )
