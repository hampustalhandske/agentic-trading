# test_engine_missing_bars.py

"""AT-125 calendar: a firm run decides on every XNYS session. An instrument with no bar on a
session gets no signal and is never filled that day. One never decided yet (starting mid-window)
gets no decision; one held keeps its last decided target as that day's decision (edgelab's rule
(c)), which fills at the next session with a bar, so over a two-session gap the order goes
unfilled twice and the position is held. The book values a held equity on such a day at its
last close, so the day's NAV stays finite and the move lands on its next bar; a held future is
not re-marked, so its P&L for the day is zero."""

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest
from pydantic import PrivateAttr

from agentic_trading.agents.base import AnalystAgent, ManagerAgent
from agentic_trading.agents.commander import Commander
from agentic_trading.agents.portfolio_construction import PortfolioConstructor
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.backtest.engine import BacktestEngine, _valuation_prices
from agentic_trading.backtest.portfolio import Portfolio
from agentic_trading.data.loader import _with_rolls_since_previous_row
from agentic_trading.data.trading_calendar import sessions_between
from agentic_trading.scoring.track_record import TrackRecord
from agentic_trading.scoring.weighting import EqualWeight

_SESSIONS = pd.DatetimeIndex(sessions_between(date(2016, 1, 4), date(2016, 1, 22)))
# LATE's first bar, and the session GAP has no bar on (Monday 2016-01-18 is a holiday).
_LATE_START = _SESSIONS[4]
_GAP_DAY = _SESSIONS[6]


class _AlternatingAnalyst(AnalystAgent):
    """Long on every asset, at a confidence that alternates by day so the book trades daily;
    records each (asset, day) it is asked about and whether its frame had a missing close for
    that asset."""

    agent_id: str = "alternating_analyst"
    _asked: list[tuple[str, date, bool]] = PrivateAttr(default_factory=list)

    def generate_signal(
        self, asset: str, market_data: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> AgentSignal:
        closes = market_data[(asset, "close")]
        self._asked.append(
            (asset, as_of_date.date(), bool(closes[closes.index <= as_of_date].isna().any()))
        )
        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction="long",
            confidence=0.4 if as_of_date.day % 2 else 0.8,
            rationale="stub",
        )


class _ViewConstructor(PortfolioConstructor):
    def construct(self, views, returns_history, as_of_date, previous_weights=None):
        return {asset: 0.3 * view.net_signal for asset, view in views.items()}


class _PassThroughManager(ManagerAgent):
    def scale_factor(self, signal, market_data, as_of_date) -> float:
        return 1.0


def _frame() -> pd.DataFrame:
    """SPY on every session, LATE from `_LATE_START`, GAP on every session but `_GAP_DAY`;
    each `log_return` is over the previous close the asset has, as the session-aligned loader
    gives it."""
    rng = np.random.default_rng(7)
    columns: dict[tuple[str, str], pd.Series] = {}
    for asset in ("SPY", "LATE", "GAP"):
        close = pd.Series(
            100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, len(_SESSIONS)))), _SESSIONS
        )
        if asset == "LATE":
            close[close.index < _LATE_START] = np.nan
        if asset == "GAP":
            close[_GAP_DAY] = np.nan
        present = close.dropna()
        log_return = np.log(present / present.shift(1)).reindex(_SESSIONS)
        columns[(asset, "open")] = close * 0.999
        columns[(asset, "close")] = close
        columns[(asset, "log_return")] = log_return
    return pd.DataFrame(columns)


def _run() -> tuple[object, _AlternatingAnalyst]:
    analyst = _AlternatingAnalyst()
    manager = _PassThroughManager(
        agent_id="pass_through_manager",
        managed_analysts=[analyst],
        weighting_strategy=EqualWeight(),
        portfolio_constructor=_ViewConstructor(),
    )
    frame = _frame()
    engine = BacktestEngine(
        commander=Commander(managers=[manager], weighting_strategy=EqualWeight()),
        portfolio=Portfolio(cash=1_000_000.0, positions={}),
        track_record=TrackRecord(),
        data_loader=lambda _assets, _start, _end: frame,
        start_date=_SESSIONS[0].date(),
        end_date=_SESSIONS[-1].date(),
        assets=["SPY", "LATE", "GAP"],
        execution_lag_days=1,
    )
    return engine.run(), analyst


def test_every_session_is_decided_with_a_finite_nav() -> None:
    """The run values and decides on every session, and every NAV is finite."""
    result, _ = _run()

    assert [day for day, _ in result.portfolio_value_history] == [s.date() for s in _SESSIONS]
    assert all(math.isfinite(value) for _, value in result.portfolio_value_history)
    assert list(result.decisions_by_day) == [s.date() for s in _SESSIONS]


def test_no_bar_means_no_signal_and_no_fill() -> None:
    """LATE before its first bar gets no decision; GAP on its missing session gets its decision
    of the session before, carried. Neither is asked about then, and the analyst never sees a
    missing close. GAP's pending order is left unfilled on the gap day; the carried decision
    fills at the next session."""
    result, analyst = _run()
    gap = _GAP_DAY.date()
    before = _SESSIONS[5].date()

    for day, decisions in result.decisions_by_day.items():
        assets = {decision.asset for decision in decisions}
        assert ("LATE" in assets) == (day >= _LATE_START.date())
        assert "GAP" in assets

    def weight(day: date, asset: str) -> float:
        return next(d.target_weight for d in result.decisions_by_day[day] if d.asset == asset)

    assert weight(gap, "GAP") == weight(before, "GAP") != 0.0
    asked = {(asset, day) for asset, day, _ in analyst._asked}
    assert ("GAP", gap) not in asked
    assert not any(asset == "LATE" and day < _LATE_START.date() for asset, day in asked)
    assert not any(saw_missing for _, _, saw_missing in analyst._asked)
    # A trade is dated by the decision it fills: the one of the session before the gap would
    # fill on the gap day and does not; the carried one fills on the session after.
    trades = {(trade.asset, trade.as_of_date) for trade in result.trade_log}
    assert ("GAP", before) not in trades
    assert ("GAP", gap) in trades
    assert not any(asset == "LATE" and day < _LATE_START.date() for asset, day in trades)
    assert any(
        order.asset == "GAP" and order.as_of_date == gap and order.reason == "no_fill_price"
        for order in result.unfilled_orders
    )


def _gap_frame(assets: tuple[str, ...], gaps: dict[str, list[pd.Timestamp]]) -> pd.DataFrame:
    """Every asset on every session but its `gaps`, each `log_return` over the asset's previous
    close."""
    rng = np.random.default_rng(5)
    columns: dict[tuple[str, str], pd.Series] = {}
    for asset in assets:
        close = pd.Series(
            100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, len(_SESSIONS)))), _SESSIONS
        )
        for day in gaps.get(asset, []):
            close[day] = np.nan
        present = close.dropna()
        columns[(asset, "open")] = close * 0.999
        columns[(asset, "close")] = close
        columns[(asset, "log_return")] = np.log(present / present.shift(1)).reindex(_SESSIONS)
    return pd.DataFrame(columns)


def _engine(frame: pd.DataFrame) -> BacktestEngine:
    manager = _PassThroughManager(
        agent_id="pass_through_manager",
        managed_analysts=[_AlternatingAnalyst()],
        weighting_strategy=EqualWeight(),
        portfolio_constructor=_ViewConstructor(),
    )
    return BacktestEngine(
        commander=Commander(managers=[manager], weighting_strategy=EqualWeight()),
        portfolio=Portfolio(cash=1_000_000.0, positions={}),
        track_record=TrackRecord(),
        data_loader=lambda _assets, _start, _end: frame,
        start_date=_SESSIONS[0].date(),
        end_date=_SESSIONS[-1].date(),
        assets=list(dict.fromkeys(frame.columns.get_level_values(0))),
        execution_lag_days=1,
    )


def test_held_future_is_not_re_marked_on_a_missing_session() -> None:
    """A held future keeps its position and its mark on its missing session, so it pays and
    books no P&L that day; at its next bar it is marked to that bar again."""
    frame = _gap_frame(("SPY", "GC=F"), {"GC=F": [_GAP_DAY]})
    engine = _engine(frame)
    marks: dict[date, tuple[float, float]] = {}

    def record(_index: int, _total: int, day: date, _value: float) -> None:
        position = engine.portfolio.marked_positions.get("GC=F")
        if position is not None:
            marks[day] = (position.quantity, position.mark)

    result = engine.run(on_day_complete=record)
    gap, before, after = _GAP_DAY.date(), _SESSIONS[5].date(), _SESSIONS[7].date()

    assert all(math.isfinite(value) for _, value in result.portfolio_value_history)
    assert marks[gap] == marks[before]
    assert marks[before][0] != 0.0
    bar = [float(frame.loc[_SESSIONS[7], ("GC=F", field)]) for field in ("open", "close")]
    assert any(marks[after][1] == pytest.approx(price) for price in bar)


def test_held_equity_over_a_two_session_gap_is_held() -> None:
    """Over two missing sessions the equity's decision is carried on both, its order goes
    unfilled on both and its shares are held; it trades again at its next bar."""
    gaps = [_SESSIONS[6], _SESSIONS[7]]
    frame = _gap_frame(("SPY", "GAP2"), {"GAP2": gaps})
    engine = _engine(frame)
    shares: dict[date, float] = {}

    def record(_index: int, _total: int, day: date, _value: float) -> None:
        shares[day] = engine.portfolio.positions.get("GAP2", 0.0)

    result = engine.run(on_day_complete=record)
    first, second, before = gaps[0].date(), gaps[1].date(), _SESSIONS[5].date()

    def weight(day: date) -> float:
        return next(d.target_weight for d in result.decisions_by_day[day] if d.asset == "GAP2")

    assert weight(first) == weight(second) == weight(before) != 0.0
    assert shares[first] == shares[second] == shares[before] != 0.0
    unfilled = {
        order.as_of_date
        for order in result.unfilled_orders
        if order.asset == "GAP2" and order.reason == "no_fill_price"
    }
    assert {first, second} <= unfilled
    trades = {trade.as_of_date for trade in result.trade_log if trade.asset == "GAP2"}
    assert before not in trades and first not in trades
    assert second in trades


def test_valuation_carries_the_last_close_and_skips_unpriced() -> None:
    """A held asset without a close is valued at its last close before the day; an asset with
    no close yet has no valuation price; a priced asset keeps its close."""
    frame = _frame()
    day = _GAP_DAY
    closes = {asset: float(frame.loc[day, (asset, "close")]) for asset in ("SPY", "LATE", "GAP")}
    early = {asset: float(frame.loc[_SESSIONS[1], (asset, "close")]) for asset in ("SPY", "LATE")}

    assert _valuation_prices(frame, day, closes, set()) == {
        "SPY": closes["SPY"],
        "LATE": closes["LATE"],
        "GAP": pytest.approx(float(frame.loc[_SESSIONS[5], ("GAP", "close")])),
    }
    assert _valuation_prices(frame, _SESSIONS[1], early, set()) == {"SPY": early["SPY"]}


def test_a_roll_inside_a_gap_is_charged_once_on_the_next_bar() -> None:
    """A front-contract roll dated on a day the session frame does not hold (the future traded on
    an exchange holiday, then missed a session) is charged once, on the future's next bar."""
    holiday = pd.Timestamp("2016-01-18")
    gap, next_bar = _SESSIONS[10], _SESSIONS[11]
    sessions = _gap_frame(("SPY", "GC=F"), {"GC=F": [gap]})
    lake = sessions.copy()
    lake.loc[holiday] = sessions.loc[_SESSIONS[9]]
    lake = lake.sort_index()
    lake[("GC=F", "roll")] = 0.0
    lake.loc[holiday, ("GC=F", "roll")] = 1.0
    lake.loc[gap, ("GC=F", "roll")] = np.nan
    frame = _with_rolls_since_previous_row(lake.loc[_SESSIONS], lake, ["GC=F"])
    result = _engine(frame).run()

    rolls = [t for t in result.trade_log if t.asset == "GC=F" and t.action == "roll"]
    assert [t.as_of_date for t in rolls] == [next_bar.date()]


def test_fx_carry_accrues_each_calendar_day_once_across_a_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A held pair accrues carry over every calendar day of the window exactly once, its
    missing session included."""
    frame = _gap_frame(("SPY", "EURUSD=X"), {"EURUSD=X": [_GAP_DAY]})
    rates = pd.DataFrame(
        {"EUR": [0.01] * len(_SESSIONS), "USD": [0.03] * len(_SESSIONS)}, index=_SESSIONS
    )
    engine = _engine(frame)
    engine.policy_rates = rates
    accrued: list[int] = []
    apply_carry = Portfolio.apply_carry

    def recording(self: Portfolio, carry_rates: dict[str, float], days_held: int) -> float:
        if self.marked_positions.get("EURUSD=X") is not None:
            accrued.append(days_held)
        return apply_carry(self, carry_rates, days_held)

    monkeypatch.setattr(Portfolio, "apply_carry", recording)
    engine.run()
    held_from = next(
        day for day in range(len(_SESSIONS)) if len(accrued) == len(_SESSIONS) - 1 - day
    )

    assert sum(accrued) == (_SESSIONS[-1] - _SESSIONS[held_from]).days
