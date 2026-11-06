# test_futures_book.py

"""A futures position is held without cash: it earns its notional times the
continuous series' daily return, settled into cash at every mark, pays no
stock borrow when short and no margin financing on its notional, and pays
one round-trip fill cost on the notional it holds through each roll."""

import itertools
from datetime import date

import numpy as np
import pandas as pd
import pytest

from agentic_trading.agents.schemas import MAX_LEVERAGE, BacktestResult, PortfolioDecision
from agentic_trading.backtest.engine import BacktestEngine
from agentic_trading.backtest.portfolio import MarkedPosition, Portfolio
from agentic_trading.config import settings
from agentic_trading.scoring.track_record import TrackRecord

_CAPITAL = 1_000_000.0
_DAY = date(2024, 1, 2)
_COMMISSION = 5e-4
_SLIPPAGE_BPS = 5.0
_ROUND_TRIP_RATE = 2.0 * (_COMMISSION + _SLIPPAGE_BPS / 10_000.0)


def _book(
    cash: float = _CAPITAL, commission_rate: float = 0.0, slippage_bps: float = 0.0
) -> Portfolio:
    return Portfolio(
        cash=cash, positions={}, commission_rate=commission_rate, slippage_bps=slippage_bps
    )


def _decision(asset: str, weight: float, day: date = _DAY) -> PortfolioDecision:
    return PortfolioDecision(
        asset=asset, as_of_date=day, target_weight=weight, contributing_signals=[]
    )


def test_a_future_fill_pays_no_cash_for_its_notional() -> None:
    """A long and a short fill leave cash untouched (no costs here) and hold the notional."""
    for weight in (0.5, -0.5):
        book = _book()
        trade = book.apply_decision(_decision("CL=F", weight), 50.0, _CAPITAL)

        assert book.cash == _CAPITAL
        assert book.positions == {}
        assert book.marked_positions["CL=F"] == MarkedPosition(
            quantity=weight * _CAPITAL / 50.0, mark=50.0
        )
        assert trade.action == ("buy" if weight > 0 else "sell")
        assert trade.notional == pytest.approx(abs(weight) * _CAPITAL)
        assert book.total_value({"CL=F": 50.0}) == _CAPITAL


def test_a_future_fill_pays_the_equity_fill_cost_rates_only() -> None:
    """Cash falls by the commission and slippage on the traded notional, nothing more."""
    book = _book(commission_rate=_COMMISSION, slippage_bps=_SLIPPAGE_BPS)
    trade = book.apply_decision(_decision("CL=F", 0.5), 50.0, _CAPITAL)

    slippage = 500_000.0 * _SLIPPAGE_BPS / 10_000.0
    commission = 500_000.0 * (1.0 + _SLIPPAGE_BPS / 10_000.0) * _COMMISSION
    assert trade.slippage_cost == pytest.approx(slippage)
    assert trade.transaction_cost == pytest.approx(commission)
    assert book.cash == pytest.approx(_CAPITAL - slippage - commission)
    assert book.marked_positions["CL=F"].mark == 50.0


def test_a_future_earns_its_notional_times_the_daily_return() -> None:
    """Each mark settles quantity x the level's change into cash and re-marks the position."""
    book = _book()
    book.apply_decision(_decision("CL=F", 0.5), 50.0, _CAPITAL)

    pnl = book.settle_marked_positions({"CL=F": 51.0})

    assert pnl == pytest.approx(500_000.0 * (51.0 / 50.0 - 1.0))
    assert book.cash == pytest.approx(_CAPITAL + pnl)
    assert book.marked_positions["CL=F"].mark == 51.0
    assert book.total_value({"CL=F": 51.0}) == pytest.approx(_CAPITAL + pnl)
    assert book.settle_marked_positions({"CL=F": 51.0}) == 0.0


def test_total_value_includes_a_move_not_yet_settled() -> None:
    book = _book()
    book.apply_decision(_decision("CL=F", -0.5), 50.0, _CAPITAL)

    assert book.total_value({"CL=F": 49.0}) == pytest.approx(_CAPITAL + 10_000.0 * 1.0)
    assert book.cash == _CAPITAL


def test_a_constant_price_future_has_zero_pnl_and_zero_financing() -> None:
    """At a constant level the future earns nothing, and its notional draws no financing:
    the book's financing is exactly an all-cash book's."""
    book = _book()
    book.apply_decision(_decision("CL=F", 1.0), 50.0, _CAPITAL)
    prices = {"CL=F": 50.0}

    assert book.settle_marked_positions(prices) == 0.0
    assert book.apply_financing(prices, 0.0, 3) == 0.0
    cash_only = _book().apply_financing({}, 0.05, 3)
    assert book.apply_financing(prices, 0.05, 3) == pytest.approx(cash_only)


def test_a_short_future_pays_no_borrow() -> None:
    """A short future is charged no stock borrow, where an equity short of the same
    notional is."""
    future = _book()
    future.apply_decision(_decision("CL=F", -0.5), 50.0, _CAPITAL)
    assert future.apply_financing({"CL=F": 50.0}, 0.0, 30) == 0.0
    assert future.cash == _CAPITAL

    equity = _book(cash=_CAPITAL + 500_000.0)
    equity.positions = {"XOM": -10_000.0}
    assert equity.apply_financing({"XOM": 50.0}, 0.0, 30) == pytest.approx(
        500_000.0 * settings.firm.short_borrow_rate_annual * 30 / 365
    )


@pytest.mark.parametrize("weight", [0.5, -0.5])
def test_a_roll_charges_one_round_trip_on_the_held_notional(weight: float) -> None:
    book = _book(commission_rate=_COMMISSION, slippage_bps=_SLIPPAGE_BPS)
    book.marked_positions = {"CL=F": MarkedPosition(quantity=weight * _CAPITAL / 50.0, mark=50.0)}

    records = book.charge_rolls(_DAY, {"CL=F": 50.0}, {"CL=F": 1.0}, _CAPITAL)

    held = abs(weight) * _CAPITAL
    assert len(records) == 1
    record = records[0]
    assert record.action == "roll"
    assert record.asset == "CL=F"
    assert record.delta_shares == 0.0
    assert record.notional == pytest.approx(2.0 * held)
    assert record.transaction_cost == pytest.approx(2.0 * held * _COMMISSION)
    assert record.slippage_cost == pytest.approx(2.0 * held * _SLIPPAGE_BPS / 10_000.0)
    assert book.cash == pytest.approx(_CAPITAL - held * _ROUND_TRIP_RATE)
    assert book.marked_positions["CL=F"].quantity == weight * _CAPITAL / 50.0


@pytest.mark.parametrize("quantity", [10_000.0, -10_000.0])
def test_a_roll_at_a_near_zero_nav_records_a_bounded_target_weight(quantity: float) -> None:
    """A held notional far above a near-zero NAV still yields a valid record: the target
    weight is clipped to ±MAX_LEVERAGE, the previous weight is the actual one, and the
    full round trip is charged."""
    book = _book(commission_rate=_COMMISSION, slippage_bps=_SLIPPAGE_BPS)
    book.marked_positions = {"CL=F": MarkedPosition(quantity=quantity, mark=50.0)}

    records = book.charge_rolls(_DAY, {"CL=F": 50.0}, {"CL=F": 1.0}, 1e-9)

    sign = 1.0 if quantity > 0 else -1.0
    assert records[0].target_weight == sign * MAX_LEVERAGE
    assert records[0].previous_weight == pytest.approx(quantity * 50.0 / 1e-9)
    assert book.cash == pytest.approx(_CAPITAL - 500_000.0 * _ROUND_TRIP_RATE)


def test_a_non_roll_day_charges_nothing() -> None:
    book = _book(commission_rate=_COMMISSION, slippage_bps=_SLIPPAGE_BPS)
    book.marked_positions = {"CL=F": MarkedPosition(quantity=10_000.0, mark=50.0)}

    assert book.charge_rolls(_DAY, {"CL=F": 50.0}, {"CL=F": 0.0}, _CAPITAL) == []
    assert book.charge_rolls(_DAY, {"CL=F": 50.0}, {}, _CAPITAL) == []
    assert book.charge_rolls(_DAY, {"CL=F": 50.0}, {"GC=F": 1.0}, _CAPITAL) == []
    assert book.cash == _CAPITAL


class _FixedCommander:
    """Targets the same weights every day."""

    def __init__(self, weights: dict[str, float]) -> None:
        self.weights = weights

    def decide(
        self,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
        scoreboard: object,
        instrument_returns: dict[str, float] | None = None,
    ) -> list[PortfolioDecision]:
        return [
            _decision(asset, weight, as_of_date.date()) for asset, weight in self.weights.items()
        ]


def _future_frame(close: list[float], rolls: list[float]) -> pd.DataFrame:
    index = pd.bdate_range("2024-01-02", periods=len(close))
    level = pd.Series(close, index=index)
    return pd.DataFrame(
        {
            ("CL=F", "open"): level,
            ("CL=F", "close"): level,
            ("CL=F", "log_return"): np.log(level / level.shift(1)),
            ("CL=F", "roll"): pd.Series(rolls, index=index),
        }
    )


def _run(frame: pd.DataFrame, portfolio: Portfolio, rate: float | None = None) -> BacktestResult:
    days = frame.index
    return BacktestEngine(
        commander=_FixedCommander({"CL=F": 0.5}),  # type: ignore[arg-type]
        portfolio=portfolio,
        track_record=TrackRecord(),
        data_loader=lambda _assets, _start, _end: frame,
        start_date=days[0].date(),
        end_date=days[-1].date(),
        assets=["CL=F"],
        risk_free_rate_series=(
            None
            if rate is None
            else pd.Series(rate / settings.firm.trading_days_per_year, index=days)
        ),
    ).run()


def test_engine_constant_price_future_keeps_nav_and_pays_no_financing() -> None:
    result = _run(_future_frame([50.0] * 6, [0.0] * 6), _book())

    assert [value for _, value in result.portfolio_value_history] == [_CAPITAL] * 6
    assert result.total_financing_cost == 0.0
    assert result.total_execution_cost == 0.0


def test_engine_future_nav_follows_notional_times_return() -> None:
    """With zero costs each day's NAV is the previous one plus half of it times the return."""
    close = [50.0, 51.0, 49.5, 49.5, 52.0]
    result = _run(_future_frame(close, [0.0] * 5), _book())

    expected = [_CAPITAL]
    for previous, current in itertools.pairwise(close):
        expected.append(expected[-1] * (1.0 + 0.5 * (current / previous - 1.0)))
    values = [value for _, value in result.portfolio_value_history]
    assert values == pytest.approx(expected, rel=1e-12)


def test_engine_charges_one_round_trip_on_each_roll_day_only() -> None:
    rolls = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    result = _run(
        _future_frame([50.0] * 6, rolls),
        _book(commission_rate=_COMMISSION, slippage_bps=_SLIPPAGE_BPS),
    )

    roll_records = [trade for trade in result.trade_log if trade.action == "roll"]
    roll_day = pd.bdate_range("2024-01-02", periods=6)[3].date()
    assert [record.as_of_date for record in roll_records] == [roll_day]
    held_before = [
        trade
        for trade in result.trade_log
        if trade.action in {"buy", "sell"} and trade.as_of_date < roll_day
    ][-1].target_shares
    roll_cost = roll_records[0].transaction_cost + roll_records[0].slippage_cost
    assert roll_cost == pytest.approx(abs(held_before) * 50.0 * _ROUND_TRIP_RATE)
