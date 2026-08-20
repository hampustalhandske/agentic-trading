"""Unit tests for agentic_trading.backtest.portfolio.

`apply_decision(self, decision: PortfolioDecision, price: float, portfolio_value: float) ->
TradeRecord`: `target_weight` is a fraction of *total* portfolio value, and a single
asset's price is not enough to derive total portfolio value (that requires prices for
every currently-held position), so the caller (BacktestEngine) supplies `portfolio_value`
directly. The returned TradeRecord is the single source of truth for "what did the
portfolio actually do" that both the CLI and the GUI's trade blotter are built from.
"""

from datetime import date

import pytest

from agentic_trading.agents.schemas import PortfolioDecision
from agentic_trading.backtest.portfolio import Portfolio


def _decision(asset: str = "AAPL", target_weight: float = 0.5) -> PortfolioDecision:
    return PortfolioDecision(
        asset=asset,
        as_of_date=date(2024, 1, 2),
        target_weight=target_weight,
        contributing_signals=[],
    )


def test_apply_decision_buys_new_position_and_debits_cash_exactly() -> None:
    """Target dollars are derived from portfolio_value, not from price alone."""
    portfolio = Portfolio(cash=10_000.0, positions={})

    portfolio.apply_decision(_decision(target_weight=0.5), price=100.0, portfolio_value=10_000.0)

    assert portfolio.positions["AAPL"] == pytest.approx(50.0)
    assert portfolio.cash == pytest.approx(10_000.0 - 50.0 * 100.0)


def test_apply_decision_zero_target_weight_fully_liquidates_existing_position() -> None:
    """A target_weight of 0.0 brings the position to exactly zero, crediting cash back."""
    portfolio = Portfolio(cash=1_000.0, positions={"AAPL": 40.0})

    portfolio.apply_decision(_decision(target_weight=0.0), price=100.0, portfolio_value=5_000.0)

    assert portfolio.positions["AAPL"] == pytest.approx(0.0)
    assert portfolio.cash == pytest.approx(1_000.0 + 40.0 * 100.0)


def test_apply_decision_negative_target_weight_produces_short_position() -> None:
    """A negative target_weight is a short: negative shares, cash credited on entry."""
    portfolio = Portfolio(cash=10_000.0, positions={})

    portfolio.apply_decision(_decision(target_weight=-0.3), price=50.0, portfolio_value=10_000.0)

    assert portfolio.positions["AAPL"] == pytest.approx(-60.0)
    assert portfolio.cash == pytest.approx(10_000.0 + 60.0 * 50.0)


def test_total_value_sums_cash_with_long_short_and_zero_positions() -> None:
    """Longs add, shorts subtract, and a zero-quantity position contributes nothing."""
    portfolio = Portfolio(cash=1_000.0, positions={"AAPL": 10.0, "TSLA": -5.0, "MSFT": 0.0})
    prices = {"AAPL": 100.0, "TSLA": 200.0, "MSFT": 50.0}

    result = portfolio.total_value(prices)

    assert result == pytest.approx(1_000.0 + 10.0 * 100.0 - 5.0 * 200.0 + 0.0 * 50.0)


def test_total_value_missing_price_for_held_asset_raises() -> None:
    """A `prices` dict missing an entry for a held asset must raise, not silently skip it."""
    portfolio = Portfolio(cash=1_000.0, positions={"AAPL": 10.0})

    with pytest.raises(KeyError):
        portfolio.total_value({})


def test_apply_decision_returns_a_trade_record_describing_a_buy() -> None:
    portfolio = Portfolio(cash=10_000.0, positions={})

    trade = portfolio.apply_decision(
        _decision(target_weight=0.5), price=100.0, portfolio_value=10_000.0
    )

    assert trade.action == "buy"
    assert trade.asset == "AAPL"
    assert trade.as_of_date == date(2024, 1, 2)
    assert trade.target_weight == pytest.approx(0.5)
    assert trade.previous_weight == pytest.approx(0.0)
    assert trade.target_shares == pytest.approx(50.0)
    assert trade.delta_shares == pytest.approx(50.0)
    assert trade.price == pytest.approx(100.0)
    assert trade.notional == pytest.approx(5_000.0)


def test_apply_decision_returns_a_trade_record_describing_a_sell() -> None:
    portfolio = Portfolio(cash=1_000.0, positions={"AAPL": 40.0})

    trade = portfolio.apply_decision(
        _decision(target_weight=0.0), price=100.0, portfolio_value=5_000.0
    )

    assert trade.action == "sell"
    assert trade.previous_weight == pytest.approx((40.0 * 100.0) / 5_000.0)
    assert trade.delta_shares == pytest.approx(-40.0)
    assert trade.notional == pytest.approx(4_000.0)


def test_apply_decision_returns_a_trade_record_describing_a_hold() -> None:
    """No change in target shares from the current position must report action="hold"."""
    portfolio = Portfolio(cash=1_000.0, positions={"AAPL": 50.0})

    trade = portfolio.apply_decision(
        _decision(target_weight=0.5), price=100.0, portfolio_value=10_000.0
    )

    assert trade.action == "hold"
    assert trade.delta_shares == pytest.approx(0.0)
    assert trade.notional == pytest.approx(0.0)
