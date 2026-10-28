# test_option_fill_tradeable.py

"""`Portfolio.apply_option_decision` only trades against a mark flagged `tradeable`."""

from datetime import date

import pytest

from agentic_trading.agents.schemas import OptionContract, PortfolioDecision
from agentic_trading.backtest.portfolio import Portfolio
from agentic_trading.data.loaders.option_marks import OptionMark

_CONTRACT = OptionContract(
    underlying="AAPL", expiry=date(2024, 4, 5), strike=180.0, option_type="call"
)


def _mark(*, tradeable: bool) -> OptionMark:
    """A 20-day-old model mark with a positive bid/ask."""
    return OptionMark(
        contract=_CONTRACT,
        as_of_date=date(2024, 1, 22),
        snapshot_date=date(2024, 1, 2),
        mid=1.05,
        bid=1.00,
        ask=1.10,
        implied_volatility=0.25,
        delta=0.5,
        gamma=0.02,
        vega=0.1,
        theta=-0.05,
        is_model_mark=True,
        is_expired=False,
        tradeable=tradeable,
    )


def _decision(target_weight: float) -> PortfolioDecision:
    return PortfolioDecision(
        asset=_CONTRACT.contract_id,
        as_of_date=date(2024, 1, 22),
        target_weight=target_weight,
        contributing_signals=[],
        instrument=_CONTRACT,
    )


@pytest.mark.parametrize("target_weight", [0.02, -0.02])
def test_untradeable_mark_holds_while_tradeable_mark_fills(target_weight: float) -> None:
    """An untradeable mark with a live-looking bid/ask holds untouched; the same
    mark flagged tradeable fills."""
    held = Portfolio(cash=100_000.0, positions={}, commission_rate=0.0, slippage_bps=0.0)
    trade = held.apply_option_decision(
        _decision(target_weight), _mark(tradeable=False), portfolio_value=100_000.0
    )

    assert trade.action == "hold"
    assert trade.delta_shares == 0.0
    assert trade.notional == 0.0
    assert held.cash == 100_000.0
    assert held.option_positions == {}

    filled = Portfolio(cash=100_000.0, positions={}, commission_rate=0.0, slippage_bps=0.0)
    trade = filled.apply_option_decision(
        _decision(target_weight), _mark(tradeable=True), portfolio_value=100_000.0
    )

    assert trade.action == ("buy" if target_weight > 0 else "sell")
    assert filled.cash != 100_000.0
    assert _CONTRACT.contract_id in filled.option_positions
