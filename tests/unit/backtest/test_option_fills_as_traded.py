# test_option_fills_as_traded.py

"""`Portfolio.apply_option_decision` applies per-contract and per-share rules in as-traded units."""

from datetime import date

import pytest

from agentic_trading.agents.schemas import OptionContract, PortfolioDecision
from agentic_trading.backtest.portfolio import OptionPosition, Portfolio
from agentic_trading.config import settings
from agentic_trading.data.loaders.option_marks import OptionMark

_NAV = 1_000_000.0
_CONTRACT = OptionContract(
    underlying="AMZN", expiry=date(2022, 6, 17), strike=3.0, option_type="call"
)


def _mark(mid: float, bid: float, ask: float) -> OptionMark:
    return OptionMark(
        contract=_CONTRACT,
        as_of_date=date(2022, 3, 1),
        snapshot_date=date(2022, 3, 1),
        mid=mid,
        bid=bid,
        ask=ask,
        implied_volatility=0.25,
        delta=0.5,
        gamma=0.02,
        vega=0.1,
        theta=-0.05,
        is_model_mark=False,
        is_expired=False,
        tradeable=True,
    )


def _decision(target_weight: float) -> PortfolioDecision:
    return PortfolioDecision(
        asset=_CONTRACT.contract_id,
        as_of_date=date(2022, 3, 1),
        target_weight=target_weight,
        contributing_signals=[],
        instrument=_CONTRACT,
    )


def _portfolio() -> Portfolio:
    return Portfolio(cash=_NAV, positions={}, commission_rate=0.0, slippage_bps=0.0)


def test_min_premium_check_uses_as_traded_premium() -> None:
    """A rescaled mid of 0.04 (as-traded 0.80) passes the minimum premium only with f=20."""
    assert settings.options.min_premium_per_share == pytest.approx(0.05)
    mark = _mark(0.04, 0.038, 0.042)

    split = _portfolio()
    trade = split.apply_option_decision(_decision(0.01), mark, _NAV, split_factor=20.0)
    assert trade.action == "buy"
    assert _CONTRACT.contract_id in split.option_positions

    unsplit = _portfolio()
    trade = unsplit.apply_option_decision(_decision(0.01), mark, _NAV, split_factor=1.0)
    assert trade.action == "hold"
    assert unsplit.cash == _NAV
    assert unsplit.option_positions == {}


def test_split_fill_rounds_to_as_traded_contracts_and_charges_as_traded_commission() -> None:
    """A 1% target buys 125 as-traded contracts: 2,500 rescaled, 125 commissions."""
    mark = _mark(0.04, 0.038, 0.042)
    multiplier = settings.options.contract_multiplier
    commission = settings.options.commission_per_contract
    real = round(0.01 * _NAV / (0.04 * 20.0 * multiplier))
    assert real == 125

    portfolio = _portfolio()
    trade = portfolio.apply_option_decision(_decision(0.01), mark, _NAV, split_factor=20.0)

    position = portfolio.option_positions[_CONTRACT.contract_id]
    assert position.contracts == 2_500
    assert trade.delta_shares == pytest.approx(2_500.0)
    assert portfolio.cash == pytest.approx(_NAV - 2_500 * 0.042 * multiplier - 125 * commission)
    assert trade.transaction_cost == pytest.approx(125 * commission)

    portfolio.apply_option_decision(_decision(0.0), mark, _NAV, split_factor=20.0)
    assert portfolio.option_positions == {}
    expected = (
        _NAV
        - 2_500 * 0.042 * multiplier
        - 125 * commission
        + 2_500 * 0.038 * multiplier
        - 125 * commission
    )
    assert portfolio.cash == pytest.approx(expected)


def test_unsplit_underlying_behaves_as_before() -> None:
    """With the default factor and with split_factor=1.0 the fill is the plain contract fill."""
    mark = _mark(1.05, 1.00, 1.10)
    multiplier = settings.options.contract_multiplier
    commission = settings.options.commission_per_contract
    contracts = round(0.01 * _NAV / (1.05 * multiplier))
    expected_cash = _NAV - contracts * 1.10 * multiplier - contracts * commission

    default = _portfolio()
    default.apply_option_decision(_decision(0.01), mark, _NAV)
    explicit = _portfolio()
    explicit.apply_option_decision(_decision(0.01), mark, _NAV, split_factor=1.0)

    for portfolio in (default, explicit):
        assert portfolio.option_positions[_CONTRACT.contract_id].contracts == contracts
        assert portfolio.cash == pytest.approx(expected_cash)


def test_rebalance_band_compares_rescaled_contracts_consistently() -> None:
    """Re-issuing the same target on a held split position is a hold."""
    mark = _mark(0.04, 0.038, 0.042)
    portfolio = _portfolio()
    portfolio.option_positions[_CONTRACT.contract_id] = OptionPosition(
        contract=_CONTRACT, contracts=2_500, entry_price=0.042, last_mid=0.04
    )
    cash = portfolio.cash

    trade = portfolio.apply_option_decision(_decision(0.01), mark, _NAV, split_factor=20.0)

    assert trade.action == "hold"
    assert portfolio.cash == cash
    assert portfolio.option_positions[_CONTRACT.contract_id].contracts == 2_500
