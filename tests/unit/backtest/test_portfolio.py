"""Unit tests for agentic_trading.backtest.portfolio.

`apply_decision(self, decision: PortfolioDecision, price: float, portfolio_value: float) ->
TradeRecord`: `target_weight` is a fraction of *total* portfolio value, and a single
asset's price is not enough to derive total portfolio value (that requires prices for
every currently-held position), so the caller (BacktestEngine) supplies `portfolio_value`
directly. The returned TradeRecord is the single source of truth for "what did the
portfolio actually do" that both the CLI and the GUI's trade blotter are built from.
"""

from datetime import date
from typing import Literal

import pytest

from agentic_trading.agents.schemas import OptionContract, PortfolioDecision
from agentic_trading.backtest.portfolio import OptionPosition, Portfolio
from agentic_trading.config import settings, temporary_overrides
from agentic_trading.data.loaders.option_marks import OptionMark


def _portfolio(cash: float, positions: dict[str, float] | None = None) -> Portfolio:
    """A portfolio with commission and slippage switched off, so the fill
    arithmetic below can be asserted exactly."""
    return Portfolio(cash=cash, positions=positions or {}, commission_rate=0.0, slippage_bps=0.0)


def _decision(asset: str = "AAPL", target_weight: float = 0.5) -> PortfolioDecision:
    return PortfolioDecision(
        asset=asset,
        as_of_date=date(2024, 1, 2),
        target_weight=target_weight,
        contributing_signals=[],
    )


def test_apply_decision_buys_new_position_and_debits_cash_exactly() -> None:
    """Target dollars are derived from portfolio_value, not from price alone."""
    portfolio = _portfolio(cash=10_000.0)

    portfolio.apply_decision(_decision(target_weight=0.5), price=100.0, portfolio_value=10_000.0)

    assert portfolio.positions["AAPL"] == pytest.approx(50.0)
    assert portfolio.cash == pytest.approx(10_000.0 - 50.0 * 100.0)


def test_apply_decision_zero_target_weight_fully_liquidates_existing_position() -> None:
    """A target_weight of 0.0 brings the position to exactly zero, crediting cash back."""
    portfolio = _portfolio(cash=1_000.0, positions={"AAPL": 40.0})

    portfolio.apply_decision(_decision(target_weight=0.0), price=100.0, portfolio_value=5_000.0)

    assert portfolio.positions["AAPL"] == pytest.approx(0.0)
    assert portfolio.cash == pytest.approx(1_000.0 + 40.0 * 100.0)


def test_apply_decision_negative_target_weight_produces_short_position() -> None:
    """A negative target_weight is a short: negative shares, cash credited on entry."""
    portfolio = _portfolio(cash=10_000.0)

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
    portfolio = _portfolio(cash=10_000.0)

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
    portfolio = _portfolio(cash=1_000.0, positions={"AAPL": 40.0})

    trade = portfolio.apply_decision(
        _decision(target_weight=0.0), price=100.0, portfolio_value=5_000.0
    )

    assert trade.action == "sell"
    assert trade.previous_weight == pytest.approx((40.0 * 100.0) / 5_000.0)
    assert trade.delta_shares == pytest.approx(-40.0)
    assert trade.notional == pytest.approx(4_000.0)


def test_apply_decision_returns_a_trade_record_describing_a_hold() -> None:
    """No change in target shares from the current position must report action="hold"."""
    portfolio = _portfolio(cash=1_000.0, positions={"AAPL": 50.0})

    trade = portfolio.apply_decision(
        _decision(target_weight=0.5), price=100.0, portfolio_value=10_000.0
    )

    assert trade.action == "hold"
    assert trade.delta_shares == pytest.approx(0.0)
    assert trade.notional == pytest.approx(0.0)


def test_constructor_defaults_read_commission_and_slippage_from_settings() -> None:
    """Omitting (or passing None for) commission_rate/slippage_bps resolves to
    `settings.firm` at construction time, so a settings override applied
    before construction is what the portfolio trades on."""
    with temporary_overrides({"firm.commission_rate": 0.002, "firm.slippage_bps": 12.0}):
        defaulted = Portfolio(cash=1.0, positions={})
        explicit_none = Portfolio(cash=1.0, positions={}, commission_rate=None, slippage_bps=None)

    assert defaulted.commission_rate == pytest.approx(0.002)
    assert defaulted.slippage_bps == pytest.approx(12.0)
    assert explicit_none.commission_rate == pytest.approx(0.002)
    assert explicit_none.slippage_bps == pytest.approx(12.0)
    assert Portfolio(cash=1.0, positions={}).commission_rate == pytest.approx(
        settings.firm.commission_rate
    )


def test_apply_financing_only_credits_interest_when_long_with_positive_cash() -> None:
    """A long-only book with positive cash pays nothing and earns the
    risk-free rate on its cash, so the net charge is the negative credit."""
    portfolio = _portfolio(cash=10_000.0, positions={"AAPL": 10.0})

    cost = portfolio.apply_financing({"AAPL": 100.0}, risk_free_rate_annual=0.05, days_held=3)

    credit = 10_000.0 * 0.05 * 3 / 365.0
    assert cost == pytest.approx(-credit)
    assert portfolio.cash == pytest.approx(10_000.0 + credit)


def test_apply_financing_charges_stock_borrow_on_short_notional() -> None:
    """notional * short_borrow_rate_annual * days / 365, on the short leg only."""
    portfolio = _portfolio(cash=13_000.0, positions={"AAPL": -60.0, "MSFT": 10.0})

    with temporary_overrides({"firm.short_borrow_rate_annual": 0.01}):
        cost = portfolio.apply_financing(
            {"AAPL": 50.0, "MSFT": 100.0}, risk_free_rate_annual=0.0, days_held=3
        )

    expected = 60.0 * 50.0 * 0.01 * 3 / 365.0
    assert cost == pytest.approx(expected)
    assert portfolio.cash == pytest.approx(13_000.0 - expected)


def test_apply_financing_charges_margin_interest_on_negative_cash() -> None:
    """Borrowed cash pays risk-free + margin_spread_annual; positive cash pays nothing."""
    portfolio = _portfolio(cash=-2_000.0, positions={"AAPL": 120.0})

    with temporary_overrides({"firm.margin_spread_annual": 0.02}):
        cost = portfolio.apply_financing({"AAPL": 100.0}, risk_free_rate_annual=0.04, days_held=1)

    expected = 2_000.0 * (0.04 + 0.02) * 1 / 365.0
    assert cost == pytest.approx(expected)
    assert portfolio.cash == pytest.approx(-2_000.0 - expected)


def test_apply_financing_zero_day_gap_charges_nothing() -> None:
    portfolio = _portfolio(cash=-2_000.0, positions={"AAPL": -10.0})

    assert portfolio.apply_financing({"AAPL": 100.0}, 0.05, days_held=0) == pytest.approx(0.0)
    assert portfolio.cash == pytest.approx(-2_000.0)


def test_fill_is_capped_at_adv_participation_and_pays_impact() -> None:
    """With volume known, a fill larger than the participation cap is cut
    to the cap (a partial fill toward the target) and pays square-root
    impact on top of slippage; a small fill pays impact but is not cut."""
    from agentic_trading.config import settings, temporary_overrides

    portfolio = Portfolio(cash=1_000_000.0, positions={}, commission_rate=0.0, slippage_bps=0.0)
    with temporary_overrides({"firm.max_adv_participation": 0.05}):
        trade = portfolio.apply_decision(
            _decision(target_weight=1.0),
            price=100.0,
            portfolio_value=1_000_000.0,
            adv_shares=10_000.0,
            daily_vol=0.02,
        )
    assert trade.delta_shares == pytest.approx(500.0)
    assert portfolio.positions["AAPL"] == pytest.approx(500.0)
    assert trade.impact_bps > 0.0
    assert trade.price > 100.0

    small = Portfolio(cash=1_000_000.0, positions={}, commission_rate=0.0, slippage_bps=0.0)
    trade = small.apply_decision(
        _decision(target_weight=0.001),
        price=100.0,
        portfolio_value=1_000_000.0,
        adv_shares=1_000_000.0,
        daily_vol=0.02,
    )
    assert trade.delta_shares == pytest.approx(10.0)
    assert 0.0 < trade.impact_bps < settings.firm.market_impact_coefficient * 0.02 * 1e4


def test_fill_without_volume_data_pays_no_impact_and_is_not_capped() -> None:
    portfolio = Portfolio(cash=1_000_000.0, positions={}, commission_rate=0.0, slippage_bps=0.0)
    trade = portfolio.apply_decision(
        _decision(target_weight=0.5), price=100.0, portfolio_value=1_000_000.0
    )
    assert trade.delta_shares == pytest.approx(5_000.0)
    assert trade.impact_bps == 0.0


# ---------------------------------------------------------------------------
# Listed-option ledger
# ---------------------------------------------------------------------------


def _contract(
    option_type: Literal["call", "put"] = "call",
    strike: float = 180.0,
    expiry: date = date(2024, 4, 5),
) -> OptionContract:
    return OptionContract(underlying="AAPL", expiry=expiry, strike=strike, option_type=option_type)


def _mark(
    contract: OptionContract,
    bid: float = 3.9,
    ask: float = 4.1,
    *,
    as_of_date: date = date(2024, 1, 2),
    tradeable: bool = True,
    is_expired: bool = False,
) -> OptionMark:
    return OptionMark(
        contract=contract,
        as_of_date=as_of_date,
        snapshot_date=as_of_date,
        mid=(bid + ask) / 2.0,
        bid=bid,
        ask=ask,
        implied_volatility=0.25,
        delta=0.5,
        gamma=0.02,
        vega=0.1,
        theta=-0.05,
        is_model_mark=False,
        is_expired=is_expired,
        tradeable=tradeable,
    )


def _option_decision(contract: OptionContract, target_weight: float) -> PortfolioDecision:
    return PortfolioDecision(
        asset=contract.contract_id,
        as_of_date=date(2024, 1, 2),
        target_weight=target_weight,
        contributing_signals=[],
        instrument=contract,
    )


def test_option_buy_fills_at_the_ask_with_per_contract_commission() -> None:
    """2% of a 100k NAV is 2,000 of premium notional; at a 4.00 mid and a
    100 multiplier that is 5 contracts, paid at the 4.10 ask plus the
    per-contract commission."""
    contract = _contract()
    portfolio = _portfolio(cash=100_000.0)

    with temporary_overrides(
        {"options.commission_per_contract": 0.65, "options.contract_multiplier": 100}
    ):
        trade = portfolio.apply_option_decision(
            _option_decision(contract, 0.02), _mark(contract), portfolio_value=100_000.0
        )

    assert trade.action == "buy"
    assert trade.instrument == contract
    assert trade.asset == contract.contract_id
    assert trade.contracts == pytest.approx(5.0)
    assert trade.price == pytest.approx(4.1)
    assert trade.premium == pytest.approx(-2_050.0)
    assert trade.transaction_cost == pytest.approx(3.25)
    assert trade.notional == pytest.approx(2_050.0)
    assert trade.previous_weight == pytest.approx(0.0)
    assert portfolio.cash == pytest.approx(100_000.0 - 2_050.0 - 3.25)
    position = portfolio.option_positions[contract.contract_id]
    assert position.contracts == 5
    assert position.entry_price == pytest.approx(4.1)
    assert position.last_mid == pytest.approx(4.0)


def test_option_sell_fills_at_the_bid_and_opens_a_short() -> None:
    contract = _contract()
    portfolio = _portfolio(cash=100_000.0)

    with temporary_overrides(
        {"options.commission_per_contract": 0.65, "options.max_short_premium_fraction": 0.05}
    ):
        trade = portfolio.apply_option_decision(
            _option_decision(contract, -0.02), _mark(contract), portfolio_value=100_000.0
        )

    assert trade.action == "sell"
    assert trade.contracts == pytest.approx(-5.0)
    assert trade.price == pytest.approx(3.9)
    assert trade.premium == pytest.approx(1_950.0)
    assert portfolio.cash == pytest.approx(100_000.0 + 1_950.0 - 3.25)
    assert portfolio.option_positions[contract.contract_id].contracts == -5


def test_option_short_is_clipped_to_the_short_premium_cap() -> None:
    """A 5% cap on a 100k NAV allows 5,000 of short premium: 12 contracts
    at a 4.00 mid, not the 25 the decision asked for; an existing short in
    another contract counts against the same cap."""
    contract = _contract()
    other = _contract(option_type="put")
    portfolio = _portfolio(cash=100_000.0)

    with temporary_overrides({"options.max_short_premium_fraction": 0.05}):
        trade = portfolio.apply_option_decision(
            _option_decision(contract, -0.10), _mark(contract), portfolio_value=100_000.0
        )
        assert trade.contracts == pytest.approx(-12.0)
        assert portfolio.option_positions[contract.contract_id].contracts == -12

        other_trade = portfolio.apply_option_decision(
            _option_decision(other, -0.10), _mark(other, 1.9, 2.1), portfolio_value=100_000.0
        )

    # 5,000 - 12 x 4.00 x 100 = 200 of headroom: one 2.00 contract.
    assert other_trade.contracts == pytest.approx(-1.0)


def test_option_short_cap_never_forces_a_buyback_of_an_existing_short() -> None:
    contract = _contract()
    portfolio = _portfolio(cash=100_000.0)
    with temporary_overrides({"options.max_short_premium_fraction": 0.05}):
        portfolio.apply_option_decision(
            _option_decision(contract, -0.04), _mark(contract), portfolio_value=100_000.0
        )
        assert portfolio.option_positions[contract.contract_id].contracts == -10

        # NAV halves: the existing short is now over the cap, and a request to
        # grow it is held at the current size rather than partially covered.
        trade = portfolio.apply_option_decision(
            _option_decision(contract, -0.10), _mark(contract), portfolio_value=50_000.0
        )

    assert trade.action == "hold"
    assert portfolio.option_positions[contract.contract_id].contracts == -10


def test_untradeable_mark_produces_a_hold_and_leaves_the_book_unchanged() -> None:
    """A sell (here: opening a short) needs a quoted bid; a zero-bid mark
    is a hold that leaves the book untouched."""
    contract = _contract()
    portfolio = _portfolio(cash=100_000.0)

    trade = portfolio.apply_option_decision(
        _option_decision(contract, -0.02),
        _mark(contract, 0.0, 0.05, tradeable=False),
        portfolio_value=100_000.0,
    )

    assert trade.action == "hold"
    assert trade.contracts == 0.0
    assert trade.premium == 0.0
    assert portfolio.cash == pytest.approx(100_000.0)
    assert contract.contract_id not in portfolio.option_positions


def test_option_decision_without_an_instrument_is_rejected() -> None:
    portfolio = _portfolio(cash=100_000.0)
    with pytest.raises(ValueError):
        portfolio.apply_option_decision(_decision(), _mark(_contract()), 100_000.0)


def test_total_value_marks_options_at_mid_and_falls_back_to_last_mid() -> None:
    contract = _contract()
    portfolio = _portfolio(cash=10_000.0, positions={"AAPL": 10.0})
    portfolio.option_positions[contract.contract_id] = OptionPosition(
        contract=contract, contracts=3, entry_price=4.1, last_mid=4.0
    )

    with temporary_overrides({"options.contract_multiplier": 100}):
        marked = portfolio.total_value(
            {"AAPL": 180.0}, {contract.contract_id: _mark(contract, 4.9, 5.1)}
        )
        unmarked = portfolio.total_value({"AAPL": 180.0})

    assert marked == pytest.approx(10_000.0 + 1_800.0 + 3 * 5.0 * 100)
    assert portfolio.option_positions[contract.contract_id].last_mid == pytest.approx(5.0)
    assert unmarked == pytest.approx(marked)


def test_settle_expired_closes_positions_at_intrinsic_without_commission() -> None:
    expiry = date(2024, 1, 5)
    long_call = _contract(expiry=expiry)
    short_put = _contract(option_type="put", expiry=expiry)
    live = _contract(expiry=date(2024, 4, 5))
    portfolio = _portfolio(cash=10_000.0)
    portfolio.option_positions = {
        long_call.contract_id: OptionPosition(contract=long_call, contracts=2, entry_price=3.0),
        short_put.contract_id: OptionPosition(contract=short_put, contracts=-1, entry_price=2.0),
        live.contract_id: OptionPosition(contract=live, contracts=1, entry_price=4.0),
    }
    marks = {
        long_call.contract_id: _mark(long_call, 10.0, 10.0, as_of_date=expiry, is_expired=True),
        short_put.contract_id: _mark(short_put, 0.0, 0.0, as_of_date=expiry, is_expired=True),
    }

    with temporary_overrides({"options.commission_per_contract": 0.65}):
        records = portfolio.settle_expired(marks, expiry)

    assert portfolio.cash == pytest.approx(10_000.0 + 2 * 10.0 * 100)
    assert set(portfolio.option_positions) == {live.contract_id}
    assert {record.asset for record in records} == {long_call.contract_id, short_put.contract_id}
    call_record = next(record for record in records if record.asset == long_call.contract_id)
    assert call_record.action == "expire"
    assert call_record.contracts == pytest.approx(-2.0)
    assert call_record.price == pytest.approx(10.0)
    assert call_record.premium == pytest.approx(2_000.0)
    assert call_record.transaction_cost == 0.0
    put_record = next(record for record in records if record.asset == short_put.contract_id)
    assert put_record.premium == pytest.approx(0.0)
    assert put_record.contracts == pytest.approx(1.0)


def test_settle_expired_without_a_mark_uses_the_last_known_mid() -> None:
    contract = _contract(expiry=date(2024, 1, 5))
    portfolio = _portfolio(cash=0.0)
    portfolio.option_positions[contract.contract_id] = OptionPosition(
        contract=contract, contracts=-1, entry_price=3.0, last_mid=2.5
    )

    records = portfolio.settle_expired({}, date(2024, 1, 8))

    assert len(records) == 1
    assert portfolio.cash == pytest.approx(-250.0)
    assert portfolio.option_positions == {}


def test_apply_financing_charges_margin_interest_on_short_option_margin() -> None:
    """A short option's margin is premium plus `short_margin_fraction` of
    the underlying notional, financed at risk-free + spread; a long option
    costs nothing."""
    short = _contract(option_type="put")
    long = _contract()
    portfolio = _portfolio(cash=10_000.0)
    portfolio.option_positions = {
        short.contract_id: OptionPosition(
            contract=short, contracts=-5, entry_price=4.0, last_mid=4.0
        ),
        long.contract_id: OptionPosition(contract=long, contracts=2, entry_price=3.0, last_mid=3.0),
    }

    with temporary_overrides(
        {"options.short_margin_fraction": 0.2, "firm.margin_spread_annual": 0.015}
    ):
        cost = portfolio.apply_financing(
            {}, risk_free_rate_annual=0.04, days_held=3, spots={"AAPL": 180.0}
        )

    margin = 5 * (4.0 + 0.2 * 180.0) * 100
    cash_credit = 10_000.0 * 0.04 * 3 / 365.0
    expected = margin * (0.04 + 0.015) * 3 / 365.0 - cash_credit
    assert cost == pytest.approx(expected)
    assert portfolio.cash == pytest.approx(10_000.0 - expected)


def test_apply_financing_takes_the_underlying_spot_from_prices_when_spots_is_omitted() -> None:
    short = _contract(option_type="put")
    portfolio = _portfolio(cash=10_000.0)
    portfolio.option_positions = {
        short.contract_id: OptionPosition(
            contract=short, contracts=-1, entry_price=4.0, last_mid=4.0
        ),
    }

    with temporary_overrides(
        {"options.short_margin_fraction": 0.2, "firm.margin_spread_annual": 0.0}
    ):
        cost = portfolio.apply_financing({"AAPL": 100.0}, risk_free_rate_annual=0.05, days_held=1)

    cash_credit = 10_000.0 * 0.05 / 365.0
    assert cost == pytest.approx((4.0 + 20.0) * 100 * 0.05 / 365.0 - cash_credit)


def test_option_fill_respects_rebalance_band_and_side_aware_quotes() -> None:
    """A small change to a held option position is a hold; a buy needs
    only an ask (a zero bid on a deep-OTM quote does not block buying);
    closing to zero always fills."""
    from datetime import date

    from agentic_trading.agents.schemas import OptionContract
    from agentic_trading.config import settings, temporary_overrides
    from agentic_trading.data.loaders.option_marks import OptionMark

    contract = OptionContract(
        underlying="SPY", expiry=date(2024, 3, 15), strike=450.0, option_type="put"
    )

    def _mark(bid: float, ask: float) -> OptionMark:
        return OptionMark(
            contract=contract,
            as_of_date=date(2024, 2, 1),
            snapshot_date=date(2024, 2, 1),
            mid=(bid + ask) / 2 if bid > 0 else ask / 2,
            bid=bid,
            ask=ask,
            implied_volatility=0.2,
            delta=-0.1,
            gamma=0.0,
            vega=0.0,
            theta=0.0,
            is_model_mark=False,
            is_expired=False,
            tradeable=bid > 0,
        )

    def _decision(weight: float) -> PortfolioDecision:
        return PortfolioDecision(
            asset=contract.contract_id,
            as_of_date=date(2024, 2, 1),
            target_weight=weight,
            contributing_signals=[],
            instrument=contract,
        )

    portfolio = Portfolio(cash=1_000_000.0, positions={}, commission_rate=0.0, slippage_bps=0.0)
    blocked = portfolio.apply_option_decision(_decision(0.01), _mark(0.0, 2.0), 1_000_000.0)
    assert blocked.action == "hold"  # a one-sided quote is no market
    first = portfolio.apply_option_decision(_decision(0.01), _mark(2.0, 2.0), 1_000_000.0)
    assert first.action == "buy" and first.contracts == 50
    with temporary_overrides({"options.rebalance_band": 0.25}):
        nudged = portfolio.apply_option_decision(_decision(0.011), _mark(2.0, 2.0), 1_000_000.0)
    assert nudged.action == "hold"
    closed = portfolio.apply_option_decision(_decision(0.0), _mark(2.0, 2.0), 1_000_000.0)
    assert closed.action == "sell" and closed.contracts == -50
    assert settings.options.rebalance_band > 0.0

    capped = Portfolio(cash=1_000_000.0, positions={}, commission_rate=0.0, slippage_bps=0.0)
    with temporary_overrides({"options.max_notional_fraction": 1.0}):
        trade = capped.apply_option_decision(
            _decision(0.05), _mark(0.10, 0.10), 1_000_000.0, spot=450.0
        )
    # 5% of NAV in a 10-cent put would be 5,000 contracts; the notional cap
    # (100% of NAV / (450 x 100)) allows 22.
    assert trade.contracts == 22
