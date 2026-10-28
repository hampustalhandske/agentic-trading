# portfolio.py

import math
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator
from quantcore.execution.financing import margin_financing_cost, short_borrow_cost
from quantcore.execution.market_impact import square_root_market_impact

from agentic_trading.agents.schemas import OptionContract, PortfolioDecision, TradeRecord
from agentic_trading.config import settings
from agentic_trading.data.loaders.option_marks import OptionMark
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)


class OptionPosition(BaseModel):
    """An open listed-option position: signed `contracts` (long positive,
    short negative), the per-share premium it was opened at, and the last
    mid it was marked at (what it is valued at on a day without a mark).

    Contracts, premiums and strike are on the split-adjusted basis the
    option marks use: one as-traded contract held across later splits of
    the underlying counts as `split_factor` contracts here (see
    `Portfolio.apply_option_decision`), so the count need not be whole."""

    contract: OptionContract
    contracts: float
    entry_price: float = Field(ge=0.0)
    last_mid: float = Field(ge=0.0, default=0.0)


class Portfolio(BaseModel):
    """Deterministic portfolio state: cash, per-asset equity positions and
    per-contract option positions. No LLM calls, no randomness — plain
    accounting.

    `commission_rate`/`slippage_bps` default to `settings.firm`'s values at
    construction time; passing `None` explicitly resolves to the same
    settings values."""

    cash: float
    positions: dict[str, float]
    option_positions: dict[str, OptionPosition] = Field(default_factory=dict)
    commission_rate: float = Field(default_factory=lambda: settings.firm.commission_rate)
    slippage_bps: float = Field(default_factory=lambda: settings.firm.slippage_bps)

    @field_validator("commission_rate", "slippage_bps", mode="before")
    @classmethod
    def _none_means_settings_default(cls, value: object, info: ValidationInfo) -> object:
        if value is not None or info.field_name is None:
            return value
        return getattr(settings.firm, info.field_name)

    def apply_decision(
        self,
        decision: PortfolioDecision,
        price: float,
        portfolio_value: float,
        adv_shares: float | None = None,
        daily_vol: float | None = None,
    ) -> TradeRecord:
        """Fills toward `decision.target_weight` for `decision.asset` at
        `price`, updating cash and positions in place, and returns the
        TradeRecord every blotter is built from. Slippage and commission
        apply to every fill. When `adv_shares` (trailing average daily
        volume) and `daily_vol` are known, the fill is capped at
        `settings.firm.max_adv_participation` of ADV (a partial fill; the
        remainder is re-targeted by the next decision) and pays square-root
        market impact (`quantcore.execution.market_impact`) on top of
        slippage. Without volume data neither applies."""
        current_shares = self.positions.get(decision.asset, 0.0)
        previous_weight = (
            0.0 if portfolio_value == 0 else (current_shares * price) / portfolio_value
        )
        target_shares = (decision.target_weight * portfolio_value) / price
        delta_shares = target_shares - current_shares
        impact_bps = 0.0
        if adv_shares is not None and adv_shares > 0.0 and daily_vol is not None:
            max_shares = settings.firm.max_adv_participation * adv_shares
            if abs(delta_shares) > max_shares:
                delta_shares = math.copysign(max_shares, delta_shares)
            if delta_shares != 0.0:
                impact_bps = square_root_market_impact(
                    abs(delta_shares),
                    adv_shares,
                    max(daily_vol, 0.0),
                    settings.firm.market_impact_coefficient,
                )
        slippage = (self.slippage_bps + impact_bps) / 10_000.0
        execution_price = price * (1.0 + slippage if delta_shares > 0 else 1.0 - slippage)
        notional = abs(delta_shares * execution_price)
        transaction_cost = notional * self.commission_rate
        self.cash -= delta_shares * execution_price
        self.cash -= transaction_cost
        self.positions[decision.asset] = current_shares + delta_shares

        action: Literal["buy", "sell", "hold"] = "hold"
        if delta_shares > 0:
            action = "buy"
        elif delta_shares < 0:
            action = "sell"

        return TradeRecord(
            as_of_date=decision.as_of_date,
            asset=decision.asset,
            action=action,
            target_weight=decision.target_weight,
            previous_weight=previous_weight,
            target_shares=target_shares,
            delta_shares=delta_shares,
            price=execution_price,
            notional=notional,
            transaction_cost=transaction_cost,
            impact_bps=impact_bps,
        )

    def apply_option_decision(
        self,
        decision: PortfolioDecision,
        mark: OptionMark,
        portfolio_value: float,
        spot: float | None = None,
        split_factor: float = 1.0,
    ) -> TradeRecord:
        """Fills toward `decision.target_weight` (signed premium notional
        over NAV) in whole as-traded contracts of `decision.instrument`,
        buying at `mark.ask` and selling at `mark.bid`, paying
        `settings.options.commission_per_contract` per as-traded contract
        traded. A mark whose `tradeable` flag is false is a hold, leaving
        cash and positions unchanged, whatever bid/ask it carries. Any
        fill needs a two-sided quote with at least
        `settings.options.min_premium_per_share` of as-traded premium (a
        close to zero needs only a bid), and a contract count is capped so
        its underlying notional stays within
        `settings.options.max_notional_fraction` of NAV when `spot` is
        known. A change smaller than `settings.options.rebalance_band` of
        the held contracts is a hold (closing to zero always fills). A
        fill that would open or grow a short past
        `settings.options.max_short_premium_fraction` of NAV (short
        premium outstanding over every short position) is clipped to the
        largest whole number of as-traded contracts within the cap.

        `mark`, `spot` and the held positions are on the split-adjusted
        basis of the underlying's price history; `split_factor` (positive)
        converts that basis to the contract as traded on the fill day: the
        product of `1 / ratio` over the underlying's splits after that day
        (1.0 with none). An as-traded premium is `mark.mid * split_factor`,
        one as-traded contract is held as `split_factor` contracts, and an
        as-traded contract count is the held count over `split_factor`.
        Every per-share and per-contract rule (the minimum premium, whole
        contract rounding, the notional and short-premium caps, the
        commission) applies in as-traded units; positions, `TradeRecord`
        contract counts and prices stay on the split-adjusted basis."""
        if decision.instrument is None:
            raise ValueError("apply_option_decision requires a decision with an instrument")
        if split_factor <= 0.0:
            raise ValueError(f"split_factor must be positive, got {split_factor}")
        contract = decision.instrument
        contract_id = contract.contract_id
        multiplier = float(settings.options.contract_multiplier)
        position = self.option_positions.get(contract_id)
        current_contracts = position.contracts if position is not None else 0.0
        previous_weight = (
            0.0
            if portfolio_value == 0
            else (current_contracts * mark.mid * multiplier) / portfolio_value
        )

        def _hold() -> TradeRecord:
            return TradeRecord(
                as_of_date=decision.as_of_date,
                asset=contract_id,
                action="hold",
                target_weight=decision.target_weight,
                previous_weight=previous_weight,
                target_shares=float(current_contracts),
                delta_shares=0.0,
                price=mark.mid,
                notional=0.0,
                instrument=contract,
            )

        if not mark.tradeable or mark.is_expired or mark.mid <= 0.0:
            return _hold()

        as_traded_mid = mark.mid * split_factor
        target_as_traded = round(
            decision.target_weight * portfolio_value / (as_traded_mid * multiplier)
        )
        if spot is not None and spot > 0.0:
            max_as_traded = int(
                settings.options.max_notional_fraction
                * portfolio_value
                / (spot * split_factor * multiplier)
            )
            target_as_traded = max(-max_as_traded, min(max_as_traded, target_as_traded))
        target_contracts = target_as_traded * split_factor
        if (
            current_contracts != 0
            and target_contracts != 0
            and abs(target_contracts - current_contracts)
            < settings.options.rebalance_band * abs(current_contracts)
        ):
            return _hold()
        if target_contracts != 0 and (
            mark.bid <= 0.0
            or mark.ask <= 0.0
            or as_traded_mid < settings.options.min_premium_per_share
        ):
            return _hold()
        if target_contracts == 0 and mark.bid <= 0.0:
            return _hold()
        if target_contracts < min(current_contracts, 0):
            other_short_premium = sum(
                abs(other.contracts) * other.last_mid * multiplier
                for other_id, other in self.option_positions.items()
                if other.contracts < 0 and other_id != contract_id
            )
            cap = settings.options.max_short_premium_fraction * portfolio_value
            allowed = (
                max(math.floor((cap - other_short_premium) / (as_traded_mid * multiplier)), 0)
                * split_factor
            )
            clipped = max(target_contracts, min(current_contracts, -allowed))
            if clipped != target_contracts:
                logger.info(
                    "Short premium cap: %s target %g contracts clipped to %g "
                    "(cap %.2f, other short premium %.2f)",
                    contract_id,
                    target_contracts,
                    clipped,
                    cap,
                    other_short_premium,
                )
                target_contracts = clipped

        delta_contracts = target_contracts - current_contracts
        if delta_contracts == 0:
            return _hold()

        fill_price = mark.ask if delta_contracts > 0 else mark.bid
        premium = -delta_contracts * fill_price * multiplier
        transaction_cost = (
            abs(delta_contracts) / split_factor * settings.options.commission_per_contract
        )
        self.cash += premium - transaction_cost

        if target_contracts == 0:
            self.option_positions.pop(contract_id, None)
        else:
            entry_price = fill_price
            if position is not None and (position.contracts > 0) == (target_contracts > 0):
                if abs(target_contracts) > abs(position.contracts):
                    entry_price = (
                        abs(position.contracts) * position.entry_price
                        + abs(delta_contracts) * fill_price
                    ) / abs(target_contracts)
                else:
                    entry_price = position.entry_price
            self.option_positions[contract_id] = OptionPosition(
                contract=contract,
                contracts=target_contracts,
                entry_price=entry_price,
                last_mid=mark.mid,
            )

        return TradeRecord(
            as_of_date=decision.as_of_date,
            asset=contract_id,
            action="buy" if delta_contracts > 0 else "sell",
            target_weight=decision.target_weight,
            previous_weight=previous_weight,
            target_shares=float(target_contracts),
            delta_shares=float(delta_contracts),
            price=fill_price,
            notional=abs(delta_contracts) * fill_price * multiplier,
            transaction_cost=transaction_cost,
            instrument=contract,
            contracts=float(delta_contracts),
            premium=premium,
        )

    def settle_expired(self, marks: dict[str, OptionMark], as_of_date: date) -> list[TradeRecord]:
        """Closes every option position whose expiry is on or before
        `as_of_date` at intrinsic value (the expired mark's mid, or the
        position's last mid without one), with no commission, and returns
        one "expire" record per settled position."""
        multiplier = float(settings.options.contract_multiplier)
        records: list[TradeRecord] = []
        for contract_id in [
            contract_id
            for contract_id, position in self.option_positions.items()
            if position.contract.expiry <= as_of_date
        ]:
            position = self.option_positions.pop(contract_id)
            mark = marks.get(contract_id)
            intrinsic = mark.mid if mark is not None else position.last_mid
            settlement = position.contracts * intrinsic * multiplier
            self.cash += settlement
            records.append(
                TradeRecord(
                    as_of_date=as_of_date,
                    asset=contract_id,
                    action="expire",
                    target_weight=0.0,
                    previous_weight=0.0,
                    target_shares=0.0,
                    delta_shares=float(-position.contracts),
                    price=intrinsic,
                    notional=abs(settlement),
                    instrument=position.contract,
                    contracts=float(-position.contracts),
                    premium=settlement,
                )
            )
        return records

    def apply_financing(
        self,
        prices: dict[str, float],
        risk_free_rate_annual: float,
        days_held: int,
        spots: dict[str, float] | None = None,
    ) -> float:
        """Charges `days_held` calendar days of financing to cash and returns
        the amount charged: stock borrow on the total short notional (at
        `prices`) at `settings.firm.short_borrow_rate_annual`, plus margin
        interest at `risk_free_rate_annual + settings.firm.margin_spread_annual`
        on a negative cash balance and on short-option margin (per short
        position, |contracts| x (last mid + `short_margin_fraction` x
        underlying spot) x multiplier; the spot comes from `spots`, then
        `prices`, and counts as 0 when neither has the underlying). A flat
        book with non-negative cash, or a zero-day gap, costs nothing."""
        if days_held <= 0:
            return 0.0
        short_notional = sum(
            -quantity * prices[asset] for asset, quantity in self.positions.items() if quantity < 0
        )
        borrowed_cash = -self.cash if self.cash < 0 else 0.0
        multiplier = float(settings.options.contract_multiplier)
        short_option_margin = 0.0
        for position in self.option_positions.values():
            if position.contracts >= 0:
                continue
            underlying = position.contract.underlying
            spot = (spots or {}).get(underlying, prices.get(underlying, 0.0))
            short_option_margin += (
                abs(position.contracts)
                * (position.last_mid + settings.options.short_margin_fraction * spot)
                * multiplier
            )
        cost = 0.0
        if short_notional > 0.0:
            cost += short_borrow_cost(
                short_notional, settings.firm.short_borrow_rate_annual, days_held
            )
        margin_rate = max(risk_free_rate_annual, 0.0) + settings.firm.margin_spread_annual
        if borrowed_cash > 0.0:
            cost += margin_financing_cost(borrowed_cash, margin_rate, days_held)
        if short_option_margin > 0.0:
            cost += margin_financing_cost(short_option_margin, margin_rate, days_held)
        self.cash -= cost
        return cost

    def total_value(
        self, prices: dict[str, float], option_marks: dict[str, OptionMark] | None = None
    ) -> float:
        """Cash plus the mark-to-market value of every position: equities
        at `prices` (asset -> current price; every asset in `positions`
        must have an entry) and options at contracts x mid x multiplier,
        using the mark in `option_marks` when there is one (which also
        becomes the position's `last_mid`) and the last known mid
        otherwise."""
        multiplier = float(settings.options.contract_multiplier)
        option_value = 0.0
        for contract_id, position in self.option_positions.items():
            mark = (option_marks or {}).get(contract_id)
            if mark is not None:
                position.last_mid = mark.mid
            option_value += position.contracts * position.last_mid * multiplier
        return (
            self.cash
            + sum(quantity * prices[asset] for asset, quantity in self.positions.items())
            + option_value
        )
