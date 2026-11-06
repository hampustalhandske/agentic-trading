# portfolio.py

import math
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator
from quantcore.execution.financing import margin_financing_cost, short_borrow_cost
from quantcore.execution.market_impact import square_root_market_impact

from agentic_trading.agents.schemas import OptionContract, PortfolioDecision, TradeRecord
from agentic_trading.config import settings
from agentic_trading.data.instruments import (
    currency_pair,
    futures_root,
    is_marked,
    quote_unit_value_usd,
    unit_value_usd,
)
from agentic_trading.data.loaders.option_marks import OptionMark
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)


class MarkedPosition(BaseModel):
    """An open futures or spot FX position (see `data/instruments.py`),
    held without paying cash for it: signed `quantity` (long positive) in
    units of the instrument's price, one unit of the continuous futures
    level or one unit of the pair's base currency, and `mark`, the price
    its profit and loss was last settled into cash at. Settling at a new
    price `S` pays `quantity x (S - mark)` in quote currency, converted to
    USD at `S` (`quote_unit_value_usd`); its USD notional is `quantity x
    unit_value_usd(mark)`."""

    quantity: float
    mark: float = Field(gt=0.0)


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
    """Deterministic portfolio state: cash, per-asset equity positions,
    per-asset futures and spot FX positions (`marked_positions`) and
    per-contract option positions. No LLM calls, no randomness — plain
    accounting.

    `commission_rate`/`slippage_bps` default to `settings.firm`'s values at
    construction time; passing `None` explicitly resolves to the same
    settings values."""

    cash: float
    positions: dict[str, float]
    marked_positions: dict[str, MarkedPosition] = Field(default_factory=dict)
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
        slippage. Without volume data neither applies.

        Slippage and impact move the execution price away from `price`,
        the record's `reference_price`; the record books each as
        |delta_shares| x `price` x its bps over 10,000 (`slippage_cost`,
        `impact_cost`) and the commission on the executed notional as
        `transaction_cost`.

        A future or spot FX pair (`data/instruments.is_marked`) fills
        through `_apply_marked_decision` instead."""
        if is_marked(decision.asset):
            return self._apply_marked_decision(
                decision, price, portfolio_value, adv_shares, daily_vol
            )
        current_shares = self.positions.get(decision.asset, 0.0)
        previous_weight = (
            0.0 if portfolio_value == 0 else (current_shares * price) / portfolio_value
        )
        target_shares = (decision.target_weight * portfolio_value) / price
        delta_shares, impact_bps = _capped_delta_and_impact(
            target_shares - current_shares, adv_shares, daily_vol
        )
        slippage = (self.slippage_bps + impact_bps) / 10_000.0
        execution_price = price * (1.0 + slippage if delta_shares > 0 else 1.0 - slippage)
        reference_notional = abs(delta_shares) * price
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
            reference_price=price,
            transaction_cost=transaction_cost,
            slippage_cost=reference_notional * self.slippage_bps / 10_000.0,
            impact_cost=reference_notional * impact_bps / 10_000.0,
            impact_bps=impact_bps,
        )

    def _apply_marked_decision(
        self,
        decision: PortfolioDecision,
        price: float,
        portfolio_value: float,
        adv_shares: float | None,
        daily_vol: float | None,
    ) -> TradeRecord:
        """Fills a future or spot FX pair toward `decision.target_weight` at
        `price` (positive), paying no cash for the position. The target is
        `target_weight x portfolio_value` of USD notional, that over
        `unit_value_usd(asset, price)` in units. A held position is first
        settled to `price` (its move from its mark is paid into cash), and
        the position after the fill is marked at `price`.

        The ADV cap, slippage, impact and commission apply as in
        `apply_decision`, on the traded USD notional: the record books
        |delta| x `unit_value_usd(asset, price)` x the slippage and impact
        bps over 10,000 (`slippage_cost`, `impact_cost`) and the commission
        on that notional moved by the slippage (`transaction_cost`). Those
        three costs are the only cash the fill moves; `price` on the record
        is the execution price, `price` moved by the slippage."""
        asset = decision.asset
        unit_value = unit_value_usd(asset, price)
        position = self.marked_positions.get(asset)
        current = 0.0
        if position is not None:
            current = position.quantity
            self.cash += self._marked_pnl(asset, position, price)
            position.mark = price
        previous_weight = 0.0 if portfolio_value == 0 else current * unit_value / portfolio_value
        target = decision.target_weight * portfolio_value / unit_value
        delta, impact_bps = _capped_delta_and_impact(target - current, adv_shares, daily_vol)
        slippage = (self.slippage_bps + impact_bps) / 10_000.0
        price_move = 1.0 + slippage if delta > 0 else 1.0 - slippage
        reference_notional = abs(delta) * unit_value
        notional = reference_notional * price_move
        transaction_cost = notional * self.commission_rate
        slippage_cost = reference_notional * self.slippage_bps / 10_000.0
        impact_cost = reference_notional * impact_bps / 10_000.0
        self.cash -= transaction_cost + slippage_cost + impact_cost
        quantity = current + delta
        if quantity == 0.0:
            self.marked_positions.pop(asset, None)
        else:
            self.marked_positions[asset] = MarkedPosition(quantity=quantity, mark=price)

        action: Literal["buy", "sell", "hold"] = "hold"
        if delta > 0:
            action = "buy"
        elif delta < 0:
            action = "sell"

        return TradeRecord(
            as_of_date=decision.as_of_date,
            asset=asset,
            action=action,
            target_weight=decision.target_weight,
            previous_weight=previous_weight,
            target_shares=target,
            delta_shares=delta,
            price=price * price_move,
            notional=notional,
            reference_price=price,
            transaction_cost=transaction_cost,
            slippage_cost=slippage_cost,
            impact_cost=impact_cost,
            impact_bps=impact_bps,
        )

    @staticmethod
    def _marked_pnl(asset: str, position: MarkedPosition, price: float) -> float:
        """The USD profit of `position` in `asset` moving from its mark to
        `price`: `quantity x (price - mark)` in quote currency, converted
        to USD at `price`."""
        return position.quantity * (price - position.mark) * quote_unit_value_usd(asset, price)

    def settle_marked_positions(self, prices: dict[str, float]) -> float:
        """Pays every futures and spot FX position's move from its mark to
        its price in `prices` into cash, re-marks it at that price, and
        returns the total paid (negative for a loss). A position whose
        price is missing, NaN or not positive keeps its mark and pays
        nothing. For a future that is the notional times the level's
        return; for a pair it is the USD profit of `MarkedPosition`."""
        total = 0.0
        for asset, position in self.marked_positions.items():
            price = prices.get(asset)
            if price is None or not math.isfinite(price) or price <= 0.0:
                continue
            total += self._marked_pnl(asset, position, price)
            position.mark = price
        self.cash += total
        return total

    def charge_rolls(
        self,
        as_of_date: date,
        prices: dict[str, float],
        rolls: dict[str, float],
        portfolio_value: float,
    ) -> list[TradeRecord]:
        """Charges each held future one round trip, an exit and an entry
        leg, per front-contract roll on `as_of_date` in `rolls` (asset to
        the number of rolls; see the price frame's `roll` field), on the
        notional it holds through the roll: |quantity| x its price in
        `prices` (its mark when `prices` has no usable one). Each leg pays
        `commission_rate` and `slippage_bps` on that notional, the per-
        notional rates of an equity fill, and cash falls by the total. A
        spot FX position, a future without a roll and a zero count are not
        charged.

        Returns one "roll" record per charged future: the position is
        unchanged (`delta_shares` 0, both weights its weight in
        `portfolio_value`), `notional` is the notional of every leg traded,
        and the commission and slippage are its `transaction_cost` and
        `slippage_cost`."""
        records: list[TradeRecord] = []
        for asset, position in self.marked_positions.items():
            count = rolls.get(asset, 0.0)
            if futures_root(asset) is None or not math.isfinite(count) or count <= 0.0:
                continue
            price = prices.get(asset, position.mark)
            if not math.isfinite(price) or price <= 0.0:
                price = position.mark
            held = abs(position.quantity) * price
            traded = 2.0 * count * held
            transaction_cost = traded * self.commission_rate
            slippage_cost = traded * self.slippage_bps / 10_000.0
            self.cash -= transaction_cost + slippage_cost
            weight = 0.0 if portfolio_value == 0 else position.quantity * price / portfolio_value
            records.append(
                TradeRecord(
                    as_of_date=as_of_date,
                    asset=asset,
                    action="roll",
                    target_weight=weight,
                    previous_weight=weight,
                    target_shares=position.quantity,
                    delta_shares=0.0,
                    price=price,
                    notional=traded,
                    reference_price=price,
                    transaction_cost=transaction_cost,
                    slippage_cost=slippage_cost,
                )
            )
        return records

    def _option_target_contracts(
        self,
        decision: PortfolioDecision,
        mark: OptionMark,
        portfolio_value: float,
        spot: float | None,
        split_factor: float,
    ) -> float:
        """The split-adjusted contract count `decision.target_weight` maps
        to at `mark.mid`: whole as-traded contracts, capped so the
        underlying notional stays within `settings.options.max_notional_fraction`
        of NAV when `spot` is known. `mark.mid` must be positive."""
        multiplier = float(settings.options.contract_multiplier)
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
        return target_as_traded * split_factor

    def option_target_differs_from_held(
        self,
        decision: PortfolioDecision,
        mark: OptionMark,
        portfolio_value: float,
        spot: float | None = None,
        split_factor: float = 1.0,
    ) -> bool:
        """Whether `decision` asks for a contract count other than the one
        held, sized at `mark.mid` exactly as `apply_option_decision` sizes
        it. A `hold_contracts` row with a non-zero target weight on a held
        contract never differs. A mark with no positive mid cannot size a
        target, so any non-zero target weight or held position counts as
        different."""
        if decision.instrument is None:
            raise ValueError(
                "option_target_differs_from_held requires a decision with an instrument"
            )
        position = self.option_positions.get(decision.instrument.contract_id)
        current_contracts = position.contracts if position is not None else 0.0
        if _keeps_held_count(decision, current_contracts):
            return False
        if mark.mid <= 0.0:
            return decision.target_weight != 0.0 or current_contracts != 0.0
        target_contracts = self._option_target_contracts(
            decision, mark, portfolio_value, spot, split_factor
        )
        return target_contracts != current_contracts

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
        traded. The record's `reference_price` is `mark.mid`, and its
        `spread_cost` is |delta contracts| x |fill price - mid| x the
        contract multiplier. A mark whose `tradeable` flag is false is a
        hold, leaving cash and positions unchanged, whatever bid/ask it
        carries, and so is a `hold_contracts` decision with a non-zero
        target weight on a contract already held. Any fill needs a
        two-sided quote with at least
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
                reference_price=mark.mid,
                instrument=contract,
            )

        if (
            _keeps_held_count(decision, current_contracts)
            or not mark.tradeable
            or mark.is_expired
            or mark.mid <= 0.0
        ):
            return _hold()

        as_traded_mid = mark.mid * split_factor
        target_contracts = self._option_target_contracts(
            decision, mark, portfolio_value, spot, split_factor
        )
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
            reference_price=mark.mid,
            transaction_cost=transaction_cost,
            spread_cost=abs(delta_contracts) * abs(fill_price - mark.mid) * multiplier,
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
                    reference_price=intrinsic,
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
        """Accrues `days_held` calendar days of financing on cash and returns
        the net charge: charges minus the interest credited, negative when
        the credit is larger. Cash changes by minus the returned amount.

        Charges are stock borrow on the total short equity notional (at
        `prices`) at `settings.firm.short_borrow_rate_annual`, plus margin interest at
        `risk_free_rate_annual + settings.firm.margin_spread_annual` on a
        negative cash balance and on short-option margin (per short
        position, |contracts| x (last mid + `short_margin_fraction` x
        underlying spot) x multiplier; the spot comes from `spots`, then
        `prices`, and counts as 0 when neither has the underlying). The
        credit is interest at `risk_free_rate_annual` on a positive cash
        balance. Short-sale proceeds sit in that cash, so a short earns the
        risk-free rate net of its borrow charge. Futures and spot FX
        positions (`marked_positions`) pay no borrow and no margin interest
        on their notional; FX carry is `apply_carry`. Every accrual is
        simple actual/365 interest; a negative `risk_free_rate_annual`
        counts as 0. A zero-day gap returns 0.0 and leaves cash unchanged."""
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
        risk_free_rate = max(risk_free_rate_annual, 0.0)
        margin_rate = risk_free_rate + settings.firm.margin_spread_annual
        if borrowed_cash > 0.0:
            cost += margin_financing_cost(borrowed_cash, margin_rate, days_held)
        if short_option_margin > 0.0:
            cost += margin_financing_cost(short_option_margin, margin_rate, days_held)
        # The deposit credit is the same simple actual/365 carry as a margin
        # charge, so it reuses quantcore's formula with the sign flipped.
        credit = (
            margin_financing_cost(self.cash, risk_free_rate, days_held) if self.cash > 0.0 else 0.0
        )
        net_cost = cost - credit
        self.cash -= net_cost
        return net_cost

    def apply_carry(self, carry_rates: dict[str, float], days_held: int) -> float:
        """Accrues `days_held` calendar days of carry on every spot FX
        position in `carry_rates` (asset to its base currency's policy rate
        minus its quote currency's, annual decimal, either sign) and returns
        the net charge, negative when carry is credited; cash changes by
        minus the returned amount. Each position accrues on its USD notional
        at its mark, |notional| x |rate| x days / 365 (simple actual/365,
        `margin_financing_cost`), credited when the position and the rate
        have the same sign and charged otherwise, so a long base position
        earns the base rate minus the quote rate and a short pays it. A
        future, a pair without a finite rate and a zero-day gap accrue
        nothing."""
        if days_held <= 0:
            return 0.0
        carry = 0.0
        for asset, position in self.marked_positions.items():
            rate = carry_rates.get(asset)
            if rate is None or not math.isfinite(rate) or currency_pair(asset) is None:
                continue
            notional = position.quantity * unit_value_usd(asset, position.mark)
            amount = margin_financing_cost(abs(notional), abs(rate), days_held)
            carry += amount if (notional > 0.0) == (rate > 0.0) else -amount
        self.cash += carry
        return -carry

    def total_value(
        self, prices: dict[str, float], option_marks: dict[str, OptionMark] | None = None
    ) -> float:
        """Cash plus the mark-to-market value of every position: equities
        at `prices` (asset -> current price; every asset in `positions`
        must have an entry), futures and spot FX at their move from their
        mark to `prices` not yet settled into cash (0 for one `prices` does
        not price, and for every one right after `settle_marked_positions`),
        and options at contracts x mid x multiplier, using the mark in
        `option_marks` when there is one (which also becomes the position's
        `last_mid`) and the last known mid otherwise."""
        multiplier = float(settings.options.contract_multiplier)
        option_value = 0.0
        for contract_id, position in self.option_positions.items():
            mark = (option_marks or {}).get(contract_id)
            if mark is not None:
                position.last_mid = mark.mid
            option_value += position.contracts * position.last_mid * multiplier
        unsettled = 0.0
        for asset, marked in self.marked_positions.items():
            price = prices.get(asset)
            if price is not None and math.isfinite(price) and price > 0.0:
                unsettled += self._marked_pnl(asset, marked, price)
        return (
            self.cash
            + sum(quantity * prices[asset] for asset, quantity in self.positions.items())
            + unsettled
            + option_value
        )


def _capped_delta_and_impact(
    delta: float, adv_shares: float | None, daily_vol: float | None
) -> tuple[float, float]:
    """`delta` (units to trade) capped at `settings.firm.max_adv_participation`
    of `adv_shares`, and the square-root market impact in bps of trading
    it. Without positive `adv_shares` and a `daily_vol`, `delta` as given
    and 0.0 impact."""
    impact_bps = 0.0
    if adv_shares is not None and adv_shares > 0.0 and daily_vol is not None:
        max_shares = settings.firm.max_adv_participation * adv_shares
        if abs(delta) > max_shares:
            delta = math.copysign(max_shares, delta)
        if delta != 0.0:
            impact_bps = square_root_market_impact(
                abs(delta),
                adv_shares,
                max(daily_vol, 0.0),
                settings.firm.market_impact_coefficient,
            )
    return delta, impact_bps


def _keeps_held_count(decision: PortfolioDecision, current_contracts: float) -> bool:
    """Whether `decision` asks the book to keep the `current_contracts` it
    holds: a `hold_contracts` row with a non-zero target weight on a
    contract with a position held."""
    return decision.hold_contracts and current_contracts != 0.0 and decision.target_weight != 0.0
