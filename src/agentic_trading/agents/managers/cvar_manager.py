# cvar_manager.py

from __future__ import annotations

from datetime import date
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field, PrivateAttr

from agentic_trading.agents._market_math import raw_spot, vol_regime_scale_factor
from agentic_trading.agents.base import ManagerAgent
from agentic_trading.agents.schemas import (
    MAX_LEVERAGE,
    AgentSignal,
    ManagerSignal,
    OptionContract,
    PortfolioDecision,
)
from agentic_trading.config import settings
from agentic_trading.data.loaders import option_marks
from agentic_trading.logging_config import get_logger
from agentic_trading.scoring.scoreboard import PerformanceScoreboard

logger = get_logger(__name__)

# The only underlyings with a real historical option chain in the lakehouse
# (`table=dolthub_option_chain`, verified directly against the lake: these
# eleven carry a multi-year daily-ish series; every other underlying there
# is an 8-day 2019 stub and does not count as coverage). The pod is scoped
# to this tuple so an options analyst never prices an instrument it has no
# real market quote for. Re-verify against the lake when the chain pull
# grows.
OPTIONS_ASSETS: tuple[str, ...] = (
    "AAPL",
    "AMZN",
    "COIN",
    "GOOGL",
    "JPM",
    "META",
    "MSFT",
    "NVDA",
    "SPY",
    "TSLA",
    "XOM",
)

# The analyst whose "long" (implied vol rich) signal is expressed as a
# short ATM straddle on top of the pod's directional leg.
VRP_ANALYST_ID = "heston_vol_surface_analyst"
# A directional leg with less delta than this is too far from the money
# to carry the view; the equity weight is kept instead.
MIN_DIRECTIONAL_DELTA = 0.05
# Straddle premium per unit of VRP confidence when the pod holds no
# directional weight in the underlying.
STRADDLE_BASE_FRACTION = 0.02


class _HeldExpression(BaseModel):
    """The option legs this pod holds on one underlying, each as contract
    id to premium weight: the long directional leg (`directional`, at
    most one entry), the short straddle legs (`straddle`, negative), and
    the equity delta-hedge weight carried against the straddle. The two
    expressions may share a contract (an ATM call is both the long leg
    of a bullish view and one leg of the straddle); `net_weights` is
    what gets emitted."""

    directional: dict[str, float] = Field(default_factory=dict)
    straddle: dict[str, float] = Field(default_factory=dict)
    hedge_weight: float = 0.0

    @property
    def directional_id(self) -> str | None:
        return next(iter(self.directional), None)

    @property
    def straddle_ids(self) -> list[str]:
        return list(self.straddle)

    @property
    def is_empty(self) -> bool:
        return not self.directional and not self.straddle

    def net_weights(self) -> dict[str, float]:
        weights = dict(self.directional)
        for contract_id, weight in self.straddle.items():
            weights[contract_id] = weights.get(contract_id, 0.0) + weight
        return weights


class ConditionalValueAtRiskManager(ManagerAgent):
    """Scales analyst confidence down when the asset's own volatility is
    elevated relative to its trailing history. Scoped
    (`assets=OPTIONS_ASSETS` in `roster.py`) to the underlyings with a
    real option chain in the lakehouse.

    `construct_portfolio` expresses every sized underlying weight as a
    listed option instead of stock: a long view buys a near-tenor ATM
    call, a short view a near-tenor ATM put, each sized so the premium
    carries the same delta exposure the stock position would have
    (capped at the stock weight). A rich variance risk premium from
    `heston_vol_surface_analyst` (direction "long" at or above
    `settings.options.short_straddle_min_confidence`) additionally sells
    an ATM straddle, delta-hedged with an equity weight on the
    underlying. Legs are held while at least
    `settings.options.roll_min_dte` days remain and the view's sign is
    unchanged, otherwise closed and re-selected. Whenever no tradeable
    contract exists the underlying's equity weight is kept."""

    agent_id: str = "cvar_manager"

    _held: dict[str, _HeldExpression] = PrivateAttr(default_factory=dict)
    _opposite_days: dict[str, int] = PrivateAttr(default_factory=dict)

    def scale_factor(
        self,
        signal: AgentSignal,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> float:
        """Volatility-regime dampening for `signal.asset` (see
        `_market_math.vol_regime_scale_factor`): near 1.0 in a normal
        regime, halved at a 1.5-sigma vol spike."""
        return vol_regime_scale_factor(market_data, as_of_date, signal.asset)

    def construct_portfolio(
        self,
        assets: list[str],
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
        scoreboard: PerformanceScoreboard,
    ) -> list[PortfolioDecision]:
        decisions = super().construct_portfolio(assets, market_data, as_of_date, scoreboard)
        day = as_of_date.date()
        equity_rows: list[PortfolioDecision] = []
        option_rows: list[PortfolioDecision] = []
        for decision in decisions:
            asset = decision.asset
            held = self._held.get(asset, _HeldExpression())
            in_scope = self.assets is None or asset in self.assets
            spot = raw_spot(market_data, as_of_date, asset) if in_scope else None
            if spot is None or spot <= 0.0:
                if not held.is_empty:
                    logger.info("%s: no spot for %s — closing its option legs.", day, asset)
                option_rows.extend(_close_rows(held, day, decision.contributing_signals))
                self._held.pop(asset, None)
                equity_rows.append(decision)
                continue

            weight = decision.target_weight
            expression = _HeldExpression()
            equity_weight = weight

            directional = self._directional_leg(asset, weight, spot, day, held)
            if directional is not None:
                contract_id, fraction = directional
                expression.directional[contract_id] = fraction
                equity_weight = 0.0

            vrp_confidence = _vrp_confidence(decision.contributing_signals)
            straddle = None
            if held.straddle and not _vrp_reversed(decision.contributing_signals):
                # A short straddle is held until the premium view reverses
                # (or the legs roll), not re-sized on every confidence tick.
                straddle = self._straddle_legs(asset, None, spot, day, held)
            elif vrp_confidence is not None:
                magnitude = abs(weight) if weight != 0.0 else STRADDLE_BASE_FRACTION
                straddle = self._straddle_legs(asset, vrp_confidence * magnitude, spot, day, held)
            if straddle is not None:
                expression.straddle, expression.hedge_weight = straddle
                equity_weight += expression.hedge_weight

            emitted = expression.net_weights()
            for contract_id in held.net_weights():
                if contract_id not in emitted:
                    option_rows.append(
                        _option_row(contract_id, day, 0.0, decision.contributing_signals)
                    )
            for contract_id, fraction in emitted.items():
                option_rows.append(
                    _option_row(contract_id, day, fraction, decision.contributing_signals)
                )
            if expression.is_empty:
                self._held.pop(asset, None)
            else:
                self._held[asset] = expression
            equity_rows.append(
                decision.model_copy(
                    update={"target_weight": max(min(equity_weight, MAX_LEVERAGE), -MAX_LEVERAGE)}
                )
            )
        return equity_rows + option_rows

    def _directional_leg(
        self, asset: str, weight: float, spot: float, day: date, held: _HeldExpression
    ) -> tuple[str, float] | None:
        """`(contract id, premium weight)` of the long single leg carrying
        `weight`'s delta exposure — the held leg while it still has the
        view's sign and enough days to expiry, otherwise a freshly
        selected near-tenor ATM contract. `None` when the view is flat
        or no tradeable contract exists, in which case the caller keeps
        the equity weight."""
        held_id = held.directional_id
        if held_id is None and abs(weight) < settings.options.min_expression_weight:
            return None
        option_type: Literal["call", "put"] = "call" if weight > 0.0 else "put"
        contract: OptionContract | None = None
        if held_id is not None:
            held_contract = OptionContract.parse(held_id)
            held_side: Literal["call", "put"] = held_contract.option_type
            if weight == 0.0 or option_type != held_side:
                # Only a confirmed reversal switches side; a flat or
                # unconfirmed opposite view keeps the held leg.
                streak = self._opposite_days.get(asset, 0) + 1
                self._opposite_days[asset] = streak
                if streak < settings.options.expression_confirmation_days:
                    option_type = held_side
                    weight = held.directional[held_id] if weight == 0.0 else weight
            else:
                self._opposite_days[asset] = 0
            if (
                held_contract.option_type == option_type
                and (held_contract.expiry - day).days >= settings.options.roll_min_dte
            ):
                contract = held_contract
        elif weight == 0.0:
            return None
        if contract is None:
            contract = _select_contract(asset, option_type, spot, spot, day)
            if contract is None:
                logger.info(
                    "%s: no listed %s on %s — keeping the equity weight %.4f.",
                    day,
                    option_type,
                    asset,
                    weight,
                )
                return None
        mark = option_marks.load_option_mark(contract, day, spot)
        if mark is None or not mark.tradeable or mark.mid <= 0.0:
            if held_id is not None and contract.contract_id == held_id:
                logger.info("%s: %s has no live mark — holding it.", day, held_id)
                return held_id, held.directional[held_id]
            logger.info(
                "%s: %s is untradeable — keeping the equity weight %.4f.",
                day,
                contract.contract_id,
                weight,
            )
            return None
        if abs(mark.delta) < MIN_DIRECTIONAL_DELTA:
            logger.info(
                "%s: %s delta %.3f too small to carry the view — keeping the equity weight %.4f.",
                day,
                contract.contract_id,
                mark.delta,
                weight,
            )
            return None
        if held_id is not None and contract.contract_id == held_id:
            # A held leg's premium fraction is frozen at whatever it was
            # when the contract was opened or rolled: recomputing it from
            # today's floating conviction score, even while holding the
            # same contract, is what turns a "hold until roll" position
            # into a stream of small, spread-crossing resizes every day
            # the pod's underlying weight moves (see KNOWN_LIMITATIONS
            # §6). Only opening a fresh contract (a first entry, a roll,
            # or a confirmed side flip) re-sizes.
            return held_id, held.directional[held_id]
        fraction = min(abs(weight) * mark.mid / (abs(mark.delta) * spot), abs(weight))
        return contract.contract_id, fraction

    def _straddle_legs(
        self,
        asset: str,
        notional_fraction: float | None,
        spot: float,
        day: date,
        held: _HeldExpression,
    ) -> tuple[dict[str, float], float] | None:
        """Short-straddle legs (`{contract id: negative premium weight}`)
        whose underlying notional is `notional_fraction` of NAV (premium =
        notional x (call mid + put mid) / spot, so a straddle on a 1x view
        at 50% confidence costs a few percent of NAV in premium, not 50%),
        split across the legs by mid so both hold the same number of
        contracts, plus the equity weight that neutralizes their delta.
        `notional_fraction=None` keeps the held straddle's premium (a
        hold, or a roll at the same size). Reuses the held straddle while
        it has enough days to expiry; `None` when no tradeable pair
        exists."""
        held_ids = held.straddle_ids
        contracts: list[OptionContract] = []
        if len(held_ids) == 2:
            held_contracts = [OptionContract.parse(cid) for cid in held_ids]
            if all(
                (contract.expiry - day).days >= settings.options.roll_min_dte
                for contract in held_contracts
            ):
                contracts = held_contracts
        if not contracts:
            call = _select_contract(asset, "call", spot, spot, day)
            if call is None:
                logger.info("%s: no listed straddle on %s — not sold.", day, asset)
                return None
            contracts = [
                call,
                OptionContract(
                    underlying=asset, expiry=call.expiry, strike=call.strike, option_type="put"
                ),
            ]
        marks = [option_marks.load_option_mark(contract, day, spot) for contract in contracts]
        if any(mark is None or not mark.tradeable or mark.mid <= 0.0 for mark in marks):
            if [contract.contract_id for contract in contracts] == held_ids:
                logger.info("%s: held straddle on %s has no live mark — holding it.", day, asset)
                return dict(held.straddle), held.hedge_weight
            logger.info("%s: straddle on %s is untradeable — not sold.", day, asset)
            return None
        live = [mark for mark in marks if mark is not None]
        total_mid = sum(mark.mid for mark in live)
        fraction = (
            sum(abs(value) for value in held.straddle.values())
            if notional_fraction is None
            else notional_fraction * total_mid / spot
        )
        legs: dict[str, float] = {}
        hedge_weight = 0.0
        for contract, mark in zip(contracts, live, strict=True):
            leg_fraction = fraction * mark.mid / total_mid
            legs[contract.contract_id] = -leg_fraction
            hedge_weight += leg_fraction * mark.delta * spot / mark.mid
        return legs, hedge_weight


def _select_contract(
    asset: str, option_type: Literal["call", "put"], target_strike: float, spot: float, day: date
) -> OptionContract | None:
    """The `option_type` contract nearest `target_strike` at the listed
    expiry after `day` nearest `day` plus the target tenor, both read from
    the chain as published by `day` (`option_marks.published_chain_date`);
    `None` when the chain lists no such expiry or no quoted strike."""
    chain_day = option_marks.published_chain_date(day)
    tenor_days = (day - chain_day).days + settings.options.target_tenor_days
    expiry = option_marks.select_expiry(asset, chain_day, tenor_days, after=day)
    if expiry is None:
        return None
    strike = option_marks.nearest_strike(asset, expiry, option_type, target_strike, chain_day)
    if strike is None:
        return None
    return OptionContract(underlying=asset, expiry=expiry, strike=strike, option_type=option_type)


def _vrp_confidence(signals: list[ManagerSignal]) -> float | None:
    """The VRP analyst's confidence when its signal qualifies for a short
    straddle (active, "long", at or above the configured floor)."""
    for signal in signals:
        if (
            signal.source_agent_id == VRP_ANALYST_ID
            and signal.status == "active"
            and signal.direction == "long"
            and signal.confidence >= settings.options.short_straddle_min_confidence
        ):
            return signal.confidence
    return None


def _vrp_reversed(signals: list[ManagerSignal]) -> bool:
    """True when the VRP analyst actively says implied vol is cheap
    (direction "short"), the only read that closes a held straddle."""
    return any(
        signal.source_agent_id == VRP_ANALYST_ID
        and signal.status == "active"
        and signal.direction == "short"
        for signal in signals
    )


def _option_row(
    contract_id: str, day: date, weight: float, signals: list[ManagerSignal]
) -> PortfolioDecision:
    return PortfolioDecision(
        asset=contract_id,
        as_of_date=day,
        target_weight=weight,
        contributing_signals=signals,
        instrument=OptionContract.parse(contract_id),
    )


def _close_rows(
    held: _HeldExpression, day: date, signals: list[ManagerSignal]
) -> list[PortfolioDecision]:
    return [_option_row(contract_id, day, 0.0, signals) for contract_id in held.net_weights()]


CVaRManager = ConditionalValueAtRiskManager
