# tail_hedge_manager.py

from datetime import date

import pandas as pd
from pydantic import PrivateAttr
from quantcore.pricing.options_positions import protective_put_sizing

from agentic_trading.agents._market_math import raw_spot
from agentic_trading.agents.base import ManagerAgent
from agentic_trading.agents.schemas import (
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

# The single underlying the sleeve protects with listed puts; a short in
# the ETF itself is the linear fallback on a day with no usable chain.
TAIL_HEDGE_ASSETS: tuple[str, ...] = ("SPY",)
_DAYS_PER_YEAR = 365.0
# `protective_put_sizing` works in whole contracts, so the protection a
# premium budget buys is logged against a nominal NAV large enough for
# contract rounding not to dominate; every logged figure is a fraction.
_NOMINAL_NAV = 1_000_000.0


class TailHedgeManager(ManagerAgent):
    """Scoped (`self.assets = TAIL_HEDGE_ASSETS`) to a single-underlying
    protective-put insurance sleeve — `vix_tail_hedge_analyst` is this
    pod's only analyst. Funded by a fixed, protected Commander-level
    budget carve-out (`settings.commander.tail_hedge_budget_fraction`)
    rather than the trailing-Sharpe/risk-parity allocation every other
    pod gets, since a hedge that gets defunded by its own (usually
    negative-carry) performance during a calm stretch defeats its own
    purpose — see `Commander.decide()`.

    `construct_portfolio` expresses the sleeve as a long SPY put
    (`settings.options.tail_hedge_strike_moneyness` of spot, near
    `settings.options.tail_hedge_tenor_days` tenor), sized in premium as the
    budget fraction times the analyst's confidence, and rolled once the
    held contract is inside `settings.options.roll_min_dte`. The SPY
    equity leg is emitted at 0.0 while a put is held; on a day with no
    usable chain the sleeve falls back to the short-SPY equity weight
    the base class sized, so protection never disappears silently.

    `scale_factor` always returns 1.0, deliberately not the EWMA-vol
    dampening every other pod's `scale_factor` uses: the analyst's own
    VIX z-score already *is* the risk-regime read this sleeve exists to
    act on, and dampening confidence by SPY's own realized volatility
    would shrink the hedge exactly when a real stress regime should be
    making it larger."""

    agent_id: str = "tail_hedge_manager"

    # The put contract currently held, `None` when the sleeve is in its
    # equity fallback or has not yet bought protection.
    _held_contract_id: str | None = PrivateAttr(default=None)

    def scale_factor(
        self,
        signal: AgentSignal,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> float:
        return 1.0

    def construct_portfolio(
        self,
        assets: list[str],
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
        scoreboard: PerformanceScoreboard,
    ) -> list[PortfolioDecision]:
        decisions = super().construct_portfolio(assets, market_data, as_of_date, scoreboard)
        underlying = TAIL_HEDGE_ASSETS[0]
        equity = next((decision for decision in decisions if decision.asset == underlying), None)
        if equity is None:
            return decisions

        confidence = _hedge_confidence(equity.contributing_signals)
        if confidence is None:
            logger.info(
                "%s: tail hedge has no active short signal — no put bought.", as_of_date.date()
            )
            return decisions

        day = as_of_date.date()
        budget = settings.commander.tail_hedge_budget_fraction * confidence
        held = self._held_contract_id
        rolling = held is None or _days_to_expiry(held, day) < settings.options.roll_min_dte
        selected = self._select_put(market_data, as_of_date, budget) if rolling else held
        if selected is None:
            if held is not None and _days_to_expiry(held, day) > 0:
                logger.info("%s: no usable SPY chain to roll %s into — holding it.", day, held)
                selected = held
            else:
                logger.info(
                    "%s: no usable SPY put chain — tail hedge falls back to the short-SPY "
                    "equity weight %.4f.",
                    day,
                    equity.target_weight,
                )
                self._held_contract_id = None
                return decisions

        option_rows: list[PortfolioDecision] = []
        if held is not None and held != selected:
            option_rows.append(_option_decision(held, day, 0.0, equity.contributing_signals))
        option_rows.append(
            _option_decision(
                selected,
                day,
                self._premium_weight(selected, market_data, as_of_date, budget),
                equity.contributing_signals,
            )
        )
        self._held_contract_id = selected

        return [
            decision.model_copy(update={"target_weight": 0.0})
            if decision.asset == underlying
            else decision
            for decision in decisions
        ] + option_rows

    def _premium_weight(
        self, contract_id: str, market_data: pd.DataFrame, as_of_date: pd.Timestamp, budget: float
    ) -> float:
        """Premium weight that buys puts on
        `settings.options.tail_hedge_notional_fraction` of NAV in underlying
        notional, capped by the budget: contracts stay fixed as the put
        decays instead of growing to spend a fixed premium. The budget
        itself when no mark is available."""
        contract = OptionContract.parse(contract_id)
        spot = raw_spot(market_data, as_of_date, contract.underlying)
        mark = (
            option_marks.load_option_mark(contract, as_of_date.date(), spot)
            if spot is not None and spot > 0.0
            else None
        )
        if mark is None or spot is None or mark.mid <= 0.0:
            return budget
        return min(budget, settings.options.tail_hedge_notional_fraction * mark.mid / spot)

    def _select_put(
        self, market_data: pd.DataFrame, as_of_date: pd.Timestamp, budget: float
    ) -> str | None:
        """The contract id of a tradeable near-tenor OTM SPY put, or
        `None` when the chain offers no expiry after `as_of_date`, strike or
        live mark. The expiry nearest `as_of_date` plus the hedge tenor and
        the strike come from the chain as published by `as_of_date`
        (`option_marks.published_chain_date`); the mark is the day's own."""
        underlying = TAIL_HEDGE_ASSETS[0]
        day = as_of_date.date()
        spot = raw_spot(market_data, as_of_date, underlying)
        if spot is None or spot <= 0.0:
            return None
        chain_day = option_marks.published_chain_date(day)
        tenor_days = (day - chain_day).days + settings.options.tail_hedge_tenor_days
        expiry = option_marks.select_expiry(underlying, chain_day, tenor_days, after=day)
        if expiry is None:
            return None
        strike = option_marks.nearest_strike(
            underlying,
            expiry,
            "put",
            spot * settings.options.tail_hedge_strike_moneyness,
            chain_day,
        )
        if strike is None:
            return None
        contract = OptionContract(
            underlying=underlying, expiry=expiry, strike=strike, option_type="put"
        )
        mark = option_marks.load_option_mark(contract, day, spot)
        if mark is None or not mark.tradeable or mark.mid <= 0.0:
            return None
        _log_protection(contract, mark, spot, day, budget)
        return contract.contract_id


def _hedge_confidence(signals: list[ManagerSignal]) -> float | None:
    """The analyst's own short-signal confidence, floored at the base
    protection level; `None` when no active short signal exists."""
    active = [
        signal for signal in signals if signal.status == "active" and signal.direction == "short"
    ]
    if not active:
        return None
    return max(
        max(signal.raw_confidence for signal in active),
        settings.analysts.tail_hedge_base_confidence,
    )


def _days_to_expiry(contract_id: str, day: date) -> int:
    return (OptionContract.parse(contract_id).expiry - day).days


def _option_decision(
    contract_id: str, day: date, weight: float, signals: list[ManagerSignal]
) -> PortfolioDecision:
    return PortfolioDecision(
        asset=contract_id,
        as_of_date=day,
        target_weight=weight,
        contributing_signals=signals,
        instrument=OptionContract.parse(contract_id),
    )


def _log_protection(
    contract: OptionContract, mark: option_marks.OptionMark, spot: float, day: date, budget: float
) -> None:
    """Logs the protection level the premium budget buys at this strike
    and implied volatility (quantcore `protective_put_sizing` against
    `_NOMINAL_NAV`, reported as fractions of NAV)."""
    time_to_expiry = (contract.expiry - day).days / _DAYS_PER_YEAR
    if mark.implied_volatility <= 0.0 or not 0.0 < budget < 1.0:
        return
    sizing = protective_put_sizing(
        _NOMINAL_NAV,
        spot,
        contract.strike,
        mark.implied_volatility,
        time_to_expiry,
        0.0,
        settings.options.tail_hedge_target_protection,
        budget,
    )
    logger.info(
        "%s: tail hedge %s at IV %.3f — premium budget %.4f of NAV implies %.2f%% protection "
        "(target %.0f%%, breakeven %.2f)",
        day,
        contract.contract_id,
        mark.implied_volatility,
        budget,
        sizing.protection_pct_achieved * 100.0,
        settings.options.tail_hedge_target_protection * 100.0,
        sizing.breakeven,
    )
