# portfolio.py

from typing import Literal

from pydantic import BaseModel

from agentic_trading.agents.schemas import PortfolioDecision, TradeRecord


class Portfolio(BaseModel):
    """Deterministic portfolio state: cash and per-asset positions. No LLM
    calls, no randomness — plain accounting."""

    cash: float
    positions: dict[str, float]
    commission_rate: float = 0.0
    slippage_bps: float = 0.0

    def apply_decision(
        self, decision: PortfolioDecision, price: float, portfolio_value: float
    ) -> TradeRecord:
        """Fills `decision.target_weight` for `decision.asset` at `price`,
        updating cash and positions in place, and returns a TradeRecord
        describing what happened — the single source of truth every trade
        blotter/trade-level report (CLI or GUI) is built from. Order-fill
        mechanics (partial fills, slippage, transaction costs) are decided
        by the eventual implementation, not by this signature."""
        current_shares = self.positions.get(decision.asset, 0.0)
        previous_weight = (
            0.0 if portfolio_value == 0 else (current_shares * price) / portfolio_value
        )
        target_shares = (decision.target_weight * portfolio_value) / price
        delta_shares = target_shares - current_shares
        slippage = self.slippage_bps / 10_000.0
        execution_price = price * (1.0 + slippage if delta_shares > 0 else 1.0 - slippage)
        notional = abs(delta_shares * execution_price)
        transaction_cost = notional * self.commission_rate
        self.cash -= delta_shares * execution_price
        self.cash -= transaction_cost
        self.positions[decision.asset] = target_shares

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
        )

    def total_value(self, prices: dict[str, float]) -> float:
        """Cash plus the mark-to-market value of every position, using
        `prices` (asset -> current price). Every asset in `positions` must
        have a corresponding entry in `prices`."""
        return self.cash + sum(
            quantity * prices[asset] for asset, quantity in self.positions.items()
        )
