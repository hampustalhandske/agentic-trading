# dividend_payout_analyst.py

from typing import Literal

import pandas as pd

from agentic_trading.agents._market_math import clamp
from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.data.loaders.lakehouse_loader import load_dividends_from_lakehouse
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

_MIN_CUT_SEVERITY_FOR_FULL_CONFIDENCE = 0.5


class DividendPayoutAnalyst(AnalystAgent):
    """A real distress/quality signal, not a return-series statistic: a
    dividend cut is a well-known early warning of financial distress (it
    directly costs a company real cash flow flexibility to announce), and
    a raise is a real positive statement about a board's confidence in
    forward earnings — both are decisions company management makes with
    money, not signals derived from price action. Compares the two most
    recent real dividend payments (`dolthub_stock_dividends`, via
    `data/loaders/lakehouse_loader.py`) for `asset`: a cut signals short,
    a raise signals long, an unchanged payout is flat. This is
    independent of `adj_close` already reflecting dividends in realized
    returns (a separate, existing correctness property) — this analyst
    is the first place in the roster that reads the payout itself as a
    signal rather than only its price impact."""

    agent_id: str = "dividend_payout_analyst"

    def generate_signal(
        self,
        asset: str,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> AgentSignal:
        series = load_dividends_from_lakehouse(asset, as_of_date.date())
        if series is None or series.size < 2:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="insufficient dividend history to compare payouts",
                status="unavailable",
            )

        latest = float(series.iloc[-1])
        previous = float(series.iloc[-2])
        if previous <= 0.0:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="prior payout was non-positive — cannot compute a change ratio",
                status="unavailable",
            )

        change_ratio = (latest - previous) / previous
        direction: Literal["long", "short", "flat"]
        if change_ratio > 0.0:
            direction = "long"
        elif change_ratio < 0.0:
            direction = "short"
        else:
            direction = "flat"
        confidence = clamp(abs(change_ratio) / _MIN_CUT_SEVERITY_FOR_FULL_CONFIDENCE)

        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction=direction,
            confidence=confidence,
            rationale=(
                f"dividend payout changed {change_ratio:+.1%} "
                f"({previous:.4f} -> {latest:.4f} per share)"
            ),
        )
