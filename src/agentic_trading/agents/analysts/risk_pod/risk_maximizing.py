# risk_maximizing.py

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from agentic_trading.agents._market_math import clamp, market_slice, vol_regime_zscore
from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.schemas import AgentSignal, SignalStatus
from agentic_trading.config import settings
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

# A capitulation needs a down move over the last trading week to buy into.
CAPITULATION_WINDOW = 5
# Confidence starts at 0.2 at the spike threshold and adds 0.5 per extra
# sigma, capped so one vol read never dominates a pod.
BASE_CONFIDENCE = 0.2
CONFIDENCE_PER_SIGMA = 0.5
MAX_CONFIDENCE = 0.8


class RiskMaximizingAnalyst(AnalystAgent):
    """Volatility as opportunity, the contrarian complement to
    `RiskMinimizingAnalyst`: buys a capitulation — a vol-regime z-score
    (`_market_math.vol_regime_zscore`) at or above
    `settings.analysts.vol_zscore_short_threshold` while the last week's
    return is negative — and otherwise has no view."""

    agent_id: str = "risk_maximizing_analyst"

    def generate_signal(
        self,
        asset: str,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> AgentSignal:
        returns = market_slice(market_data, as_of_date, asset).log_returns
        z_score = vol_regime_zscore(returns)
        direction: Literal["long", "short", "flat"] = "flat"
        confidence = 0.0
        status: SignalStatus = "active"
        if z_score is None:
            status = "unavailable"
            rationale = "insufficient history to place today's volatility in its own regime"
        else:
            recent_move = float(np.sum(returns[-CAPITULATION_WINDOW:]))
            threshold = settings.analysts.vol_zscore_short_threshold
            if z_score >= threshold and recent_move < 0.0:
                direction = "long"
                confidence = clamp(
                    BASE_CONFIDENCE + CONFIDENCE_PER_SIGMA * (z_score - threshold),
                    0.0,
                    MAX_CONFIDENCE,
                )
            rationale = (
                f"vol-regime z-score={z_score:.2f}, {CAPITULATION_WINDOW}d return={recent_move:.4f}"
            )
        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction=direction,
            confidence=confidence,
            rationale=rationale,
            status=status,
        )
