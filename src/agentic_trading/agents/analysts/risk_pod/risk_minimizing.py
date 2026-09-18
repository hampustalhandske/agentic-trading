# risk_minimizing.py

from __future__ import annotations

from typing import Literal

import pandas as pd

from agentic_trading.agents._market_math import clamp, market_slice, vol_regime_zscore
from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.schemas import AgentSignal, SignalStatus
from agentic_trading.config import settings
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

# Confidence reaches its cap at a 3-sigma vol-regime reading; the cap keeps
# a single vol read from ever dominating a pod on its own.
FULL_CONFIDENCE_ZSCORE = 3.0
MAX_CONFIDENCE = 0.8


class RiskMinimizingAnalyst(AnalystAgent):
    """The low-volatility premium as a directional view: where today's
    conditional volatility sits against the asset's own trailing vol
    history (`_market_math.vol_regime_zscore`, an EWMA log-vol z-score
    over the last year). A calm regime (z at or below
    `settings.analysts.vol_zscore_long_threshold`) is long; a genuine
    spike (z at or above `vol_zscore_short_threshold`) carries negative
    expected return and is short; in between there is no view."""

    agent_id: str = "risk_minimizing_analyst"

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
            if z_score <= settings.analysts.vol_zscore_long_threshold:
                direction = "long"
            elif z_score >= settings.analysts.vol_zscore_short_threshold:
                direction = "short"
            if direction != "flat":
                confidence = clamp(abs(z_score) / FULL_CONFIDENCE_ZSCORE, 0.0, MAX_CONFIDENCE)
            rationale = f"vol-regime z-score={z_score:.2f} vs own trailing EWMA log-vol path"
        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction=direction,
            confidence=confidence,
            rationale=rationale,
            status=status,
        )
