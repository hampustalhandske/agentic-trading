# options_skew_sentiment_analyst.py

from typing import Literal

import numpy as np
import pandas as pd

from agentic_trading.agents._market_math import clamp
from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.data.loaders.cboe_loader import load_put_call_ratio
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

WINDOW = 20
MIN_HISTORY_POINTS = 20
Z_CAP = 3.0


class OptionsSkewSentimentAnalyst(AnalystAgent):
    """CBOE total put/call ratio (`data/loaders/cboe_loader.py`) as a
    market-wide crowd fear/greed positioning read — distinct from the
    CVaR pod's per-name theoretical option pricing, this reads real
    aggregate market positioning, not a modeled price. Convention: a
    z-score-elevated put/call ratio is read as *contrarian bullish*
    (crowded bearish positioning is itself fuel for a reversal), the same
    contrarian convention `short_interest_squeeze_analyst` already applies
    to crowded short positioning elsewhere in this project — kept
    consistent rather than picking the opposite (fear-following) reading
    silently. A depressed ratio (z-score negative) is the mirror case:
    crowd complacency/greed, read as contrarian bearish."""

    agent_id: str = "options_skew_sentiment_analyst"

    def generate_signal(
        self,
        asset: str,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> AgentSignal:
        try:
            history = load_put_call_ratio(as_of_date)
        except Exception as exc:
            logger.debug("%s: CBOE lookup failed (%s) — holding.", self.agent_id, exc)
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="CBOE put/call ratio lookup failed — holding.",
            )

        if "put_call_ratio" not in history.columns:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="CBOE data missing a put/call ratio column — holding.",
            )

        series = history["put_call_ratio"].dropna().tail(WINDOW + 1)
        if len(series) < MIN_HISTORY_POINTS:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="insufficient put/call ratio history for a z-score read — holding.",
            )

        values = series.to_numpy(dtype=np.float64)
        baseline, latest = values[:-1], values[-1]
        std = float(np.std(baseline, ddof=1))
        z_score = 0.0 if std == 0.0 else float((latest - np.mean(baseline)) / std)
        sample_count = len(series)

        direction: Literal["long", "short", "flat"] = (
            "long" if z_score > 0.0 else "short" if z_score < 0.0 else "flat"
        )
        confidence = clamp(min(abs(z_score), Z_CAP) / Z_CAP)

        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction=direction,
            confidence=confidence,
            rationale=(
                f"put/call ratio z-score={z_score:.2f} — contrarian {direction} bias "
                f"[sample_count={sample_count}]"
            ),
        )
