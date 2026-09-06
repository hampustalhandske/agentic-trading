# retail_attention_analyst.py

import numpy as np
import pandas as pd

from agentic_trading.agents._market_math import clamp
from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.data.loaders.trends_loader import load_search_interest
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

MIN_HISTORY_POINTS = 14
Z_CAP = 3.0


class RetailAttentionAnalyst(AnalystAgent):
    """Google Trends search-interest via `data/loaders/trends_loader.py`:
    an unusual spike in retail search attention for the ticker (a z-score
    on the latest interest reading vs. its own trailing history) is read
    as a retail-attention/momentum proxy — the literature on search-volume
    spikes (e.g. Da, Engelberg & Gao 2011) finds they precede short-term
    positive returns, so this analyst only ever signals `long` on a
    positive spike (`flat` otherwise) rather than treating low search
    interest as a bearish signal, which the literature doesn't support."""

    agent_id: str = "retail_attention_analyst"

    def generate_signal(
        self,
        asset: str,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> AgentSignal:
        try:
            interest = load_search_interest(asset, as_of_date)
        except Exception as exc:
            logger.debug(
                "%s: Google Trends lookup failed for %s (%s) — holding.",
                self.agent_id,
                asset,
                exc,
            )
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="Google Trends lookup failed — holding.",
            )

        series = interest["interest"].dropna() if "interest" in interest.columns else pd.Series()
        if len(series) < MIN_HISTORY_POINTS:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="insufficient Google Trends history for a z-score read — holding.",
            )

        values = series.to_numpy(dtype=np.float64)
        baseline, latest = values[:-1], values[-1]
        std = float(np.std(baseline, ddof=1))
        z_score = 0.0 if std == 0.0 else float((latest - np.mean(baseline)) / std)
        latest_date = series.index[-1]
        sample_count = len(series)

        if z_score <= 0.0:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale=(
                    f"search interest not elevated (z={z_score:.2f}) "
                    f"[source_as_of={pd.Timestamp(latest_date).date().isoformat()}] "
                    f"[sample_count={sample_count}]"
                ),
            )

        confidence = clamp(min(z_score, Z_CAP) / Z_CAP)
        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction="long",
            confidence=confidence,
            rationale=(
                f"search interest spike (z={z_score:.2f}) — retail attention long bias "
                f"[source_as_of={pd.Timestamp(latest_date).date().isoformat()}] "
                f"[sample_count={sample_count}]"
            ),
        )
