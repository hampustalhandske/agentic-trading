# vix_tail_hedge_analyst.py

import pandas as pd

from agentic_trading.agents._market_math import clamp
from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.config import settings
from agentic_trading.data.loaders.lakehouse_loader import load_price_history_from_lakehouse
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

VIX_TICKER = "^VIX"


class VixTailHedgeAnalyst(AnalystAgent):
    """A systematic tail-risk insurance signal, not a directional call.
    Reads `^VIX`'s own trailing price history directly from the
    lakehouse — independent of whatever asset it's asked about, the
    same cross-asset lookup pattern `FamaMacbethAnalyst` already uses
    for its factor universe — and produces a `short` signal on the
    asset it's given (always SPY in practice; see
    `TailHedgeManager.assets`), scaled by how many standard deviations
    above its own trailing mean VIX's current level sits. Protection
    is always on: confidence floors at
    `settings.analysts.tail_hedge_base_confidence` when VIX is calm (a
    standing insurance sleeve) and rises toward 1.0 as VIX becomes
    genuinely elevated — never flips to `long`; this sleeve only ever
    hedges, it doesn't try to time a VIX mean-reversion trade. Holds
    flat only when there isn't enough VIX history to place the current
    level in its own regime."""

    agent_id: str = "vix_tail_hedge_analyst"

    def generate_signal(
        self,
        asset: str,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> AgentSignal:
        window_start = (as_of_date - pd.Timedelta(days=settings.analysts.vix_lookback_days)).date()
        frame = load_price_history_from_lakehouse([VIX_TICKER], window_start, as_of_date.date())
        if frame is None or VIX_TICKER not in frame.columns.get_level_values(0):
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="no ^VIX price history in the lakehouse",
                status="unavailable",
            )

        vix_close = frame[(VIX_TICKER, "close")].dropna()
        if vix_close.size < settings.analysts.min_vix_observations:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="insufficient ^VIX history for a trailing z-score",
                status="unavailable",
            )

        mean = float(vix_close.mean())
        std = float(vix_close.std(ddof=1))
        current_level = float(vix_close.iloc[-1])
        z_score = 0.0 if std < 1e-8 else (current_level - mean) / std
        stress_confidence = clamp(max(z_score, 0.0) / settings.analysts.vix_zscore_full_confidence)
        confidence = max(settings.analysts.tail_hedge_base_confidence, stress_confidence)

        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction="short",
            confidence=confidence,
            rationale=(
                f"^VIX at {current_level:.1f}, z-score={z_score:.2f} vs its own trailing mean"
            ),
        )
