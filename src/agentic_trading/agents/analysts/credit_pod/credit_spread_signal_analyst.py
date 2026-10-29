# credit_spread_signal_analyst.py

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import Field

from agentic_trading.agents._market_math import clamp
from agentic_trading.agents.base import AnalystAgent, DataRequirement, calendar_days_for_sessions
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.data.loaders.lakehouse_loader import load_fred_vintages_from_lakehouse
from agentic_trading.data.loaders.macro_loader import BAMLH0A0HYM2
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

_RISK_WINDOW = 120
_MIN_OBSERVATIONS = 20


class CreditSpreadSignalAnalyst(AnalystAgent):
    """Market-wide credit-risk-appetite signal, not a single-issuer read:
    z-scores the ICE BofA US High Yield option-adjusted spread
    (`BAMLH0A0HYM2`, FRED) against its own trailing window, and treats a
    widening spread (deteriorating credit conditions — a real "risk-off"
    signal institutional credit desks watch) as bearish for whatever
    credit-sensitive asset it's attached to (a high-yield bond ETF is the
    natural target — see the Credit pod's own wiring in `roster.py`), a
    tightening spread as bullish. This is a market-wide overlay, not a
    per-issuer credit read — no free source publishes per-issuer credit
    ratings or CDS spreads (see this project's CLAUDE.md, data-sources
    section). `BAMLH0A0HYM2` itself only has free history back to
    2023-09-18 (an ICE licensing change, not a fetch bug — confirmed
    directly against FRED); this analyst is flat before that date
    regardless of `as_of_date`, not silently wrong. Every active signal's
    `source_as_of` is the newest `known_at` (the later of a vintage's
    publication date and its observation date) among the spread vintages
    in the trailing window."""

    agent_id: str = "credit_spread_signal_analyst"
    data_requirements: tuple[DataRequirement, ...] = Field(
        default_factory=lambda: (
            DataRequirement(
                source="fred",
                identifier=BAMLH0A0HYM2,
                asset_scoped=False,
                description="FRED BAMLH0A0HYM2",
                lookback_days=calendar_days_for_sessions(_RISK_WINDOW),
            ),
        )
    )

    def generate_signal(
        self,
        asset: str,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> AgentSignal:
        vintages = load_fred_vintages_from_lakehouse(BAMLH0A0HYM2, as_of_date.date())
        if vintages is None:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="no FRED BAMLH0A0HYM2 coverage in the lakehouse",
                status="unavailable",
            )
        used = vintages.dropna(subset=["value"]).iloc[-_RISK_WINDOW:]
        window = used["value"].to_numpy(dtype=np.float64)
        if window.size < _MIN_OBSERVATIONS:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="insufficient credit-spread history for a z-score read",
                status="unavailable",
            )
        source_as_of = used["known_at"].max().date()

        mean = float(np.mean(window))
        std = float(np.std(window, ddof=1))
        if std <= 0.0:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="zero trailing variance in the credit-spread window",
                source_as_of=source_as_of,
            )

        z_score = (float(window[-1]) - mean) / std
        direction: Literal["long", "short", "flat"]
        if z_score > 0.0:
            direction = "short"
        elif z_score < 0.0:
            direction = "long"
        else:
            direction = "flat"
        confidence = clamp(abs(z_score) / 3.0)

        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction=direction,
            confidence=confidence,
            rationale=(
                f"HY OAS spread z-score={z_score:.2f} vs its own {_RISK_WINDOW}d trailing window"
            ),
            source_as_of=source_as_of,
        )
