# earnings_surprise_analyst.py

from typing import Literal

import pandas as pd

from agentic_trading.agents._market_math import clamp
from agentic_trading.agents.base import AnalystAgent, DataRequirement
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.data.loaders.lakehouse_loader import load_earnings_surprise_from_lakehouse
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

# Post-earnings-announcement drift (PEAD) is one of the most-replicated
# academic anomalies — a real, sizable earnings surprise keeps
# predicting returns for weeks after the announcement, not just on the
# day itself (Bernard & Thomas 1989 and a large replication literature
# since). This window is deliberately wide relative to the underlying
# surprise's own real availability lag, not a same-day-only read.
_DRIFT_WINDOW_DAYS = 60
_MIN_ESTIMATE_MAGNITUDE = 0.01
_SURPRISE_SCALE = 0.10  # a +/-10% EPS surprise maps to full confidence


class EarningsSurpriseAnalyst(AnalystAgent):
    """Post-earnings-announcement drift (PEAD): a genuine earnings
    surprise (real reported EPS vs. the real consensus estimate as of
    the announcement, `dolthub_eps_history`/`dolthub_earnings_calendar`
    via `data/loaders/lakehouse_loader.py`) predicts continued drift in
    the same direction for weeks afterward, not just an immediate
    reaction — one of the most-replicated anomalies in the academic
    factor literature, which is why this lives in the Factor pod
    alongside Fama-French/Fama-MacBeth rather than Event/Credit.

    Point-in-time: the announcement date used to gate this is the real
    date from `dolthub_earnings_calendar`, not an estimated lag — see
    the loader's own docstring. Confidence decays linearly to zero as
    `as_of_date` moves from the announcement date to
    `_DRIFT_WINDOW_DAYS` afterward, so the signal fades out rather than
    persisting at full strength indefinitely."""

    agent_id: str = "earnings_surprise_analyst"
    data_requirements: tuple[DataRequirement, ...] = (
        DataRequirement(
            source="earnings",
            identifier="all",
            asset_scoped=False,
            description="Earnings announcement calendar",
            lookback_days=_DRIFT_WINDOW_DAYS,
        ),
    )

    def generate_signal(
        self,
        asset: str,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> AgentSignal:
        result = load_earnings_surprise_from_lakehouse(asset, as_of_date.date())
        if result is None:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="no real earnings-surprise coverage available",
                status="unavailable",
            )

        reported, estimate, announcement_date = result
        days_since_announcement = (as_of_date - announcement_date).days
        if days_since_announcement > _DRIFT_WINDOW_DAYS:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale=(
                    f"most recent earnings surprise is {days_since_announcement}d old — "
                    f"beyond the {_DRIFT_WINDOW_DAYS}d PEAD window"
                ),
            )

        if abs(estimate) < _MIN_ESTIMATE_MAGNITUDE:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="consensus estimate too close to zero for a stable surprise ratio",
                status="unavailable",
            )

        surprise_ratio = (reported - estimate) / abs(estimate)
        direction: Literal["long", "short", "flat"]
        if surprise_ratio > 0.0:
            direction = "long"
        elif surprise_ratio < 0.0:
            direction = "short"
        else:
            direction = "flat"

        drift_decay = clamp(1.0 - days_since_announcement / _DRIFT_WINDOW_DAYS)
        confidence = clamp(abs(surprise_ratio) / _SURPRISE_SCALE) * drift_decay

        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction=direction,
            confidence=confidence,
            rationale=(
                f"EPS surprise={surprise_ratio:+.1%} (reported={reported:.2f} vs "
                f"estimate={estimate:.2f}), {days_since_announcement}d post-announcement"
            ),
        )
