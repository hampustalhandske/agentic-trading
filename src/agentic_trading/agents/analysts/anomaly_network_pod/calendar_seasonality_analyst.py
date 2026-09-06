# calendar_seasonality_analyst.py

from typing import Literal

import numpy as np
import pandas as pd

from agentic_trading.agents._market_math import clamp
from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

MIN_SAMPLES = 20
TURN_OF_MONTH_WINDOW_DAYS = 3
SANTA_RALLY_START_DAY = 27
JANUARY_EFFECT_END_DAY = 5
T_STAT_CAP = 3.0


def _asset_history(market_data: pd.DataFrame, as_of_date: pd.Timestamp, asset: str) -> pd.DataFrame:
    history = market_data.loc[market_data.index <= as_of_date]
    if isinstance(history.columns, pd.MultiIndex):
        if asset not in history.columns.get_level_values(0):
            return pd.DataFrame()
        sliced = history.xs(asset, axis=1, level=0)
        return sliced.to_frame() if isinstance(sliced, pd.Series) else sliced
    return history


class CalendarSeasonalityAnalyst(AnalystAgent):
    """Pure price-derived, no new data source: day-of-week, turn-of-month,
    January-effect, and Santa-Claus-rally anomalies from the documented
    calendar-effect literature, computed directly from the asset's own
    price history already in `market_data`. A special calendar regime
    (turn-of-month / January / Santa rally) for the next trading day takes
    priority over the plain day-of-week effect when it applies, since the
    literature treats those as distinct, typically stronger effects;
    otherwise falls back to the day-of-week mean. Requires at least
    `MIN_SAMPLES` historical observations of the matched regime, else
    holds flat — a calendar effect estimated from a handful of days isn't
    a real signal, only noise that happens to have a label."""

    agent_id: str = "calendar_seasonality_analyst"

    def _regime_mask(
        self, index: pd.DatetimeIndex, next_day: pd.Timestamp
    ) -> tuple[str, pd.Series]:
        month, day = next_day.month, next_day.day
        days_in_month = next_day.days_in_month

        if month == 12 and day >= SANTA_RALLY_START_DAY:
            mask = (index.month == 12) & (index.day >= SANTA_RALLY_START_DAY)
            return "santa-claus-rally", pd.Series(mask, index=index)
        if month == 1 and day <= JANUARY_EFFECT_END_DAY:
            mask = (index.month == 1) & (index.day <= JANUARY_EFFECT_END_DAY)
            return "january-effect", pd.Series(mask, index=index)
        if day <= TURN_OF_MONTH_WINDOW_DAYS or day >= days_in_month - TURN_OF_MONTH_WINDOW_DAYS + 1:
            mask = (index.day <= TURN_OF_MONTH_WINDOW_DAYS) | (
                index.day >= index.days_in_month - TURN_OF_MONTH_WINDOW_DAYS + 1
            )
            return "turn-of-month", pd.Series(mask, index=index)

        mask = index.dayofweek == next_day.dayofweek
        return "day-of-week", pd.Series(mask, index=index)

    def generate_signal(
        self,
        asset: str,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> AgentSignal:
        history = _asset_history(market_data, as_of_date, asset)
        if "log_return" not in history.columns:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="no price history available — holding.",
            )

        returns = history["log_return"].dropna()
        next_day = as_of_date + pd.tseries.offsets.BDay(1)
        regime_name, mask = self._regime_mask(pd.DatetimeIndex(returns.index), next_day)
        matched = returns.loc[mask.reindex(returns.index, fill_value=False)]
        sample_count = len(matched)

        if sample_count < MIN_SAMPLES:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale=(
                    f"only {sample_count} historical {regime_name} observations — "
                    f"below the {MIN_SAMPLES} required — holding. [sample_count={sample_count}]"
                ),
            )

        values = matched.to_numpy(dtype=np.float64)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1))
        t_stat = 0.0 if std == 0.0 else mean / (std / np.sqrt(sample_count))

        direction: Literal["long", "short", "flat"] = (
            "long" if mean > 0.0 else "short" if mean < 0.0 else "flat"
        )
        confidence = clamp(min(abs(t_stat), T_STAT_CAP) / T_STAT_CAP)

        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction=direction,
            confidence=confidence,
            rationale=(
                f"{regime_name} effect: mean log-return={mean:.5f} over {sample_count} "
                f"historical observations (t={t_stat:.2f}) [sample_count={sample_count}]"
            ),
        )
