# pairs_cointegration_analyst.py

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import Field, PrivateAttr
from quantcore.statistics.cointegration import spread_zscore

from agentic_trading.agents._market_math import clamp, market_slice
from agentic_trading.agents.base import AnalystAgent, DataRequirement
from agentic_trading.agents.schemas import AgentSignal, SignalStatus
from agentic_trading.config import settings
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

MIN_OVERLAP_DAYS = 30


def spread_direction(
    current_z: float, open_direction: Literal["long", "short"] | None
) -> Literal["long", "short", "flat"]:
    """Gatev-Goetzmann-Rouwenhorst entry/exit on a spread z-score: enter
    against the spread beyond `z_entry_threshold` (spread above its mean
    means the asset is rich, so short it), hold an open position while
    the spread is still outside `z_exit_threshold`, and go flat once it
    has reverted inside the exit band."""
    if current_z > settings.analysts.z_entry_threshold:
        return "short"
    if current_z < -settings.analysts.z_entry_threshold:
        return "long"
    if open_direction is not None and abs(current_z) > settings.analysts.z_exit_threshold:
        return open_direction
    return "flat"


# A position that is open (freshly entered or held inside the entry band
# but outside the exit band) always carries at least this much confidence,
# so the pod keeps the trade on until the spread actually reverts instead
# of closing it the day |z| drifts back under the entry threshold.
HELD_POSITION_MIN_CONFIDENCE = 0.25


def spread_confidence(current_z: float, direction: Literal["long", "short", "flat"]) -> float:
    """Confidence of a spread trade: 0 when flat; otherwise grows from
    `HELD_POSITION_MIN_CONFIDENCE` at the entry threshold toward 1.0 two
    sigma beyond it."""
    if direction == "flat":
        return 0.0
    beyond_entry = clamp((abs(current_z) - settings.analysts.z_entry_threshold) / 2.0)
    return max(HELD_POSITION_MIN_CONFIDENCE, beyond_entry)


class PairsCointegrationAnalyst(AnalystAgent):
    """Trades the asset's cointegration spread against its configured
    partner ticker(s): `spread = log_p(asset) - hedge_ratio * log_p(partner)`,
    z-scored over `settings.analysts.z_window`, entered against the
    spread beyond `z_entry_threshold`, held while outside
    `z_exit_threshold`, and flattened once it reverts inside it.

    Which pairs are cointegrated, and with what hedge ratio, is decided
    by `StatArbManager`'s walk-forward Engle-Granger re-screen, not here —
    it fills `partner_tickers_by_asset` and `hedge_ratio_by_pair` on this
    instance. Both are keyed per-asset (not one global partner list
    applied to every ticker uniformly) because a single shared instance
    covers every asset in a run, and different assets legitimately pair
    with different partners or with none at all.

    `hedge_ratio_by_pair` is keyed `(asset, partner)` and already oriented
    for that asset's spread: for the pair's dependent leg it is the
    Engle-Granger beta itself (`log_p(dep) - beta * log_p(other)`), for the
    other leg it is `1 / beta` (`log_p(other) - (1 / beta) * log_p(dep)`),
    so each side of a pair trades the same relationship from its own
    point of view. A pair without an entry falls back to a unit ratio."""

    agent_id: str = "pairs_cointegration_analyst"
    data_requirements: tuple[DataRequirement, ...] = (
        DataRequirement(
            source="sec_sic",
            identifier="{ticker}",
            asset_scoped=True,
            description="SEC industry classification (SIC code)",
        ),
    )

    partner_tickers_by_asset: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    hedge_ratio_by_pair: dict[tuple[str, str], float] = Field(default_factory=dict)
    _open_direction_by_asset: dict[str, Literal["long", "short"]] = PrivateAttr(
        default_factory=dict
    )

    def _flat(
        self,
        asset: str,
        as_of_date: pd.Timestamp,
        rationale: str,
        status: SignalStatus = "active",
    ) -> AgentSignal:
        self._open_direction_by_asset.pop(asset, None)
        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction="flat",
            confidence=0.0,
            rationale=rationale,
            status=status,
        )

    def generate_signal(
        self,
        asset: str,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> AgentSignal:
        partner_tickers = self.partner_tickers_by_asset.get(asset, ())
        if not partner_tickers:
            return self._flat(
                asset, as_of_date, f"no partner ticker(s) configured for {asset}", "unavailable"
            )

        asset_returns = market_slice(market_data, as_of_date, asset).log_returns[
            -settings.analysts.risk_window_long :
        ]
        partner_returns = [
            market_slice(market_data, as_of_date, ticker).log_returns[
                -settings.analysts.risk_window_long :
            ]
            for ticker in partner_tickers
        ]
        length = min(asset_returns.size, *(series.size for series in partner_returns))
        if length < MIN_OVERLAP_DAYS:
            return self._flat(
                asset,
                as_of_date,
                "insufficient overlapping history for a cointegration spread",
                "unavailable",
            )

        asset_prices = np.cumsum(asset_returns[-length:])
        hedged_partner_prices = [
            self.hedge_ratio_by_pair.get((asset, ticker), 1.0) * np.cumsum(series[-length:])
            for ticker, series in zip(partner_tickers, partner_returns, strict=True)
        ]
        spread = asset_prices - np.mean(hedged_partner_prices, axis=0)

        z_window = min(settings.analysts.z_window, spread.size)
        current_z = float(spread_zscore(spread, z_window)[-1])
        direction = spread_direction(current_z, self._open_direction_by_asset.get(asset))
        if direction == "flat":
            self._open_direction_by_asset.pop(asset, None)
        else:
            self._open_direction_by_asset[asset] = direction
        hedge_ratios = ", ".join(
            f"{ticker}={self.hedge_ratio_by_pair.get((asset, ticker), 1.0):.3f}"
            for ticker in partner_tickers
        )

        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction=direction,
            confidence=spread_confidence(current_z, direction),
            rationale=f"cointegration spread vs {hedge_ratios}, z={current_z:.2f}",
        )
