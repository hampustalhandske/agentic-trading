# kalman_hedge_ratio_analyst.py

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import Field, PrivateAttr
from quantcore.filtering.kalman import dynamic_hedge_ratio
from quantcore.statistics.cointegration import spread_zscore

from agentic_trading.agents._market_math import RefitCache, market_slice
from agentic_trading.agents.analysts.stat_arb_pod.pairs_cointegration_analyst import (
    spread_confidence,
    spread_direction,
)
from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.config import settings
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)


class KalmanHedgeRatioAnalyst(AnalystAgent):
    """Same partner-ticker spread idea as PairsCointegrationAnalyst, but
    the hedge ratio is time-varying via a Kalman filter rather than a
    static OLS beta — the spread is built from the *latest filtered*
    hedge ratio, which is this analyst's differentiator. Refit at most
    once every `_refit_cache.refit_interval_days` simulated days per
    asset — see `RefitCache`'s docstring. Entry/exit follow the pod's
    shared `spread_direction` rule: enter against the spread beyond
    `z_entry_threshold`, hold while outside `z_exit_threshold`, flat once
    it reverts inside.

    `partner_ticker_by_asset` is keyed per-asset, same reasoning as
    `PairsCointegrationAnalyst.partner_tickers_by_asset` — a single
    shared instance covers every asset in a run, and different assets
    pair with different partners or none at all."""

    agent_id: str = "kalman_hedge_ratio_analyst"

    partner_ticker_by_asset: dict[str, str] = Field(default_factory=dict)
    _refit_cache: RefitCache = PrivateAttr(default_factory=RefitCache)
    _open_direction_by_asset: dict[str, Literal["long", "short"]] = PrivateAttr(
        default_factory=dict
    )

    def generate_signal(
        self,
        asset: str,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> AgentSignal:
        partner_ticker = self.partner_ticker_by_asset.get(asset)
        if partner_ticker is None:
            self._open_direction_by_asset.pop(asset, None)
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale=f"no partner ticker configured for {asset}",
                status="unavailable",
            )

        y_returns = market_slice(market_data, as_of_date, asset).log_returns[
            -settings.analysts.risk_window_long :
        ]
        x_returns = market_slice(market_data, as_of_date, partner_ticker).log_returns[
            -settings.analysts.risk_window_long :
        ]
        length = min(y_returns.size, x_returns.size)
        if length < 30:
            self._open_direction_by_asset.pop(asset, None)
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="insufficient overlapping history for a Kalman hedge ratio",
                status="unavailable",
            )

        y_prices = np.cumsum(y_returns[-length:])
        x_prices = np.cumsum(x_returns[-length:])
        hedge_ratios = self._refit_cache.get_or_refit(
            asset, as_of_date, lambda: dynamic_hedge_ratio(y_prices, x_prices)
        )
        latest_beta = float(hedge_ratios[-1])
        spread = y_prices - latest_beta * x_prices

        z_window = min(settings.analysts.z_window, spread.size)
        current_z = float(spread_zscore(spread, z_window)[-1])

        direction = spread_direction(current_z, self._open_direction_by_asset.get(asset))
        if direction == "flat":
            self._open_direction_by_asset.pop(asset, None)
        else:
            self._open_direction_by_asset[asset] = direction
        confidence = spread_confidence(current_z, direction)

        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction=direction,
            confidence=confidence,
            rationale=(
                f"Kalman hedge ratio vs {partner_ticker}={latest_beta:.3f}, "
                f"spread z={current_z:.2f}"
            ),
        )
