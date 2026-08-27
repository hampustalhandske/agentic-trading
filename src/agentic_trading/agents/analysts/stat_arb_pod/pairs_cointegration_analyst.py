# pairs_cointegration_analyst.py

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import Field
from quantcore.statistics.cointegration import (
    adf_test,
    engle_granger_test,
    johansen_trace_test,
    spread_zscore,
)

from agentic_trading.agents._market_math import clamp, market_slice
from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

Z_ENTRY_THRESHOLD = 1.0
Z_WINDOW = 60


class PairsCointegrationAnalyst(AnalystAgent):
    """Tests the asset against a configurable list of candidate partner
    tickers for cointegration (Engle-Granger for 2 series, Johansen for
    3+), only signaling when the test rejects no-cointegration at
    `significance_level`."""

    agent_id: str = "pairs_cointegration_analyst"

    partner_tickers: tuple[str, ...] = Field(default_factory=tuple)
    significance_level: float = 0.05
    _risk_window: int = 250

    def generate_signal(
        self,
        asset: str,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> AgentSignal:
        if not self.partner_tickers:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="no partner tickers configured",
            )

        asset_returns = market_slice(market_data, as_of_date, asset).log_returns[
            -self._risk_window :
        ]
        partner_returns = [
            market_slice(market_data, as_of_date, ticker).log_returns[-self._risk_window :]
            for ticker in self.partner_tickers
        ]
        length = min(asset_returns.size, *(series.size for series in partner_returns))
        if length < 30:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="insufficient overlapping history for a cointegration test",
            )

        asset_prices = np.cumsum(asset_returns[-length:])
        partner_prices = [np.cumsum(series[-length:]) for series in partner_returns]

        _, asset_p_value = adf_test(asset_prices)
        if asset_p_value < self.significance_level:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale=(
                    f"{asset} already looks stationary (ADF p={asset_p_value:.3f}) — no "
                    "cointegration framing applies"
                ),
            )

        if len(partner_prices) == 1:
            _, p_value, _ = engle_granger_test(asset_prices, partner_prices[0])
            cointegrated = p_value < self.significance_level
            spread = asset_prices - partner_prices[0]
            test_detail = f"Engle-Granger p={p_value:.3f}"
        else:
            system = np.column_stack([asset_prices, *partner_prices])
            trace_stats, critical_values, _ = johansen_trace_test(system)
            cointegrated = bool(trace_stats[0] > critical_values[0])
            spread = asset_prices - np.mean(partner_prices, axis=0)
            test_detail = f"Johansen trace={trace_stats[0]:.3f} vs crit={critical_values[0]:.3f}"

        if not cointegrated:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale=f"no cointegration detected ({test_detail})",
            )

        z_window = min(Z_WINDOW, spread.size)
        current_z = float(spread_zscore(spread, z_window)[-1])
        direction: Literal["long", "short", "flat"]
        if current_z < -Z_ENTRY_THRESHOLD:
            direction = "long"
        elif current_z > Z_ENTRY_THRESHOLD:
            direction = "short"
        else:
            direction = "flat"
        confidence = clamp((abs(current_z) - Z_ENTRY_THRESHOLD) / 2.0)

        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction=direction,
            confidence=confidence,
            rationale=(
                f"cointegrated with {self.partner_tickers} ({test_detail}), "
                f"spread z={current_z:.2f}"
            ),
        )
