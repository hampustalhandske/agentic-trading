# ou_mean_reversion_analyst.py

import math
from typing import Literal

import numpy as np
import pandas as pd
from quantcore.core.stochastic import simulate_ou_paths
from quantcore.statistics.cointegration import ou_half_life, spread_zscore

from agentic_trading.agents._market_math import clamp, market_slice
from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

MIN_HALF_LIFE_DAYS = 1.0
MAX_ZSCORE_WINDOW = 250
Z_ENTRY_THRESHOLD = 1.0


class OUMeanReversionAnalyst(AnalystAgent):
    """De-trends the asset's own log price (or a configurable benchmark
    spread) and signals from how far the resulting spread has drifted from
    its own mean, checked against a simulated OU path's directional
    agreement."""

    agent_id: str = "ou_mean_reversion_analyst"

    benchmark: str | None = None
    _risk_window: int = 250

    def _build_spread(
        self, market_data: pd.DataFrame, as_of_date: pd.Timestamp, asset: str
    ) -> np.ndarray:
        asset_slice = market_slice(market_data, as_of_date, asset)
        log_prices = np.cumsum(asset_slice.log_returns)
        if self.benchmark is None:
            window = min(len(log_prices), self._risk_window)
            if window < 2:
                return np.array([])
            trailing = log_prices[-window:]
            return trailing - float(np.mean(trailing))

        benchmark_slice = market_slice(market_data, as_of_date, self.benchmark)
        benchmark_log_prices = np.cumsum(benchmark_slice.log_returns)
        length = min(len(log_prices), len(benchmark_log_prices), self._risk_window)
        if length < 2:
            return np.array([])
        return log_prices[-length:] - benchmark_log_prices[-length:]

    def generate_signal(
        self,
        asset: str,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> AgentSignal:
        spread = self._build_spread(market_data, as_of_date, asset)
        if spread.size < 10:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="insufficient history to build an OU spread",
            )

        half_life = max(float(ou_half_life(spread)), MIN_HALF_LIFE_DAYS)
        z_window = int(min(max(round(4 * half_life), 5), MAX_ZSCORE_WINDOW, spread.size))
        z_scores = spread_zscore(spread, z_window)
        current_z = float(z_scores[-1])

        direction: Literal["long", "short", "flat"]
        if current_z < -Z_ENTRY_THRESHOLD:
            direction = "long"
        elif current_z > Z_ENTRY_THRESHOLD:
            direction = "short"
        else:
            direction = "flat"
        confidence = clamp((abs(current_z) - Z_ENTRY_THRESHOLD) / 2.0)

        if direction != "flat":
            theta = math.log(2) / half_life
            mu = float(np.mean(spread))
            sigma = float(np.std(spread, ddof=1))
            simulated = simulate_ou_paths(
                theta, mu, sigma, spread[-1], t=half_life * 4, n_paths=500, n_steps=50, seed=42
            )
            reverted_toward_mean = simulated[:, -1] - spread[-1]
            simulated_direction_agrees = (
                direction == "long" and float(np.mean(reverted_toward_mean > 0)) > 0.5
            ) or (direction == "short" and float(np.mean(reverted_toward_mean < 0)) > 0.5)
            if not simulated_direction_agrees:
                confidence = 0.0

        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction=direction,
            confidence=confidence,
            rationale=f"OU spread z-score={current_z:.2f}, half-life={half_life:.1f}d",
        )
