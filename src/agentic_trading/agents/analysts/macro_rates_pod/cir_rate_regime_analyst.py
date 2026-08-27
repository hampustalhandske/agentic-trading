# cir_rate_regime_analyst.py

import math
from typing import Literal

import numpy as np
import pandas as pd
from quantcore.core.stochastic import simulate_cir_paths
from quantcore.statistics.cointegration import ou_half_life

from agentic_trading.agents._market_math import clamp
from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.data.loaders.macro_loader import DGS3MO, load_fred_series
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

MIN_HALF_LIFE_DAYS = 1.0
SIMULATION_HORIZON_DAYS = 30


class CIRRateRegimeAnalyst(AnalystAgent):
    """Not itself a single-asset price-direction read — its role is a
    rates-regime signal, still emitted as an AgentSignal on `asset`.
    Estimates the 3-month T-bill's mean-reversion speed by applying the
    OU/AR(1) half-life estimator to the rate series (quantcore has no
    CIR-specific fitter), simulates the rate forward via CIR, and derives
    direction from whether simulated future rates trend up or down —
    scaled by `rate_sensitivity` (assumed uniform across assets unless
    configured otherwise, a deliberate simplification)."""

    agent_id: str = "cir_rate_regime_analyst"

    rate_sensitivity: float = -1.0
    _risk_window: int = 250

    def generate_signal(
        self,
        asset: str,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> AgentSignal:
        try:
            rate_series = load_fred_series(DGS3MO, as_of_date).dropna().to_numpy(dtype=np.float64)
        except Exception as exc:
            logger.debug("%s: FRED DGS3MO lookup failed (%s) — returning flat.", self.agent_id, exc)
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="FRED DGS3MO lookup failed",
            )

        rates = rate_series[-self._risk_window :] / 100.0
        if rates.size < 30:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="insufficient rate history for a CIR regime read",
            )

        half_life = max(float(ou_half_life(rates)), MIN_HALF_LIFE_DAYS)
        kappa = math.log(2) / half_life
        theta = float(np.mean(rates))
        sigma = float(np.std(rates, ddof=1))
        r0 = float(rates[-1])

        simulated = simulate_cir_paths(
            r0,
            kappa,
            theta,
            sigma,
            t=SIMULATION_HORIZON_DAYS / 365.0,
            n_paths=500,
            n_steps=SIMULATION_HORIZON_DAYS,
            seed=42,
        )
        simulated_terminal = float(np.mean(simulated[:, -1]))
        rate_trend = simulated_terminal - r0

        raw_direction = 1.0 if rate_trend >= 0.0 else -1.0
        signed = raw_direction * self.rate_sensitivity
        direction: Literal["long", "short", "flat"]
        if signed > 0:
            direction = "long"
        elif signed < 0:
            direction = "short"
        else:
            direction = "flat"

        confidence = clamp(abs(rate_trend) / max(sigma, 1e-8))
        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction=direction,
            confidence=confidence,
            rationale=(
                f"CIR-simulated {SIMULATION_HORIZON_DAYS}d rate trend={rate_trend:.4f} "
                f"(half-life={half_life:.1f}d, sensitivity={self.rate_sensitivity})"
            ),
        )
