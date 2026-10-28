# cir_rate_regime_analyst.py

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import PrivateAttr
from quantcore.core.stochastic import cir_fit, simulate_cir_paths

from agentic_trading.agents._market_math import RefitCache, clamp, derived_seed
from agentic_trading.agents.base import AnalystAgent, DataRequirement
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.config import settings
from agentic_trading.data.loaders.lakehouse_loader import load_fred_series_from_lakehouse
from agentic_trading.data.loaders.macro_loader import DGS3MO
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

SIMULATION_HORIZON_DAYS = 30
# Daily FRED observations -> annualized CIR time unit, matching
# simulate_cir_paths' own t=SIMULATION_HORIZON_DAYS/365.0 convention below.
_DAILY_DT_YEARS = 1.0 / 365.0


class CIRRateRegimeAnalyst(AnalystAgent):
    """Not itself a single-asset price-direction read — its role is a
    rates-regime signal, still emitted as an AgentSignal on `asset`.
    Fits the 3-month T-bill's own CIR(kappa, theta, sigma) parameters
    directly from the rate series (`quantcore.core.stochastic.cir_fit`,
    a CKLS/GLS moment estimator), simulates the rate forward via CIR,
    and derives direction from whether simulated future rates trend up
    or down — scaled by `rate_sensitivity` (assumed uniform across
    assets unless configured otherwise, a deliberate simplification).
    The fit is only re-run at most once every
    `_refit_cache.refit_interval_days` simulated days per asset — see
    `RefitCache`'s docstring."""

    agent_id: str = "cir_rate_regime_analyst"

    rate_sensitivity: float = -1.0
    _refit_cache: RefitCache = PrivateAttr(default_factory=RefitCache)
    data_requirements: tuple[DataRequirement, ...] = (
        DataRequirement(
            source="fred", identifier=DGS3MO, asset_scoped=False, description="FRED DGS3MO"
        ),
    )

    def generate_signal(
        self,
        asset: str,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> AgentSignal:
        rate_history = load_fred_series_from_lakehouse(DGS3MO, as_of_date.date())
        if rate_history is None:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="no FRED DGS3MO coverage in the lakehouse",
                status="unavailable",
            )
        rate_series = rate_history.dropna().to_numpy(dtype=np.float64)

        rates = rate_series[-settings.analysts.risk_window_long :] / 100.0
        if rates.size < 30:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="insufficient rate history for a CIR regime read",
                status="unavailable",
            )

        try:
            fitted = self._refit_cache.get_or_refit(
                asset, as_of_date, lambda: cir_fit(rates, _DAILY_DT_YEARS)
            )
        except (ValueError, RuntimeError) as exc:
            logger.debug(
                "%s: cir_fit could not fit a mean-reverting CIR process to the trailing rate "
                "window for %s (%s) — the series may be trending rather than reverting, which "
                "is expected during sustained hiking/cutting cycles; returning flat.",
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
                rationale="rate series is trending, not mean-reverting — no CIR regime read",
            )
        kappa, theta, sigma = fitted.kappa, fitted.theta, fitted.sigma
        r0 = float(rates[-1])

        simulated = simulate_cir_paths(
            r0,
            kappa,
            theta,
            sigma,
            t=SIMULATION_HORIZON_DAYS / 365.0,
            n_paths=500,
            n_steps=SIMULATION_HORIZON_DAYS,
            seed=derived_seed(self.agent_id, asset, as_of_date.date()),
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

        sample_sigma = float(np.std(rates, ddof=1))
        confidence = clamp(abs(rate_trend) / max(sample_sigma, 1e-8))
        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction=direction,
            confidence=confidence,
            rationale=(
                f"CIR-simulated {SIMULATION_HORIZON_DAYS}d rate trend={rate_trend:.4f} "
                f"(kappa={kappa:.3f}, theta={theta:.4f}, sensitivity={self.rate_sensitivity})"
            ),
        )
