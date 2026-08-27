# heston_vol_surface_analyst.py

from typing import Literal

import numpy as np
import pandas as pd
from quantcore.core.stochastic import simulate_heston_paths

from agentic_trading.agents._market_math import clamp, market_slice
from agentic_trading.agents.analysts.cvar_pod._theoretical_options import (
    annualized_garch_volatility,
)
from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

TERM_STRUCTURE_WINDOWS: tuple[int, ...] = (10, 30, 60, 90)
MIN_WINDOW_OBSERVATIONS = TERM_STRUCTURE_WINDOWS[-1] + 1
CALIBRATION_PATHS = 200
CALIBRATION_STEPS = 30
TRADING_DAYS_PER_YEAR = 252
VOL_DIVERGENCE_THRESHOLD = 0.02


def _realized_vol_term_structure(returns: np.ndarray) -> np.ndarray:
    values = []
    for window in TERM_STRUCTURE_WINDOWS:
        windowed = returns[-window:]
        values.append(float(np.std(windowed, ddof=1)) * float(np.sqrt(TRADING_DAYS_PER_YEAR)))
    return np.asarray(values, dtype=np.float64)


def _simulated_vol_term_structure(
    v0: float, kappa: float, theta: float, xi: float, rho: float, spot: float
) -> np.ndarray:
    _, variance_paths = simulate_heston_paths(
        spot,
        v0,
        0.0,
        kappa,
        theta,
        xi,
        rho,
        t=1.0,
        n_paths=CALIBRATION_PATHS,
        n_steps=CALIBRATION_STEPS,
        seed=42,
    )
    log_returns = np.diff(np.log(np.maximum(variance_paths, 1e-12)), axis=1)
    values = []
    for window in TERM_STRUCTURE_WINDOWS:
        horizon_steps = max(1, round(window / 365.0 * CALIBRATION_STEPS))
        horizon_steps = min(horizon_steps, log_returns.shape[1])
        segment = log_returns[:, :horizon_steps]
        values.append(float(np.std(segment)) * float(np.sqrt(TRADING_DAYS_PER_YEAR)))
    return np.asarray(values, dtype=np.float64)


def _calibrate_heston(
    realized_term_structure: np.ndarray, spot: float
) -> tuple[dict[str, float], float]:
    from scipy.optimize import minimize

    initial_variance = float(realized_term_structure[0]) ** 2

    def objective(params: np.ndarray) -> float:
        kappa, theta, xi, rho = params
        simulated = _simulated_vol_term_structure(
            initial_variance, abs(kappa), abs(theta), abs(xi), np.clip(rho, -0.99, 0.99), spot
        )
        return float(np.sum((simulated - realized_term_structure) ** 2))

    result = minimize(
        objective,
        x0=np.array([1.5, initial_variance, 0.3, -0.5]),
        method="Nelder-Mead",
        options={"maxiter": 50, "xatol": 1e-3, "fatol": 1e-6},
    )
    kappa, theta, xi, rho = result.x
    params = {
        "v0": initial_variance,
        "kappa": abs(float(kappa)),
        "theta": abs(float(theta)),
        "xi": abs(float(xi)),
        "rho": float(np.clip(rho, -0.99, 0.99)),
    }
    return params, float(result.fun)


class HestonVolSurfaceAnalyst(AnalystAgent):
    """Compares two *model* volatility regimes, not model-vs-market.

    There is no real market IV skew to calibrate against historically (see
    `data/loaders/options_loader.py`), so this analyst instead calibrates
    Heston parameters against the asset's own historical realized-vol term
    structure (10/30/60/90-day trailing realized vol, treated as a
    synthetic term structure) and compares the resulting long-run
    (theta-implied) forward vol against the short-horizon GARCH read from
    `_theoretical_options`. This reads whether the model's own term
    structure implies rising or falling forward vol relative to the
    current GARCH read — it is not a market-skew signal.
    """

    agent_id: str = "heston_vol_surface_analyst"

    _risk_window: int = 90

    def generate_signal(
        self,
        asset: str,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> AgentSignal:
        slice_ = market_slice(market_data, as_of_date, asset)
        returns = slice_.log_returns[-self._risk_window :]
        if slice_.close is None or returns.size < MIN_WINDOW_OBSERVATIONS:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="insufficient history to fit a realized-vol term structure",
            )

        realized_term_structure = _realized_vol_term_structure(returns)
        garch_vol = annualized_garch_volatility(returns, self.agent_id, asset)

        try:
            params, residual = _calibrate_heston(realized_term_structure, slice_.close)
        except Exception as exc:
            logger.debug(
                "%s: Heston calibration failed for %s (%s) — returning flat.",
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
                rationale="Heston term-structure calibration failed",
            )

        heston_forward_vol = float(np.sqrt(max(params["theta"], 0.0)))
        divergence = heston_forward_vol - garch_vol

        direction: Literal["long", "short", "flat"]
        if divergence > VOL_DIVERGENCE_THRESHOLD:
            direction = "short"
        elif divergence < -VOL_DIVERGENCE_THRESHOLD:
            direction = "long"
        else:
            direction = "flat"

        fit_quality = clamp(1.0 / (1.0 + residual))
        confidence = clamp(fit_quality * min(abs(divergence) / VOL_DIVERGENCE_THRESHOLD, 1.0))

        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction=direction,
            confidence=confidence,
            rationale=(
                f"Heston forward vol {heston_forward_vol:.4f} vs GARCH read {garch_vol:.4f} "
                f"(calibration residual={residual:.6f})"
            ),
        )
