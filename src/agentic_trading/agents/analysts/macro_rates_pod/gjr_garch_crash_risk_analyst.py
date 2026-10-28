# gjr_garch_crash_risk_analyst.py

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import PrivateAttr
from quantcore.risk import ConvergenceError
from quantcore.risk.egarch import GJRGARCHParams, gjr_garch_11_variance, gjr_garch_fit

from agentic_trading.agents._market_math import RefitCache, clamp, market_slice
from agentic_trading.agents.analysts.macro_rates_pod._garch_checks import (
    check_gjr_params,
    check_sigma_path,
    fit_or_convergence_error,
)
from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.schemas import AgentSignal, SignalStatus
from agentic_trading.config import settings
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)


class GJRGARCHCrashRiskAnalyst(AnalystAgent):
    """A narrow crash-risk detector, not a general directional view: fits
    GJR-GARCH(1,1) (`quantcore.risk.egarch.gjr_garch_fit`, Gaussian MLE)
    to the last `settings.analysts.risk_window_garch` returns of the
    asset, re-fitting at most once every `garch_refit_interval_days` per
    asset and computing the conditional-variance path daily from the
    cached parameters; a fit that raises `ConvergenceError` is cached the
    same way, so it is not retried before the interval ends. Only signals
    defensively when conditional vol has jumped above `vol_jump_threshold`
    times its trailing average, and that jump is attributable to the
    fitted asymmetric (bad-news) term rather than the symmetric ARCH term.

    Stays flat, with the failing check named in the rationale, whenever
    the window is shorter than `min_garch_observations`, the fit fails or
    does not converge (`ConvergenceError`), the parameters fail
    `check_gjr_params`, or the computed path fails `check_sigma_path`."""

    agent_id: str = "gjr_garch_crash_risk_analyst"
    _refit_cache: RefitCache = PrivateAttr(
        default_factory=lambda: RefitCache(
            refit_interval_days=settings.analysts.garch_refit_interval_days
        )
    )

    def _flat(
        self,
        asset: str,
        as_of_date: pd.Timestamp,
        rationale: str,
        status: SignalStatus = "active",
    ) -> AgentSignal:
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
        returns = market_slice(market_data, as_of_date, asset).log_returns[
            -settings.analysts.risk_window_garch :
        ]
        if returns.size < settings.analysts.min_garch_observations:
            return self._flat(
                asset, as_of_date, "insufficient history for GJR-GARCH fit", "unavailable"
            )

        try:
            fitted: GJRGARCHParams | ConvergenceError = self._refit_cache.get_or_refit(
                asset, as_of_date, lambda: fit_or_convergence_error(gjr_garch_fit, returns)
            )
        except (RuntimeError, ValueError) as exc:
            logger.debug(
                "%s: GJR-GARCH MLE fit failed for %s (%s) — returning flat.",
                self.agent_id,
                asset,
                exc,
            )
            return self._flat(asset, as_of_date, "GJR-GARCH MLE fit failed", "error")

        if isinstance(fitted, ConvergenceError):
            logger.debug(
                "%s: GJR-GARCH MLE fit did not converge for %s (%s) — returning flat.",
                self.agent_id,
                asset,
                fitted.optimizer_message,
            )
            return self._flat(
                asset,
                as_of_date,
                "GJR-GARCH parameters rejected: MLE fit did not converge",
                "unavailable",
            )

        sample_std = float(np.std(returns, ddof=1))
        failed_check = check_gjr_params(fitted, sample_std)
        if failed_check is not None:
            return self._flat(
                asset, as_of_date, f"GJR-GARCH parameters rejected: {failed_check}", "unavailable"
            )

        try:
            variance_path = gjr_garch_11_variance(
                returns, fitted.omega, fitted.alpha, fitted.gamma, fitted.beta
            )
        except (RuntimeError, ValueError, ZeroDivisionError) as exc:
            logger.debug(
                "%s: GJR-GARCH variance computation failed for %s (%s) — returning flat.",
                self.agent_id,
                asset,
                exc,
            )
            return self._flat(asset, as_of_date, "GJR-GARCH variance computation failed", "error")

        conditional_std = np.sqrt(np.maximum(variance_path, 0.0))
        failed_check = check_sigma_path(conditional_std, sample_std)
        if failed_check is not None:
            return self._flat(
                asset, as_of_date, f"GJR-GARCH path rejected: {failed_check}", "unavailable"
            )

        current_vol = float(conditional_std[-1])
        average_vol = float(np.sqrt(np.mean(variance_path)))
        jump_ratio = current_vol / average_vol if average_vol > 0 else 1.0

        last_shock = float(returns[-1])
        asymmetric_term = fitted.gamma * (last_shock**2) * (1.0 if last_shock < 0 else 0.0)
        symmetric_term = fitted.alpha * (last_shock**2)
        driven_by_bad_news = asymmetric_term > symmetric_term

        direction: Literal["long", "short", "flat"]
        if jump_ratio > settings.analysts.vol_jump_threshold and driven_by_bad_news:
            direction = "short"
            confidence = clamp(
                (jump_ratio - settings.analysts.vol_jump_threshold)
                / settings.analysts.vol_jump_threshold
            )
            rationale = f"vol jump ratio={jump_ratio:.2f}, driven by asymmetric bad-news term"
        else:
            direction = "flat"
            confidence = 0.0
            rationale = f"no crash-risk jump detected (vol jump ratio={jump_ratio:.2f})"

        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction=direction,
            confidence=confidence,
            rationale=rationale,
        )
