# egarch_asymmetric_vol_analyst.py

import math
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import PrivateAttr
from quantcore.risk.egarch import EGARCHParams, egarch_11_variance, egarch_fit

from agentic_trading.agents._market_math import RefitCache, clamp, market_slice
from agentic_trading.agents.analysts.macro_rates_pod._garch_checks import (
    check_egarch_params,
    check_sigma_path,
)
from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.schemas import AgentSignal, SignalStatus
from agentic_trading.config import settings
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

EXPECTED_ABS_STANDARD_NORMAL = math.sqrt(2.0 / math.pi)


class EGARCHAsymmetricVolAnalyst(AnalystAgent):
    """Fits EGARCH(1,1) (`quantcore.risk.egarch.egarch_fit`, Gaussian MLE)
    to the last `settings.analysts.risk_window_garch` returns of the asset,
    re-fitting at most once every `garch_refit_interval_days` per asset
    and computing the conditional-variance path daily from the cached
    parameters. Signals from the sign and magnitude of the fitted leverage
    term's realized contribution on the most recent observation — a
    strongly negative recent shock combined with a positive fitted gamma
    signals elevated near-term downside vol.

    Stays flat, with the failing check named in the rationale, whenever
    the window is shorter than `min_garch_observations`, the fit fails or
    does not converge, the parameters fail `check_egarch_params`, or the
    computed path fails `check_sigma_path`."""

    agent_id: str = "egarch_asymmetric_vol_analyst"
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
                asset, as_of_date, "insufficient history for EGARCH fit", "unavailable"
            )

        try:
            fitted: EGARCHParams = self._refit_cache.get_or_refit(
                asset, as_of_date, lambda: egarch_fit(returns)
            )
        except (RuntimeError, ValueError) as exc:
            logger.debug(
                "%s: EGARCH MLE fit failed for %s (%s) — returning flat.",
                self.agent_id,
                asset,
                exc,
            )
            return self._flat(asset, as_of_date, "EGARCH MLE fit failed", "error")

        sample_std = float(np.std(returns, ddof=1))
        failed_check = check_egarch_params(fitted, sample_std)
        if failed_check is not None:
            return self._flat(
                asset, as_of_date, f"EGARCH parameters rejected: {failed_check}", "unavailable"
            )

        try:
            log_variance_path = egarch_11_variance(
                returns, fitted.omega, fitted.alpha, fitted.gamma, fitted.beta
            )
        except (RuntimeError, ValueError, ZeroDivisionError) as exc:
            logger.debug(
                "%s: EGARCH variance computation failed for %s (%s) — returning flat.",
                self.agent_id,
                asset,
                exc,
            )
            return self._flat(asset, as_of_date, "EGARCH variance computation failed", "error")

        # egarch_11_variance returns ln(sigma_t^2) — exponentiate before
        # taking sqrt.
        with np.errstate(over="ignore"):
            conditional_std = np.sqrt(np.exp(log_variance_path))
        failed_check = check_sigma_path(conditional_std, sample_std)
        if failed_check is not None:
            return self._flat(
                asset, as_of_date, f"EGARCH path rejected: {failed_check}", "unavailable"
            )

        z_last = float(returns[-1] / conditional_std[-1])
        leverage_contribution = fitted.gamma * z_last

        direction: Literal["long", "short", "flat"]
        if leverage_contribution < 0.0:
            direction = "short"
            confidence = clamp(abs(z_last) - EXPECTED_ABS_STANDARD_NORMAL)
        else:
            direction = "flat"
            confidence = 0.0

        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction=direction,
            confidence=confidence,
            rationale=(
                f"EGARCH leverage contribution gamma*z_t-1={leverage_contribution:.4f} "
                f"(z_t-1={z_last:.3f})"
            ),
        )
