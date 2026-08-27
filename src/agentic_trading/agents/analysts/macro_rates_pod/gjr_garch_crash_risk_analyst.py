# gjr_garch_crash_risk_analyst.py

from typing import Literal

import numpy as np
import pandas as pd
from quantcore.risk.egarch import gjr_garch_11_variance

from agentic_trading.agents._market_math import clamp, market_slice
from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

# quantcore has no GJR-GARCH MLE fitter — these are documented fixed
# parameters, not a fitted estimate (same honest-approximation pattern as
# EGARCHAsymmetricVolAnalyst). A real fitter would be quantcore-shaped
# future work.
FIXED_OMEGA = 0.05
FIXED_ALPHA = 0.05
FIXED_GAMMA = 0.1
FIXED_BETA = 0.85
VOL_JUMP_THRESHOLD = 1.5


class GJRGARCHCrashRiskAnalyst(AnalystAgent):
    """A narrow crash-risk detector, not a general directional view: only
    signals defensively when conditional vol has jumped materially above
    its trailing average, and that jump is attributable to the asymmetric
    (bad-news) term specifically rather than the symmetric ARCH term."""

    agent_id: str = "gjr_garch_crash_risk_analyst"

    _risk_window: int = 60

    def generate_signal(
        self,
        asset: str,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> AgentSignal:
        returns = market_slice(market_data, as_of_date, asset).log_returns[-self._risk_window :]
        if returns.size < 10:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="insufficient history for GJR-GARCH crash-risk read",
            )

        try:
            variance_path = gjr_garch_11_variance(
                returns, FIXED_OMEGA, FIXED_ALPHA, FIXED_GAMMA, FIXED_BETA
            )
        except (RuntimeError, ValueError) as exc:
            logger.debug(
                "%s: GJR-GARCH variance computation failed for %s (%s) — returning flat.",
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
                rationale="GJR-GARCH variance computation failed",
            )

        current_vol = float(np.sqrt(max(variance_path[-1], 0.0)))
        average_vol = float(np.sqrt(np.mean(variance_path)))
        jump_ratio = current_vol / average_vol if average_vol > 0 else 1.0

        last_shock = float(returns[-1])
        asymmetric_term = FIXED_GAMMA * (last_shock**2) * (1.0 if last_shock < 0 else 0.0)
        symmetric_term = FIXED_ALPHA * (last_shock**2)
        driven_by_bad_news = asymmetric_term > symmetric_term

        direction: Literal["long", "short", "flat"]
        if jump_ratio > VOL_JUMP_THRESHOLD and driven_by_bad_news:
            direction = "short"
            confidence = clamp((jump_ratio - VOL_JUMP_THRESHOLD) / VOL_JUMP_THRESHOLD)
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
