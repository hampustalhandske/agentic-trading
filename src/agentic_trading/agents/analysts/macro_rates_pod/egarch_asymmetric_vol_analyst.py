# egarch_asymmetric_vol_analyst.py

import math
from typing import Literal

import numpy as np
import pandas as pd
from quantcore.risk.egarch import egarch_11_variance

from agentic_trading.agents._market_math import clamp, market_slice
from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

# quantcore has no EGARCH MLE fitter — these are documented fixed parameters
# (a reasonable, stable EGARCH(1,1) parameterization with positive gamma to
# express the leverage effect) rather than a fitted estimate. A real fitter
# for this model would be quantcore-shaped future work, not something to
# reimplement here.
FIXED_OMEGA = -0.1
FIXED_ALPHA = 0.15
FIXED_GAMMA = 0.1
FIXED_BETA = 0.9
EXPECTED_ABS_STANDARD_NORMAL = math.sqrt(2.0 / math.pi)


class EGARCHAsymmetricVolAnalyst(AnalystAgent):
    """Signals from the sign and magnitude of the EGARCH leverage term's
    realized contribution on the most recent observation — a strongly
    negative recent shock combined with the fixed positive gamma term
    signals elevated near-term downside vol."""

    agent_id: str = "egarch_asymmetric_vol_analyst"

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
                rationale="insufficient history for EGARCH asymmetric-vol read",
            )

        try:
            variance_path = egarch_11_variance(
                returns, FIXED_OMEGA, FIXED_ALPHA, FIXED_GAMMA, FIXED_BETA
            )
        except (RuntimeError, ValueError) as exc:
            logger.debug(
                "%s: EGARCH variance computation failed for %s (%s) — returning flat.",
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
                rationale="EGARCH variance computation failed",
            )

        conditional_std = np.sqrt(np.maximum(variance_path, 1e-12))
        z_last = float(returns[-1] / conditional_std[-1])
        leverage_contribution = FIXED_GAMMA * z_last

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
