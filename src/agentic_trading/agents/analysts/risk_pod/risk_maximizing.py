# risk_maximizing.py

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from quantcore.risk import conditional_value_at_risk, fit_garch_11, garch_11_variance, value_at_risk

from agentic_trading.agents._market_math import clamp, market_slice
from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)


class RiskMaximizingAnalyst(AnalystAgent):
    """Favors assets with higher GARCH-modeled volatility — proposes
    directional views that seek out volatility as opportunity rather than
    treating it as something to avoid."""

    agent_id: str = "risk_maximizing_analyst"

    _risk_window: int = 60
    _confidence_level: float = 0.95

    def generate_signal(
        self,
        asset: str,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> AgentSignal:
        slice_ = market_slice(market_data, as_of_date, asset)
        returns = slice_.log_returns[-self._risk_window :]
        direction: Literal["long", "short", "flat"]
        if returns.size < 2:
            direction = "flat"
            confidence = 0.5
            rationale = "insufficient history for quantcore risk estimate"
        else:
            mean_return = float(np.mean(returns))
            volatility = float(np.std(returns, ddof=1))
            try:
                omega, alpha, beta = fit_garch_11(returns)
                centered = returns - mean_return
                variance_path = garch_11_variance(centered, omega, alpha, beta)
                volatility = float(np.sqrt(max(float(variance_path[-1]), 0.0)))
            except (RuntimeError, ValueError) as exc:
                logger.debug(
                    "%s: GARCH fit failed for %s on %s (%s) — falling back to sample stdev "
                    "volatility=%.4f.",
                    self.agent_id,
                    asset,
                    as_of_date.date(),
                    exc,
                    volatility,
                )

            try:
                var = value_at_risk(returns, self._confidence_level, volatility=volatility)
                cvar = conditional_value_at_risk(
                    returns, self._confidence_level, volatility=volatility
                )
            except ValueError as exc:
                logger.debug(
                    "%s: VaR/CVaR computation failed for %s on %s (%s) — falling back to "
                    "volatility=%.4f for both.",
                    self.agent_id,
                    asset,
                    as_of_date.date(),
                    exc,
                    volatility,
                )
                var = volatility
                cvar = volatility

            tail_risk = max(0.0, float(var)) + max(0.0, float(cvar)) + volatility
            direction = "long" if mean_return >= 0.0 else "short"
            confidence = clamp(min(1.0, tail_risk))
            rationale = (
                f"quantcore high-tail-risk bias (VaR={var:.4f}, "
                f"CVaR={cvar:.4f}, vol={volatility:.4f})"
            )
        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction=direction,
            confidence=confidence,
            rationale=rationale,
        )
