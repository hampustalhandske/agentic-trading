# fama_french_factor_analyst.py

from typing import Literal

import numpy as np
import pandas as pd
from quantcore.statistics.regression import factor_loadings

from agentic_trading.agents._market_math import clamp, market_slice
from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.data.loaders.factor_loader import load_fama_french_factors
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

RECENT_FACTOR_WINDOW = 21
T_STAT_CONFIDENCE_SCALE = 4.0


class FamaFrenchFactorAnalyst(AnalystAgent):
    """Regresses the asset's trailing excess returns on Fama-French factors
    and signals from whether the asset's tilt toward the strongest recent
    factor is positive and statistically significant. `confidence` is set
    from the loading's Newey-West t-stat per the convention this pod's
    manager (`FactorManager`) relies on: `clamp(abs(t_stat) / 4.0)`."""

    agent_id: str = "fama_french_factor_analyst"

    model: str = "3factor"
    _risk_window: int = 120

    def generate_signal(
        self,
        asset: str,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> AgentSignal:
        returns = market_slice(market_data, as_of_date, asset).log_returns[-self._risk_window :]
        try:
            factors = load_fama_french_factors(as_of_date, model=self.model)
        except Exception as exc:
            logger.debug("%s: factor download failed (%s) — returning flat.", self.agent_id, exc)
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="Fama-French factor download failed",
            )

        factor_columns = [column for column in factors.columns if column != "RF"]
        length = min(returns.size, len(factors))
        if length < 30:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="insufficient overlapping history for a factor regression",
            )

        aligned_factors = factors.iloc[-length:]
        excess_returns = returns[-length:] - aligned_factors["RF"].to_numpy(dtype=np.float64)
        factor_matrix = aligned_factors[factor_columns].to_numpy(dtype=np.float64)

        betas, t_stats = factor_loadings(excess_returns, factor_matrix)

        recent_factor_returns = aligned_factors[factor_columns].iloc[-RECENT_FACTOR_WINDOW:].sum()
        strongest_factor_index = int(np.argmax(np.abs(recent_factor_returns.to_numpy())))
        strongest_beta = float(betas[strongest_factor_index])
        strongest_t_stat = float(t_stats[strongest_factor_index])
        strongest_factor_name = factor_columns[strongest_factor_index]
        recent_factor_sign = (
            1.0 if recent_factor_returns.iloc[strongest_factor_index] >= 0 else -1.0
        )

        direction: Literal["long", "short", "flat"] = "flat"
        confidence = clamp(abs(strongest_t_stat) / T_STAT_CONFIDENCE_SCALE)
        if confidence > 0.0:
            direction = "long" if (strongest_beta * recent_factor_sign) >= 0 else "short"

        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction=direction,
            confidence=confidence,
            rationale=(
                f"tilt toward {strongest_factor_name}: beta={strongest_beta:.3f}, "
                f"t={strongest_t_stat:.2f}"
            ),
        )
