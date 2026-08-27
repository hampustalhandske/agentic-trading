# greeks_delta_hedge_analyst.py

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import Field
from quantcore.pricing.greeks import bs_gamma

from agentic_trading.agents._market_math import clamp, market_slice
from agentic_trading.agents.analysts.cvar_pod._theoretical_options import (
    TIME_TO_MATURITY,
    annualized_garch_volatility,
    moneyness_strikes,
    risk_free_rate,
)
from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

MOMENTUM_WINDOW = 5
GAMMA_PINNING_THRESHOLD = 0.0


class GreeksDeltaHedgeAnalyst(AnalystAgent):
    """Model-implied gamma-exposure proxy, not a market-positioning read.

    Prices the shared theoretical moneyness grid and sums Black-Scholes
    gamma across it, weighted by `strike_weights` (default: equal-weighted,
    can be reconfigured to weight the ATM strike more heavily). There is no
    real open interest to weight by here (see
    `data/loaders/options_loader.py`), so this is not the open-interest-
    weighted "dealer GEX" concept from the options-flow literature — that
    fundamentally requires real chain data this project doesn't have access
    to historically.
    """

    agent_id: str = "greeks_delta_hedge_analyst"

    _risk_window: int = 60
    strike_weights: tuple[float, ...] = Field(default=(1.0, 1.0, 1.0, 1.0, 1.0))

    def generate_signal(
        self,
        asset: str,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> AgentSignal:
        slice_ = market_slice(market_data, as_of_date, asset)
        returns = slice_.log_returns[-self._risk_window :]
        if slice_.close is None or returns.size < 2:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="insufficient history for theoretical gamma-exposure proxy",
            )

        volatility = annualized_garch_volatility(returns, self.agent_id, asset)
        rate = risk_free_rate(as_of_date, self.agent_id)
        strikes = moneyness_strikes(slice_.close)
        gammas = [
            bs_gamma(slice_.close, strike, rate, volatility, TIME_TO_MATURITY) for strike in strikes
        ]
        weights = np.asarray(self.strike_weights[: len(gammas)], dtype=np.float64)
        aggregate_gamma = float(np.average(gammas, weights=weights))

        momentum = returns[-MOMENTUM_WINDOW:]
        momentum_sign: Literal["long", "short"] = (
            "long" if float(np.sum(momentum)) >= 0.0 else "short"
        )

        direction: Literal["long", "short", "flat"]
        if aggregate_gamma > GAMMA_PINNING_THRESHOLD:
            direction = "flat"
            confidence = clamp(1.0 - min(aggregate_gamma, 1.0))
            rationale = f"strongly positive model gamma exposure ({aggregate_gamma:.4f}) — pinning"
        else:
            direction = momentum_sign
            confidence = clamp(min(abs(aggregate_gamma), 1.0))
            rationale = (
                f"negative model gamma exposure ({aggregate_gamma:.4f}) — acceleration risk "
                f"in the direction of recent {MOMENTUM_WINDOW}-day momentum"
            )

        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction=direction,
            confidence=confidence,
            rationale=rationale,
        )
