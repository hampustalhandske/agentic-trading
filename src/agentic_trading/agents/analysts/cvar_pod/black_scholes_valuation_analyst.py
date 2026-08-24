# valuation_black_scholes.py

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from quantcore.pricing import black_scholes_call, black_scholes_put
from quantcore.risk import fit_garch_11, garch_11_variance

from agentic_trading.agents._market_math import clamp, market_slice
from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)


class BlackScholesValuationAnalyst(AnalystAgent):
    """Risk-neutral volatility-divergence valuation: prices a short-dated
    at-the-money option on the asset via quantcore's closed-form
    Black-Scholes pricer, once using GARCH-fitted (realized) volatility
    and once using a fixed benchmark volatility, and treats the
    divergence between the two prices as a mispricing signal — proposes
    a directional view from whichever volatility regime the option price
    implies relative to the benchmark, rather than a direct market-price-
    vs-fair-value comparison."""

    agent_id: str = "black_scholes_valuation_analyst"

    _risk_window: int = 60
    _time_to_maturity: float = 0.25
    _benchmark_volatility: float = 0.20

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
                confidence=0.5,
                rationale="insufficient history for quantcore Black-Scholes valuation",
            )

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

        fitted_call = black_scholes_call(
            slice_.close,
            slice_.close,
            mean_return,
            volatility,
            self._time_to_maturity,
        )
        benchmark_call = black_scholes_call(
            slice_.close,
            slice_.close,
            mean_return,
            self._benchmark_volatility,
            self._time_to_maturity,
        )
        fitted_put = black_scholes_put(
            slice_.close,
            slice_.close,
            mean_return,
            volatility,
            self._time_to_maturity,
        )
        benchmark_put = black_scholes_put(
            slice_.close,
            slice_.close,
            mean_return,
            self._benchmark_volatility,
            self._time_to_maturity,
        )
        edge = (fitted_call - benchmark_call) - (fitted_put - benchmark_put)
        direction: Literal["long", "short", "flat"] = "long" if edge >= 0.0 else "short"
        confidence = clamp(abs(edge) / max(slice_.close, 1e-12))
        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction=direction,
            confidence=confidence,
            rationale=(
                "quantcore Black-Scholes edge "
                f"call={fitted_call:.4f}/{benchmark_call:.4f}, "
                f"put={fitted_put:.4f}/{benchmark_put:.4f}"
            ),
        )
