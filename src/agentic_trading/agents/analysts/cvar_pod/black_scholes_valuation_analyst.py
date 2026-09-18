# valuation_black_scholes.py

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import PrivateAttr
from quantcore.pricing import black_scholes_call, black_scholes_put

from agentic_trading.agents._market_math import (
    RefitCache,
    clamp,
    conditional_daily_volatility,
    market_slice,
)
from agentic_trading.agents.analysts.cvar_pod._options_data import (
    real_atm_options_by_expiration,
)
from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.config import settings
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

TRADING_DAYS_PER_YEAR = 252


class BlackScholesValuationAnalyst(AnalystAgent):
    """Risk-neutral volatility-divergence valuation: prices a short-dated
    at-the-money option on the asset via quantcore's closed-form
    Black-Scholes pricer with this analyst's own GARCH-fitted volatility,
    and compares it against the same option priced off the **real
    market-quoted ATM implied volatility** (nearest available expiration,
    `_options_data.real_atm_options_by_expiration`) — a model-vs-market
    read. Flat with zero confidence when no real snapshot exists for the
    date. The GARCH leg refits at most once every
    `_refit_cache.refit_interval_days` simulated days per asset — see
    `RefitCache`'s docstring."""

    agent_id: str = "black_scholes_valuation_analyst"

    _time_to_maturity: float = 0.25
    _refit_cache: RefitCache = PrivateAttr(default_factory=RefitCache)

    def _annualized_garch_volatility(
        self, returns: np.ndarray, refit_cache: RefitCache, cache_key: str, as_of_date: pd.Timestamp
    ) -> float:
        daily_volatility = conditional_daily_volatility(
            returns, refit_cache, cache_key, as_of_date, self.agent_id
        )
        return daily_volatility * float(np.sqrt(TRADING_DAYS_PER_YEAR))

    def generate_signal(
        self,
        asset: str,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> AgentSignal:
        slice_ = market_slice(market_data, as_of_date, asset)
        returns = slice_.log_returns[-settings.analysts.risk_window_short :]
        if slice_.close is None or returns.size < 2:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="insufficient history for quantcore Black-Scholes valuation",
                status="unavailable",
            )

        real_options = real_atm_options_by_expiration(asset, as_of_date, slice_.close)
        if real_options is None or real_options.empty:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="no real option-chain snapshot in the lakehouse for this date",
                status="unavailable",
            )
        benchmark_volatility = float(real_options.iloc[0]["implied_volatility"])
        benchmark_label = (
            f"real market ATM IV ({int(real_options.iloc[0]['days_to_expiration'])}d exp)"
        )

        mean_return = float(np.mean(returns))
        fitted_volatility = self._annualized_garch_volatility(
            returns, self._refit_cache, asset, as_of_date
        )

        fitted_call = black_scholes_call(
            slice_.close,
            slice_.close,
            mean_return,
            fitted_volatility,
            self._time_to_maturity,
        )
        benchmark_call = black_scholes_call(
            slice_.close,
            slice_.close,
            mean_return,
            benchmark_volatility,
            self._time_to_maturity,
        )
        fitted_put = black_scholes_put(
            slice_.close,
            slice_.close,
            mean_return,
            fitted_volatility,
            self._time_to_maturity,
        )
        benchmark_put = black_scholes_put(
            slice_.close,
            slice_.close,
            mean_return,
            benchmark_volatility,
            self._time_to_maturity,
        )
        # By put-call parity, call - put = S - K*e^{-rT} for any fixed
        # S/K/r/T regardless of volatility — since both legs here share the
        # same spot-equals-strike pair, (fitted_call - fitted_put) and
        # (benchmark_call - benchmark_put) are identically equal, so a
        # call-minus-put spread cancels to zero on every day. The call
        # price alone already carries the full vol-divergence signal (to
        # first order, fitted_call - benchmark_call ~= vega * (fitted_vol -
        # benchmark_vol)) — by the same parity identity, fitted_put -
        # benchmark_put is numerically identical to this, so using the put
        # leg instead would lose nothing and gain nothing.
        edge = fitted_call - benchmark_call
        direction: Literal["long", "short", "flat"] = "long" if edge >= 0.0 else "short"
        confidence = clamp(abs(edge) / max(slice_.close, 1e-12))
        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction=direction,
            confidence=confidence,
            rationale=(
                f"quantcore Black-Scholes edge: {settings.analysts.risk_window_short}d fitted vol="
                f"{fitted_volatility:.3f} vs {benchmark_label}={benchmark_volatility:.3f}, "
                f"call={fitted_call:.4f}/{benchmark_call:.4f}, "
                f"put={fitted_put:.4f}/{benchmark_put:.4f}"
            ),
        )
