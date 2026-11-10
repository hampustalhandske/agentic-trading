# valuation_black_scholes.py

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import Field, PrivateAttr
from quantcore.pricing import black_scholes_call, black_scholes_put

from agentic_trading.agents._market_math import (
    RefitCache,
    clamp,
    conditional_daily_volatility,
    market_slice,
    raw_spot,
)
from agentic_trading.agents.analysts.cvar_pod._options_data import (
    chain_snapshot_date,
    options_chain_requirements,
    real_atm_options_by_expiration,
)
from agentic_trading.agents.base import AnalystAgent, DataRequirement
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.config import settings
from agentic_trading.data.loaders.lakehouse_loader import load_fred_vintages_from_lakehouse
from agentic_trading.data.loaders.macro_loader import DGS3MO
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
    read. Spot and strike are the unadjusted close
    (`_market_math.raw_spot`), the basis option strikes are quoted on.
    Both legs price at the risk-free rate r, the latest DGS3MO value
    known on the decision day (`load_fred_vintages_from_lakehouse`),
    converted from percent to an annual decimal rate.

    Unavailable (flat, zero confidence) when no real snapshot or no
    close exists for the date, when the fitted or the chain volatility
    is not a positive finite number, or when no DGS3MO value is known.
    The GARCH leg refits at most once every
    `_refit_cache.refit_interval_days` simulated days per asset — see
    `RefitCache`'s docstring.

    Every active signal's `source_as_of` is the newer of the date of the
    chain snapshot it priced off (`_options_data.chain_snapshot_date`,
    the snapshot's own date rather than the later decision date the
    publication lag first makes it visible on) and the `known_at` of the
    DGS3MO value r came from."""

    agent_id: str = "black_scholes_valuation_analyst"
    # No `lookback_days` on DGS3MO: the latest value is read whatever its
    # age, so there is no fixed window short of the warm-up.
    data_requirements: tuple[DataRequirement, ...] = Field(
        default_factory=lambda: (
            *options_chain_requirements(),
            DataRequirement(
                source="fred", identifier=DGS3MO, asset_scoped=False, description="FRED DGS3MO"
            ),
        )
    )

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
        returns = market_slice(market_data, as_of_date, asset).log_returns[
            -settings.analysts.risk_window_short :
        ]
        spot = raw_spot(market_data, as_of_date, asset)
        if spot is None or returns.size < 2:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="insufficient history for quantcore Black-Scholes valuation",
                status="unavailable",
            )

        real_options = real_atm_options_by_expiration(asset, as_of_date, spot)
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
        snapshot_date = chain_snapshot_date(real_options)

        fitted_volatility = self._annualized_garch_volatility(
            returns, self._refit_cache, asset, as_of_date
        )
        for label, volatility in (("fitted", fitted_volatility), ("chain", benchmark_volatility)):
            if not (math.isfinite(volatility) and volatility > 0.0):
                return self.unavailable_signal(
                    asset,
                    as_of_date,
                    f"{label} volatility {volatility} is not a positive finite number",
                    source_as_of=snapshot_date,
                )

        vintages = load_fred_vintages_from_lakehouse(DGS3MO, as_of_date.date())
        known_rates = None if vintages is None else vintages.dropna(subset=["value"])
        if known_rates is None or known_rates.empty:
            return self.unavailable_signal(
                asset,
                as_of_date,
                "no FRED DGS3MO value known on this date for the risk-free rate",
                source_as_of=snapshot_date,
            )
        latest_rate = known_rates.iloc[-1]
        risk_free_rate = float(latest_rate["value"]) / 100.0
        source_as_of = max(snapshot_date, pd.Timestamp(latest_rate["known_at"]).date())

        fitted_call = black_scholes_call(
            spot,
            spot,
            risk_free_rate,
            fitted_volatility,
            self._time_to_maturity,
        )
        benchmark_call = black_scholes_call(
            spot,
            spot,
            risk_free_rate,
            benchmark_volatility,
            self._time_to_maturity,
        )
        fitted_put = black_scholes_put(
            spot,
            spot,
            risk_free_rate,
            fitted_volatility,
            self._time_to_maturity,
        )
        benchmark_put = black_scholes_put(
            spot,
            spot,
            risk_free_rate,
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
        confidence = clamp(abs(edge) / max(spot, 1e-12))
        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction=direction,
            confidence=confidence,
            source_as_of=source_as_of,
            rationale=(
                f"quantcore Black-Scholes edge: {settings.analysts.risk_window_short}d fitted vol="
                f"{fitted_volatility:.3f} vs {benchmark_label}={benchmark_volatility:.3f}, "
                f"r={risk_free_rate:.4f}, "
                f"call={fitted_call:.4f}/{benchmark_call:.4f}, "
                f"put={fitted_put:.4f}/{benchmark_put:.4f}"
            ),
        )
