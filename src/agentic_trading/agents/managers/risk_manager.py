# risk_manager.py

from __future__ import annotations

import numpy as np
import pandas as pd
from quantcore.risk import fit_garch_11, garch_11_variance, value_at_risk

from agentic_trading.agents._market_math import clamp, market_slice
from agentic_trading.agents.base import ManagerAgent
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.data.loaders.tiingo_loader import load_tiingo_price_history
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

_TIINGO_DIVERGENCE_SCALE = 0.02
"""Relative price divergence (Tiingo close vs. the primary yfinance/Stooq
close) at which the Tiingo cross-check factor roughly halves confidence —
2% is well outside ordinary vendor rounding/timing noise between two
independent EOD sources, so a divergence this large signals an actual
data-quality problem in one of them rather than routine discrepancy."""


class RiskManager(ManagerAgent):
    """Scales analyst confidence down for assets with elevated quantcore
    VaR/CVaR or an unfavorable GARCH volatility regime, and closer to 1.0
    for calm assets. Also cross-checks the primary price against Tiingo's
    independent, IEX-sourced close (`data/loaders/tiingo_loader.py`) and
    dampens confidence further when the two diverge — a divergence is a
    data-quality signal in its own right, not just another price source
    to fall back to (see CLAUDE.md's data-sources section). Does not
    decide direction — only how much confidence an analyst's signal
    deserves once risk and data quality are accounted for."""

    agent_id: str = "risk_manager"

    _risk_window: int = 60
    _confidence_level: float = 0.95

    def _tiingo_scale_factor(
        self,
        signal: AgentSignal,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> float:
        try:
            tiingo_history = load_tiingo_price_history(signal.asset, as_of_date)
        except Exception as exc:
            logger.debug(
                "%s: Tiingo cross-check lookup failed for %s (%s) — no divergence dampening "
                "applied.",
                self.agent_id,
                signal.asset,
                exc,
            )
            return 1.0
        if tiingo_history.empty:
            return 1.0

        primary_close = market_slice(market_data, as_of_date, signal.asset).close
        if primary_close is None or primary_close == 0.0:
            return 1.0

        tiingo_close = float(tiingo_history["close"].iloc[-1])
        relative_divergence = abs(tiingo_close - primary_close) / primary_close
        return clamp(1.0 / (1.0 + relative_divergence / _TIINGO_DIVERGENCE_SCALE))

    def _risk_load(
        self, signal: AgentSignal, market_data: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> float:
        slice_ = market_slice(market_data, as_of_date, signal.asset)
        returns = slice_.log_returns[-self._risk_window :]
        if returns.size < 2:
            return 0.0

        volatility = float(np.std(returns, ddof=1))
        try:
            omega, alpha, beta = fit_garch_11(returns)
            centered = returns - float(np.mean(returns))
            variance_path = garch_11_variance(centered, omega, alpha, beta)
            volatility = float(np.sqrt(max(float(variance_path[-1]), 0.0)))
        except (RuntimeError, ValueError) as exc:
            logger.debug(
                "%s: GARCH fit failed for %s on %s (%s) — falling back to sample stdev "
                "volatility=%.4f.",
                self.agent_id,
                signal.asset,
                as_of_date.date(),
                exc,
                volatility,
            )

        try:
            var = value_at_risk(returns, self._confidence_level, volatility=volatility)
        except ValueError as exc:
            logger.debug(
                "%s: VaR computation failed for %s on %s (%s) — falling back to volatility=%.4f.",
                self.agent_id,
                signal.asset,
                as_of_date.date(),
                exc,
                volatility,
            )
            var = volatility

        return max(0.0, float(var)) + volatility

    def scale_factor(
        self,
        signal: AgentSignal,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> float:
        """Computes this manager's dampening multiplier for one analyst
        signal, using quantcore's VaR and GARCH volatility regime for
        `signal.asset` as of `as_of_date`, further dampened by any
        Tiingo/primary-source price divergence. Must be >= 0."""
        risk_load = self._risk_load(signal, market_data, as_of_date)
        risk_factor = clamp(1.0 / (1.0 + risk_load))
        tiingo_factor = self._tiingo_scale_factor(signal, market_data, as_of_date)
        return risk_factor * tiingo_factor
