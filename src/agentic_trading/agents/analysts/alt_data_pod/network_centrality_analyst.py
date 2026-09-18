# lead_lag_network_centrality_analyst.py

from datetime import timedelta
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import Field, PrivateAttr

from agentic_trading.agents._market_math import RefitCache, clamp, load_fixed_universe_returns
from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.data.loaders.lakehouse_loader import (
    load_price_history_from_lakehouse,
    load_sp500_constituents_asof,
)
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

# Emergency fallback only — used when the lakehouse (real, point-in-time
# S&P 500 membership, hundreds of names) isn't reachable. A 5-name graph
# is not a real cross-section for a network-centrality read and should
# never be what actually runs once the lakehouse is available — see
# `fama_macbeth_analyst.py`'s identical fallback for the same reasoning.
DEFAULT_UNIVERSE: tuple[str, ...] = ("AAPL", "MSFT", "GOOGL", "AMZN", "JPM")
WINDOW = 60
LAG = 1
RETURN_CAP = 0.05
# A broad universe will always have some tickers fail to load on any
# given day; below this many usable names the cross-section isn't broad
# enough to be worth preferring over the fixed fallback.
_MIN_UNIVERSE_SIZE = 30
_UNIVERSE_HISTORY_DAYS = 250


class LeadLagNetworkCentralityAnalyst(AnalystAgent):
    """The second analyst (after `fama_macbeth_analyst`) that legitimately
    needs more than one ticker's data: builds a rolling lag-1 cross-
    correlation graph across a universe — edge (i, j) is the correlation
    between asset i's return at t-1 and asset j's return at t, i.e. how
    much i "leads" j — and derives two things from it for `asset`: a
    network-centrality read (the mean absolute lead-lag correlation
    between `asset` and every other node, a simple degree-centrality
    proxy — no graph library needed, since this reduces to a row/column
    average over a correlation matrix) used for confidence, and a
    directional forecast (each other asset's most recent return, weighted
    by how strongly it historically leads `asset`). Distinct methodology
    from the factor_pod regressions: this is a graph/network-science read
    of the cross-sectional return structure, not a linear factor model.

    The universe is resolved fresh each call (refit at
    `_universe_refit_cache.refit_interval_days`) as the real, point-in-time
    S&P 500 membership plus `asset` — the same "real membership, fixed
    list as an emergency fallback only" pattern `fama_macbeth_analyst.py`
    uses, and for the same reason: a 5-name graph is not a real
    cross-section. Falls back to the fixed `universe` field (via
    `load_fixed_universe_returns()`, independent of whatever assets the
    current backtest's own `market_data` happens to include) only when
    the lakehouse can't resolve real membership or too few names have
    usable history."""

    agent_id: str = "lead_lag_network_centrality_analyst"

    universe: tuple[str, ...] = Field(default=DEFAULT_UNIVERSE)
    _universe_refit_cache: RefitCache = PrivateAttr(default_factory=RefitCache)

    def _load_broad_lakehouse_universe(
        self, as_of_date: pd.Timestamp
    ) -> dict[str, np.ndarray] | None:
        """Real, point-in-time S&P 500 membership plus a single batch
        lakehouse query for all of their trailing returns — the expensive
        fetch `_universe_refit_cache` gates, cached under one universe-
        wide key so every asset this pod scores on the same day (and
        within the same refit interval) shares one fetch, the same shape
        `fama_macbeth_analyst.py`'s identical helper uses. Returns `None`
        (not raises) on any failure or a too-thin resulting cross-section,
        so a cache miss during a lakehouse outage degrades to the fixed-
        universe fallback for `refit_interval_days` rather than retrying
        daily. Deliberately does not force any specific asset into the
        membership list — an asset real S&P membership doesn't include
        is resolved by the caller falling back to the fixed universe for
        that one asset, not by mutating the shared cached fetch."""
        members = load_sp500_constituents_asof(as_of_date.date())
        if members is None:
            return None
        window_start = (as_of_date - timedelta(days=_UNIVERSE_HISTORY_DAYS)).date()
        frame = load_price_history_from_lakehouse(list(members), window_start, as_of_date.date())
        if frame is None:
            return None
        candidate_returns = {
            str(ticker): frame[(ticker, "log_return")].dropna().to_numpy()
            for ticker in frame.columns.get_level_values(0).unique()
        }
        usable = {
            ticker: series
            for ticker, series in candidate_returns.items()
            if series.size >= WINDOW + LAG
        }
        if len(usable) < _MIN_UNIVERSE_SIZE:
            return None
        return usable

    def generate_signal(
        self,
        asset: str,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> AgentSignal:
        try:
            broad = self._universe_refit_cache.get_or_refit(
                "broad_universe",
                as_of_date,
                lambda: self._load_broad_lakehouse_universe(as_of_date),
            )
        except (RuntimeError, ValueError) as exc:
            logger.debug(
                "%s: broad lakehouse universe fetch failed (%s) — falling back to the fixed "
                "%d-name universe.",
                self.agent_id,
                exc,
                len(self.universe),
            )
            broad = None

        if broad is not None and asset in broad:
            universe = tuple(broad)
            returns = broad
        else:
            universe = self.universe if asset in self.universe else (*self.universe, asset)
            try:
                returns = load_fixed_universe_returns(universe, as_of_date)
            except Exception as exc:
                logger.debug(
                    "%s: universe price history download failed (%s) — holding.",
                    self.agent_id,
                    exc,
                )
                return AgentSignal(
                    agent_id=self.agent_id,
                    asset=asset,
                    as_of_date=as_of_date.date(),
                    direction="flat",
                    confidence=0.0,
                    rationale="universe price history lookup failed",
                    status="error",
                )

            if len(returns) < len(universe):
                return AgentSignal(
                    agent_id=self.agent_id,
                    asset=asset,
                    as_of_date=as_of_date.date(),
                    direction="flat",
                    confidence=0.0,
                    rationale=f"insufficient price history for part of universe {universe}",
                    status="unavailable",
                )

        min_length = min(series.size for series in returns.values())
        if min_length < WINDOW + LAG or asset not in returns:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale=(
                    f"insufficient cross-sectional history across universe {universe} — holding."
                ),
                status="unavailable",
            )

        aligned = {ticker: series[-(WINDOW + LAG) :] for ticker, series in returns.items()}
        asset_current = aligned[asset][LAG:]

        lead_lag_corr: dict[str, float] = {}
        for ticker in universe:
            if ticker == asset:
                continue
            leader_lagged = aligned[ticker][:-LAG]
            with np.errstate(invalid="ignore"):
                corr = float(np.corrcoef(leader_lagged, asset_current)[0, 1])
            lead_lag_corr[ticker] = 0.0 if np.isnan(corr) else corr

        if not lead_lag_corr:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="universe contains no other tickers to build a lead-lag graph — holding.",
                status="unavailable",
            )

        centrality = float(np.mean([abs(corr) for corr in lead_lag_corr.values()]))
        predicted_return = float(
            sum(corr * float(aligned[ticker][-1]) for ticker, corr in lead_lag_corr.items())
        )

        direction: Literal["long", "short", "flat"] = (
            "long" if predicted_return > 0.0 else "short" if predicted_return < 0.0 else "flat"
        )
        confidence = clamp(centrality) * clamp(abs(predicted_return) / RETURN_CAP)

        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction=direction,
            confidence=confidence,
            rationale=(
                f"lead-lag centrality={centrality:.3f}, predicted return={predicted_return:.5f} "
                f"across universe {universe}"
            ),
            sample_count=min_length,
        )
