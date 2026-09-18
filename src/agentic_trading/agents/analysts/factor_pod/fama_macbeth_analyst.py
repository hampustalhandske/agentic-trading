# fama_macbeth_analyst.py

from datetime import timedelta
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import Field, PrivateAttr
from quantcore.statistics.regression import factor_loadings, fama_macbeth_regression

from agentic_trading.agents._market_math import RefitCache, clamp, load_fixed_universe_returns
from agentic_trading.agents.base import AnalystAgent, DataRequirement
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.data.loaders.lakehouse_loader import (
    load_fama_french_factors_from_lakehouse,
    load_price_history_from_lakehouse,
    load_sp500_constituents_asof,
)
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

# Emergency fallback only — used when the lakehouse (real, point-in-time
# S&P 500 membership, hundreds of names) isn't reachable. Fama-MacBeth's
# statistical power depends on cross-sectional breadth; this 5-name list
# is not a real cross-section and should never be what actually runs
# once the lakehouse is available.
DEFAULT_UNIVERSE: tuple[str, ...] = ("AAPL", "MSFT", "GOOGL", "AMZN", "JPM")
_RISK_WINDOW = 120
# A broad universe will always have some tickers fail to load on any
# given day (thin history, a real gap) — requiring every single one to
# succeed would make the whole analyst flat on the first failure across
# hundreds of names. This
# is the minimum surviving cross-section size to still run the
# regression on, not "all of them."
_MIN_UNIVERSE_SIZE = 30


class FamaMacbethAnalyst(AnalystAgent):
    """The one analyst whose signal legitimately needs more than one
    ticker's data: estimates per-asset factor loadings across a
    cross-sectional `universe`, runs a Fama-MacBeth cross-sectional
    regression for the estimated risk premia, and predicts `asset`'s
    expected return from its own loadings times those premia.

    `universe` is resolved fresh each call as the real, point-in-time
    S&P 500 membership as of `as_of_date` (`data/loaders/lakehouse_loader.py`),
    not a fixed list — Fama-MacBeth's statistical power comes from
    genuine cross-sectional breadth (hundreds of names, the way the
    original methodology and institutional usage both run it), which a
    handful of megacaps cannot provide regardless of which five are
    picked. Falls back to `self.universe` (a small static list) only if
    the lakehouse is unreachable that day."""

    agent_id: str = "fama_macbeth_analyst"

    universe: tuple[str, ...] = Field(default=DEFAULT_UNIVERSE)
    model: str = "3factor"
    # The broad S&P 500 universe and its trailing returns barely change
    # day to day — re-querying the lakehouse for ~500 names on every
    # single simulated day, for every asset this analyst is asked about,
    # is the same "refitting a slow-moving model daily" waste every other
    # RefitCache user in this project already avoids. Cached once per
    # `refit_interval_days`, shared across every asset's call that day
    # (not per-asset), since the broad universe itself doesn't depend on
    # which specific traded asset is asking.
    _universe_refit_cache: RefitCache = PrivateAttr(
        default_factory=lambda: RefitCache(refit_interval_days=5)
    )
    data_requirements: tuple[DataRequirement, ...] = (
        DataRequirement(
            source="ken_french_factors",
            identifier="3factor",
            asset_scoped=False,
            description="Ken French 3-factor returns",
        ),
    )

    def generate_signal(
        self,
        asset: str,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> AgentSignal:
        broad_universe_result = self._universe_refit_cache.get_or_refit(
            "broad_universe", as_of_date, lambda: self._load_broad_lakehouse_universe(as_of_date)
        )
        if broad_universe_result is not None:
            resolution = self._resolve_with_broad_universe(broad_universe_result, asset, as_of_date)
        else:
            candidate_universe = (
                self.universe if asset in self.universe else (*self.universe, asset)
            )
            resolution = self._fixed_universe_returns(candidate_universe, asset, as_of_date)

        if isinstance(resolution, AgentSignal):
            return resolution

        universe, universe_returns = resolution
        factors = load_fama_french_factors_from_lakehouse(as_of_date.date(), model=self.model)
        if factors is None:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="no Ken French factor coverage in the lakehouse",
                status="unavailable",
            )
        factor_columns = [column for column in factors.columns if column != "RF"]

        min_length = min([len(factors), *(series.size for series in universe_returns.values())])
        if min_length < 30 or asset not in universe_returns:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="insufficient cross-sectional history for a Fama-MacBeth regression",
                status="unavailable",
            )

        aligned_factors = factors.iloc[-min_length:]
        factor_matrix = aligned_factors[factor_columns].to_numpy(dtype=np.float64)
        risk_free = aligned_factors["RF"].to_numpy(dtype=np.float64)

        betas_by_ticker: dict[str, np.ndarray] = {}
        panel_returns = []
        for ticker in universe:
            excess = universe_returns[ticker][-min_length:] - risk_free
            # quantcore returns the intercept (alpha) first; only the per-
            # factor loadings belong in the cross-sectional stage.
            loadings, _t_stats = factor_loadings(excess, factor_matrix)
            betas_by_ticker[ticker] = loadings[1:]
            panel_returns.append(excess)

        panel = np.column_stack(panel_returns)
        beta_matrix = np.vstack([betas_by_ticker[ticker] for ticker in universe])
        premia, _t_stats = fama_macbeth_regression(panel, beta_matrix)

        predicted_return = float(np.dot(betas_by_ticker[asset], premia))
        realized_vol = float(np.std(universe_returns[asset][-min_length:], ddof=1))

        direction: Literal["long", "short", "flat"] = "long" if predicted_return >= 0.0 else "short"
        confidence = clamp(abs(predicted_return) / max(realized_vol, 1e-8))

        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction=direction,
            confidence=confidence,
            rationale=(
                f"Fama-MacBeth predicted return={predicted_return:.5f} across a "
                f"{len(universe)}-name cross-section"
            ),
        )

    def _load_broad_lakehouse_universe(
        self, as_of_date: pd.Timestamp
    ) -> tuple[tuple[str, ...], dict[str, np.ndarray]] | None:
        """The expensive fetch `_universe_refit_cache` gates: real,
        point-in-time S&P 500 membership plus a single batch lakehouse
        query for all of their trailing returns. Returns `None` (not
        raises) on any failure, so a cache miss during a lakehouse outage
        degrades to the fixed-universe fallback for `refit_interval_days`
        rather than retrying every day."""
        lakehouse_universe = load_sp500_constituents_asof(as_of_date.date())
        if lakehouse_universe is None:
            return None
        window_start = (as_of_date - timedelta(days=250)).date()
        frame = load_price_history_from_lakehouse(
            lakehouse_universe, window_start, as_of_date.date()
        )
        if frame is None:
            return None
        # A ticker present in the frame isn't necessarily usable — a
        # corrupted or thin price series (confirmed live: some lakehouse
        # rows carry a real `close` but a broken all-zero `adj_close`,
        # masked to NaN by the loader) can leave a ticker with almost no
        # real return observations. Filtering on this BEFORE computing
        # `min_length` in the caller matters: an all-NaN ticker included
        # here would otherwise drag the cross-sectional `min_length` down
        # to 0 for every asset's regression that day, not just its own.
        candidate_returns = {
            str(ticker): frame[(ticker, "log_return")].dropna().to_numpy()[-_RISK_WINDOW:]
            for ticker in frame.columns.get_level_values(0).unique()
        }
        universe_returns = {
            ticker: returns
            for ticker, returns in candidate_returns.items()
            if returns.size >= _RISK_WINDOW // 2
        }
        if len(universe_returns) < _MIN_UNIVERSE_SIZE:
            return None
        return tuple(universe_returns), universe_returns

    def _resolve_with_broad_universe(
        self,
        broad_universe_result: tuple[tuple[str, ...], dict[str, np.ndarray]],
        asset: str,
        as_of_date: pd.Timestamp,
    ) -> AgentSignal | tuple[tuple[str, ...], dict[str, np.ndarray]]:
        """Per-call, cheap: reuses the cached broad-universe returns as
        they stand, only doing a fresh (single-ticker, fast) lakehouse
        lookup if `asset` itself isn't already in the cached set — most
        of this project's traded assets are themselves S&P 500 members,
        so this extra lookup is the exception, not the common path."""
        resolved_tickers, universe_returns = broad_universe_result
        if asset in universe_returns:
            return resolved_tickers, universe_returns

        window_start = (as_of_date - timedelta(days=250)).date()
        asset_frame = load_price_history_from_lakehouse([asset], window_start, as_of_date.date())
        if asset_frame is None or asset not in asset_frame.columns.get_level_values(0):
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale=f"{asset} not resolvable in the lakehouse cross-section",
                status="unavailable",
            )
        merged_returns = dict(universe_returns)
        merged_returns[asset] = (
            asset_frame[(asset, "log_return")].dropna().to_numpy()[-_RISK_WINDOW:]
        )
        return (*resolved_tickers, asset), merged_returns

    def _fixed_universe_returns(
        self, universe: tuple[str, ...], asset: str, as_of_date: pd.Timestamp
    ) -> AgentSignal | tuple[tuple[str, ...], dict[str, np.ndarray]]:
        """Fallback used only when the broad lakehouse S&P 500 universe
        itself is unreachable that day: `load_fixed_universe_returns`
        over a small fixed list, only usable at that scale — every
        ticker must succeed, since this path is never handed hundreds of
        names where partial failure is expected."""
        try:
            universe_returns = {
                ticker: series[-_RISK_WINDOW:]
                for ticker, series in load_fixed_universe_returns(universe, as_of_date).items()
            }
        except Exception as exc:
            logger.debug(
                "%s: universe price history download failed (%s) — returning flat.",
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
        if len(universe_returns) < len(universe):
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale=f"insufficient price history for part of universe {universe}",
                status="unavailable",
            )
        return universe, universe_returns
