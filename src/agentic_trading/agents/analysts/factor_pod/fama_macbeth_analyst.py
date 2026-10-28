# fama_macbeth_analyst.py

from datetime import timedelta
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import Field, PrivateAttr
from quantcore.statistics.regression import factor_loadings, fama_macbeth_regression

from agentic_trading.agents._market_math import (
    RefitCache,
    balanced_panel,
    clamp,
    load_fixed_universe_returns,
)
from agentic_trading.agents.base import AnalystAgent, DataRequirement, calendar_days_for_sessions
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.data.loaders.lakehouse_loader import (
    load_fama_french_factors_from_lakehouse,
    load_price_history_from_lakehouse,
    load_security_log_returns,
    load_sp500_member_securities_asof,
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
# Fewest reference dates (the asset's return and the factors both
# present) the time-series and cross-sectional stages run on.
_MIN_JOINED_OBSERVATIONS = 30


class FamaMacbethAnalyst(AnalystAgent):
    """The one analyst whose signal legitimately needs more than one
    ticker's data: estimates per-asset factor loadings across a
    cross-sectional `universe`, runs a Fama-MacBeth cross-sectional
    regression for the estimated risk premia, and predicts `asset`'s
    expected return from its own loadings times those premia.

    Both stages run on one balanced, date-aligned panel: the reference
    dates are the last `_RISK_WINDOW` days on which `asset`'s return and
    the factors both exist (joined by date), and only universe tickers
    with a return on every reference date are kept — a ticker missing a
    date is dropped, never the date. `sample_count` is the number of
    reference dates; the rationale reports how many tickers were kept.

    `universe` is resolved fresh each call as the real, point-in-time
    S&P 500 membership as of `as_of_date`, each member priced by its
    lake `security_id` (`_load_broad_lakehouse_universe`), not a fixed
    list — Fama-MacBeth's statistical power comes from
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
    # Factors over the `_RISK_WINDOW` reference dates; membership only as
    # of each decision date.
    data_requirements: tuple[DataRequirement, ...] = Field(
        default_factory=lambda: (
            DataRequirement(
                source="ken_french_factors",
                identifier="3factor",
                asset_scoped=False,
                description="Ken French 3-factor returns",
                lookback_days=calendar_days_for_sessions(_RISK_WINDOW),
            ),
            DataRequirement(
                source="sp500_membership",
                identifier="all",
                asset_scoped=False,
                description="S&P 500 membership history",
                lookback_days=0,
            ),
        )
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

        if asset not in universe_returns:
            return self._insufficient_history(asset, as_of_date)
        reference = (
            pd.concat(
                {"asset": universe_returns[asset].to_frame("return"), "factors": factors},
                axis=1,
                join="inner",
            )
            .dropna()
            .iloc[-_RISK_WINDOW:]
        )
        if len(reference) < _MIN_JOINED_OBSERVATIONS:
            return self._insufficient_history(asset, as_of_date)
        panel_frame = balanced_panel(universe, universe_returns, reference.index)
        kept = tuple(panel_frame.columns)
        if len(kept) <= len(factor_columns):
            return self._insufficient_history(asset, as_of_date)
        joined = pd.concat(
            {"returns": panel_frame, "factors": reference["factors"]}, axis=1, join="inner"
        )

        joined_returns = joined["returns"]
        joined_factors = joined["factors"]
        factor_matrix = joined_factors[factor_columns].to_numpy(dtype=np.float64)
        risk_free = joined_factors["RF"].to_numpy(dtype=np.float64)

        betas_by_ticker: dict[str, np.ndarray] = {}
        panel_returns = []
        for ticker in kept:
            excess = joined_returns[ticker].to_numpy(dtype=np.float64) - risk_free
            # quantcore returns the intercept (alpha) first; only the per-
            # factor loadings belong in the cross-sectional stage.
            loadings, _t_stats = factor_loadings(excess, factor_matrix)
            betas_by_ticker[ticker] = loadings[1:]
            panel_returns.append(excess)

        panel = np.column_stack(panel_returns)
        beta_matrix = np.vstack([betas_by_ticker[ticker] for ticker in kept])
        premia, _t_stats = fama_macbeth_regression(panel, beta_matrix)

        predicted_return = float(np.dot(betas_by_ticker[asset], premia))
        realized_vol = float(np.std(joined_returns[asset].to_numpy(dtype=np.float64), ddof=1))

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
                f"{len(kept)}-name cross-section ({len(universe) - len(kept)} of "
                f"{len(universe)} names dropped for a missing date)"
            ),
            sample_count=len(joined),
        )

    def _insufficient_history(self, asset: str, as_of_date: pd.Timestamp) -> AgentSignal:
        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction="flat",
            confidence=0.0,
            rationale="insufficient cross-sectional history for a Fama-MacBeth regression",
            status="unavailable",
        )

    def _load_broad_lakehouse_universe(
        self, as_of_date: pd.Timestamp
    ) -> tuple[tuple[str, ...], dict[str, pd.Series]] | None:
        """The expensive fetch `_universe_refit_cache` gates: the
        point-in-time S&P 500 members on `as_of_date` as securities
        (`load_sp500_member_securities_asof`) and one batch query for
        their trailing 250 calendar days of split- and
        distribution-adjusted log returns by `security_id`
        (`load_security_log_returns`). Returns `(tickers, returns)` with
        both keyed by each member's ticker on `as_of_date`.

        A member needs at least `_RISK_WINDOW // 2` returns: a thin series
        would empty the caller's date-joined panel for every asset that
        day, not just its own. Members left out, including those with no
        returns at all, are counted in one INFO line. Returns `None` when
        the lake has no membership for the date or fewer than
        `_MIN_UNIVERSE_SIZE` members qualify, so the caller uses the
        fixed-universe fallback for `refit_interval_days`. A failed lake
        request propagates."""
        members = load_sp500_member_securities_asof(as_of_date.date())
        if members is None:
            return None
        window_start = (as_of_date - timedelta(days=250)).date()
        returns_by_security = load_security_log_returns(
            [member.security_id for member in members], window_start, as_of_date.date()
        )
        min_returns = _RISK_WINDOW // 2
        universe_returns: dict[str, pd.Series] = {}
        for member in members:
            returns = returns_by_security.get(member.security_id)
            if returns is not None and returns.size >= min_returns:
                universe_returns[member.ticker] = returns
        excluded = len(members) - len(universe_returns)
        if excluded:
            logger.info(
                "%s: left %d of %d S&P 500 members on %s out of the cross-section for "
                "fewer than %d returns since %s",
                self.agent_id,
                excluded,
                len(members),
                as_of_date.date(),
                min_returns,
                window_start,
            )
        if len(universe_returns) < _MIN_UNIVERSE_SIZE:
            return None
        return tuple(universe_returns), universe_returns

    def _resolve_with_broad_universe(
        self,
        broad_universe_result: tuple[tuple[str, ...], dict[str, pd.Series]],
        asset: str,
        as_of_date: pd.Timestamp,
    ) -> AgentSignal | tuple[tuple[str, ...], dict[str, pd.Series]]:
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
        merged_returns[asset] = asset_frame[(asset, "log_return")].dropna()
        return (*resolved_tickers, asset), merged_returns

    def _fixed_universe_returns(
        self, universe: tuple[str, ...], asset: str, as_of_date: pd.Timestamp
    ) -> AgentSignal | tuple[tuple[str, ...], dict[str, pd.Series]]:
        """Fallback used only when the broad lakehouse S&P 500 universe
        itself is unreachable that day: `load_fixed_universe_returns`
        over a small fixed list, only usable at that scale — every
        ticker must succeed, since this path is never handed hundreds of
        names where partial failure is expected."""
        try:
            universe_returns = load_fixed_universe_returns(universe, as_of_date)
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
