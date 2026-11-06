# fama_macbeth_analyst.py

import threading
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from pydantic import Field, PrivateAttr
from quantcore.statistics.regression import factor_loadings, fama_macbeth_regression

from agentic_trading.agents._market_math import RefitCache, balanced_panel, clamp
from agentic_trading.agents.base import AnalystAgent, DataRequirement, calendar_days_for_sessions
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.data.loaders.lakehouse_loader import (
    ken_french_public_date,
    load_fama_french_factors_from_lakehouse,
    load_price_history_from_lakehouse,
    load_security_log_returns,
    load_sp500_member_securities_asof,
)
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

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


@dataclass(frozen=True)
class _CrossSection:
    """The cross-sectional stage of one balanced panel, shared by every
    asset whose call builds that panel: the kept tickers in universe
    order, each one's factor loadings (intercept dropped), the
    Fama-MacBeth risk premia, the panel's returns by ticker on the
    reference dates, the number of those dates and the newest of them."""

    kept: tuple[str, ...]
    betas_by_ticker: dict[str, np.ndarray]
    premia: np.ndarray
    joined_returns: pd.DataFrame
    sample_count: int
    last_date: pd.Timestamp


def _cross_section_key(
    universe: tuple[str, ...],
    universe_returns: dict[str, pd.Series],
    reference: pd.DataFrame,
    factor_columns: list[str],
) -> Hashable:
    """Everything `_cross_section` reads, as a hashable key: the universe
    tuple and returns dict by identity, the reference dates with their
    dtype, the factor columns, and the reference factor rows as their
    float64 bytes, which is the precision the stage reads them at."""
    reference_factors = reference["factors"]
    return (
        id(universe),
        id(universe_returns),
        str(reference.index.dtype),
        tuple(reference.index),
        tuple(factor_columns),
        tuple(reference_factors.columns),
        reference_factors.to_numpy(dtype=np.float64).tobytes(),
    )


def _cross_section(
    universe: tuple[str, ...],
    universe_returns: dict[str, pd.Series],
    reference: pd.DataFrame,
    factor_columns: list[str],
) -> _CrossSection | None:
    """Builds the balanced panel of `universe` on `reference`'s dates
    (`balanced_panel`), estimates every kept ticker's factor loadings on
    its excess returns, then the Fama-MacBeth premia across them. Returns
    `None` when no more tickers than factors are kept."""
    panel_frame = balanced_panel(universe, universe_returns, reference.index)
    kept = tuple(panel_frame.columns)
    if len(kept) <= len(factor_columns):
        return None
    joined = pd.concat(
        {"returns": panel_frame, "factors": reference["factors"]}, axis=1, join="inner"
    )

    # A top-level key of a two-level column index selects a frame.
    joined_returns = cast(pd.DataFrame, joined["returns"])
    joined_factors = joined["factors"]
    factor_matrix = joined_factors[factor_columns].to_numpy(dtype=np.float64)
    risk_free = joined_factors["RF"].to_numpy(dtype=np.float64)

    betas_by_ticker: dict[str, np.ndarray] = {}
    panel_returns = []
    for ticker in kept:
        excess = joined_returns[ticker].to_numpy(dtype=np.float64) - risk_free
        # quantcore returns the intercept (alpha) first; only the per-
        # factor loadings belong in the cross-sectional stage.
        betas_by_ticker[ticker] = factor_loadings(excess, factor_matrix).loadings[1:]
        panel_returns.append(excess)

    panel = np.column_stack(panel_returns)
    beta_matrix = np.vstack([betas_by_ticker[ticker] for ticker in kept])
    premia, _t_stats = fama_macbeth_regression(panel, beta_matrix)
    return _CrossSection(
        kept=kept,
        betas_by_ticker=betas_by_ticker,
        premia=premia,
        joined_returns=joined_returns,
        sample_count=len(joined),
        last_date=joined.index.max(),
    )


class _CrossSectionCache:
    """The `_CrossSection`s of one decision day, keyed by
    `_cross_section_key`; a `None` entry records a panel too narrow for
    the regression. A call for another day drops every entry first, so
    the cache holds one day's stages. Each entry also holds the universe
    tuple and returns dict its key names by `id`, so neither can be freed
    and its id reused while the entry stands. One lock covers each lookup
    and the computation it may run, since a manager's worker thread can
    call the analyst while another's does.

    A runtime memo, not strategy state: it pickles and deep-copies as a
    new empty cache, so `carried_state` snapshots none of it and a
    restored analyst recomputes each stage on first use."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._day: pd.Timestamp | None = None
        self._entries: dict[
            Hashable, tuple[tuple[str, ...], dict[str, pd.Series], _CrossSection | None]
        ] = {}

    def __reduce__(self) -> tuple[Any, ...]:
        return (_CrossSectionCache, ())

    @property
    def day(self) -> pd.Timestamp | None:
        """The decision day the entries belong to; `None` before any call."""
        return self._day

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def get_or_compute(
        self,
        day: pd.Timestamp,
        key: Hashable,
        universe: tuple[str, ...],
        universe_returns: dict[str, pd.Series],
        compute: Callable[[], _CrossSection | None],
    ) -> _CrossSection | None:
        """The stage stored under `key` for `day`, computed with `compute`
        and stored when there is none. `universe` and `universe_returns`
        are the objects `key` names by identity. An exception from
        `compute` propagates and stores nothing."""
        with self._lock:
            if day != self._day:
                self._entries.clear()
                self._day = day
            if key in self._entries:
                return self._entries[key][2]
            stage = compute()
            self._entries[key] = (universe, universe_returns, stage)
            return stage


class FamaMacbethAnalyst(AnalystAgent):
    """The one analyst whose signal legitimately needs more than one
    ticker's data: estimates per-asset factor loadings across a
    cross-sectional universe, runs a Fama-MacBeth cross-sectional
    regression for the estimated risk premia, and predicts `asset`'s
    expected return from its own loadings times those premia.

    Both stages run on one balanced, date-aligned panel: the reference
    dates are the last `_RISK_WINDOW` days on which `asset`'s return and
    the factors both exist (joined by date), and only universe tickers
    with a return on every reference date are kept — a ticker missing a
    date is dropped, never the date. `sample_count` is the number of
    reference dates; the rationale reports how many tickers were kept.
    The cross-sectional stage depends on the day's universe, the
    reference dates and the factor rows, not on `asset`: it runs once per
    distinct set of those on a decision day and is shared by every asset
    that builds the same panel (`_CrossSectionCache`).

    The universe is resolved fresh each call as the real, point-in-time
    S&P 500 membership as of `as_of_date`, each member priced by its
    lake `security_id` (`_load_broad_lakehouse_universe`) —
    Fama-MacBeth's statistical power comes from genuine cross-sectional
    breadth (hundreds of names, the way the original methodology and
    institutional usage both run it). A day with no membership in the
    lake, or fewer than `_MIN_UNIVERSE_SIZE` members with enough
    history, is unavailable. A lake request that raises propagates to
    `ManagerAgent.collect_signals`, which records an error signal;
    missing names, factors or history are unavailable.

    An active signal's `source_as_of` is the public date
    (`ken_french_public_date`) of the newest factor row the regression
    used. The price bars are known at the close of their own date, on or
    before `as_of_date`, so they do not move it."""

    agent_id: str = "fama_macbeth_analyst"

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
    # The ticker of each `security_id` in the cached universe, set with it.
    _universe_tickers_by_security: dict[int, str] = PrivateAttr(default_factory=dict)
    # The day's cross-sectional stages, shared by every asset whose call
    # builds the same panel; never part of `carried_state`.
    _cross_section_cache: _CrossSectionCache = PrivateAttr(default_factory=_CrossSectionCache)
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
        if broad_universe_result is None:
            return self.unavailable_signal(
                asset,
                as_of_date,
                f"no point-in-time S&P 500 cross-section of at least {_MIN_UNIVERSE_SIZE} "
                "members with enough history",
            )
        resolution = self._resolve_with_broad_universe(broad_universe_result, asset, as_of_date)
        if isinstance(resolution, AgentSignal):
            return resolution

        universe, universe_returns = resolution
        factors = load_fama_french_factors_from_lakehouse(as_of_date.date(), model=self.model)
        if factors is None:
            return self.unavailable_signal(
                asset, as_of_date, "no Ken French factor coverage in the lakehouse"
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
        stage = self._cross_section_cache.get_or_compute(
            as_of_date,
            _cross_section_key(universe, universe_returns, reference, factor_columns),
            universe,
            universe_returns,
            lambda: _cross_section(universe, universe_returns, reference, factor_columns),
        )
        if stage is None:
            return self._insufficient_history(asset, as_of_date)
        kept = stage.kept

        predicted_return = float(np.dot(stage.betas_by_ticker[asset], stage.premia))
        realized_vol = float(np.std(stage.joined_returns[asset].to_numpy(dtype=np.float64), ddof=1))

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
            sample_count=stage.sample_count,
            source_as_of=ken_french_public_date(stage.last_date.date()),
        )

    def _insufficient_history(self, asset: str, as_of_date: pd.Timestamp) -> AgentSignal:
        return self.unavailable_signal(
            asset, as_of_date, "insufficient cross-sectional history for a Fama-MacBeth regression"
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

        Records the ticker of each kept member's `security_id` for
        `_resolve_with_broad_universe`.

        A member needs at least `_RISK_WINDOW // 2` returns: a thin series
        would empty the caller's date-joined panel for every asset that
        day, not just its own. Members left out, including those with no
        returns at all, are counted in one INFO line. Returns `None` when
        the lake has no membership for the date or fewer than
        `_MIN_UNIVERSE_SIZE` members qualify, which the caller reports as
        unavailable for `refit_interval_days`. A failed lake request
        propagates."""
        members = load_sp500_member_securities_asof(as_of_date.date())
        if members is None:
            return None
        window_start = (as_of_date - timedelta(days=250)).date()
        returns_by_security = load_security_log_returns(
            [member.security_id for member in members], window_start, as_of_date.date()
        )
        min_returns = _RISK_WINDOW // 2
        universe_returns: dict[str, pd.Series] = {}
        tickers_by_security: dict[int, str] = {}
        for member in members:
            returns = returns_by_security.get(member.security_id)
            if returns is not None and returns.size >= min_returns:
                universe_returns[member.ticker] = returns
                tickers_by_security[member.security_id] = member.ticker
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
        self._universe_tickers_by_security = tickers_by_security
        return tuple(universe_returns), universe_returns

    def _resolve_with_broad_universe(
        self,
        broad_universe_result: tuple[tuple[str, ...], dict[str, pd.Series]],
        asset: str,
        as_of_date: pd.Timestamp,
    ) -> AgentSignal | tuple[tuple[str, ...], dict[str, pd.Series]]:
        """Per-call, cheap: reuses the cached broad-universe returns as
        they stand, only doing a fresh (single-asset, fast) lakehouse
        lookup if `asset` itself isn't already in the cached set — most
        of this project's traded assets are themselves S&P 500 members,
        so this extra lookup is the exception, not the common path.

        A registered-firm equity (`roster.FIRM_V1_EQUITY_SECURITY_IDS`) is
        matched by its `security_id`: when the cached universe holds that
        security under another ticker (Alphabet class A as GOOG while the
        asset is GOOGL), that member is the asset, renamed to `asset` in
        place, so the security enters the cross-section once; otherwise it
        is read by id (`load_security_log_returns`, the universe's own
        basis). Any other asset is read by ticker
        (`load_price_history_from_lakehouse`)."""
        # roster imports this module, so its constants are read at call time.
        from agentic_trading.agents.roster import FIRM_V1_EQUITY_SECURITY_IDS

        resolved_tickers, universe_returns = broad_universe_result
        if asset in universe_returns:
            return resolved_tickers, universe_returns

        security_id = FIRM_V1_EQUITY_SECURITY_IDS.get(asset)
        member_ticker = (
            None if security_id is None else self._universe_tickers_by_security.get(security_id)
        )
        if member_ticker is not None and member_ticker in universe_returns:
            renamed = tuple(
                asset if ticker == member_ticker else ticker for ticker in resolved_tickers
            )
            return renamed, {
                asset if ticker == member_ticker else ticker: returns
                for ticker, returns in universe_returns.items()
            }

        window_start = (as_of_date - timedelta(days=250)).date()
        asset_returns: pd.Series | None = None
        if security_id is not None:
            asset_returns = load_security_log_returns(
                [security_id], window_start, as_of_date.date()
            ).get(security_id)
        else:
            asset_frame = load_price_history_from_lakehouse(
                [asset], window_start, as_of_date.date()
            )
            if asset_frame is not None and asset in asset_frame.columns.get_level_values(0):
                asset_returns = asset_frame[(asset, "log_return")].dropna()
        if asset_returns is None:
            return self.unavailable_signal(
                asset, as_of_date, f"{asset} not resolvable in the lakehouse cross-section"
            )
        merged_returns = dict(universe_returns)
        merged_returns[asset] = asset_returns
        return (*resolved_tickers, asset), merged_returns
