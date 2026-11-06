# lead_lag_network_centrality_analyst.py

from datetime import timedelta
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import PrivateAttr

from agentic_trading.agents._market_math import RefitCache, balanced_panel, clamp
from agentic_trading.agents.base import AnalystAgent, DataRequirement
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.data.loaders.lakehouse_loader import (
    load_security_log_returns,
    load_sp500_member_securities_asof,
)
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

WINDOW = 60
LAG = 1
RETURN_CAP = 0.05
# A broad universe will always have some tickers fail to load on any
# given day; below this many usable names the cross-section isn't broad
# enough for a network-centrality read.
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
    The graph is built on one balanced, date-aligned panel: the reference
    dates are `asset`'s own last `WINDOW + LAG` return dates, and only
    universe names with a return on every one of them are kept — a name
    missing a date is dropped, never the date. `sample_count` is the
    number of reference dates; the rationale reports the names kept.

    The universe is resolved fresh each call (refit at
    `_universe_refit_cache.refit_interval_days`) as the real, point-in-time
    S&P 500 membership, each member priced by its lake `security_id`
    (`_load_broad_lakehouse_universe`), the same cross-section
    `fama_macbeth_analyst.py` uses. A day with no membership in the lake,
    or fewer than `_MIN_UNIVERSE_SIZE` members with usable history, is
    unavailable, as is an `asset` outside that membership or too little
    history to build the graph. A fetch that raises is an error.

    An active signal's `source_as_of` is the latest reference date: S&P
    500 membership changes are announced before they take effect, so the
    newest data the graph rests on is its latest price bar."""

    agent_id: str = "lead_lag_network_centrality_analyst"
    data_requirements: tuple[DataRequirement, ...] = (
        DataRequirement(
            source="sp500_membership",
            identifier="all",
            asset_scoped=False,
            description="S&P 500 membership history",
            lookback_days=0,
        ),
    )

    _universe_refit_cache: RefitCache = PrivateAttr(default_factory=RefitCache)

    def _load_broad_lakehouse_universe(
        self, as_of_date: pd.Timestamp
    ) -> dict[str, pd.Series] | None:
        """The expensive fetch `_universe_refit_cache` gates, cached under
        one universe-wide key so every asset scored within a refit
        interval shares it: the point-in-time S&P 500 members on
        `as_of_date` as securities (`load_sp500_member_securities_asof`)
        and one batch query for their trailing `_UNIVERSE_HISTORY_DAYS`
        calendar days of split- and distribution-adjusted log returns by
        `security_id` (`load_security_log_returns`), keyed by each
        member's ticker on `as_of_date`.

        A member needs at least `WINDOW + LAG` returns; members left out,
        including those with no returns at all, are counted in one INFO
        line. Returns `None` when the lake has no membership for the date
        or fewer than `_MIN_UNIVERSE_SIZE` members qualify, which the
        caller reports as unavailable for `refit_interval_days`. A failed
        lake request propagates to `generate_signal`, which returns an
        error signal for it. Never adds `asset` to the membership: an
        asset outside it is unavailable, leaving the shared cached fetch
        intact."""
        members = load_sp500_member_securities_asof(as_of_date.date())
        if members is None:
            return None
        window_start = (as_of_date - timedelta(days=_UNIVERSE_HISTORY_DAYS)).date()
        returns_by_security = load_security_log_returns(
            [member.security_id for member in members], window_start, as_of_date.date()
        )
        min_returns = WINDOW + LAG
        usable: dict[str, pd.Series] = {}
        for member in members:
            series = returns_by_security.get(member.security_id)
            if series is not None and series.size >= min_returns:
                usable[member.ticker] = series
        excluded = len(members) - len(usable)
        if excluded:
            logger.info(
                "%s: left %d of %d S&P 500 members on %s out of the universe for "
                "fewer than %d returns since %s",
                self.agent_id,
                excluded,
                len(members),
                as_of_date.date(),
                min_returns,
                window_start,
            )
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
            logger.warning(
                "%s: broad lakehouse universe fetch failed for %s on %s (%s).",
                self.agent_id,
                asset,
                as_of_date.date(),
                exc,
            )
            return self.error_signal(asset, as_of_date, "broad S&P 500 universe lookup failed")

        if broad is None:
            return self.unavailable_signal(
                asset,
                as_of_date,
                f"no point-in-time S&P 500 cross-section of at least {_MIN_UNIVERSE_SIZE} "
                "members with enough history",
            )
        if asset not in broad:
            return self.unavailable_signal(
                asset,
                as_of_date,
                f"{asset} is not an S&P 500 member with enough history on {as_of_date.date()}",
            )
        universe = tuple(broad)
        returns = broad

        asset_returns = returns[asset].dropna() if asset in returns else pd.Series(dtype=float)
        reference_dates = asset_returns.index[-(WINDOW + LAG) :]
        if len(reference_dates) < WINDOW + LAG:
            return self.unavailable_signal(
                asset,
                as_of_date,
                f"insufficient cross-sectional history across universe {universe} — holding.",
            )

        panel = balanced_panel(universe, returns, reference_dates)
        kept = tuple(panel.columns)
        aligned = {ticker: panel[ticker].to_numpy(dtype=np.float64) for ticker in kept}
        asset_current = aligned[asset][LAG:]

        lead_lag_corr: dict[str, float] = {}
        for ticker in kept:
            if ticker == asset:
                continue
            leader_lagged = aligned[ticker][:-LAG]
            with np.errstate(invalid="ignore"):
                corr = float(np.corrcoef(leader_lagged, asset_current)[0, 1])
            lead_lag_corr[ticker] = 0.0 if np.isnan(corr) else corr

        if not lead_lag_corr:
            return self.unavailable_signal(
                asset,
                as_of_date,
                "universe contains no other tickers to build a lead-lag graph — holding.",
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
                f"across {len(kept)} of {len(universe)} universe names {kept}"
            ),
            sample_count=len(reference_dates),
            source_as_of=reference_dates[-1].date(),
        )
