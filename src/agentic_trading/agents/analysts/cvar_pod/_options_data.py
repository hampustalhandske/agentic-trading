# _options_data.py

"""Shared market-data inputs for the CVaR (options) pod.

Two things every options analyst in this pod reads from one place, so
they all see the same numbers on the same day:

- **Real option chain**: the lakehouse's historical chain snapshot for
  the asset as published by the decision date
  (`real_atm_options_by_expiration`), reduced to one ATM row
  per real expiration. The pod is scoped (`cvar_manager.OPTIONS_ASSETS`)
  to tickers with real chain coverage, and an analyst that still finds
  no snapshot for a given date returns a flat, zero-confidence signal —
  never a model-implied substitute quote.
- **Model volatility**: GARCH(1,1) 1-step-ahead conditional variance,
  annualized (`annualized_garch_volatility`) — the analyst's own
  volatility view that gets compared against the real market quote.
"""

import numpy as np
import pandas as pd

from agentic_trading.agents._market_math import RefitCache, conditional_daily_volatility
from agentic_trading.agents.base import DataRequirement
from agentic_trading.config import settings
from agentic_trading.data.loaders.lakehouse_loader import (
    DEFAULT_OPTIONS_SNAPSHOT_LOOKBACK_DAYS,
    load_options_snapshot_from_lakehouse,
)
from agentic_trading.data.loaders.option_marks import published_chain_date
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

TRADING_DAYS_PER_YEAR = 252

# Shared across every CVaR-pod analyst that calls annualized_garch_volatility
# (a plain module-level function, not bound to any one analyst instance) —
# keyed by (agent_id, asset), so each analyst still gets its own independent
# refit cadence rather than accidentally sharing one analyst's fit with
# another's.
_GARCH_REFIT_CACHE = RefitCache()


def annualized_garch_volatility(
    returns: np.ndarray, agent_id: str, asset: str, as_of_date: pd.Timestamp
) -> float:
    """Annualized (sqrt 252) one-step-ahead conditional volatility from
    `_market_math.conditional_daily_volatility` (a sanity-checked
    GARCH(1,1), EWMA when the fit is unusable), refit at most once every
    `_GARCH_REFIT_CACHE.refit_interval_days` simulated days per
    (agent_id, asset)."""
    daily_volatility = conditional_daily_volatility(
        returns, _GARCH_REFIT_CACHE, f"{agent_id}:{asset}", as_of_date, agent_id
    )
    return daily_volatility * float(np.sqrt(TRADING_DAYS_PER_YEAR))


def options_chain_requirements() -> tuple[DataRequirement, ...]:
    """The option-chain `DataRequirement` of every analyst reading
    `real_atm_options_by_expiration`: the snapshot it sees on a decision
    date is at most `settings.options.publication_lag_days` plus
    `DEFAULT_OPTIONS_SNAPSHOT_LOOKBACK_DAYS` calendar days old."""
    return (
        DataRequirement(
            source="options_chain",
            identifier="{ticker}",
            asset_scoped=True,
            description="Option chain snapshots",
            lookback_days=settings.options.publication_lag_days
            + DEFAULT_OPTIONS_SNAPSHOT_LOOKBACK_DAYS,
        ),
    )


def real_atm_options_by_expiration(
    asset: str, as_of_date: pd.Timestamp, spot: float
) -> pd.DataFrame | None:
    """Real market options data where it exists (see
    `data/loaders/lakehouse_loader.py::load_options_snapshot_from_lakehouse`
    — 11 tickers as of the current pull, no open interest/volume), reduced
    to one row per real expiration in the nearest available snapshot:
    the strike closest to `spot`, that strike's call/put implied
    volatility averaged (put-call parity means they're already close;
    averaging is noise reduction, not a new assumption), and the call
    leg's real delta/gamma. Only a snapshot already published by
    `as_of_date` is visible: one dated on or before
    `option_marks.published_chain_date(as_of_date)`. `days_to_expiration` is
    measured from the snapshot's own date, not `as_of_date` (they differ
    by at least the publication lag and up to the lag plus the loader's
    `lookback_days`, since coverage isn't daily-dense).

    Returns `None` if `asset` has no real snapshot for this date; every
    CVaR-pod analyst calling this then reports flat with zero confidence
    rather than pricing off a model-implied substitute quote."""
    snapshot = load_options_snapshot_from_lakehouse(asset, published_chain_date(as_of_date.date()))
    if snapshot is None or snapshot.empty:
        return None

    snapshot = snapshot.copy()
    snapshot["moneyness_distance"] = (snapshot["strike"] - spot).abs()
    snapshot_date = snapshot["date"].iloc[0]

    rows = []
    for expiration, group in snapshot.groupby("expiration"):
        nearest_strike = group.loc[group["moneyness_distance"].idxmin(), "strike"]
        at_strike = group[group["strike"] == nearest_strike]
        call_row = at_strike[at_strike["option_type"] == "call"]
        if call_row.empty:
            continue
        iv_values = at_strike["implied_volatility"].dropna()
        if iv_values.empty:
            continue
        days_to_expiration = max(int((expiration - snapshot_date).days), 1)
        rows.append(
            {
                "expiration": expiration,
                "days_to_expiration": days_to_expiration,
                "strike": float(np.asarray(nearest_strike, dtype=np.float64)),
                "implied_volatility": float(iv_values.mean()),
                "delta": float(call_row["delta"].iloc[0]),
                "gamma": float(call_row["gamma"].iloc[0]),
            }
        )
    if not rows:
        return None
    return pd.DataFrame(rows).sort_values("days_to_expiration").reset_index(drop=True)
