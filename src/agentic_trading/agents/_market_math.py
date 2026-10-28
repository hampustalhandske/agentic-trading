from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, cast

import numpy as np
import pandas as pd

from agentic_trading.config import settings
from agentic_trading.data.loaders.lakehouse_loader import load_price_history_from_lakehouse
from agentic_trading.data.point_in_time import get_point_in_time_data
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

# A fixed, far-past start so the universe price query returns full
# history; safely predates any realistic backtest start date.
_UNIVERSE_HISTORY_START = date(2000, 1, 1)


@dataclass(frozen=True)
class MarketSlice:
    close: float | None
    log_returns: np.ndarray


def _slice_history(
    market_data: pd.DataFrame,
    as_of_date: pd.Timestamp,
    asset: str | None = None,
) -> pd.DataFrame:
    history = market_data.loc[market_data.index <= as_of_date]
    if isinstance(history, pd.Series):
        history = history.to_frame().T
    if asset is None or not isinstance(history.columns, pd.MultiIndex):
        return history.copy()

    if asset not in history.columns.get_level_values(0):
        return pd.DataFrame(index=history.index)

    sliced = history.xs(asset, axis=1, level=0)
    if isinstance(sliced, pd.Series):
        return sliced.to_frame()
    return sliced.copy()


def _log_return_series(history: pd.DataFrame) -> pd.Series:
    """Date-indexed log returns of a single-asset `history`: its own
    `log_return` column when present, otherwise log differences of
    `close`/`Close`; empty when it has neither."""
    if "log_return" in history.columns:
        return history["log_return"].dropna()
    for field in ("close", "Close"):
        if field in history.columns:
            close_series = history[field].dropna()
            if close_series.empty:
                break
            return pd.Series(
                np.log(close_series / close_series.shift(1)), index=close_series.index
            ).dropna()
    return pd.Series(dtype=float)


def log_return_series(
    market_data: pd.DataFrame,
    as_of_date: pd.Timestamp,
    asset: str | None = None,
) -> pd.Series:
    """Date-indexed log returns of `asset` on bars at or before
    `as_of_date` — the same values as `market_slice(...).log_returns`,
    keeping the dates so a caller can join them to another dated source
    instead of aligning by position."""
    return _log_return_series(_slice_history(market_data, as_of_date, asset))


def market_slice(
    market_data: pd.DataFrame,
    as_of_date: pd.Timestamp,
    asset: str | None = None,
) -> MarketSlice:
    history = _slice_history(market_data, as_of_date, asset)

    close: float | None = None
    for field in ("close", "Close"):
        if field in history.columns:
            close_series = history[field].dropna()
            if not close_series.empty:
                close = float(close_series.iloc[-1])
            break

    return_series = _log_return_series(history)
    return MarketSlice(close=close, log_returns=return_series.to_numpy(dtype=np.float64))


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return float(max(lower, min(upper, value)))


def derived_seed(analyst_id: str, asset: str, as_of: date) -> int:
    """A reproducible 32-bit seed for one analyst's simulation of `asset`
    on `as_of`: the first 16 hex digits of the sha256 of
    `"{settings.analysts.mc_seed}|{analyst_id}|{asset}|{YYYY-MM-DD}"`,
    reduced modulo 2**32. A `datetime` or `pd.Timestamp` counts as its
    calendar date. The base seed is read on every call, so an override of
    `analysts.mc_seed` takes effect immediately."""
    day = as_of.date() if isinstance(as_of, datetime) else as_of
    key = f"{settings.analysts.mc_seed}|{analyst_id}|{asset}|{day.isoformat()}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:16], 16) % 2**32


def load_fixed_universe_returns(
    universe: tuple[str, ...], as_of_date: pd.Timestamp
) -> dict[str, pd.Series]:
    """Point-in-time-safe, date-indexed log returns (NaNs dropped) for a
    fixed cross-sectional `universe` a few analysts need for their own
    methodology (e.g. a Fama-MacBeth cross-sectional regression, a
    lead-lag correlation graph) — fetched
    independently of whatever assets the current backtest's own
    `market_data` happens to include, via the lakehouse, not
    `market_slice()`.

    Using `market_data` instead would silently fail for any universe ticker
    not in the backtest's own asset list: `market_slice()` returns an empty
    series rather than raising when a ticker isn't a `market_data` column,
    which would read as "insufficient history" regardless of how much real
    history the other universe tickers had.

    Lakehouse-only, like every other price read in this project — a
    universe ticker the lakehouse doesn't cover is simply omitted from
    the result, the same shape a genuine data gap already produces for
    callers, rather than falling back to a live price API."""
    raw = load_price_history_from_lakehouse(
        list(universe), _UNIVERSE_HISTORY_START, as_of_date.date()
    )
    if raw is None:
        return {}
    filtered = get_point_in_time_data(raw, as_of_date)
    tickers_present = set(filtered.columns.get_level_values(0))
    return {
        ticker: filtered[ticker]["log_return"].dropna().astype(np.float64)
        for ticker in universe
        if ticker in tickers_present
    }


def balanced_panel(
    universe: tuple[str, ...],
    universe_returns: dict[str, pd.Series],
    reference_dates: pd.Index,
) -> pd.DataFrame:
    """Returns of every `universe` ticker that has a value on each of
    `reference_dates`, one column per kept ticker in `universe` order,
    indexed by `reference_dates`. A ticker missing any of those dates is
    dropped whole, so one gappy or recently listed name never removes a
    date from the other names."""
    columns = {}
    for ticker in universe:
        values = universe_returns[ticker].reindex(reference_dates)
        if not values.isna().any():
            columns[ticker] = values
    return pd.DataFrame(columns, index=reference_dates)


def single_asset_from_attrs(market_data: pd.DataFrame) -> str:
    """The one asset a non-MultiIndex `market_data` frame covers, read
    from `market_data.attrs["asset"]` — the caller must have set this
    when building a single-asset frame, since there is no column-level
    ticker to read it from instead. Raises rather than guessing, since a
    silent fallback here would attribute every asset's return to a
    hardcoded, potentially wrong ticker."""
    asset = market_data.attrs.get("asset")
    if asset is None:
        raise ValueError(
            "market_data has non-MultiIndex columns but no attrs['asset'] set — "
            "cannot determine which asset this frame covers."
        )
    return cast(str, asset)


def wide_log_returns(assets: list[str], market_data: pd.DataFrame) -> pd.DataFrame:
    """Reshapes `market_data` into one wide `log_return` frame (columns =
    `assets`, already point-in-time-sliced since `market_data` itself is),
    the shape `PortfolioConstructor` needs for a covariance matrix — the
    same reshape every pod manager's `construct_portfolio()` needs, so it
    lives here rather than being duplicated per manager. Returns an empty
    frame for an empty `assets` list rather than raising, since a
    scoped pod (`ManagerAgent.assets`) can legitimately have no assets in
    scope on a given day."""
    if not assets:
        return pd.DataFrame()
    if isinstance(market_data.columns, pd.MultiIndex):
        frame = pd.concat({asset: market_data[asset]["log_return"] for asset in assets}, axis=1)
    else:
        frame = market_data[["log_return"]].rename(columns={"log_return": assets[0]})
    return pd.DataFrame(frame.dropna())


class RefitCache:
    """Reuses a model fit across `refit_interval_days` of simulated days
    instead of refitting it from scratch on every single one. Most of the
    statistical models in this project (GARCH, HMM regime detection, a
    Kalman-filtered hedge ratio, a cointegration test, ARIMA, a VAR
    system) don't need daily refitting to stay valid — real desks commonly
    refit on a weekly-or-slower cadence and score every day off the same
    fitted parameters; a model refit every single simulated day is mostly
    reacting to refit-to-refit estimation noise, not genuine regime
    change. Cuts backtest compute roughly `refit_interval_days`x for
    whichever analyst/manager owns the instance, on top of that realism
    argument.

    Keyed by an arbitrary caller-chosen `key` (typically the asset), not
    a single global cadence — one `RefitCache` instance can track several
    keys' independent refit schedules (e.g. one per asset an analyst
    covers)."""

    def __init__(self, refit_interval_days: int = 5) -> None:
        self.refit_interval_days = refit_interval_days
        self._cache: dict[str, tuple[pd.Timestamp, Any]] = {}

    def get_or_refit(self, key: str, as_of_date: pd.Timestamp, refit_fn: Callable[[], Any]) -> Any:
        cached = self._cache.get(key)
        if cached is not None:
            last_refit_date, fitted_params = cached
            if (as_of_date - last_refit_date).days < self.refit_interval_days:
                return fitted_params

        fitted_params = refit_fn()
        self._cache[key] = (as_of_date, fitted_params)
        return fitted_params


def vol_regime_zscore(returns: np.ndarray) -> float | None:
    """Where today's conditional volatility sits relative to the asset's
    own trailing vol history: the z-score of the last log EWMA
    (RiskMetrics, `settings.managers.ewma_lambda`) volatility against the
    log-vol path over the last `settings.managers.vol_regime_window`
    returns. A z of 0 is a normal regime, +2 a genuine spike, -1 a calm
    stretch. Returns `None` with fewer than
    `settings.managers.min_vol_regime_observations` returns, when the
    path is too short for a stable std."""
    from quantcore.risk.egarch import ewma_variance

    from agentic_trading.config import settings

    window = returns[-settings.managers.vol_regime_window :]
    if window.size < settings.managers.min_vol_regime_observations:
        return None
    centered = window - float(np.mean(window))
    variance_path = ewma_variance(centered, settings.managers.ewma_lambda)
    log_vol = 0.5 * np.log(np.maximum(variance_path, 1e-18))
    spread = float(np.std(log_vol, ddof=1))
    if not np.isfinite(spread) or spread < 1e-12:
        return None
    return float(np.clip((log_vol[-1] - float(np.mean(log_vol))) / spread, -3.0, 4.0))


def vol_regime_scale_factor(
    market_data: pd.DataFrame, as_of_date: pd.Timestamp, asset: str
) -> float:
    """A pod manager's volatility-regime dampening multiplier for one
    asset: a logistic in `vol_regime_zscore` centred at
    `settings.managers.vol_zscore_midpoint` with slope
    `settings.managers.vol_zscore_slope`, floored at
    `settings.managers.min_scale_factor`. Roughly 0.95 in a normal
    regime, 0.5 at the midpoint, near the floor in a 3-sigma spike; 1.0
    when there is not enough history to place today's vol at all."""
    from agentic_trading.config import settings

    z = vol_regime_zscore(market_slice(market_data, as_of_date, asset).log_returns)
    if z is None:
        return 1.0
    cfg = settings.managers
    factor = 1.0 / (1.0 + math.exp(cfg.vol_zscore_slope * (z - cfg.vol_zscore_midpoint)))
    return clamp(factor, cfg.min_scale_factor, 1.0)


# A GARCH(1,1) fit whose implied unconditional vol is outside this band
# around the sample vol is a degenerate optimizer result, not a model.
_GARCH_UNCONDITIONAL_VOL_BAND = (0.3, 3.0)


def conditional_daily_volatility(
    returns: np.ndarray,
    refit_cache: RefitCache,
    cache_key: str,
    as_of_date: pd.Timestamp,
    agent_id: str,
) -> float:
    """One-step-ahead daily conditional volatility of `returns`: a
    GARCH(1,1) fit (`quantcore.risk.fit_garch_11`, refit through
    `refit_cache` at its own cadence) when the fit is usable — converged,
    persistence `alpha + beta` in (0, `settings.analysts.max_garch_persistence`],
    and unconditional vol within 0.3x-3x of the sample vol — otherwise
    the RiskMetrics EWMA (`settings.managers.ewma_lambda`) volatility,
    which needs no estimation and cannot degenerate."""
    from quantcore.risk import fit_garch_11, garch_11_variance
    from quantcore.risk.egarch import ewma_variance

    from agentic_trading.config import settings

    centered = returns - float(np.mean(returns))
    sample_variance = float(np.var(centered, ddof=1))
    ewma_volatility = float(
        np.sqrt(max(float(ewma_variance(centered, settings.managers.ewma_lambda)[-1]), 0.0))
    )
    try:
        omega, alpha, beta = refit_cache.get_or_refit(
            cache_key, as_of_date, lambda: fit_garch_11(returns)
        )
        persistence = alpha + beta
        if not 0.0 < persistence <= settings.analysts.max_garch_persistence:
            raise ValueError(f"persistence {persistence:.4f} outside (0, max]")
        unconditional_ratio = math.sqrt(omega / (1.0 - persistence) / max(sample_variance, 1e-18))
        low, high = _GARCH_UNCONDITIONAL_VOL_BAND
        if not low <= unconditional_ratio <= high:
            raise ValueError(f"unconditional vol {unconditional_ratio:.2f}x sample vol")
        variance_path = garch_11_variance(centered, omega, alpha, beta)
        return float(np.sqrt(max(float(variance_path[-1]), 0.0)))
    except (RuntimeError, ValueError) as exc:
        logger.debug(
            "%s: GARCH(1,1) unusable for %s on %s (%s) — using EWMA volatility %.4f.",
            agent_id,
            cache_key,
            as_of_date.date(),
            exc,
            ewma_volatility,
        )
        return ewma_volatility


def raw_spot(market_data: pd.DataFrame, as_of_date: pd.Timestamp, asset: str) -> float | None:
    """The unadjusted close of `asset` on the last bar at or before
    `as_of_date` — the price basis option strikes are quoted on. Falls
    back to the (adjusted) `close` when the frame carries no `raw_close`
    field; `None` with no bar at all."""
    history = market_data.loc[market_data.index <= as_of_date]
    if history.empty:
        return None
    if isinstance(history.columns, pd.MultiIndex):
        field = "raw_close" if (asset, "raw_close") in history.columns else "close"
        series = history[(asset, field)]
    else:
        field = "raw_close" if "raw_close" in history.columns else "close"
        series = history[field]
    series = series.dropna()
    if series.empty:
        return None
    return float(series.iloc[-1])
