# _theoretical_options.py

"""Shared theoretical option-pricing convention for the CVaR pod.

Every backtestable options-pricing analyst in this pod prices a synthetic,
model-implied option rather than reading a real market quote — there is no
free source for historical option chains (see
`data/loaders/options_loader.py`), so any analyst built around an observed
chain can only run for `as_of_date == today`. This module is the one place
that convention lives, so every analyst applies exactly the same tenor,
strike grid, volatility input, and risk-free rate:

- **Tenor**: fixed 30 calendar days to expiry — the same constant-maturity
  convention CBOE's own VIX methodology uses, so results are comparable to
  how "30-day IV" is understood everywhere else in finance.
- **Strikes**: a moneyness grid (`K = S * moneyness`), not fixed dollar
  spacing or a fixed-delta grid — the standard way to parameterize a vol
  surface without observed market strikes.
- **Volatility**: GARCH(1,1) 1-step-ahead conditional variance, annualized —
  the textbook substitute for implied volatility when no market quote
  exists (Hull, *Options, Futures, and Other Derivatives*).
- **Risk-free rate**: FRED's 3-month T-bill yield (DGS3MO), point-in-time
  via `load_fred_series`, falling back to a fixed constant if FRED fails.
"""

import numpy as np
from quantcore.risk import fit_garch_11, garch_11_variance

from agentic_trading.data.loaders.macro_loader import DGS3MO, load_fred_series
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

TENOR_DAYS = 30
TIME_TO_MATURITY = TENOR_DAYS / 365.0
MONEYNESS_GRID: tuple[float, ...] = (0.90, 0.95, 1.00, 1.05, 1.10)
FALLBACK_RISK_FREE_RATE = 0.02
TRADING_DAYS_PER_YEAR = 252


def annualized_garch_volatility(returns: np.ndarray, agent_id: str, asset: str) -> float:
    """GARCH(1,1) 1-step-ahead conditional volatility on `returns`,
    annualized by sqrt(252). Falls back to the sample stdev, annualized the
    same way, if the GARCH fit fails."""
    mean_return = float(np.mean(returns))
    daily_volatility = float(np.std(returns, ddof=1))
    try:
        omega, alpha, beta = fit_garch_11(returns)
        centered = returns - mean_return
        variance_path = garch_11_variance(centered, omega, alpha, beta)
        daily_volatility = float(np.sqrt(max(float(variance_path[-1]), 0.0)))
    except (RuntimeError, ValueError) as exc:
        logger.debug(
            "%s: GARCH fit failed for %s (%s) — falling back to sample stdev volatility=%.4f.",
            agent_id,
            asset,
            exc,
            daily_volatility,
        )
    return daily_volatility * float(np.sqrt(TRADING_DAYS_PER_YEAR))


def risk_free_rate(as_of_date, agent_id: str) -> float:  # type: ignore[no-untyped-def]
    """Point-in-time 3-month T-bill yield from FRED, falling back to a
    fixed constant (documented here as 2%) if the FRED call fails."""
    try:
        series = load_fred_series(DGS3MO, as_of_date)
        latest = series.dropna()
        if not latest.empty:
            return float(latest.iloc[-1]) / 100.0
    except Exception as exc:
        logger.debug(
            "%s: FRED DGS3MO lookup failed (%s) — falling back to %.2f%%.",
            agent_id,
            exc,
            FALLBACK_RISK_FREE_RATE * 100,
        )
    return FALLBACK_RISK_FREE_RATE


def moneyness_strikes(spot: float) -> list[float]:
    return [spot * moneyness for moneyness in MONEYNESS_GRID]
