# _garch_checks.py

import math
from collections.abc import Callable

import numpy as np
from quantcore.risk import ConvergenceError
from quantcore.risk.egarch import EGARCHParams, GJRGARCHParams

from agentic_trading.config import settings

# A fitted unconditional vol this far from the sample std means the MLE
# landed somewhere the data doesn't support, however good the likelihood.
UNCONDITIONAL_VOL_RATIO_BOUNDS = (0.3, 3.0)
# EGARCH beta below this is a near-memoryless process; the recursion is
# then reacting to single-day noise rather than tracking a vol regime.
MIN_EGARCH_BETA = 0.5
# Bounds on the computed conditional-std path: an intra-window max/min
# ratio this large, or an end-point this far from the sample std, comes
# from a recursion that has blown up, not from a real vol regime.
MAX_SIGMA_PATH_RANGE_RATIO = 20.0
FINAL_SIGMA_RATIO_BOUNDS = (0.2, 5.0)


def fit_or_convergence_error[FittedParams: (EGARCHParams, GJRGARCHParams)](
    fit: Callable[[np.ndarray], FittedParams], returns: np.ndarray
) -> FittedParams | ConvergenceError:
    """Runs `fit(returns)` (`egarch_fit` or `gjr_garch_fit`) and returns the
    fitted parameters, or the `ConvergenceError` it raised when no starting
    point converged but a best attempt exists (`params` set). Returning the
    error as a value lets `RefitCache` keep a non-converged outcome for the
    whole refit interval, like any fit. A `ConvergenceError` without a best
    attempt (no starting point gave a finite likelihood) is re-raised, as is
    every other exception."""
    try:
        return fit(returns)
    except ConvergenceError as exc:
        if exc.params is None:
            raise
        return exc


def _ratio_outside(value: float, reference: float, bounds: tuple[float, float]) -> bool:
    if not math.isfinite(value) or reference <= 0.0:
        return True
    ratio = value / reference
    return ratio < bounds[0] or ratio > bounds[1]


def check_gjr_params(fitted: GJRGARCHParams, sample_std: float) -> str | None:
    """Returns the name of the first sanity check a GJR-GARCH(1,1) fit
    fails, or None if it passes all of them: positivity and
    non-negativity constraints on the coefficients, persistence
    `alpha + beta + gamma/2` at most `max_garch_persistence`, and the
    implied unconditional vol within `UNCONDITIONAL_VOL_RATIO_BOUNDS` of
    `sample_std`."""
    if fitted.omega <= 0.0:
        return "omega <= 0"
    if fitted.alpha < 0.0 or fitted.beta < 0.0:
        return "negative alpha or beta"
    if fitted.alpha + fitted.gamma < 0.0:
        return "alpha + gamma < 0"
    persistence = fitted.alpha + fitted.beta + 0.5 * fitted.gamma
    if persistence > settings.analysts.max_garch_persistence:
        return f"persistence {persistence:.4f} above max"
    unconditional_vol = math.sqrt(fitted.omega / (1.0 - persistence))
    if _ratio_outside(unconditional_vol, sample_std, UNCONDITIONAL_VOL_RATIO_BOUNDS):
        return f"implied unconditional vol {unconditional_vol:.4f} far from sample std"
    return None


def check_egarch_params(fitted: EGARCHParams, sample_std: float) -> str | None:
    """Returns the name of the first sanity check an EGARCH(1,1) fit fails,
    or None if it passes all of them: beta within
    `[MIN_EGARCH_BETA, max_garch_persistence]`, and the implied
    unconditional vol `exp(omega / (2 (1 - beta)))` within
    `UNCONDITIONAL_VOL_RATIO_BOUNDS` of `sample_std`."""
    if fitted.beta < MIN_EGARCH_BETA:
        return f"beta {fitted.beta:.4f} below minimum"
    if fitted.beta > settings.analysts.max_garch_persistence:
        return f"beta {fitted.beta:.4f} above max persistence"
    unconditional_vol = math.exp(fitted.omega / (2.0 * (1.0 - fitted.beta)))
    if _ratio_outside(unconditional_vol, sample_std, UNCONDITIONAL_VOL_RATIO_BOUNDS):
        return f"implied unconditional vol {unconditional_vol:.4f} far from sample std"
    return None


def check_sigma_path(conditional_std: np.ndarray, sample_std: float) -> str | None:
    """Returns the name of the first sanity check a conditional-std path
    fails, or None if it passes all of them: every value finite and
    positive, max/min ratio below `MAX_SIGMA_PATH_RANGE_RATIO`, and the
    final value within `FINAL_SIGMA_RATIO_BOUNDS` of `sample_std`."""
    if conditional_std.size == 0 or not np.all(np.isfinite(conditional_std)):
        return "non-finite conditional variance path"
    lowest = float(np.min(conditional_std))
    highest = float(np.max(conditional_std))
    if lowest <= 0.0 or highest / lowest >= MAX_SIGMA_PATH_RANGE_RATIO:
        return f"conditional vol path range ratio {highest / max(lowest, 1e-12):.1f} too wide"
    if _ratio_outside(float(conditional_std[-1]), sample_std, FINAL_SIGMA_RATIO_BOUNDS):
        return f"final conditional vol {conditional_std[-1]:.4f} far from sample std"
    return None
