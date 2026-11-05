# _garch_checks.py

import math
from collections.abc import Callable
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel
from quantcore.risk import ConvergenceError
from quantcore.risk.egarch import EGARCHParams, GJRGARCHParams

from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.schemas import AgentSignal
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


class CheckFailure(BaseModel):
    """The first sanity check a GARCH fit or conditional-vol path fails.
    `kind` is "error" when the fit or recursion produced something
    inadmissible (a failure of the model) and "abstain" when the result is
    admissible but too extreme to trade on (the model declines to bet);
    `reason` names the check."""

    kind: Literal["error", "abstain"]
    reason: str


def check_gjr_params(fitted: GJRGARCHParams, sample_std: float) -> CheckFailure | None:
    """Returns the first sanity check a GJR-GARCH(1,1) fit fails, or None
    if it passes all of them. An inadmissible model (omega <= 0, a
    negative alpha or beta, or alpha + gamma < 0) is an "error"; a
    persistence `alpha + beta + gamma/2` above `max_garch_persistence`, or
    an implied unconditional vol outside `UNCONDITIONAL_VOL_RATIO_BOUNDS` of
    `sample_std`, is an "abstain"."""
    if fitted.omega <= 0.0:
        return CheckFailure(kind="error", reason="omega <= 0")
    if fitted.alpha < 0.0 or fitted.beta < 0.0:
        return CheckFailure(kind="error", reason="negative alpha or beta")
    if fitted.alpha + fitted.gamma < 0.0:
        return CheckFailure(kind="error", reason="alpha + gamma < 0")
    persistence = fitted.alpha + fitted.beta + 0.5 * fitted.gamma
    if persistence > settings.analysts.max_garch_persistence:
        return CheckFailure(kind="abstain", reason=f"persistence {persistence:.4f} above max")
    unconditional_vol = math.sqrt(fitted.omega / (1.0 - persistence))
    if _ratio_outside(unconditional_vol, sample_std, UNCONDITIONAL_VOL_RATIO_BOUNDS):
        return CheckFailure(
            kind="abstain",
            reason=f"implied unconditional vol {unconditional_vol:.4f} far from sample std",
        )
    return None


def check_egarch_params(fitted: EGARCHParams, sample_std: float) -> CheckFailure | None:
    """Returns the first sanity check an EGARCH(1,1) fit fails, or None if
    it passes all of them. A non-finite omega or beta is an "error" (the
    fit returned no usable model). Every other failure is an "abstain":
    beta outside `[MIN_EGARCH_BETA, max_garch_persistence]`, or the
    implied unconditional vol `exp(omega / (2 (1 - beta)))` outside
    `UNCONDITIONAL_VOL_RATIO_BOUNDS` of `sample_std`. The vol check
    compares logs, `omega / (2 (1 - beta))` against the log bounds, so a
    fitted vol far above the band is judged on a finite number and never
    overflows."""
    if not (math.isfinite(fitted.omega) and math.isfinite(fitted.beta)):
        return CheckFailure(kind="error", reason="non-finite EGARCH omega or beta")
    if fitted.beta < MIN_EGARCH_BETA:
        return CheckFailure(kind="abstain", reason=f"beta {fitted.beta:.4f} below minimum")
    if fitted.beta > settings.analysts.max_garch_persistence:
        return CheckFailure(kind="abstain", reason=f"beta {fitted.beta:.4f} above max persistence")
    log_vol = fitted.omega / (2.0 * (1.0 - fitted.beta))
    low, high = UNCONDITIONAL_VOL_RATIO_BOUNDS
    if sample_std <= 0.0 or not (
        math.log(low * sample_std) <= log_vol <= math.log(high * sample_std)
    ):
        return CheckFailure(
            kind="abstain",
            reason=(
                f"implied unconditional log-vol {log_vol:.4f} far from log sample std "
                f"{math.log(sample_std) if sample_std > 0.0 else float('-inf'):.4f}"
            ),
        )
    return None


def check_sigma_path(conditional_std: np.ndarray, sample_std: float) -> CheckFailure | None:
    """Returns the first sanity check a conditional-std path fails, or None
    if it passes all of them. An empty path, or one with a non-finite or
    non-positive value, is an "error" (the recursion broke); a max/min
    ratio at or above `MAX_SIGMA_PATH_RANGE_RATIO`, or a final value
    outside `FINAL_SIGMA_RATIO_BOUNDS` of `sample_std`, is an "abstain"."""
    if conditional_std.size == 0 or not np.all(np.isfinite(conditional_std)):
        return CheckFailure(kind="error", reason="non-finite conditional variance path")
    lowest = float(np.min(conditional_std))
    highest = float(np.max(conditional_std))
    if lowest <= 0.0:
        return CheckFailure(
            kind="error", reason=f"non-positive conditional vol {lowest:.4g} in path"
        )
    if highest / lowest >= MAX_SIGMA_PATH_RANGE_RATIO:
        return CheckFailure(
            kind="abstain",
            reason=f"conditional vol path range ratio {highest / lowest:.1f} too wide",
        )
    if _ratio_outside(float(conditional_std[-1]), sample_std, FINAL_SIGMA_RATIO_BOUNDS):
        return CheckFailure(
            kind="abstain",
            reason=f"final conditional vol {conditional_std[-1]:.4f} far from sample std",
        )
    return None


def rejection_signal(
    analyst: AnalystAgent,
    asset: str,
    as_of_date: pd.Timestamp,
    rationale: str,
    failure: CheckFailure,
) -> AgentSignal:
    """The flat signal `analyst` returns when a fit or path fails a sanity
    check: an error signal for a `failure` of kind "error", an abstention
    for one of kind "abstain"."""
    build = analyst.error_signal if failure.kind == "error" else analyst.abstain_signal
    return build(asset, as_of_date, rationale)
