# test_garch_fit_outcomes.py

"""The EGARCH and GJR-GARCH analysts tell the two `ConvergenceError` cases of quantcore's fits
apart. SciPy's SLSQP (the optimizer `arch` runs for both) exits with status 0 exactly when its
message is "Optimization terminated successfully", and quantcore raises either for a non-zero
status or for a converged fit outside the model's stationary region. An optimizer failure is an
error signal naming the optimizer message; a converged fit that is non-stationary (EGARCH
|beta| >= 1, GJR persistence alpha + beta + gamma/2 >= 1) is an abstention giving that value;
a converged fit outside the admissible region for another reason is an error. Each outcome is
cached for the refit interval like any fit."""

import importlib
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
from quantcore.risk import ConvergenceError
from quantcore.risk.egarch import egarch_fit, gjr_garch_fit
from scipy.optimize import minimize

from agentic_trading.agents.analysts.macro_rates_pod._garch_checks import (
    SLSQP_SUCCESS_MESSAGE,
)
from agentic_trading.agents.schemas import AgentSignal

PACKAGE = "agentic_trading.agents.analysts.macro_rates_pod"
_ANALYSTS = {
    "egarch": ("egarch_asymmetric_vol_analyst", "EGARCHAsymmetricVolAnalyst", "egarch_fit"),
    "gjr": ("gjr_garch_crash_risk_analyst", "GJRGARCHCrashRiskAnalyst", "gjr_garch_fit"),
}
_LABEL = {"egarch": "EGARCH", "gjr": "GJR-GARCH"}


def _module(name: str) -> ModuleType:
    return importlib.import_module(f"{PACKAGE}.{_ANALYSTS[name][0]}")


def _market_data(returns: np.ndarray) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    dates = pd.date_range("2020-01-01", periods=returns.size, freq="D")
    close = 100.0 * np.exp(np.cumsum(returns))
    return pd.DataFrame({"close": close, "log_return": returns}, index=dates), dates


def _noise(periods: int = 400) -> np.ndarray:
    return np.random.default_rng(3).normal(0.0, 0.01, periods)


def _signals(
    monkeypatch: pytest.MonkeyPatch, name: str, error: ConvergenceError, days: int = 1
) -> tuple[list[AgentSignal], int]:
    """The `name` analyst's signals on the last `days` days of a noise series, its fit
    stubbed to raise `error`, and how many times the fit ran."""
    _, class_name, fit_name = _ANALYSTS[name]
    module = _module(name)
    calls: list[int] = []

    def _fit(_returns: np.ndarray) -> object:
        calls.append(1)
        raise error

    monkeypatch.setattr(module, fit_name, _fit)
    market_data, dates = _market_data(_noise())
    analyst = getattr(module, class_name)()
    return [analyst.generate_signal("AAPL", market_data, day) for day in dates[-days:]], len(calls)


def _error(params: tuple[float, ...], message: str) -> ConvergenceError:
    return ConvergenceError(
        "did not converge to a feasible point",
        params=params,
        log_likelihood=1_000.0,
        optimizer_message=message,
    )


# Inside each model's stationary region, and on either side of it.
_STATIONARY = {"egarch": (-0.2, 0.1, -0.05, 0.95), "gjr": (1e-6, 0.05, 0.1, 0.8)}
_NON_STATIONARY = {"egarch": (-0.01, 0.13, 0.04, 1.0), "gjr": (2e-6, 0.06, 0.12, 0.94)}
_NON_STATIONARY_VALUE = {"egarch": "|beta| 1.000000", "gjr": "persistence 1.060000"}


def test_slsqp_success_message_is_scipys_status_zero_message() -> None:
    """The message the classification keys on is the one SciPy's SLSQP gives on status 0,
    and only then."""
    converged = minimize(lambda x: float((x[0] - 1.0) ** 2), [0.0], method="SLSQP")
    capped = minimize(
        lambda x: float((x[0] - 1.0) ** 2), [0.0], method="SLSQP", options={"maxiter": 1}
    )

    assert converged.status == 0
    assert converged.message == SLSQP_SUCCESS_MESSAGE
    assert capped.status != 0
    assert capped.message != SLSQP_SUCCESS_MESSAGE


@pytest.mark.parametrize("name", ["egarch", "gjr"])
@pytest.mark.parametrize("region", ["stationary", "non_stationary"])
def test_optimizer_failure_is_an_error_naming_the_optimizer_message(
    monkeypatch: pytest.MonkeyPatch, name: str, region: str
) -> None:
    """A non-zero optimizer status is an error whatever the best attempt's parameters, and the
    rationale names the optimizer message."""
    params = _STATIONARY[name] if region == "stationary" else _NON_STATIONARY[name]
    signals, _ = _signals(monkeypatch, name, _error(params, "Iteration limit reached"))

    (signal,) = signals
    assert signal.status == "error"
    assert signal.direction == "flat" and signal.confidence == 0.0
    assert signal.rationale == (
        f"{_LABEL[name]} parameters rejected: MLE fit did not converge "
        "(optimizer: Iteration limit reached)"
    )


@pytest.mark.parametrize("name", ["egarch", "gjr"])
def test_converged_non_stationary_fit_abstains_with_its_persistence(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    """A converged fit outside the stationary region is an abstention (active, flat,
    confidence 0) whose rationale says it is non-stationary and gives the persistence or
    beta."""
    signals, _ = _signals(monkeypatch, name, _error(_NON_STATIONARY[name], SLSQP_SUCCESS_MESSAGE))

    (signal,) = signals
    assert signal.status == "active"
    assert signal.direction == "flat" and signal.confidence == 0.0
    assert signal.rationale == (
        f"{_LABEL[name]} parameters rejected: converged fit is non-stationary "
        f"({_NON_STATIONARY_VALUE[name]} >= 1)"
    )


@pytest.mark.parametrize(
    ("name", "params"),
    [
        ("egarch", (float("nan"), 0.1, -0.05, 1.2)),
        ("gjr", (0.0, 0.06, 0.12, 0.94)),
        ("gjr", (1e-6, -0.2, 0.1, 1.2)),
    ],
    ids=["egarch_non_finite", "gjr_zero_omega", "gjr_negative_alpha"],
)
def test_converged_inadmissible_fit_is_an_error(
    monkeypatch: pytest.MonkeyPatch, name: str, params: tuple[float, ...]
) -> None:
    """A converged fit outside the admissible region for a reason other than
    non-stationarity is an error."""
    signals, _ = _signals(monkeypatch, name, _error(params, SLSQP_SUCCESS_MESSAGE))

    (signal,) = signals
    assert signal.status == "error"
    assert signal.rationale.startswith(
        f"{_LABEL[name]} parameters rejected: converged fit outside the admissible region"
    )


@pytest.mark.parametrize("name", ["egarch", "gjr"])
@pytest.mark.parametrize("message", ["Iteration limit reached", SLSQP_SUCCESS_MESSAGE])
def test_both_outcomes_are_cached_for_the_refit_interval(
    monkeypatch: pytest.MonkeyPatch, name: str, message: str
) -> None:
    """Either outcome runs the fit once over three days inside the refit interval and gives
    the same status each day."""
    signals, calls = _signals(monkeypatch, name, _error(_NON_STATIONARY[name], message), days=3)

    assert calls == 1
    expected = "error" if message != SLSQP_SUCCESS_MESSAGE else "active"
    assert [signal.status for signal in signals] == [expected] * 3
    assert len({signal.rationale for signal in signals}) == 1


def _integrating_returns() -> np.ndarray:
    """300 days of Gaussian returns whose vol grows twelvefold, steadily: every
    GARCH-family fit on them converges to an integrated process."""
    rng = np.random.default_rng(2)
    return rng.normal(0.0, 1.0, 300) * 0.005 * np.exp(np.linspace(0.0, 2.5, 300))


@pytest.mark.parametrize("name", ["egarch", "gjr"])
def test_real_integrating_series_converges_non_stationary_and_abstains(name: str) -> None:
    """On a real series with steadily growing vol, quantcore's fit converges (SLSQP status 0)
    outside the stationary region, and the analyst abstains on it."""
    returns = _integrating_returns()
    fit = egarch_fit if name == "egarch" else gjr_garch_fit
    with pytest.raises(ConvergenceError) as raised:
        fit(returns)
    assert raised.value.optimizer_message == SLSQP_SUCCESS_MESSAGE

    market_data, dates = _market_data(returns)
    signal = getattr(_module(name), _ANALYSTS[name][1])().generate_signal(
        "AAPL", market_data, dates[-1]
    )

    assert signal.status == "active"
    assert signal.direction == "flat" and signal.confidence == 0.0
    assert "converged fit is non-stationary" in signal.rationale
