# test_garch_convergence_cache.py

"""Non-converged GARCH-family fits are cached for the refit interval; fits with no best attempt
are retried."""

import importlib

import numpy as np
import pandas as pd
import pytest
from quantcore.risk import ConvergenceError

from agentic_trading.agents.analysts.macro_rates_pod._garch_checks import fit_or_convergence_error

PACKAGE = "agentic_trading.agents.analysts.macro_rates_pod"


def _error(params: tuple[float, ...] | None) -> ConvergenceError:
    return ConvergenceError(
        "no start converged",
        params=params,
        log_likelihood=0.0,
        optimizer_message="Iteration limit reached",
    )


def _raiser(error: Exception):
    def _fit(_returns):
        raise error

    return _fit


def _market_data(periods: int = 800) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    returns = np.random.default_rng(1).normal(0.0, 0.01, periods)
    dates = pd.bdate_range("2020-01-01", periods=periods)
    close = 100.0 * np.exp(np.cumsum(returns))
    return pd.DataFrame({"close": close, "log_return": returns}, index=dates), dates


def test_fit_or_convergence_error_returns_error_with_params_and_raises_without() -> None:
    """A ConvergenceError with params is returned; without params it and other errors re-raise."""
    with_params = _error((1.0, 2.0, 3.0, 4.0))
    without_params = _error(None)

    assert fit_or_convergence_error(_raiser(with_params), np.zeros(3)) is with_params
    assert fit_or_convergence_error(lambda _r: "fitted", np.zeros(3)) == "fitted"
    with pytest.raises(ConvergenceError) as raised:
        fit_or_convergence_error(_raiser(without_params), np.zeros(3))
    assert raised.value is without_params
    with pytest.raises(ValueError, match="boom"):
        fit_or_convergence_error(_raiser(ValueError("boom")), np.zeros(3))


@pytest.mark.parametrize(
    ("module_name", "class_name", "fit_name"),
    [
        ("egarch_asymmetric_vol_analyst", "EGARCHAsymmetricVolAnalyst", "egarch_fit"),
        ("gjr_garch_crash_risk_analyst", "GJRGARCHCrashRiskAnalyst", "gjr_garch_fit"),
    ],
)
def test_analyst_caches_non_converged_fit_but_retries_fit_without_params(
    monkeypatch: pytest.MonkeyPatch, module_name: str, class_name: str, fit_name: str
) -> None:
    """Non-converged fits run once per interval and read unavailable; paramless ones retry daily."""
    mod = importlib.import_module(f"{PACKAGE}.{module_name}")
    market_data, dates = _market_data()
    days = dates[-3:]
    calls: list[int] = []

    def _patch(params: tuple[float, ...] | None) -> None:
        calls.clear()

        def _fit(_returns):
            calls.append(1)
            raise _error(params)

        monkeypatch.setattr(mod, fit_name, _fit)

    _patch((1e-6, 0.05, 0.1, 0.8))
    analyst = getattr(mod, class_name)()
    signals = [analyst.generate_signal("AAPL", market_data, day) for day in days]

    assert len(calls) == 1
    for signal in signals:
        assert signal.direction == "flat"
        assert signal.status == "unavailable"
        assert "did not converge" in signal.rationale

    _patch(None)
    analyst = getattr(mod, class_name)()
    signals = [analyst.generate_signal("AAPL", market_data, day) for day in days]

    assert len(calls) == 3
    for signal in signals:
        assert signal.direction == "flat"
        assert signal.status == "error"
