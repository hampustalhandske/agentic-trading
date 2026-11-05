# test_quantcore_explicit_options.py

"""GARCH-family variance paths run on the window's demeaned residuals from `variance_backcast`,
and the statsmodels-delegated quantcore calls pass their options explicitly."""

from typing import Any

import numpy as np
import pandas as pd
import pytest
import quantcore.risk
from quantcore.risk import (
    EGARCHParams,
    GJRGARCHParams,
    egarch_11_variance,
    garch_11_variance,
    gjr_garch_11_variance,
    variance_backcast,
)

from agentic_trading.agents._market_math import RefitCache, clamp, conditional_daily_volatility
from agentic_trading.agents.analysts.factor_pod import newey_west_significance_analyst
from agentic_trading.agents.analysts.macro_rates_pod import (
    arima_forecast_analyst,
    egarch_asymmetric_vol_analyst,
    gjr_garch_crash_risk_analyst,
)
from agentic_trading.agents.analysts.stat_arb_pod import _pairs_screening
from agentic_trading.config import settings

DRIFT = 0.002


class _Spy:
    """Records the positional and keyword arguments of each call and delegates to `fn`."""

    def __init__(self, fn: Any) -> None:
        self.fn = fn
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((args, kwargs))
        return self.fn(*args, **kwargs)


def _drifting_returns(periods: int = 800, seed: int = 1) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, 0.01, periods) + DRIFT


def _market_data(returns: np.ndarray) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    dates = pd.bdate_range("2020-01-01", periods=returns.size)
    close = 100.0 * np.exp(np.cumsum(returns))
    return pd.DataFrame({"close": close, "log_return": returns}, index=dates), dates


def test_egarch_path_and_shock_use_demeaned_residuals(monkeypatch: pytest.MonkeyPatch) -> None:
    """EGARCH runs on epsilon from variance_backcast; z_t-1 is epsilon[-1] / sigma[-1]."""
    mod = egarch_asymmetric_vol_analyst
    returns = _drifting_returns()
    market_data, dates = _market_data(returns)
    window = returns[-settings.analysts.risk_window_garch :]
    epsilon = window - window.mean()
    sample_std = float(np.std(window, ddof=1))
    beta = 0.95
    params = EGARCHParams(
        omega=2.0 * (1.0 - beta) * float(np.log(sample_std)),
        alpha=0.1,
        gamma=0.1,
        beta=beta,
        log_likelihood=0.0,
    )
    spy = _Spy(mod.egarch_11_variance)
    monkeypatch.setattr(mod, "egarch_fit", lambda _returns: params)
    monkeypatch.setattr(mod, "egarch_11_variance", spy)

    signal = mod.EGARCHAsymmetricVolAnalyst().generate_signal("AAPL", market_data, dates[-1])

    assert len(spy.calls) == 1
    args, kwargs = spy.calls[0]
    np.testing.assert_allclose(args[0], epsilon)
    assert args[1:] == (params.omega, params.alpha, params.gamma, params.beta)
    assert kwargs["presample_variance"] == pytest.approx(variance_backcast(window))

    log_variance = egarch_11_variance(
        epsilon,
        params.omega,
        params.alpha,
        params.gamma,
        params.beta,
        presample_variance=variance_backcast(window),
    )
    sigma_last = float(np.sqrt(np.exp(log_variance[-1])))
    z_last = float(epsilon[-1] / sigma_last)
    leverage = params.gamma * z_last
    assert f"{window[-1] / sigma_last:.3f}" != f"{z_last:.3f}"
    assert signal.status == "active"
    assert signal.rationale == (
        f"EGARCH leverage contribution gamma*z_t-1={leverage:.4f} (z_t-1={z_last:.3f})"
    )
    expected_confidence = (
        clamp(abs(z_last) - mod.EXPECTED_ABS_STANDARD_NORMAL) if leverage < 0.0 else 0.0
    )
    assert signal.confidence == pytest.approx(expected_confidence)


def test_gjr_path_and_shock_use_demeaned_residuals(monkeypatch: pytest.MonkeyPatch) -> None:
    """GJR-GARCH runs on epsilon from variance_backcast and judges bad news on epsilon[-1]."""
    mod = gjr_garch_crash_risk_analyst
    returns = _drifting_returns()
    returns[-2] = -0.06
    returns[-1] = 0.0005
    market_data, dates = _market_data(returns)
    window = returns[-settings.analysts.risk_window_garch :]
    epsilon = window - window.mean()
    sample_std = float(np.std(window, ddof=1))
    params = GJRGARCHParams(
        omega=(1.0 - 0.98) * sample_std**2, alpha=0.03, gamma=0.1, beta=0.9, log_likelihood=0.0
    )
    spy = _Spy(mod.gjr_garch_11_variance)
    monkeypatch.setattr(mod, "gjr_garch_fit", lambda _returns: params)
    monkeypatch.setattr(mod, "gjr_garch_11_variance", spy)

    signal = mod.GJRGARCHCrashRiskAnalyst().generate_signal("AAPL", market_data, dates[-1])

    assert len(spy.calls) == 1
    args, kwargs = spy.calls[0]
    np.testing.assert_allclose(args[0], epsilon)
    assert args[1:] == (params.omega, params.alpha, params.gamma, params.beta)
    assert kwargs["presample_variance"] == pytest.approx(variance_backcast(window))

    variance = gjr_garch_11_variance(
        epsilon,
        params.omega,
        params.alpha,
        params.gamma,
        params.beta,
        presample_variance=variance_backcast(window),
    )
    jump = float(np.sqrt(variance[-1]) / np.sqrt(np.mean(variance)))
    threshold = settings.analysts.vol_jump_threshold
    assert window[-1] > 0.0 > epsilon[-1]
    assert jump > threshold
    assert signal.direction == "short"
    assert signal.rationale == f"vol jump ratio={jump:.2f}, driven by asymmetric bad-news term"
    assert signal.confidence == pytest.approx(clamp((jump - threshold) / threshold))


def test_conditional_daily_volatility_uses_backcast_on_centered_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The GARCH(1,1) path runs on centered returns with presample_variance=variance_backcast."""
    returns = _drifting_returns(500)
    centered = returns - returns.mean()
    omega, alpha, beta = 0.05 * float(np.var(centered, ddof=1)), 0.05, 0.9
    spy = _Spy(garch_11_variance)
    monkeypatch.setattr(quantcore.risk, "fit_garch_11", lambda _returns: (omega, alpha, beta))
    monkeypatch.setattr(quantcore.risk, "garch_11_variance", spy)

    volatility = conditional_daily_volatility(
        returns, RefitCache(), "AAPL", pd.Timestamp("2024-01-02"), "test_agent"
    )

    assert len(spy.calls) == 1
    args, kwargs = spy.calls[0]
    np.testing.assert_allclose(args[0], centered)
    assert args[1:] == (omega, alpha, beta)
    assert kwargs["presample_variance"] == pytest.approx(variance_backcast(returns))
    expected = garch_11_variance(
        centered, omega, alpha, beta, presample_variance=variance_backcast(returns)
    )
    assert volatility == pytest.approx(float(np.sqrt(expected[-1])))


def test_pairs_screening_passes_adf_options_explicitly(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both legs' ADF tests pass max_lags=1, trend="c" and autolag=None as keywords."""
    formation_days = 100
    rng = np.random.default_rng(0)
    common, idio_a, idio_b = (rng.normal(0.0, 0.01, 300) for _ in range(3))
    returns_a = 0.9 * common + np.sqrt(1 - 0.81) * idio_a
    returns_b = 0.9 * common + np.sqrt(1 - 0.81) * idio_b
    dates = pd.bdate_range("2023-01-01", periods=300)
    close_a = 100 * np.exp(np.cumsum(returns_a))
    close_b = 100 * np.exp(np.cumsum(returns_b))
    frame = pd.concat(
        {
            "A": pd.DataFrame({"close": close_a}, index=dates),
            "B": pd.DataFrame({"close": close_b}, index=dates),
        },
        axis=1,
    )
    spy = _Spy(_pairs_screening.adf_test)
    monkeypatch.setattr(_pairs_screening, "adf_test", spy)

    _pairs_screening.rescreen_pairs(frame, dates[-1], (("A", "B"),), formation_days)

    assert len(spy.calls) == 2
    for (args, kwargs), close in zip(spy.calls, (close_a, close_b), strict=True):
        np.testing.assert_allclose(args[0], np.cumsum(np.diff(np.log(close))[-formation_days:]))
        options = {name: kwargs.get(name, "missing") for name in ("max_lags", "trend", "autolag")}
        assert options == {"max_lags": 1, "trend": "c", "autolag": None}


def test_arima_ljung_box_calls_pass_model_df_explicitly(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pre-fit and residual Ljung-Box tests both pass model_df=0 as a keyword."""
    mod = arima_forecast_analyst
    noise = np.random.default_rng(3).normal(0.0, 0.01, 300)
    returns = np.zeros(300)
    for t in range(1, 300):
        returns[t] = 0.5 * returns[t - 1] + noise[t]
    market_data, dates = _market_data(returns)
    spy = _Spy(mod.ljung_box_test)
    monkeypatch.setattr(mod, "ljung_box_test", spy)

    mod.ARIMAForecastAnalyst(p=1, d=1, q=0).generate_signal("AAPL", market_data, dates[-1])

    assert len(spy.calls) == 2
    assert [kwargs.get("model_df", "missing") for _, kwargs in spy.calls] == [0, 0]


def test_newey_west_cov_passes_small_sample_correction_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The HAC covariance call passes small_sample_correction=False as a keyword."""
    mod = newey_west_significance_analyst
    market_data, dates = _market_data(np.random.default_rng(4).normal(0.0, 0.01, 150))
    spy = _Spy(mod.newey_west_cov)
    monkeypatch.setattr(mod, "newey_west_cov", spy)

    mod.NeweyWestSignificanceAnalyst().generate_signal("AAPL", market_data, dates[-1])

    assert len(spy.calls) == 1
    args, kwargs = spy.calls[0]
    assert len(args) == 3
    assert kwargs.get("small_sample_correction", "missing") is False
