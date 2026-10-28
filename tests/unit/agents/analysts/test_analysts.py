import numpy as np
import pandas as pd
import pytest

from agentic_trading.agents.analysts import (
    ARIMAForecastAnalyst,
    BlackScholesValuationAnalyst,
    CalendarSeasonalityAnalyst,
    CIRRateRegimeAnalyst,
    EGARCHAsymmetricVolAnalyst,
    FamaFrenchFactorAnalyst,
    FamaMacbethAnalyst,
    GBMMomentumAnalyst,
    GJRGARCHCrashRiskAnalyst,
    GreeksDeltaHedgeAnalyst,
    HestonVolSurfaceAnalyst,
    KalmanHedgeRatioAnalyst,
    LeadLagNetworkCentralityAnalyst,
    MacroVarSpilloverAnalyst,
    MonteCarloValuationAnalyst,
    NeweyWestSignificanceAnalyst,
    OUMeanReversionAnalyst,
    PairsCointegrationAnalyst,
    RegimeHMMAnalyst,
    RiskMaximizingAnalyst,
    RiskMinimizingAnalyst,
    SecFundamentalsAnalyst,
    ShortInterestSqueezeAnalyst,
)
from agentic_trading.agents.base import AnalystAgent

ALL_ANALYST_CLASSES = (
    RiskMaximizingAnalyst,
    RiskMinimizingAnalyst,
    BlackScholesValuationAnalyst,
    MonteCarloValuationAnalyst,
    GBMMomentumAnalyst,
    GreeksDeltaHedgeAnalyst,
    HestonVolSurfaceAnalyst,
    OUMeanReversionAnalyst,
    PairsCointegrationAnalyst,
    KalmanHedgeRatioAnalyst,
    RegimeHMMAnalyst,
    EGARCHAsymmetricVolAnalyst,
    GJRGARCHCrashRiskAnalyst,
    ARIMAForecastAnalyst,
    MacroVarSpilloverAnalyst,
    CIRRateRegimeAnalyst,
    FamaFrenchFactorAnalyst,
    FamaMacbethAnalyst,
    NeweyWestSignificanceAnalyst,
    SecFundamentalsAnalyst,
    ShortInterestSqueezeAnalyst,
    CalendarSeasonalityAnalyst,
    LeadLagNetworkCentralityAnalyst,
)


def test_analysts_imports() -> None:
    for analyst_cls in ALL_ANALYST_CLASSES:
        assert analyst_cls.__name__


def test_analysts_can_be_constructed_with_defaults() -> None:
    for analyst_cls in ALL_ANALYST_CLASSES:
        instance = analyst_cls()
        assert isinstance(instance.agent_id, str) and instance.agent_id


def test_analysts_behave_as_analysts() -> None:
    for analyst_cls in ALL_ANALYST_CLASSES:
        assert issubclass(analyst_cls, AnalystAgent)


def test_macro_var_spillover_drops_non_finite_rows(monkeypatch) -> None:
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    market_data = pd.DataFrame(
        {"close": np.arange(100.0, 130.0), "log_return": np.linspace(-0.01, 0.01, 30)},
        index=dates,
    )
    macro = pd.Series(np.linspace(10.0, 20.0, 30), index=dates)
    macro.iloc[5] = np.nan

    monkeypatch.setattr(
        "agentic_trading.agents.analysts.macro_rates_pod.macro_var_spillover_analyst."
        "load_fred_series_from_lakehouse",
        lambda _series_id, _as_of_date: macro,
    )
    signal = MacroVarSpilloverAnalyst(use_yield_curve_slope=False).generate_signal(
        "AAPL", market_data, dates[-1]
    )

    assert signal.direction == "flat"
    assert signal.confidence == 0.0


def test_newey_west_current_predictor_uses_todays_window_not_yesterdays(monkeypatch) -> None:
    """current_predictor_value must be today's own trailing-momentum window
    (through the most recent return), not the window ending one day earlier.
    Constructed so the two windows have opposite sign, so a regression to the
    stale-by-one-day predictor flips the reported direction."""
    from agentic_trading.agents.analysts.factor_pod import newey_west_significance_analyst as mod

    dates = pd.date_range("2024-01-01", periods=150, freq="D")
    returns = np.full(150, 1e-6)
    returns[-6] = -5.0  # only in yesterday's (stale) 5-day window
    returns[-1] = 5.0  # only in today's (correct) 5-day window
    market_data = pd.DataFrame(
        {"close": np.arange(100.0, 250.0), "log_return": returns}, index=dates
    )

    monkeypatch.setattr(
        mod, "ols", lambda _y, _x: (np.array([0.0, 1.0]), np.zeros(_x.shape[0]), 1.0)
    )
    monkeypatch.setattr(mod, "newey_west_cov", lambda *_args: np.array([[0.0, 0.0], [0.0, 0.01]]))

    signal = mod.NeweyWestSignificanceAnalyst().generate_signal("AAPL", market_data, dates[-1])

    # today's window sums to +5.0 (correct); yesterday's stale window would
    # have summed to -5.0 and flipped this to "short".
    assert signal.direction == "long"


def test_sec_fundamentals_t_stat_matches_classical_ols_standard_error(monkeypatch) -> None:
    """t_stat must come from the classical OLS SE (sqrt of the diagonal of
    residual_variance * (X'X)^-1)), not from ols()'s residuals vector
    misread as a per-coefficient standard error."""
    from agentic_trading.agents.analysts.event_credit_pod import sec_fundamentals_analyst as mod

    rng = np.random.default_rng(0)
    values = 2.0 + 0.3 * np.arange(8) + rng.normal(0.0, 0.05, 8)
    period_ends = pd.date_range("2022-03-31", periods=8, freq="91D")
    filed_dates = period_ends + pd.Timedelta(days=30)
    facts = pd.DataFrame(
        {
            "concept": ["EarningsPerShareDiluted"] * 8,
            "unit": "USD/shares",
            "period_start": period_ends - pd.Timedelta(days=90),
            "period_end": period_ends,
            "filed": filed_dates,
            "available_date": filed_dates,
            "value": values,
        }
    )
    monkeypatch.setattr(mod, "load_company_facts_from_lakehouse", lambda _asset, _as_of: facts)

    signal = mod.SecFundamentalsAnalyst().generate_signal(
        "AAPL", pd.DataFrame(), pd.Timestamp(filed_dates[-1]) + pd.Timedelta(days=1)
    )

    periods = np.asarray((period_ends - period_ends[0]).days, dtype=np.float64) / 91.3125
    design_matrix = np.column_stack([np.ones(8), periods])
    coefs, _, _, _ = np.linalg.lstsq(design_matrix, values, rcond=None)
    residuals = values - design_matrix @ coefs
    residual_variance = float(residuals @ residuals) / (8 - 2)
    cov = residual_variance * np.linalg.inv(design_matrix.T @ design_matrix)
    expected_t = float(coefs[1]) / float(np.sqrt(cov[1, 1]))

    # rationale formats t to 2 decimals, so compare at that precision
    reported_t = float(signal.rationale.split("t=")[1].split(")")[0].split(" ")[0])
    assert reported_t == pytest.approx(expected_t, abs=5e-3)


def _vol_regime_market_data(spike: bool) -> pd.DataFrame:
    """300 calm days (0.5% vol), optionally ending in a 10-day 4%-vol
    sell-off — enough history for the vol-regime z-score to place today."""
    rng = np.random.default_rng(3)
    calm = rng.normal(0.0, 0.005, 300)
    if spike:
        calm[-10:] = -np.abs(rng.normal(0.0, 0.04, 10))
    dates = pd.bdate_range("2023-01-02", periods=300)
    return pd.DataFrame({"close": 100.0 * np.exp(np.cumsum(calm)), "log_return": calm}, index=dates)


def test_risk_minimizing_is_long_calm_and_short_a_vol_spike() -> None:
    from agentic_trading.agents.analysts.risk_pod import risk_minimizing as mod

    calm = _vol_regime_market_data(spike=False)
    calm_signal = mod.RiskMinimizingAnalyst().generate_signal("AAPL", calm, calm.index[-1])
    assert calm_signal.direction in {"long", "flat"}
    assert calm_signal.confidence <= mod.MAX_CONFIDENCE

    spiked = _vol_regime_market_data(spike=True)
    spike_signal = mod.RiskMinimizingAnalyst().generate_signal("AAPL", spiked, spiked.index[-1])
    assert spike_signal.direction == "short"
    assert 0.0 < spike_signal.confidence <= mod.MAX_CONFIDENCE


def test_risk_maximizing_buys_a_capitulation_only() -> None:
    from agentic_trading.agents.analysts.risk_pod import risk_maximizing as mod

    calm = _vol_regime_market_data(spike=False)
    calm_signal = mod.RiskMaximizingAnalyst().generate_signal("AAPL", calm, calm.index[-1])
    assert calm_signal.direction == "flat"
    assert calm_signal.confidence == 0.0

    spiked = _vol_regime_market_data(spike=True)
    spike_signal = mod.RiskMaximizingAnalyst().generate_signal("AAPL", spiked, spiked.index[-1])
    assert spike_signal.direction == "long"
    assert 0.0 < spike_signal.confidence <= mod.MAX_CONFIDENCE


def test_risk_analysts_are_flat_without_enough_history() -> None:
    from agentic_trading.agents.analysts.risk_pod import risk_minimizing as mod

    dates = pd.bdate_range("2024-01-01", periods=30)
    market_data = pd.DataFrame(
        {"close": np.full(30, 100.0), "log_return": np.full(30, 0.001)}, index=dates
    )
    signal = mod.RiskMinimizingAnalyst().generate_signal("AAPL", market_data, dates[-1])
    assert signal.direction == "flat"
    assert signal.confidence == 0.0


def _simulated_gjr_returns(seed: int, n: int = 800) -> np.ndarray:
    """Daily returns from a GJR-GARCH(1,1) process with textbook equity
    parameters, so an MLE fit on them passes the analysts' sanity checks."""
    omega, alpha, gamma, beta = 2e-6, 0.05, 0.08, 0.88
    rng = np.random.default_rng(seed)
    returns = np.zeros(n)
    variance = omega / (1.0 - alpha - beta - gamma / 2.0)
    for t in range(n):
        returns[t] = np.sqrt(variance) * rng.normal()
        variance = (
            omega
            + alpha * returns[t] ** 2
            + gamma * returns[t] ** 2 * (returns[t] < 0)
            + beta * variance
        )
    return returns


def _garch_market_data(returns: np.ndarray) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    dates = pd.date_range("2020-01-01", periods=returns.size, freq="D")
    close = 100.0 * np.exp(np.cumsum(returns))
    return pd.DataFrame({"close": close, "log_return": returns}, index=dates), dates


def test_egarch_analyst_uses_real_mle_fitted_parameters(monkeypatch) -> None:
    """The analyst must size its variance recursion off a real
    `egarch_fit` MLE estimate for this asset's own trailing returns, not
    a fixed/shared parameter set — different return series should fit to
    different (omega, alpha, gamma, beta)."""
    from agentic_trading.agents.analysts.macro_rates_pod import egarch_asymmetric_vol_analyst as mod
    from agentic_trading.config import settings

    captured: dict[str, float] = {}
    real_fn = mod.egarch_11_variance

    def _spy(returns, omega, alpha, gamma, beta):
        captured.update(omega=omega, alpha=alpha, gamma=gamma, beta=beta)
        return real_fn(returns, omega, alpha, gamma, beta)

    monkeypatch.setattr(mod, "egarch_11_variance", _spy)

    returns = _simulated_gjr_returns(0)
    market_data, dates = _garch_market_data(returns)

    mod.EGARCHAsymmetricVolAnalyst().generate_signal("AAPL", market_data, dates[-1])

    fitted = mod.egarch_fit(returns[-settings.analysts.risk_window_garch :])
    assert captured["omega"] == pytest.approx(fitted.omega)
    assert captured["alpha"] == pytest.approx(fitted.alpha)
    assert captured["gamma"] == pytest.approx(fitted.gamma)
    assert captured["beta"] == pytest.approx(fitted.beta)


def test_gjr_garch_analyst_uses_real_mle_fitted_parameters(monkeypatch) -> None:
    """Same real-fit requirement as EGARCH, for the GJR-GARCH crash-risk
    analyst."""
    from agentic_trading.agents.analysts.macro_rates_pod import gjr_garch_crash_risk_analyst as mod
    from agentic_trading.config import settings

    captured: dict[str, float] = {}
    real_fn = mod.gjr_garch_11_variance

    def _spy(returns, omega, alpha, gamma, beta):
        captured.update(omega=omega, alpha=alpha, gamma=gamma, beta=beta)
        return real_fn(returns, omega, alpha, gamma, beta)

    monkeypatch.setattr(mod, "gjr_garch_11_variance", _spy)

    returns = _simulated_gjr_returns(0)
    market_data, dates = _garch_market_data(returns)

    mod.GJRGARCHCrashRiskAnalyst().generate_signal("AAPL", market_data, dates[-1])

    fitted = mod.gjr_garch_fit(returns[-settings.analysts.risk_window_garch :])
    assert captured["omega"] == pytest.approx(fitted.omega)
    assert captured["alpha"] == pytest.approx(fitted.alpha)
    assert captured["gamma"] == pytest.approx(fitted.gamma)
    assert captured["beta"] == pytest.approx(fitted.beta)


@pytest.mark.parametrize(
    ("module_name", "class_name", "fit_name"),
    [
        ("egarch_asymmetric_vol_analyst", "EGARCHAsymmetricVolAnalyst", "egarch_fit"),
        ("gjr_garch_crash_risk_analyst", "GJRGARCHCrashRiskAnalyst", "gjr_garch_fit"),
    ],
)
def test_garch_analysts_return_flat_below_minimum_observations(
    monkeypatch, module_name: str, class_name: str, fit_name: str
) -> None:
    import importlib

    from agentic_trading.config import settings

    mod = importlib.import_module(f"agentic_trading.agents.analysts.macro_rates_pod.{module_name}")

    def _fail(*_args, **_kwargs):
        raise AssertionError("no fit should be attempted below the observation floor")

    monkeypatch.setattr(mod, fit_name, _fail)
    returns = _simulated_gjr_returns(1, n=settings.analysts.min_garch_observations - 1)
    market_data, dates = _garch_market_data(returns)

    signal = getattr(mod, class_name)().generate_signal("AAPL", market_data, dates[-1])

    assert signal.direction == "flat"
    assert signal.confidence == 0.0


def test_egarch_analyst_returns_flat_on_non_convergent_fit(monkeypatch) -> None:
    from quantcore.risk import ConvergenceError

    from agentic_trading.agents.analysts.macro_rates_pod import egarch_asymmetric_vol_analyst as mod

    def _non_convergent(returns):
        raise ConvergenceError(
            "no start converged",
            params=(-0.05, 0.1, -0.05, 0.9),
            log_likelihood=0.0,
            optimizer_message="Iteration limit reached",
        )

    monkeypatch.setattr(mod, "egarch_fit", _non_convergent)
    market_data, dates = _garch_market_data(_simulated_gjr_returns(1))

    signal = mod.EGARCHAsymmetricVolAnalyst().generate_signal("AAPL", market_data, dates[-1])

    assert signal.direction == "flat"
    assert signal.confidence == 0.0
    assert "did not converge" in signal.rationale


def test_gjr_garch_analyst_returns_flat_on_non_convergent_fit(monkeypatch) -> None:
    from quantcore.risk import ConvergenceError

    from agentic_trading.agents.analysts.macro_rates_pod import gjr_garch_crash_risk_analyst as mod

    def _non_convergent(returns):
        raise ConvergenceError(
            "no start converged",
            params=(1e-6, 0.05, 0.1, 0.8),
            log_likelihood=0.0,
            optimizer_message="Iteration limit reached",
        )

    monkeypatch.setattr(mod, "gjr_garch_fit", _non_convergent)
    market_data, dates = _garch_market_data(_simulated_gjr_returns(1))

    signal = mod.GJRGARCHCrashRiskAnalyst().generate_signal("AAPL", market_data, dates[-1])

    assert signal.direction == "flat"
    assert signal.confidence == 0.0
    assert "did not converge" in signal.rationale


def test_egarch_analyst_returns_flat_on_over_persistent_fit(monkeypatch) -> None:
    """A converged fit with beta above `max_garch_persistence` is an
    effectively integrated process, not a usable vol forecast."""
    from quantcore.risk.egarch import EGARCHParams

    from agentic_trading.agents.analysts.macro_rates_pod import egarch_asymmetric_vol_analyst as mod

    over_persistent = EGARCHParams(
        omega=-0.05, alpha=0.1, gamma=-0.05, beta=0.999, log_likelihood=0.0
    )
    monkeypatch.setattr(mod, "egarch_fit", lambda returns: over_persistent)
    monkeypatch.setattr(
        mod, "egarch_11_variance", lambda *_a: (_ for _ in ()).throw(AssertionError("no path"))
    )
    market_data, dates = _garch_market_data(_simulated_gjr_returns(2))

    signal = mod.EGARCHAsymmetricVolAnalyst().generate_signal("AAPL", market_data, dates[-1])

    assert signal.direction == "flat"
    assert signal.confidence == 0.0
    assert "persistence" in signal.rationale


def test_gjr_garch_analyst_returns_flat_on_over_persistent_fit(monkeypatch) -> None:
    """A converged fit with alpha + beta + gamma/2 above
    `max_garch_persistence` is an effectively integrated process, not a
    usable vol forecast."""
    from quantcore.risk.egarch import GJRGARCHParams

    from agentic_trading.agents.analysts.macro_rates_pod import gjr_garch_crash_risk_analyst as mod

    over_persistent = GJRGARCHParams(
        omega=1e-6, alpha=0.05, gamma=0.1, beta=0.9, log_likelihood=0.0
    )
    monkeypatch.setattr(mod, "gjr_garch_fit", lambda returns: over_persistent)
    monkeypatch.setattr(
        mod,
        "gjr_garch_11_variance",
        lambda *_a: (_ for _ in ()).throw(AssertionError("no path")),
    )
    market_data, dates = _garch_market_data(_simulated_gjr_returns(2))

    signal = mod.GJRGARCHCrashRiskAnalyst().generate_signal("AAPL", market_data, dates[-1])

    assert signal.direction == "flat"
    assert signal.confidence == 0.0
    assert "persistence" in signal.rationale


def test_garch_analysts_reuse_cached_fit_within_refit_interval(monkeypatch) -> None:
    """Within `garch_refit_interval_days` of the last fit for an asset the
    cached parameters are reused; the variance path is still recomputed
    from the current window each day."""
    from agentic_trading.agents.analysts.macro_rates_pod import gjr_garch_crash_risk_analyst as mod

    real_fit = mod.gjr_garch_fit
    fit_calls: list[int] = []

    def _counting_fit(returns):
        fit_calls.append(returns.size)
        return real_fit(returns)

    monkeypatch.setattr(mod, "gjr_garch_fit", _counting_fit)
    market_data, dates = _garch_market_data(_simulated_gjr_returns(3))

    analyst = mod.GJRGARCHCrashRiskAnalyst()
    analyst.generate_signal("AAPL", market_data, dates[-3])
    analyst.generate_signal("AAPL", market_data, dates[-2])
    analyst.generate_signal("AAPL", market_data, dates[-1])

    assert len(fit_calls) == 1


def test_monte_carlo_valuation_annualizes_drift_and_vol_before_simulation(monkeypatch) -> None:
    """mean_return/volatility are daily-scale statistics; simulate_gbm_paths
    takes annualized rate/volatility (per quantcore's own docstring) — they
    must be scaled by TRADING_DAYS_PER_YEAR / sqrt(TRADING_DAYS_PER_YEAR)
    before crossing that boundary, or the simulated fair value sits within
    pure path noise of spot regardless of the asset's real drift/vol."""
    from agentic_trading.agents.analysts.cvar_pod import monte_carlo_valuation_analyst as mod

    captured: dict[str, float] = {}

    def _fake_simulate(close, rate, volatility, *_args, **_kwargs):
        captured["rate"] = rate
        captured["volatility"] = volatility
        return np.full((2, 1), close)

    monkeypatch.setattr(mod, "simulate_gbm_paths", _fake_simulate)
    monkeypatch.setattr(
        mod,
        "conditional_daily_volatility",
        lambda returns, *_a, **_k: float(np.std(returns, ddof=1)),
    )

    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    returns = np.linspace(-0.01, 0.01, 60)
    market_data = pd.DataFrame(
        {"close": np.arange(100.0, 160.0), "log_return": returns}, index=dates
    )

    mod.MonteCarloValuationAnalyst().generate_signal("AAPL", market_data, dates[-1])

    expected_rate = float(returns.mean()) * mod.TRADING_DAYS_PER_YEAR
    expected_vol = float(returns.std(ddof=1)) * mod.TRADING_DAYS_PER_YEAR**0.5
    assert captured["rate"] == pytest.approx(expected_rate)
    assert captured["volatility"] == pytest.approx(expected_vol)


def _heston_market_data(periods: int = 300) -> pd.DataFrame:
    """Flat spot with alternating +-1.26% log returns: ~20% annualized
    realized vol."""
    dates = pd.date_range("2024-01-01", periods=periods, freq="D")
    returns = np.where(np.arange(periods) % 2 == 0, 0.0126, -0.0126)
    return pd.DataFrame({"close": np.full(periods, 100.0), "log_return": returns}, index=dates)


def test_vrp_analyst_returns_flat_with_fewer_than_two_usable_expiries(monkeypatch) -> None:
    from agentic_trading.agents.analysts.cvar_pod import heston_vol_surface_analyst as mod

    market_data = _heston_market_data()
    monkeypatch.setattr(
        mod, "real_atm_options_by_expiration", lambda *_a, **_k: _real_options_frame([0.3], [0.01])
    )
    signal = mod.HestonVolSurfaceAnalyst().generate_signal(
        "AAPL", market_data, market_data.index[-1]
    )
    assert signal.direction == "flat"
    assert signal.confidence == 0.0


def test_vrp_analyst_is_long_a_rich_premium_in_contango(monkeypatch) -> None:
    """Implied vol well above the model read on an upward-sloping curve
    is long with confidence in (0, cap]; a backwardated curve is short."""
    from agentic_trading.agents.analysts.cvar_pod import heston_vol_surface_analyst as mod

    market_data = _heston_market_data()
    monkeypatch.setattr(mod, "annualized_garch_volatility", lambda *_a, **_k: 0.20)
    monkeypatch.setattr(
        mod,
        "real_atm_options_by_expiration",
        lambda *_a, **_k: _real_options_frame([0.28, 0.30, 0.32], [0.01, 0.01, 0.01]),
    )
    analyst = mod.HestonVolSurfaceAnalyst()
    signal = analyst.generate_signal("AAPL", market_data, market_data.index[-1])
    assert signal.direction == "long"
    assert 0.0 < signal.confidence <= mod.MAX_CONFIDENCE

    monkeypatch.setattr(
        mod,
        "real_atm_options_by_expiration",
        lambda *_a, **_k: _real_options_frame([0.32, 0.30, 0.28], [0.01, 0.01, 0.01]),
    )
    stressed = analyst.generate_signal("AAPL", market_data, market_data.index[-1])
    assert stressed.direction == "short"


def test_vrp_analyst_is_flat_inside_the_premium_threshold(monkeypatch) -> None:
    from agentic_trading.agents.analysts.cvar_pod import heston_vol_surface_analyst as mod

    market_data = _heston_market_data()
    monkeypatch.setattr(mod, "annualized_garch_volatility", lambda *_a, **_k: 0.29)
    monkeypatch.setattr(
        mod,
        "real_atm_options_by_expiration",
        lambda *_a, **_k: _real_options_frame([0.29, 0.30, 0.30], [0.01, 0.01, 0.01]),
    )
    signal = mod.HestonVolSurfaceAnalyst().generate_signal(
        "AAPL", market_data, market_data.index[-1]
    )
    assert signal.direction == "flat"
    assert signal.confidence == 0.0


def _real_options_frame(ivs: list[float], gammas: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "expiration": pd.to_datetime(["2024-03-15", "2024-04-19", "2024-06-21"][: len(ivs)]),
            "days_to_expiration": [30, 65, 128][: len(ivs)],
            "strike": [100.0] * len(ivs),
            "implied_volatility": ivs,
            "delta": [0.5] * len(ivs),
            "gamma": gammas,
        }
    )


def test_greeks_delta_hedge_reaches_both_pinning_and_momentum_branches(monkeypatch) -> None:
    """Below the minimum gamma history the analyst is flat; once the
    history exists, a gamma-heavy snapshot (high z) fades the 5-day move
    and a gamma-light snapshot (low z) follows it. A missing real snapshot
    reports flat."""
    from agentic_trading.agents.analysts.cvar_pod import greeks_delta_hedge_analyst as mod
    from agentic_trading.config import settings

    periods = 120
    dates = pd.date_range("2024-01-01", periods=periods, freq="D")
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0, 0.01, periods)
    returns[-5:] = 0.02  # a clear positive 5-day move against ~1% daily noise
    market_data = pd.DataFrame(
        {"close": np.full(periods, 100.0), "log_return": returns}, index=dates
    )
    analyst = mod.GreeksDeltaHedgeAnalyst()

    def _snapshot(gamma: float):
        return lambda *_a, **_k: _real_options_frame([0.2, 0.2, 0.2], [gamma, 0.001, 0.001])

    min_history = settings.analysts.gamma_min_history
    seed_gammas = np.linspace(0.010, 0.012, min_history - 1)
    for day, gamma in zip(dates[20 : 20 + min_history - 1], seed_gammas, strict=True):
        monkeypatch.setattr(mod, "real_atm_options_by_expiration", _snapshot(float(gamma)))
        thin_signal = analyst.generate_signal("AAPL", market_data, day)
        assert thin_signal.direction == "flat"
        assert thin_signal.confidence == 0.0
        assert "history" in thin_signal.rationale

    monkeypatch.setattr(mod, "real_atm_options_by_expiration", _snapshot(0.03))
    pinned_signal = analyst.generate_signal("AAPL", market_data, dates[-2])
    assert pinned_signal.direction == "short"
    assert 0.0 < pinned_signal.confidence <= 0.75
    assert "z=" in pinned_signal.rationale

    monkeypatch.setattr(mod, "real_atm_options_by_expiration", _snapshot(0.003))
    light_signal = analyst.generate_signal("AAPL", market_data, dates[-1])
    assert light_signal.direction == "long"
    assert 0.0 < light_signal.confidence <= 0.75

    monkeypatch.setattr(mod, "real_atm_options_by_expiration", lambda *_a, **_k: None)
    missing_signal = analyst.generate_signal("AAPL", market_data, dates[-1])
    assert missing_signal.direction == "flat"
    assert missing_signal.confidence == 0.0


def test_black_scholes_valuation_edge_is_not_identically_zero(monkeypatch) -> None:
    """An ATM call-minus-put spread cancels to zero by put-call parity
    regardless of volatility — the edge must come from the call price
    alone, which genuinely diverges between the fitted GARCH volatility
    and the real market ATM implied volatility. With no real snapshot the
    analyst reports flat with zero confidence, never a substitute quote."""
    from agentic_trading.agents.analysts.cvar_pod import black_scholes_valuation_analyst as mod

    dates = pd.date_range("2024-01-01", periods=300, freq="D")
    market_data = pd.DataFrame(
        {"close": np.full(300, 100.0), "log_return": np.linspace(-0.01, 0.01, 300)},
        index=dates,
    )
    monkeypatch.setattr(
        mod,
        "conditional_daily_volatility",
        lambda returns, *_a, **_k: float(np.std(returns, ddof=1)),
    )
    monkeypatch.setattr(
        mod,
        "real_atm_options_by_expiration",
        lambda *_a, **_k: _real_options_frame([0.6, 0.55], [0.01, 0.01]),
    )
    signal = mod.BlackScholesValuationAnalyst().generate_signal("AAPL", market_data, dates[-1])
    assert signal.confidence > 1e-6

    monkeypatch.setattr(mod, "real_atm_options_by_expiration", lambda *_a, **_k: None)
    missing = mod.BlackScholesValuationAnalyst().generate_signal("AAPL", market_data, dates[-1])
    assert missing.direction == "flat"
    assert missing.confidence == 0.0


def test_gbm_momentum_annualizes_drift_and_vol_before_simulation(monkeypatch) -> None:
    """mu/sigma are daily-scale statistics; simulate_gbm_paths takes
    annualized drift/volatility over a time_to_maturity in years."""
    from agentic_trading.agents.analysts.risk_pod import gbm_momentum_analyst as mod

    captured: dict[str, float] = {}

    def _fake_simulate(close, mu, sigma, *_args, **_kwargs):
        captured["mu"] = mu
        captured["sigma"] = sigma
        return np.full((2, 1), close)

    monkeypatch.setattr(mod, "simulate_gbm_paths", _fake_simulate)

    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    returns = np.linspace(-0.01, 0.01, 60)
    market_data = pd.DataFrame(
        {"close": np.arange(100.0, 160.0), "log_return": returns}, index=dates
    )

    mod.GBMMomentumAnalyst().generate_signal("AAPL", market_data, dates[-1])

    expected_mu = float(returns.mean()) * mod.TRADING_DAYS_PER_YEAR
    expected_sigma = float(returns.std(ddof=1)) * mod.TRADING_DAYS_PER_YEAR**0.5
    assert captured["mu"] == pytest.approx(expected_mu)
    assert captured["sigma"] == pytest.approx(expected_sigma)


def test_cir_rate_regime_falls_back_to_flat_on_trending_rates(monkeypatch) -> None:
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    market_data = pd.DataFrame(
        {"close": np.arange(100.0, 130.0), "log_return": np.linspace(-0.01, 0.01, 30)},
        index=dates,
    )
    # A monotonically trending rate series has no discernible mean
    # reversion — cir_fit raises RuntimeError for it by design (see
    # quantcore's core/stochastic.py), which must fall through to a
    # graceful flat signal rather than an uncaught exception.
    trending_rates = pd.Series(np.linspace(1.0, 5.0, 30), index=dates)

    monkeypatch.setattr(
        "agentic_trading.agents.analysts.macro_rates_pod.cir_rate_regime_analyst."
        "load_fred_series_from_lakehouse",
        lambda _series_id, _as_of_date: trending_rates,
    )
    signal = CIRRateRegimeAnalyst().generate_signal("AAPL", market_data, dates[-1])

    assert signal.status == "active"
    assert signal.direction == "flat"
    assert signal.confidence == 0.0


def test_cir_rate_regime_uses_real_mle_fitted_parameters(monkeypatch) -> None:
    """The analyst must simulate off a real `cir_fit` estimate for the
    actual observed rate series, not a fixed/derived shortcut."""
    from agentic_trading.agents.analysts.macro_rates_pod import cir_rate_regime_analyst as mod

    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    market_data = pd.DataFrame(
        {"close": np.arange(100.0, 160.0), "log_return": np.zeros(60)}, index=dates
    )
    # A genuinely mean-reverting rate series (oscillating around 3.0%),
    # since a trending or flat one won't produce a valid CIR fit.
    rng = np.random.default_rng(2)
    mean_reverting_rates = pd.Series(
        3.0 + 0.3 * np.sin(np.linspace(0, 6 * np.pi, 60)) + rng.normal(0.0, 0.02, 60),
        index=dates,
    )
    monkeypatch.setattr(mod, "load_fred_series_from_lakehouse", lambda *_: mean_reverting_rates)

    captured: dict[str, float] = {}
    real_fn = mod.simulate_cir_paths

    def _spy(r0, kappa, theta, sigma, **kwargs):
        captured.update(kappa=kappa, theta=theta, sigma=sigma)
        return real_fn(r0, kappa, theta, sigma, **kwargs)

    monkeypatch.setattr(mod, "simulate_cir_paths", _spy)

    mod.CIRRateRegimeAnalyst().generate_signal("AAPL", market_data, dates[-1])

    rates = mean_reverting_rates.to_numpy(dtype=np.float64)[-250:] / 100.0
    fitted = mod.cir_fit(rates, mod._DAILY_DT_YEARS)
    assert captured["kappa"] == pytest.approx(fitted.kappa)
    assert captured["theta"] == pytest.approx(fitted.theta)
    assert captured["sigma"] == pytest.approx(fitted.sigma)


def _autocorrelated_returns(seed: int, n: int = 300, phi: float = 0.4) -> np.ndarray:
    """AR(1) daily returns — enough serial dependence that a Ljung-Box
    pre-test rejects white noise."""
    rng = np.random.default_rng(seed)
    returns = np.zeros(n)
    for t in range(1, n):
        returns[t] = phi * returns[t - 1] + rng.normal(0.0, 0.01)
    return returns


def _arima_market_data(returns: np.ndarray) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    dates = pd.date_range("2023-01-01", periods=returns.size, freq="D")
    close = 100.0 * np.exp(np.cumsum(returns))
    return pd.DataFrame({"close": close, "log_return": returns}, index=dates), dates


def test_arima_forecast_stays_flat_on_white_noise_returns(monkeypatch) -> None:
    """With no serial dependence in the trailing returns (Ljung-Box
    p >= 0.05) there is nothing for an ARIMA model to forecast, and no
    order selection or fit is attempted."""
    from agentic_trading.agents.analysts.macro_rates_pod import arima_forecast_analyst as mod

    def _fail(*_args, **_kwargs):
        raise AssertionError("no fit should be attempted on white-noise returns")

    monkeypatch.setattr(mod, "select_arima_order", _fail)
    monkeypatch.setattr(mod, "arima_fit", _fail)
    market_data, dates = _arima_market_data(np.random.default_rng(3).normal(0.0, 0.01, 300))

    signal = mod.ARIMAForecastAnalyst().generate_signal("AAPL", market_data, dates[-1])

    assert signal.direction == "flat"
    assert signal.confidence == 0.0
    assert "no serial dependence to forecast (Ljung-Box p=" in signal.rationale


def test_arima_forecast_selects_order_via_aic_by_default(monkeypatch) -> None:
    """With p/d/q left unset, the analyst must select its ARIMA order via
    `select_arima_order` (AIC grid search) with the integration order
    pinned at 1 (the grid runs on returns, so quantcore searches d=0
    only), rather than a fixed (1,1,1)."""
    from agentic_trading.agents.analysts.macro_rates_pod import arima_forecast_analyst as mod

    market_data, dates = _arima_market_data(_autocorrelated_returns(3))

    captured: dict[str, object] = {}
    real_select = mod.select_arima_order

    def _spy(series, max_p, max_d, max_q, criterion):
        order = real_select(series, max_p, max_d, max_q, criterion)
        captured.update(order=order, max_p=max_p, max_d=max_d, max_q=max_q)
        return order

    monkeypatch.setattr(mod, "select_arima_order", _spy)

    analyst = mod.ARIMAForecastAnalyst()
    assert analyst.p is None and analyst.d is None and analyst.q is None
    signal = analyst.generate_signal("AAPL", market_data, dates[-1])

    assert captured == {
        "order": captured["order"],
        "max_p": mod.MAX_P,
        "max_d": 0,
        "max_q": mod.MAX_Q,
    }
    assert signal.status == "active"
    assert ",1," in signal.rationale


def test_arima_forecast_reselects_order_less_often_than_it_refits(monkeypatch) -> None:
    """Order selection runs once per `arima_order_reselect_days`; the fit
    itself is refreshed on the shorter `_refit_cache` cadence in between."""
    from agentic_trading.agents.analysts.macro_rates_pod import arima_forecast_analyst as mod

    select_calls: list[int] = []
    fit_calls: list[int] = []
    real_select, real_fit = mod.select_arima_order, mod.arima_fit

    def _select(*args):
        select_calls.append(1)
        return real_select(*args)

    def _fit(*args):
        fit_calls.append(1)
        return real_fit(*args)

    monkeypatch.setattr(mod, "select_arima_order", _select)
    monkeypatch.setattr(mod, "arima_fit", _fit)
    market_data, dates = _arima_market_data(_autocorrelated_returns(5, n=320))

    analyst = mod.ARIMAForecastAnalyst()
    for offset in (20, 10, 0):
        analyst.generate_signal("AAPL", market_data, dates[-1 - offset])

    assert len(select_calls) == 1
    assert len(fit_calls) == 3


def test_arima_forecast_honors_pinned_order_without_selecting(monkeypatch) -> None:
    """An explicitly-pinned p/d/q must skip order selection entirely."""
    from agentic_trading.agents.analysts.macro_rates_pod import arima_forecast_analyst as mod

    market_data, dates = _arima_market_data(_autocorrelated_returns(4))

    def _fail(*args, **kwargs):
        raise AssertionError("select_arima_order must not be called when p/d/q are pinned")

    monkeypatch.setattr(mod, "select_arima_order", _fail)

    analyst = mod.ARIMAForecastAnalyst(p=2, d=1, q=1)
    signal = analyst.generate_signal("AAPL", market_data, dates[-1])

    assert signal.status == "active"
    assert "ARIMA(2,1,1)" in signal.rationale


def test_macro_var_spillover_handles_solver_failure(monkeypatch) -> None:
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    market_data = pd.DataFrame(
        {"close": np.arange(100.0, 130.0), "log_return": np.linspace(-0.01, 0.01, 30)},
        index=dates,
    )
    macro = pd.Series(np.linspace(10.0, 20.0, 30), index=dates)

    monkeypatch.setattr(
        "agentic_trading.agents.analysts.macro_rates_pod.macro_var_spillover_analyst."
        "load_fred_series_from_lakehouse",
        lambda _series_id, _as_of_date: macro,
    )
    monkeypatch.setattr(
        "agentic_trading.agents.analysts.macro_rates_pod.macro_var_spillover_analyst.granger_causality_test",
        lambda *_args: (0.0, 0.01),
    )
    monkeypatch.setattr(
        "agentic_trading.agents.analysts.macro_rates_pod.macro_var_spillover_analyst.var_fit",
        lambda *_args: (_ for _ in ()).throw(np.linalg.LinAlgError("SVD did not converge")),
    )
    signal = MacroVarSpilloverAnalyst(use_yield_curve_slope=False).generate_signal(
        "AAPL", market_data, dates[-1]
    )

    assert signal.direction == "flat"
    assert signal.confidence == 0.0


def test_macro_var_spillover_selects_lag_order_via_aic_by_default(monkeypatch) -> None:
    """With n_lags left unset, the analyst must select its VAR lag order
    via `select_var_lag_order` rather than a fixed default."""
    from agentic_trading.agents.analysts.macro_rates_pod import macro_var_spillover_analyst as mod

    dates = pd.date_range("2024-01-01", periods=90, freq="D")
    rng = np.random.default_rng(5)
    market_data = pd.DataFrame(
        {"close": np.arange(100.0, 190.0), "log_return": rng.normal(0.0, 0.01, size=90)},
        index=dates,
    )
    macro = pd.Series(rng.normal(20.0, 2.0, size=90), index=dates)
    monkeypatch.setattr(mod, "load_fred_series_from_lakehouse", lambda *_: macro)

    captured: dict[str, int] = {}
    real_select = mod.select_var_lag_order

    def _spy(system, max_lags, criterion):
        n_lags = real_select(system, max_lags, criterion)
        captured["n_lags"] = n_lags
        return n_lags

    monkeypatch.setattr(mod, "select_var_lag_order", _spy)

    analyst = mod.MacroVarSpilloverAnalyst(use_yield_curve_slope=False)
    assert analyst.n_lags is None
    analyst.generate_signal("AAPL", market_data, dates[-1])

    assert "n_lags" in captured


def _hmm_market_data(n: int = 600, seed: int = 6) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, 0.01, size=n)
    return (
        pd.DataFrame({"close": np.arange(100.0, 100.0 + n), "log_return": returns}, index=dates),
        dates,
    )


def _patch_hmm(
    monkeypatch,
    transition_matrix: list[list[float]],
    variances: list[float],
    last_posterior: list[float],
    means: list[float] | None = None,
    occupancy: list[float] | None = None,
) -> None:
    """Hand-built two-state HMM parameters: `hmm_fit` returns them
    verbatim, `hmm_predict_proba` returns a posterior path whose rows
    average to `occupancy` (default balanced) and end at `last_posterior`."""
    from agentic_trading.agents.analysts.stat_arb_pod import regime_hmm_analyst as mod

    matrix = np.asarray(transition_matrix, dtype=float)
    state_means = np.asarray(means if means is not None else [0.001, -0.002], dtype=float)
    state_variances = np.asarray(variances, dtype=float)
    initial = np.array([0.5, 0.5])

    def _fit(observations, n_states):
        assert n_states == 2
        return matrix, state_means, state_variances, initial

    def _predict(observations, *_params):
        rows = observations.size
        share = np.asarray(occupancy if occupancy is not None else [0.5, 0.5], dtype=float)
        posteriors = np.tile(share, (rows, 1))
        posteriors[-1] = np.asarray(last_posterior, dtype=float)
        return posteriors

    monkeypatch.setattr(mod, "hmm_fit", _fit)
    monkeypatch.setattr(mod, "hmm_predict_proba", _predict)


def test_regime_hmm_goes_long_when_next_step_bull_probability_is_high(monkeypatch) -> None:
    """Bull = the lower-variance state (index 0 here); p_next = p_T @ A puts
    ~0.95 on it, above the action threshold, with confidence 2*|p-0.5|."""
    from agentic_trading.agents.analysts.stat_arb_pod import regime_hmm_analyst as mod

    _patch_hmm(
        monkeypatch,
        transition_matrix=[[0.95, 0.05], [0.10, 0.90]],
        variances=[1e-4, 4e-4],
        last_posterior=[1.0, 0.0],
    )
    market_data, dates = _hmm_market_data()

    signal = mod.RegimeHMMAnalyst().generate_signal("AAPL", market_data, dates[-1])

    assert signal.direction == "long"
    assert signal.confidence == pytest.approx(2 * abs(0.95 - 0.5))


def test_regime_hmm_labels_bear_as_higher_variance_state_regardless_of_index(
    monkeypatch,
) -> None:
    """The higher-variance state is bear even when it has index 0 and the
    higher mean — variance, not mean or EM ordering, defines the label."""
    from agentic_trading.agents.analysts.stat_arb_pod import regime_hmm_analyst as mod

    _patch_hmm(
        monkeypatch,
        transition_matrix=[[0.95, 0.05], [0.10, 0.90]],
        variances=[4e-4, 1e-4],
        means=[0.002, 0.001],
        last_posterior=[1.0, 0.0],
    )
    market_data, dates = _hmm_market_data()

    signal = mod.RegimeHMMAnalyst().generate_signal("AAPL", market_data, dates[-1])

    assert signal.direction == "short"


def test_regime_hmm_is_flat_between_action_thresholds(monkeypatch) -> None:
    from agentic_trading.agents.analysts.stat_arb_pod import regime_hmm_analyst as mod

    _patch_hmm(
        monkeypatch,
        transition_matrix=[[0.9, 0.1], [0.1, 0.9]],
        variances=[1e-4, 4e-4],
        last_posterior=[0.6, 0.4],
    )
    market_data, dates = _hmm_market_data()

    signal = mod.RegimeHMMAnalyst().generate_signal("AAPL", market_data, dates[-1])

    assert signal.direction == "flat"
    assert signal.confidence == pytest.approx(2 * abs(0.58 - 0.5))


@pytest.mark.parametrize(
    ("transition_matrix", "variances", "occupancy", "expected_reason"),
    [
        ([[0.95, 0.05], [0.10, 0.90]], [1e-4, 1.2e-4], None, "variance ratio"),
        ([[0.95, 0.05], [0.10, 0.90]], [1e-4, 4e-4], [0.98, 0.02], "occupancy"),
        ([[0.95, 0.05], [0.50, 0.50]], [1e-4, 4e-4], None, "persistence"),
    ],
)
def test_regime_hmm_returns_flat_for_degenerate_fits(
    monkeypatch, transition_matrix, variances, occupancy, expected_reason
) -> None:
    from agentic_trading.agents.analysts.stat_arb_pod import regime_hmm_analyst as mod

    _patch_hmm(
        monkeypatch,
        transition_matrix=transition_matrix,
        variances=variances,
        last_posterior=[1.0, 0.0],
        occupancy=occupancy,
    )
    market_data, dates = _hmm_market_data()

    signal = mod.RegimeHMMAnalyst().generate_signal("AAPL", market_data, dates[-1])

    assert signal.direction == "flat"
    assert signal.confidence == 0.0
    assert expected_reason in signal.rationale


def test_regime_hmm_needs_minimum_observations(monkeypatch) -> None:
    from agentic_trading.agents.analysts.stat_arb_pod import regime_hmm_analyst as mod
    from agentic_trading.config import settings

    market_data, dates = _hmm_market_data(n=settings.analysts.hmm_min_observations - 1)

    signal = mod.RegimeHMMAnalyst().generate_signal("AAPL", market_data, dates[-1])

    assert signal.direction == "flat"
    assert "observations" in signal.rationale


def test_regime_hmm_refits_only_after_refit_interval(monkeypatch) -> None:
    from agentic_trading.agents.analysts.stat_arb_pod import regime_hmm_analyst as mod
    from agentic_trading.config import settings

    _patch_hmm(
        monkeypatch,
        transition_matrix=[[0.95, 0.05], [0.10, 0.90]],
        variances=[1e-4, 4e-4],
        last_posterior=[1.0, 0.0],
    )
    fits: list[int] = []
    real_fit = mod.hmm_fit

    def _counting_fit(observations, n_states):
        fits.append(n_states)
        return real_fit(observations, n_states)

    monkeypatch.setattr(mod, "hmm_fit", _counting_fit)
    market_data, dates = _hmm_market_data(n=600 + settings.analysts.hmm_refit_interval_days)
    analyst = mod.RegimeHMMAnalyst()
    first = dates[599]

    analyst.generate_signal("AAPL", market_data, first)
    analyst.generate_signal("AAPL", market_data, first + pd.Timedelta(days=1))
    analyst.generate_signal(
        "AAPL", market_data, first + pd.Timedelta(days=settings.analysts.hmm_refit_interval_days)
    )

    assert fits == [2, 2]


def test_newey_west_significance_selects_lags_via_optimal_rule_by_default(monkeypatch) -> None:
    """With n_lags left unset, the analyst must select its HAC bandwidth
    via `newey_west_optimal_lags` rather than a fixed default."""
    from agentic_trading.agents.analysts.factor_pod import newey_west_significance_analyst as mod

    dates = pd.date_range("2024-01-01", periods=120, freq="D")
    rng = np.random.default_rng(8)
    market_data = pd.DataFrame(
        {"close": np.arange(100.0, 220.0), "log_return": rng.normal(0.0005, 0.01, size=120)},
        index=dates,
    )

    captured: dict[str, int] = {}
    real_fn = mod.newey_west_optimal_lags

    def _spy(residuals):
        n_lags = real_fn(residuals)
        captured["n_lags"] = n_lags
        return n_lags

    monkeypatch.setattr(mod, "newey_west_optimal_lags", _spy)

    analyst = mod.NeweyWestSignificanceAnalyst()
    assert analyst.n_lags is None
    analyst.generate_signal("AAPL", market_data, dates[-1])

    assert "n_lags" in captured


def test_fama_french_signal_is_expected_return_over_vol_not_a_t_stat(monkeypatch) -> None:
    """A strongly loaded asset in a positive-premium factor environment is
    long with a confidence bounded below the cap, and an asset with no
    significant loading is flat — the view comes from beta x premium
    scaled by the asset's own vol, so it cannot saturate on a t-stat."""
    from agentic_trading.agents.analysts.factor_pod import fama_french_factor_analyst as mod

    rng = np.random.default_rng(5)
    periods = 600
    dates = pd.bdate_range("2022-01-03", periods=periods)
    market = rng.normal(0.0004, 0.01, periods)
    factors = pd.DataFrame(
        {
            "Mkt-RF": market,
            "SMB": rng.normal(0.0, 0.004, periods),
            "HML": rng.normal(0.0, 0.004, periods),
            "RF": np.full(periods, 0.0001),
        },
        index=dates,
    )
    monkeypatch.setattr(mod, "load_fama_french_factors_from_lakehouse", lambda *_a, **_k: factors)

    loaded_returns = 1.2 * market + rng.normal(0.0, 0.003, periods) + 0.0001
    loaded = pd.DataFrame(
        {"close": 100.0 * np.exp(np.cumsum(loaded_returns)), "log_return": loaded_returns},
        index=dates,
    )
    signal = mod.FamaFrenchFactorAnalyst().generate_signal("AAPL", loaded, dates[-1])
    assert signal.direction == "long"
    assert 0.0 < signal.confidence <= mod.MAX_CONFIDENCE

    noise_returns = rng.normal(0.0, 0.01, periods)
    noise = pd.DataFrame(
        {"close": 100.0 * np.exp(np.cumsum(noise_returns)), "log_return": noise_returns},
        index=dates,
    )
    flat_signal = mod.FamaFrenchFactorAnalyst().generate_signal("KO", noise, dates[-1])
    assert flat_signal.direction == "flat"
    assert flat_signal.confidence == 0.0
