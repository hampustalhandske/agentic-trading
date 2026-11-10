# test_undefined_statistics.py

"""An analyst whose statistic is undefined on the day's data (a NaN z-score, a non-finite
half-life or p-value, a zero volatility) never trades on it: it reports `unavailable` when the
data cannot define the statistic and `error` when a fitted model's diagnostic is undefined.
Every loader is stubbed on the analyst's own module, so no test reads the lake."""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from agentic_trading.agents.analysts.event_credit_pod.short_interest_squeeze_analyst import (
    ShortInterestSqueezeAnalyst,
)
from agentic_trading.agents.analysts.macro_rates_pod import (
    arima_forecast_analyst,
    macro_var_spillover_analyst,
)
from agentic_trading.agents.analysts.macro_rates_pod.arima_forecast_analyst import (
    ARIMAForecastAnalyst,
)
from agentic_trading.agents.analysts.macro_rates_pod.macro_var_spillover_analyst import (
    MacroVarSpilloverAnalyst,
)
from agentic_trading.agents.analysts.risk_pod.gbm_momentum_analyst import GBMMomentumAnalyst
from agentic_trading.agents.analysts.stat_arb_pod import ou_mean_reversion_analyst
from agentic_trading.agents.analysts.stat_arb_pod.ou_mean_reversion_analyst import (
    OUMeanReversionAnalyst,
)
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.config import temporary_overrides
from agentic_trading.data.loaders.macro_loader import DGS3MO, DGS10, VIXCLS

_SHORT_INTEREST = "agentic_trading.agents.analysts.event_credit_pod.short_interest_squeeze_analyst"
AS_OF = pd.Timestamp("2024-06-14")


def _market(returns: np.ndarray) -> pd.DataFrame:
    """A single-asset frame of business-day `log_return`s ending AS_OF, with closes."""
    index = pd.bdate_range(end=AS_OF, periods=len(returns))
    frame = pd.DataFrame({"log_return": np.asarray(returns, dtype=np.float64)}, index=index)
    frame["close"] = 100.0 * np.exp(np.cumsum(frame["log_return"].to_numpy()))
    return frame


def _noise(size: int, seed: int = 7) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, 0.01, size)


def _flat(signal: AgentSignal) -> tuple[str, str, float]:
    return signal.status, signal.direction, signal.confidence


def _constant_short_interest(settlements: int, days_to_cover: float) -> pd.DataFrame:
    """`settlements` semimonthly settlements, every one with the same days-to-cover."""
    dates = pd.date_range("2015-01-15", periods=settlements, freq="SMS-15")
    return pd.DataFrame(
        {
            "settlement_date": dates,
            "dissemination_date": dates + pd.Timedelta(days=12),
            "short_interest_shares": 1_000_000.0,
            "days_to_cover": days_to_cover,
            "avg_daily_volume": 500_000.0,
        }
    )


@pytest.mark.parametrize("days_to_cover", [0.0, 3.0])
def test_short_interest_constant_days_to_cover_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, days_to_cover: float
) -> None:
    """A days-to-cover that never moves over the z window has no z-score."""
    history = _constant_short_interest(8, days_to_cover)
    monkeypatch.setattr(f"{_SHORT_INTEREST}.load_short_interest_from_lakehouse", lambda *_: history)

    signal = ShortInterestSqueezeAnalyst().generate_signal(
        "AAPL", pd.DataFrame(), pd.Timestamp("2015-06-30")
    )

    assert signal.status == "unavailable"
    assert signal.direction == "flat"
    assert signal.confidence == 0.0
    assert signal.source_as_of == history["dissemination_date"].max().date()
    assert signal.sample_count == len(history)


class UndefinedResultError(ValueError):
    """Stands in for quantcore's undefined-result error, a `ValueError` subclass."""


def _raising(error: BaseException) -> Any:
    def _fit(*_args: Any, **_kwargs: Any) -> Any:
        raise error

    return _fit


# A daily log return of 2**-10: its running sums, their mean and the de-meaned spread are exact
# in float64, so the spread is a straight line on every platform and BLAS build.
_DYADIC_STEP = 2.0**-10


def test_ou_straight_line_spread_is_unavailable() -> None:
    """A log price rising by the same amount every day is a straight-line spread, whose
    half-life is undefined."""
    analyst = OUMeanReversionAnalyst()
    analyst._open_direction_by_asset["AAPL"] = "long"
    market = _market(np.full(120, _DYADIC_STEP))
    steps = np.diff(analyst._build_spread(market, AS_OF, "AAPL"))
    assert steps.size > 0 and np.ptp(steps) == 0.0

    signal = analyst.generate_signal("AAPL", market, AS_OF)

    assert _flat(signal) == ("unavailable", "flat", 0.0)
    assert "straight line" in signal.rationale
    assert "AAPL" not in analyst._open_direction_by_asset


@pytest.mark.parametrize(
    "fit",
    [
        lambda _spread: float("nan"),
        lambda _spread: float("inf"),
        _raising(UndefinedResultError("half-life undefined")),
    ],
    ids=["nan", "inf", "undefined_result"],
)
def test_ou_undefined_half_life_is_unavailable(monkeypatch: pytest.MonkeyPatch, fit: Any) -> None:
    monkeypatch.setattr(ou_mean_reversion_analyst, "ou_half_life", fit)
    analyst = OUMeanReversionAnalyst()
    analyst._open_direction_by_asset["AAPL"] = "long"

    signal = analyst.generate_signal("AAPL", _market(_noise(120)), AS_OF)

    assert _flat(signal) == ("unavailable", "flat", 0.0)
    assert "AAPL" not in analyst._open_direction_by_asset


def test_ou_not_mean_reverting_abstains(monkeypatch: pytest.MonkeyPatch) -> None:
    """quantcore's phi >= 0 refusal is a spread the model declines to trade."""
    error = ValueError("estimated series does not exhibit mean reversion (phi >= 0, diverging)")
    monkeypatch.setattr(ou_mean_reversion_analyst, "ou_half_life", _raising(error))

    signal = OUMeanReversionAnalyst().generate_signal("AAPL", _market(_noise(120)), AS_OF)

    assert _flat(signal) == ("active", "flat", 0.0)


def test_ou_any_other_fit_failure_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ou_mean_reversion_analyst, "ou_half_life", _raising(ValueError("singular regression"))
    )

    signal = OUMeanReversionAnalyst().generate_signal("AAPL", _market(_noise(120)), AS_OF)

    assert _flat(signal) == ("error", "flat", 0.0)


def _stub_half_life(monkeypatch: pytest.MonkeyPatch, half_life: float) -> list[int]:
    """Stub `ou_half_life` to return `half_life`; the returned list collects the size of
    every spread it is fitted on."""
    sizes: list[int] = []

    def _fit(spread: np.ndarray) -> float:
        sizes.append(int(spread.size))
        return half_life

    monkeypatch.setattr(ou_mean_reversion_analyst, "ou_half_life", _fit)
    return sizes


def _spy_zscore(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Wrap the real `spread_zscore`; the returned list collects every window it is called
    with, so a test can tell the normal z-score path ran."""
    windows: list[int] = []
    real = ou_mean_reversion_analyst.spread_zscore

    def _zscore(spread: np.ndarray, window: int) -> np.ndarray:
        windows.append(window)
        return real(spread, window)

    monkeypatch.setattr(ou_mean_reversion_analyst, "spread_zscore", _zscore)
    return windows


@pytest.mark.parametrize("half_life", [1e8, 251.0])
def test_ou_half_life_beyond_the_fitted_window_abstains(
    monkeypatch: pytest.MonkeyPatch, half_life: float
) -> None:
    """A finite half-life longer than the 250-session fitted spread has not shown one
    half-life of reversion in the data: the analyst declines to trade it."""
    sizes = _stub_half_life(monkeypatch, half_life)
    windows = _spy_zscore(monkeypatch)
    analyst = OUMeanReversionAnalyst()
    analyst._open_direction_by_asset["AAPL"] = "long"

    signal = analyst.generate_signal("AAPL", _market(_noise(400)), AS_OF)

    assert sizes == [250]
    assert _flat(signal) == ("active", "flat", 0.0)
    assert f"{half_life:.1f}d" in signal.rationale
    assert "250-session fitted window" in signal.rationale
    assert "AAPL" not in analyst._open_direction_by_asset
    assert windows == []


def test_ou_half_life_inside_the_fitted_window_runs_the_normal_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sizes = _stub_half_life(monkeypatch, 249.0)
    windows = _spy_zscore(monkeypatch)

    signal = OUMeanReversionAnalyst().generate_signal("AAPL", _market(_noise(400)), AS_OF)

    assert sizes == [250]
    assert windows == [250]
    assert "fitted window" not in signal.rationale


@pytest.mark.parametrize(
    ("windows_allowed", "half_life", "abstains"),
    [(2.0, 499.0, False), (2.0, 501.0, True), (0.5, 124.0, False), (0.5, 126.0, True)],
)
def test_ou_half_life_bound_scales_with_max_half_life_windows(
    monkeypatch: pytest.MonkeyPatch, windows_allowed: float, half_life: float, abstains: bool
) -> None:
    _stub_half_life(monkeypatch, half_life)
    windows = _spy_zscore(monkeypatch)

    with temporary_overrides({"analysts.ou_max_half_life_windows": windows_allowed}):
        signal = OUMeanReversionAnalyst().generate_signal("AAPL", _market(_noise(400)), AS_OF)

    assert ("fitted window" in signal.rationale) is abstains
    assert (windows == []) is abstains
    if abstains:
        assert _flat(signal) == ("active", "flat", 0.0)


def test_ou_half_life_bound_uses_the_spread_length_on_short_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With less history than `risk_window_long`, the fitted window is the spread itself."""
    sizes = _stub_half_life(monkeypatch, 150.0)
    windows = _spy_zscore(monkeypatch)

    signal = OUMeanReversionAnalyst().generate_signal("AAPL", _market(_noise(120)), AS_OF)

    assert len(sizes) == 1 and sizes[0] < 150
    assert _flat(signal) == ("active", "flat", 0.0)
    assert f"{sizes[0]}-session fitted window" in signal.rationale
    assert windows == []


def test_ou_half_life_inside_a_short_spread_runs_the_normal_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sizes = _stub_half_life(monkeypatch, 100.0)
    windows = _spy_zscore(monkeypatch)

    signal = OUMeanReversionAnalyst().generate_signal("AAPL", _market(_noise(120)), AS_OF)

    assert len(sizes) == 1 and 100 < sizes[0] < 150
    assert len(windows) == 1
    assert "fitted window" not in signal.rationale


def _autocorrelated(size: int, seed: int = 11, phi: float = 0.6) -> np.ndarray:
    """An AR(1) return series with coefficient `phi`, serially dependent enough to pass the
    ARIMA pretest."""
    shocks = _noise(size, seed)
    series = np.empty(size)
    series[0] = shocks[0]
    for t in range(1, size):
        series[t] = phi * series[t - 1] + shocks[t]
    return series


def _stub_arima_fit(monkeypatch: pytest.MonkeyPatch, residuals: np.ndarray | None = None) -> None:
    """A fixed ARIMA(1,0,0) order and fit, and `residuals` as the fit's residuals when given."""
    monkeypatch.setattr(arima_forecast_analyst, "select_arima_order", lambda *_a: (1, 0, 0))
    monkeypatch.setattr(
        arima_forecast_analyst, "arima_fit", lambda *_a: (np.array([0.1]), np.array([]), 1e-4)
    )
    if residuals is not None:
        monkeypatch.setattr(arima_forecast_analyst, "arima_residuals", lambda *_a: residuals)


def _ljung_box(*p_values: float) -> Any:
    """A stand-in `ljung_box_test` returning `p_values` in order, one per call."""
    remaining = iter(p_values)
    return lambda *_args, **_kwargs: (0.0, next(remaining))


@pytest.mark.parametrize("value", [0.0, 0.01])
def test_arima_constant_pretest_returns_are_unavailable(
    monkeypatch: pytest.MonkeyPatch, value: float
) -> None:
    """A return window with no range defines no autocorrelation to test; Ljung-Box gives NaN
    on zeros and a spuriously tiny p on a non-zero constant."""
    _stub_arima_fit(monkeypatch)

    signal = ARIMAForecastAnalyst().generate_signal("TLT", _market(np.full(300, value)), AS_OF)

    assert _flat(signal) == ("unavailable", "flat", 0.0)


def test_arima_non_finite_pretest_p_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_arima_fit(monkeypatch)
    monkeypatch.setattr(arima_forecast_analyst, "ljung_box_test", _ljung_box(float("nan")))

    signal = ARIMAForecastAnalyst().generate_signal("TLT", _market(_autocorrelated(300)), AS_OF)

    assert _flat(signal) == ("unavailable", "flat", 0.0)


@pytest.mark.parametrize("value", [0.0, 0.01])
def test_arima_constant_residuals_are_an_error(
    monkeypatch: pytest.MonkeyPatch, value: float
) -> None:
    """A fitted model whose residuals have no range cannot be checked for misspecification."""
    _stub_arima_fit(monkeypatch, residuals=np.full(249, value))

    signal = ARIMAForecastAnalyst().generate_signal("TLT", _market(_autocorrelated(300)), AS_OF)

    assert _flat(signal) == ("error", "flat", 0.0)


def test_arima_non_finite_residual_p_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_arima_fit(monkeypatch, residuals=_noise(249, 3))
    monkeypatch.setattr(arima_forecast_analyst, "ljung_box_test", _ljung_box(0.001, float("nan")))

    signal = ARIMAForecastAnalyst().generate_signal("TLT", _market(_autocorrelated(300)), AS_OF)

    assert _flat(signal) == ("error", "flat", 0.0)


def test_arima_real_pretest_passes_the_autocorrelated_fixture() -> None:
    """The fixture the residual tests use clears the real pretest, so they reach the fit."""
    returns = _autocorrelated(300)[-arima_forecast_analyst.settings.analysts.arima_pretest_window :]
    lags = min(arima_forecast_analyst.LJUNG_BOX_MAX_LAGS, returns.size // 5)
    _, p_value = arima_forecast_analyst.ljung_box_test(returns, lags, model_df=0)
    assert p_value < arima_forecast_analyst.LJUNG_BOX_SIGNIFICANCE


_VAR_DATES = pd.bdate_range(end=AS_OF, periods=150)


def _fred_vintages(values: np.ndarray, index: pd.DatetimeIndex) -> pd.DataFrame:
    """`values` as FRED vintages on `index`, each known the day after its date."""
    known = index + pd.Timedelta(days=1)
    return pd.DataFrame(
        {"value": np.asarray(values, dtype=np.float64), "realtime_start": known, "known_at": known},
        index=index,
    )


def _stub_var_macro(monkeypatch: pytest.MonkeyPatch, vix: np.ndarray | None = None) -> None:
    """VIXCLS (`vix`, else noise around 20), DGS10 and DGS3MO vintages through the day before
    AS_OF."""
    index = _VAR_DATES[:-1]
    frames = {
        VIXCLS: _fred_vintages(20.0 + _noise(len(index), 0) * 100 if vix is None else vix, index),
        DGS10: _fred_vintages(4.0 + _noise(len(index), 1) * 100, index),
        DGS3MO: _fred_vintages(3.0 + _noise(len(index), 2) * 100, index),
    }
    monkeypatch.setattr(
        macro_var_spillover_analyst,
        "load_fred_vintages_from_lakehouse",
        lambda series_id, _as_of: frames[series_id],
    )
    monkeypatch.setattr(macro_var_spillover_analyst, "select_var_lag_order", lambda *_a: 1)


def test_var_non_finite_granger_p_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_var_macro(monkeypatch)
    monkeypatch.setattr(
        macro_var_spillover_analyst, "granger_causality_test", lambda *_a: (0.0, float("nan"))
    )

    signal = MacroVarSpilloverAnalyst().generate_signal("SPY", _market(_noise(150, 8)), AS_OF)

    assert _flat(signal) == ("error", "flat", 0.0)


@pytest.mark.parametrize("value", [0.0, 0.01])
def test_var_constant_asset_returns_are_an_error(
    monkeypatch: pytest.MonkeyPatch, value: float
) -> None:
    """A Granger test on a column with no range is undefined."""
    _stub_var_macro(monkeypatch)

    signal = MacroVarSpilloverAnalyst().generate_signal("SPY", _market(np.full(150, value)), AS_OF)

    assert _flat(signal) == ("error", "flat", 0.0)


@pytest.mark.parametrize("value", [0.0, 20.0])
def test_var_constant_macro_column_is_an_error(
    monkeypatch: pytest.MonkeyPatch, value: float
) -> None:
    _stub_var_macro(monkeypatch, vix=np.full(len(_VAR_DATES) - 1, value))

    signal = MacroVarSpilloverAnalyst().generate_signal("SPY", _market(_noise(150, 8)), AS_OF)

    assert _flat(signal) == ("error", "flat", 0.0)


@pytest.mark.parametrize("value", [0.0, 2.0**-7])
def test_gbm_zero_volatility_is_unavailable(value: float) -> None:
    """Returns with zero sample volatility leave no distribution to simulate; a power of two
    keeps a non-zero constant's sample std exactly 0."""
    signal = GBMMomentumAnalyst().generate_signal("AAPL", _market(np.full(120, value)), AS_OF)

    assert _flat(signal) == ("unavailable", "flat", 0.0)


def test_gbm_non_finite_volatility_is_unavailable() -> None:
    returns = _noise(120)
    returns[-3] = np.inf

    signal = GBMMomentumAnalyst().generate_signal("AAPL", _market(returns), AS_OF)

    assert _flat(signal) == ("unavailable", "flat", 0.0)
