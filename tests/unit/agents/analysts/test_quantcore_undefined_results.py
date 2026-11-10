# test_quantcore_undefined_results.py

"""Each caller of a quantcore function that can raise `quantcore.errors.UndefinedResultError`
(quantcore's `docs/UNDEFINED_RESULTS.md`) maps it to the status the data gives: `unavailable`
when the data cannot define the result, an abstention when the model declines on data it has,
and `error` where the analyst treats an undefined test of its fitted system as a model failure.
Each test drives the real quantcore function into its undefined case where the inputs can reach
it, and stubs only the lake loaders. The OU analyst's cases are in
`test_undefined_statistics.py`."""

from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import pytest
from quantcore.errors import UndefinedResultError
from quantcore.risk import fit_garch_11
from quantcore.risk.egarch import egarch_fit, gjr_garch_fit
from quantcore.statistics.regression import factor_loadings
from quantcore.time_series.arima import arima_fit
from quantcore.time_series.var_model import select_var_lag_order

from agentic_trading.agents import _market_math
from agentic_trading.agents.analysts.factor_pod import (
    fama_french_factor_analyst,
    fama_macbeth_analyst,
)
from agentic_trading.agents.analysts.macro_rates_pod import macro_var_spillover_analyst
from agentic_trading.agents.analysts.macro_rates_pod.arima_forecast_analyst import (
    ARIMAForecastAnalyst,
)
from agentic_trading.agents.analysts.macro_rates_pod.egarch_asymmetric_vol_analyst import (
    EGARCHAsymmetricVolAnalyst,
)
from agentic_trading.agents.analysts.macro_rates_pod.gjr_garch_crash_risk_analyst import (
    GJRGARCHCrashRiskAnalyst,
)
from agentic_trading.agents.analysts.stat_arb_pod import _pairs_screening
from agentic_trading.agents.schemas import VolatilityProvenance
from agentic_trading.config import settings
from agentic_trading.data.loaders.lakehouse_loader import ken_french_public_date
from agentic_trading.data.loaders.macro_loader import DGS3MO, DGS10, VIXCLS
from unit.agents.analysts.test_signal_status_meanings import (
    _FM_DEFAULTS,
    _FM_FACTORS,
    _assert_flat,
    _recorded,
    _stub,
)

AS_OF = pd.Timestamp("2024-06-14")


def _market(returns: np.ndarray) -> pd.DataFrame:
    """A single-asset frame of business-day `log_return`s ending AS_OF, with closes."""
    index = pd.bdate_range(end=AS_OF, periods=len(returns))
    frame = pd.DataFrame({"log_return": np.asarray(returns, dtype=np.float64)}, index=index)
    frame["close"] = 100.0 * np.exp(np.cumsum(frame["log_return"].to_numpy()))
    return frame


def _noise(size: int, seed: int, scale: float = 0.01) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, scale, size)


# GARCH family: returns with no range define no fit.


@pytest.mark.parametrize("value", [0.0, 0.01])
@pytest.mark.parametrize(
    ("analyst", "fit"),
    [(EGARCHAsymmetricVolAnalyst, egarch_fit), (GJRGARCHCrashRiskAnalyst, gjr_garch_fit)],
    ids=["egarch", "gjr"],
)
def test_garch_family_constant_returns_are_unavailable(
    analyst: Any, fit: Any, value: float
) -> None:
    returns = np.full(settings.analysts.risk_window_garch, value)
    with pytest.raises(UndefinedResultError):
        fit(returns)

    signal = analyst().generate_signal("SPY", _market(returns), AS_OF)

    _assert_flat(signal, "unavailable")
    assert "undefined" in signal.rationale


@pytest.mark.parametrize("value", [0.0, 0.01])
def test_conditional_volatility_of_constant_returns_falls_back_to_ewma(value: float) -> None:
    """The GARCH(1,1) fit is undefined on returns with no range; the volatility is the EWMA's,
    which is 0 there."""
    returns = np.full(300, value)
    with pytest.raises(UndefinedResultError):
        fit_garch_11(returns)
    provenance = VolatilityProvenance()

    volatility = _market_math.conditional_daily_volatility(
        returns, _market_math.RefitCache(), "SPY", AS_OF, "test", provenance
    )

    assert provenance.estimator == "ewma"
    assert volatility == 0.0


# ARIMA: a differenced series with no range has no likelihood maximum.


def test_arima_fit_on_a_differenced_series_with_no_range_is_unavailable() -> None:
    """With d pinned at 2, returns that grow by the same step every day (exact in float64) give
    a twice-differenced log price with no range: the fit is undefined."""
    returns = (np.arange(300.0) - 150.0) * 2.0**-14
    with pytest.raises(UndefinedResultError):
        arima_fit(np.cumsum(returns[-250:]), 1, 2, 0)

    signal = ARIMAForecastAnalyst(p=1, d=2, q=0).generate_signal("TLT", _market(returns), AS_OF)

    _assert_flat(signal, "unavailable")
    assert "ARIMA" in signal.rationale


def test_arima_order_selection_undefined_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """`select_arima_order` raises when every candidate's differenced series has no range."""
    from agentic_trading.agents.analysts.macro_rates_pod import arima_forecast_analyst

    monkeypatch.setattr(
        arima_forecast_analyst,
        "select_arima_order",
        _stub(UndefinedResultError("every candidate's differenced series has zero range")),
    )
    rng = np.random.default_rng(11)
    returns = np.empty(300)
    returns[0] = 0.0
    for t in range(1, 300):
        returns[t] = 0.6 * returns[t - 1] + rng.normal(0.0, 0.01)

    signal = ARIMAForecastAnalyst().generate_signal("TLT", _market(returns), AS_OF)

    _assert_flat(signal, "unavailable")


# Macro VAR: an undefined test of the VAR system is an error, as for a constant Granger input.


def _fred(values: np.ndarray, index: pd.DatetimeIndex) -> pd.DataFrame:
    known = index + pd.Timedelta(days=1)
    return pd.DataFrame(
        {"value": np.asarray(values, dtype=np.float64), "realtime_start": known, "known_at": known},
        index=index,
    )


def test_var_constant_yield_curve_slope_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A yield-curve slope that never moves is a VAR column with no range: lag selection is
    undefined, and the analyst reports the VAR fit as failed."""
    dates = pd.bdate_range(end=AS_OF, periods=150)[:-1]
    frames = {
        VIXCLS: _fred(20.0 + _noise(len(dates), 0) * 100, dates),
        DGS10: _fred(np.full(len(dates), 4.0), dates),
        DGS3MO: _fred(np.full(len(dates), 3.0), dates),
    }
    monkeypatch.setattr(
        macro_var_spillover_analyst,
        "load_fred_vintages_from_lakehouse",
        lambda series_id, _as_of: frames[series_id],
    )
    system = np.column_stack([_noise(149, 8), frames[VIXCLS]["value"], np.ones(149)])
    with pytest.raises(UndefinedResultError):
        select_var_lag_order(system, macro_var_spillover_analyst.MAX_LAGS, "aic")

    signal = macro_var_spillover_analyst.MacroVarSpilloverAnalyst().generate_signal(
        "SPY", _market(_noise(150, 8)), AS_OF
    )

    _assert_flat(signal, "error")
    assert "VAR spillover fit failed" in signal.rationale


# Factor pod: a factor or a return series with no range defines no loadings or t-statistics.


def _ff_factors(constant_column: str | None = None) -> pd.DataFrame:
    index = pd.bdate_range("2022-01-03", "2024-04-30")
    values = np.random.default_rng(14).normal(0.0, 0.01, (len(index), 3))
    frame = pd.DataFrame(values, index=index, columns=["Mkt-RF", "SMB", "HML"])
    if constant_column is not None:
        frame[constant_column] = 0.0
    frame["RF"] = 0.0001
    return frame


def test_fama_french_constant_factor_column_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factors = _ff_factors(constant_column="HML")
    monkeypatch.setattr(
        fama_french_factor_analyst, "load_fama_french_factors_from_lakehouse", _stub(factors)
    )
    with pytest.raises(UndefinedResultError):
        factor_loadings(_noise(200, 1), factors[["Mkt-RF", "SMB", "HML"]].to_numpy()[:200])

    signal = fama_french_factor_analyst.FamaFrenchFactorAnalyst().generate_signal(
        "AAPL", _market(_noise(300, 13)), AS_OF
    )

    _assert_flat(signal, "unavailable", ken_french_public_date(date(2024, 4, 30)))
    assert signal.sample_count is not None and signal.sample_count >= 120


def test_fama_french_constant_excess_returns_are_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Excess returns with no range give NaN t-statistics: no loading's significance is
    defined, which is not the same as no loading being significant."""
    factors = _ff_factors()
    monkeypatch.setattr(
        fama_french_factor_analyst, "load_fama_french_factors_from_lakehouse", _stub(factors)
    )
    regression = factor_loadings(np.zeros(200), factors[["Mkt-RF", "SMB", "HML"]].to_numpy()[:200])
    assert np.all(np.isnan(regression.t_stats))

    signal = fama_french_factor_analyst.FamaFrenchFactorAnalyst().generate_signal(
        "AAPL", _market(np.full(300, 0.0001)), AS_OF
    )

    _assert_flat(signal, "unavailable", ken_french_public_date(date(2024, 4, 30)))
    assert signal.sample_count is not None and signal.sample_count >= 120


def test_fama_macbeth_constant_factor_column_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factors = _FM_FACTORS.copy()
    factors["HML"] = 0.0
    for name, outcome in {
        **_FM_DEFAULTS,
        "load_fama_french_factors_from_lakehouse": factors,
    }.items():
        monkeypatch.setattr(fama_macbeth_analyst, name, _stub(outcome))

    signal = _recorded(fama_macbeth_analyst.FamaMacbethAnalyst(), "T000", pd.DataFrame())

    _assert_flat(signal, "unavailable")
    assert "Fama-MacBeth" in signal.rationale


def test_fama_macbeth_undefined_premia_are_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """`fama_macbeth_regression` raises when a column of betas is all zero."""
    for name, outcome in _FM_DEFAULTS.items():
        monkeypatch.setattr(fama_macbeth_analyst, name, _stub(outcome))
    monkeypatch.setattr(
        fama_macbeth_analyst,
        "fama_macbeth_regression",
        _stub(UndefinedResultError("column 0 of betas is all zero")),
    )

    signal = _recorded(fama_macbeth_analyst.FamaMacbethAnalyst(), "T000", pd.DataFrame())

    _assert_flat(signal, "unavailable")


# Pair re-screening: an undefined statistic rejects the pair.


@pytest.mark.parametrize("failing", ["adf_test", "engle_granger_test", "ou_half_life"])
def test_pair_rescreen_rejects_an_undefined_statistic(
    monkeypatch: pytest.MonkeyPatch, failing: str
) -> None:
    rng = np.random.default_rng(0)
    common = rng.normal(0.0, 0.01, 300)
    dates = pd.bdate_range("2023-01-02", periods=300)
    frame = pd.concat(
        {
            ticker: pd.DataFrame(
                {"close": 100 * np.exp(np.cumsum(0.9 * common + rng.normal(0, 0.004, 300)))},
                index=dates,
            )
            for ticker in ("A", "B")
        },
        axis=1,
    )
    monkeypatch.setattr(_pairs_screening, "adf_test", lambda *_a, **_k: (-1.0, 0.5))
    monkeypatch.setattr(_pairs_screening, "engle_granger_test", lambda *_a: (1.0, -4.0, 0.01))
    monkeypatch.setattr(_pairs_screening, "ou_half_life", lambda _spread: 15.0)
    assert _pairs_screening.rescreen_pairs(frame, dates[-1], (("A", "B"),), 100) != {}
    monkeypatch.setattr(
        _pairs_screening, failing, _stub(UndefinedResultError(f"{failing} undefined"))
    )

    assert _pairs_screening.rescreen_pairs(frame, dates[-1], (("A", "B"),), 100) == {}
