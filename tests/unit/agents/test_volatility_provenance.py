# test_volatility_provenance.py

"""`conditional_daily_volatility` reports which estimator gave its volatility through the
`VolatilityProvenance` it is handed: `garch_11` for a usable GARCH(1,1) fit, `ewma` for the
RiskMetrics fallback. A fallback on a day the fit is re-run logs one WARNING naming the caller
and the cache key; a day that reuses a cached unusable fit logs none. The CVaR-pod analysts
that read it (Black-Scholes and Monte Carlo valuation, Heston, through
`_options_data.annualized_garch_volatility`) record the estimator on their signal's
`volatility_estimator`, and a manager carries it onto its `ManagerSignal`."""

import logging
from collections.abc import Iterator
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import pytest
import quantcore.risk
from quantcore.risk import ConvergenceError
from quantcore.risk.egarch import ewma_variance

from agentic_trading.agents._market_math import (
    RefitCache,
    VolatilityProvenance,
    conditional_daily_volatility,
)
from agentic_trading.agents.analysts.cvar_pod import (
    _options_data,
    black_scholes_valuation_analyst,
    heston_vol_surface_analyst,
    monte_carlo_valuation_analyst,
)
from agentic_trading.agents.base import AnalystAgent, ManagerAgent
from agentic_trading.agents.portfolio_construction import PortfolioConstructor
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.config import settings
from agentic_trading.scoring.scoreboard import PerformanceScoreboard
from agentic_trading.scoring.track_record import TrackRecord
from agentic_trading.scoring.weighting import EqualWeight

AS_OF = pd.Timestamp("2024-06-17")
_LOGGER = "agentic_trading.agents._market_math"


@pytest.fixture
def records(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """Routes the package logger (which does not propagate) into `caplog`."""
    package_logger = logging.getLogger("agentic_trading")
    package_logger.addHandler(caplog.handler)
    previous_level = package_logger.level
    package_logger.setLevel(logging.DEBUG)
    yield caplog
    package_logger.setLevel(previous_level)
    package_logger.removeHandler(caplog.handler)


def _returns(periods: int = 300) -> np.ndarray:
    return np.random.default_rng(5).normal(0.0, 0.01, periods)


def _ewma_volatility(returns: np.ndarray) -> float:
    centered = returns - float(np.mean(returns))
    return float(np.sqrt(ewma_variance(centered, settings.managers.ewma_lambda)[-1]))


def _raise_convergence(_returns: np.ndarray) -> Any:
    raise ConvergenceError(
        "no start converged",
        params=(1e-6, 0.05, 0.9),
        log_likelihood=0.0,
        optimizer_message="Iteration limit reached",
    )


def _usable_fit(returns: np.ndarray) -> tuple[float, float, float]:
    """GARCH(1,1) parameters whose unconditional variance is the sample variance."""
    variance = float(np.var(returns - returns.mean(), ddof=1))
    return 0.05 * variance, 0.05, 0.9


def _fallback_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == _LOGGER and record.levelno == logging.WARNING
    ]


def test_a_failed_fit_falls_back_to_ewma_and_says_so(
    monkeypatch: pytest.MonkeyPatch, records: pytest.LogCaptureFixture
) -> None:
    """A fit that raises gives the EWMA volatility, marks `ewma` and warns once, naming the
    caller and the key."""
    monkeypatch.setattr(quantcore.risk, "fit_garch_11", _raise_convergence)
    returns = _returns()
    provenance = VolatilityProvenance()

    volatility = conditional_daily_volatility(
        returns, RefitCache(), "AAPL", AS_OF, "test_agent", provenance=provenance
    )

    assert volatility == pytest.approx(_ewma_volatility(returns))
    assert provenance.estimator == "ewma"
    warnings = _fallback_warnings(records)
    assert len(warnings) == 1
    assert "test_agent" in warnings[0] and "AAPL" in warnings[0] and "EWMA" in warnings[0]


def test_a_cached_unusable_fit_warns_only_on_its_refit_day(
    monkeypatch: pytest.MonkeyPatch, records: pytest.LogCaptureFixture
) -> None:
    """An over-persistent fit is cached: the day it is fitted warns, the days that reuse it
    fall back to EWMA silently, and each marks `ewma`."""
    calls: list[int] = []

    def _over_persistent(_returns: np.ndarray) -> tuple[float, float, float]:
        calls.append(1)
        return 1e-7, 0.1, 0.8999

    monkeypatch.setattr(quantcore.risk, "fit_garch_11", _over_persistent)
    cache = RefitCache(refit_interval_days=5)
    returns = _returns()
    estimators = []
    for offset in range(3):
        provenance = VolatilityProvenance()
        conditional_daily_volatility(
            returns,
            cache,
            "AAPL",
            AS_OF + pd.Timedelta(days=offset),
            "test_agent",
            provenance=provenance,
        )
        estimators.append(provenance.estimator)

    assert len(calls) == 1
    assert estimators == ["ewma"] * 3
    assert len(_fallback_warnings(records)) == 1


def test_a_usable_fit_marks_garch_and_does_not_warn(
    monkeypatch: pytest.MonkeyPatch, records: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(quantcore.risk, "fit_garch_11", _usable_fit)
    provenance = VolatilityProvenance()

    conditional_daily_volatility(
        _returns(), RefitCache(), "AAPL", AS_OF, "test_agent", provenance=provenance
    )

    assert provenance.estimator == "garch_11"
    assert _fallback_warnings(records) == []


def test_without_a_provenance_the_volatility_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The provenance is optional and does not change the value."""
    monkeypatch.setattr(quantcore.risk, "fit_garch_11", _raise_convergence)
    returns = _returns()

    assert conditional_daily_volatility(
        returns, RefitCache(), "AAPL", AS_OF, "test_agent"
    ) == pytest.approx(_ewma_volatility(returns))


def _market(periods: int = 300) -> pd.DataFrame:
    days = pd.bdate_range(end=AS_OF, periods=periods)
    returns = _returns(periods)
    return pd.DataFrame(
        {"close": 100.0 * np.exp(np.cumsum(returns)), "log_return": returns}, index=days
    )


def _chain(*_args: object, **_kwargs: object) -> pd.DataFrame:
    """Two ATM rows, at 30 and 60 days, an implied vol of 0.2 and 0.22."""
    snapshot = pd.Timestamp("2024-06-14")
    return pd.DataFrame(
        {
            "snapshot_date": [snapshot, snapshot],
            "expiration": [snapshot + pd.Timedelta(days=30), snapshot + pd.Timedelta(days=60)],
            "days_to_expiration": [30, 60],
            "strike": [100.0, 100.0],
            "implied_volatility": [0.2, 0.22],
            "delta": [0.5, 0.5],
            "gamma": [0.02, 0.02],
        }
    )


_ANALYSTS = {
    "black_scholes": (black_scholes_valuation_analyst, "BlackScholesValuationAnalyst"),
    "monte_carlo": (monte_carlo_valuation_analyst, "MonteCarloValuationAnalyst"),
    "heston": (heston_vol_surface_analyst, "HestonVolSurfaceAnalyst"),
}


def _rates(_series_id: str, _as_of_date: object) -> pd.DataFrame:
    """A one-row DGS3MO frame in percent, known before `AS_OF`, for the analysts that read r."""
    known = pd.Timestamp(AS_OF) - pd.Timedelta(days=1)
    return pd.DataFrame(
        {"value": [1.5], "realtime_start": [known], "known_at": [known]},
        index=pd.DatetimeIndex([known], name="date"),
    )


def _signal(monkeypatch: pytest.MonkeyPatch, name: str) -> AgentSignal:
    module, class_name = _ANALYSTS[name]
    if hasattr(module, "real_atm_options_by_expiration"):
        monkeypatch.setattr(module, "real_atm_options_by_expiration", _chain)
    if hasattr(module, "load_fred_vintages_from_lakehouse"):
        monkeypatch.setattr(module, "load_fred_vintages_from_lakehouse", _rates)
    return getattr(module, class_name)().generate_signal("AAPL", _market(), AS_OF)


@pytest.mark.parametrize("name", list(_ANALYSTS))
@pytest.mark.parametrize(
    ("fit", "estimator"), [(_raise_convergence, "ewma"), (_usable_fit, "garch_11")]
)
def test_cvar_pod_signals_record_the_estimator(
    monkeypatch: pytest.MonkeyPatch, name: str, fit: Any, estimator: str
) -> None:
    """Each analyst's signal records the estimator its volatility came from."""
    monkeypatch.setattr(quantcore.risk, "fit_garch_11", fit)

    signal = _signal(monkeypatch, name)

    assert signal.volatility_estimator == estimator


def test_heston_records_its_realized_vol_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the GARCH read itself raises, Heston's realized-vol fallback is recorded."""

    def _raise(*_args: object, **_kwargs: object) -> float:
        raise RuntimeError("volatility read failed")

    monkeypatch.setattr(heston_vol_surface_analyst, "annualized_garch_volatility", _raise)

    signal = _signal(monkeypatch, "heston")

    assert signal.volatility_estimator == "realized_std"


def test_annualized_garch_volatility_passes_the_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(quantcore.risk, "fit_garch_11", _raise_convergence)
    provenance = VolatilityProvenance()

    _options_data.annualized_garch_volatility(
        _returns(), RefitCache(), "heston_vol_surface_analyst", "AAPL", AS_OF, provenance
    )

    assert provenance.estimator == "ewma"


class _EwmaAnalyst(AnalystAgent):
    agent_id: str = "ewma_analyst"

    def generate_signal(
        self, asset: str, market_data: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> AgentSignal:
        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction="long",
            confidence=0.4,
            rationale="stub",
            volatility_estimator="ewma",
        )


class _PassManager(ManagerAgent):
    agent_id: str = "pass_manager"

    def scale_factor(
        self, signal: AgentSignal, market_data: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> float:
        return 1.0


def test_a_manager_carries_the_estimator_onto_its_signal() -> None:
    manager = _PassManager(
        managed_analysts=[_EwmaAnalyst()],
        weighting_strategy=EqualWeight(),
        portfolio_constructor=PortfolioConstructor(),
    )
    signal = _EwmaAnalyst().generate_signal("AAPL", _market(), AS_OF)

    (scaled,) = manager.evaluate(
        [signal], _market(), AS_OF, PerformanceScoreboard(track_record=TrackRecord())
    )

    assert scaled.volatility_estimator == "ewma"


def test_the_estimator_defaults_to_none() -> None:
    signal = AgentSignal(
        agent_id="a",
        asset="AAPL",
        as_of_date=date(2024, 1, 2),
        direction="flat",
        confidence=0.0,
        rationale="stub",
    )

    assert signal.volatility_estimator is None
