# test_commander_flat_pods.py

"""Zero-variance pods and assets in the cross-pod risk parity and the firm stress test."""

import warnings

import numpy as np
import pandas as pd
import pytest

from agentic_trading.agents.commander import (
    _apply_firm_limits,
    _cross_pod_risk_parity_weights,
    _risk_budget_weights,
)
from agentic_trading.config import temporary_overrides


def _pod_history(columns: dict[str, np.ndarray]) -> dict[str, pd.Series]:
    index = pd.bdate_range("2024-01-01", periods=len(next(iter(columns.values()))))
    return {pod: pd.Series(values, index=index) for pod, values in columns.items()}


def test_cross_pod_risk_parity_leaves_out_flat_pods() -> None:
    """A constant-return pod has no key; with fewer than 2 varying pods the result is None."""
    rng = np.random.default_rng(0)
    periods = 60
    varying = {
        "alpha": rng.normal(0.0005, 0.01, periods),
        "beta": rng.normal(0.0003, 0.02, periods),
    }
    history = _pod_history({**varying, "flat": np.zeros(periods)})

    weights = _cross_pod_risk_parity_weights(history, {"alpha", "beta", "flat"})

    assert weights is not None
    assert set(weights) == {"alpha", "beta"}
    assert all(weight > 0.0 for weight in weights.values())

    two_pods = _pod_history({"alpha": varying["alpha"], "flat": np.zeros(periods)})
    assert _cross_pod_risk_parity_weights(two_pods, {"alpha", "flat"}) is None


def test_firm_stress_test_tolerates_zero_variance_assets() -> None:
    """A book whose assets never move has an all-zero covariance: the stress test
    raises no invalid-divide warning and, with zero stressed loss, leaves the
    weights as they are."""
    periods = 300
    index = pd.bdate_range("2023-01-02", periods=periods)
    flat_a = pd.DataFrame({"close": 100.0, "log_return": np.zeros(periods)}, index=index)
    flat_b = pd.DataFrame({"close": 50.0, "log_return": np.zeros(periods)}, index=index)
    market_data = pd.concat({"AAPL": flat_a, "FLAT": flat_b}, axis=1)
    overrides = {
        "firm.max_single_name_weight": 100.0,
        "firm.max_gross_exposure": 1000.0,
        "firm.max_daily_var": 1000.0,
        "firm.max_stress_loss": 1e-4,
    }

    with temporary_overrides(overrides), warnings.catch_warnings():
        warnings.simplefilter("error")
        limited = _apply_firm_limits({"AAPL": 0.5, "FLAT": -0.5}, market_data, index[-1])

    assert limited == {"AAPL": 0.5, "FLAT": -0.5}


class _EqualWeights:
    """Weighting strategy that gives every pod the same trailing-performance weight."""

    def get_weights(self, scoreboard: object, manager_ids: list[str], as_of: object) -> dict:
        return {manager_id: 1.0 / len(manager_ids) for manager_id in manager_ids}


def test_flat_pod_is_sized_as_an_average_risk_parity_pod() -> None:
    """A pod left out of risk parity for having no return variation gets the
    average risk-parity share, not a multiple of it."""
    rng = np.random.default_rng(2)
    periods = 60
    history = _pod_history(
        {
            "alpha": rng.normal(0.0005, 0.01, periods),
            "beta": rng.normal(0.0003, 0.02, periods),
            "gamma": rng.normal(0.0001, 0.015, periods),
            "flat": np.zeros(periods),
        }
    )
    manager_ids = ["alpha", "beta", "gamma", "flat"]

    weights = _risk_budget_weights(
        history,
        manager_ids,
        scoreboard=None,  # type: ignore[arg-type]
        as_of=history["alpha"].index[-1].date(),
        weighting_strategy=_EqualWeights(),  # type: ignore[arg-type]
    )

    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["flat"] == pytest.approx(0.25)
    assert all(weights[pod] > 0.0 for pod in ("alpha", "beta", "gamma"))
