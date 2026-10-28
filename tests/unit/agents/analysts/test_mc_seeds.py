# test_mc_seeds.py

"""Monte Carlo analysts seed their simulations from `derived_seed`."""

import hashlib
from collections.abc import Callable
from datetime import date
from importlib import import_module
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from agentic_trading.agents._market_math import derived_seed
from agentic_trading.config import settings, temporary_overrides

_AS_OF = date(2024, 3, 1)


def _expected_seed(base: int, analyst_id: str, asset: str, as_of: date) -> int:
    key = f"{base}|{analyst_id}|{asset}|{as_of.isoformat()}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:16], 16) % 2**32


def test_derived_seed_is_deterministic_and_matches_sha256_formula() -> None:
    """`derived_seed` equals the documented sha256 construction and is stable across calls."""
    seed = derived_seed("some_analyst", "AAPL", _AS_OF)

    assert settings.analysts.mc_seed == 42
    assert seed == _expected_seed(42, "some_analyst", "AAPL", _AS_OF)
    assert seed == derived_seed("some_analyst", "AAPL", _AS_OF)
    assert 0 <= seed < 2**32


def test_derived_seed_changes_with_every_input() -> None:
    """Changing the analyst id, asset, date or base seed changes the seed."""
    base = derived_seed("a", "AAPL", _AS_OF)

    assert derived_seed("b", "AAPL", _AS_OF) != base
    assert derived_seed("a", "MSFT", _AS_OF) != base
    assert derived_seed("a", "AAPL", date(2024, 3, 2)) != base
    with temporary_overrides({"analysts.mc_seed": 7}):
        overridden = derived_seed("a", "AAPL", _AS_OF)
    assert overridden != base
    assert overridden == _expected_seed(7, "a", "AAPL", _AS_OF)
    assert derived_seed("a", "AAPL", _AS_OF) == base


def _market_data() -> tuple[pd.DataFrame, pd.Timestamp]:
    dates = pd.date_range("2023-09-01", "2024-03-01", freq="D")
    returns = np.random.default_rng(0).normal(0.0, 0.01, len(dates))
    frame = pd.DataFrame(
        {"close": 100.0 * np.exp(np.cumsum(returns)), "log_return": returns}, index=dates
    )
    return frame, dates[-1]


def _patch_monte_carlo(mod: Any, monkeypatch: pytest.MonkeyPatch, sim: Callable[..., Any]) -> None:
    monkeypatch.setattr(mod, "simulate_gbm_paths", sim)
    monkeypatch.setattr(mod, "conditional_daily_volatility", lambda r, *_a, **_k: 0.01)


def _patch_gbm(mod: Any, monkeypatch: pytest.MonkeyPatch, sim: Callable[..., Any]) -> None:
    monkeypatch.setattr(mod, "simulate_gbm_paths", sim)


def _patch_cir(mod: Any, monkeypatch: pytest.MonkeyPatch, sim: Callable[..., Any]) -> None:
    rates = pd.Series(4.0 + np.random.default_rng(1).normal(0.0, 0.1, 400))
    monkeypatch.setattr(mod, "load_fred_series_from_lakehouse", lambda *_a: rates)
    monkeypatch.setattr(
        mod, "cir_fit", lambda *_a: SimpleNamespace(kappa=1.0, theta=0.04, sigma=0.05)
    )
    monkeypatch.setattr(mod, "simulate_cir_paths", sim)


def _patch_ou(mod: Any, monkeypatch: pytest.MonkeyPatch, sim: Callable[..., Any]) -> None:
    monkeypatch.setattr(mod, "ou_half_life", lambda _spread: 10.0)
    monkeypatch.setattr(mod, "spread_direction", lambda *_a: "long")
    monkeypatch.setattr(mod, "simulate_ou_paths", sim)


_CASES = [
    (
        "agentic_trading.agents.analysts.cvar_pod.monte_carlo_valuation_analyst",
        "MonteCarloValuationAnalyst",
        _patch_monte_carlo,
    ),
    (
        "agentic_trading.agents.analysts.risk_pod.gbm_momentum_analyst",
        "GBMMomentumAnalyst",
        _patch_gbm,
    ),
    (
        "agentic_trading.agents.analysts.macro_rates_pod.cir_rate_regime_analyst",
        "CIRRateRegimeAnalyst",
        _patch_cir,
    ),
    (
        "agentic_trading.agents.analysts.stat_arb_pod.ou_mean_reversion_analyst",
        "OUMeanReversionAnalyst",
        _patch_ou,
    ),
]


@pytest.mark.parametrize(("module_path", "class_name", "patcher"), _CASES)
def test_analyst_passes_derived_seed_to_its_simulation(
    monkeypatch: pytest.MonkeyPatch, module_path: str, class_name: str, patcher: Callable[..., None]
) -> None:
    """Each Monte Carlo analyst seeds its simulator with `derived_seed(agent_id, asset, date)`."""
    mod = import_module(module_path)
    seeds: list[int] = []

    def _simulate(*_args: Any, **kwargs: Any) -> np.ndarray:
        seeds.append(kwargs["seed"])
        return np.ones((4, 60))

    patcher(mod, monkeypatch, _simulate)
    frame, as_of = _market_data()
    analyst = getattr(mod, class_name)()

    analyst.generate_signal("AAPL", frame, as_of)
    analyst.generate_signal("MSFT", frame, as_of)

    assert len(seeds) == 2
    assert seeds[0] == derived_seed(analyst.agent_id, "AAPL", as_of.date())
    assert seeds[1] == derived_seed(analyst.agent_id, "MSFT", as_of.date())
    assert seeds[0] != seeds[1]
    assert 42 not in seeds
