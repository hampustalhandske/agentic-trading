# test_heston_resume_identity.py

"""`HestonVolSurfaceAnalyst` resumed mid-run from its `carried_state()` produces exactly the
signals an unbroken run produces on every later day. Its GARCH(1,1) read is refit at most once
every `RefitCache.refit_interval_days` simulated days, so a resume inside a refit window gives the
same signals only when the fit it reuses travels with the snapshot.

A resume in a real firm record runs in a fresh process, where module-level state starts empty.
Each resumed run here imports its own copy of the Heston module and of the CVaR pod's
`_options_data` module, so it starts from the module-level state of a fresh process too; the
copies are removed from `sys.modules` again when the test ends."""

import importlib
import sys
from datetime import date
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from agentic_trading.agents.analysts.cvar_pod import heston_vol_surface_analyst
from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.data.loaders.lakehouse_loader import DEFAULT_OPTIONS_SNAPSHOT_LOOKBACK_DAYS

ASSET = "AAPL"
_PACKAGE = "agentic_trading.agents.analysts.cvar_pod"
_FRESH_MODULES = (f"{_PACKAGE}._options_data", f"{_PACKAGE}.heston_vol_surface_analyst")
_HISTORY_DAYS = 400
_DECISION_DAYS = 20
_INDEX = pd.bdate_range("2023-01-02", periods=_HISTORY_DAYS)
_SESSIONS = _INDEX[-_DECISION_DAYS:]
# Contango ATM implied vols 2.5 to 6 vol points above the GARCH read on `_market`'s returns.
_CHAIN_IVS = (0.18, 0.19, 0.20)


def _market() -> pd.DataFrame:
    """AAPL on a seeded GARCH(1,1) path (omega 2e-6, alpha 0.08, beta 0.90; about 1% a day), in
    the loader's (ticker, field) shape."""
    rng = np.random.default_rng(11)
    omega, alpha, beta = 2e-6, 0.08, 0.90
    variance = omega / (1.0 - alpha - beta)
    returns = np.empty(_HISTORY_DAYS)
    for day in range(_HISTORY_DAYS):
        returns[day] = np.sqrt(variance) * rng.standard_normal()
        variance = omega + alpha * returns[day] ** 2 + beta * variance
    returns[0] = 0.0
    close = 100.0 * np.exp(np.cumsum(returns))
    frame = pd.DataFrame({"close": close, "log_return": returns}, index=_INDEX)
    return pd.concat({ASSET: frame}, axis=1)


def _snapshot(
    ticker: str, as_of_date: date, lookback_days: int = DEFAULT_OPTIONS_SNAPSHOT_LOOKBACK_DAYS
) -> pd.DataFrame:
    """A chain snapshot dated `as_of_date`: a call and a put at every whole strike from 50 to 200
    for three expiries 30, 65 and 128 days out, the n-th expiry quoted at `_CHAIN_IVS[n]`."""
    stamp = pd.Timestamp(as_of_date)
    return pd.DataFrame(
        [
            {
                "date": stamp,
                "expiration": stamp + pd.Timedelta(days=days_out),
                "option_type": kind,
                "strike": float(strike),
                "implied_volatility": iv,
                "delta": 0.5 if kind == "call" else -0.5,
                "gamma": 0.02,
            }
            for days_out, iv in zip((30, 65, 128), _CHAIN_IVS, strict=True)
            for strike in range(50, 201)
            for kind in ("call", "put")
        ]
    )


def _fresh_heston_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """A newly imported copy of the Heston module, reading its chains through a newly imported
    copy of `_options_data` that serves `_snapshot`: the module-level state a fresh process
    starts from. `monkeypatch` puts the original modules back when the test ends."""
    package = importlib.import_module(_PACKAGE)
    for name in _FRESH_MODULES:
        monkeypatch.setattr(package, name.rsplit(".", 1)[1], importlib.import_module(name))
        monkeypatch.delitem(sys.modules, name)
    options_data = importlib.import_module(_FRESH_MODULES[0])
    monkeypatch.setattr(options_data, "load_options_snapshot_from_lakehouse", _snapshot)
    return importlib.import_module(_FRESH_MODULES[1])


def _signals(
    analyst: AnalystAgent, market: pd.DataFrame, sessions: pd.DatetimeIndex
) -> list[AgentSignal]:
    return [analyst.generate_signal(ASSET, market, day) for day in sessions]


def test_resume_inside_a_refit_window_reproduces_the_unbroken_signals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Snapshotted after any day k and restored onto an analyst built in a fresh process, the
    analyst's signals on every day after k equal the unbroken run's exactly; an analyst built
    there without the snapshot diverges after at least one k that falls inside a refit window."""
    market = _market()
    unbroken_analyst = _fresh_heston_module(monkeypatch).HestonVolSurfaceAnalyst()
    unbroken = _signals(unbroken_analyst, market, _SESSIONS)
    assert all(signal.status == "active" for signal in unbroken)
    assert all(signal.direction == "long" for signal in unbroken)
    assert all(
        0.0 < signal.confidence < heston_vol_surface_analyst.MAX_CONFIDENCE for signal in unbroken
    )

    unrestored_diverged = False
    for k in range(len(_SESSIONS) - 1):
        original = _fresh_heston_module(monkeypatch).HestonVolSurfaceAnalyst()
        _signals(original, market, _SESSIONS[: k + 1])
        state = original.carried_state()

        resumed = _fresh_heston_module(monkeypatch).HestonVolSurfaceAnalyst()
        resumed.restore_carried_state(state)
        assert _signals(resumed, market, _SESSIONS[k + 1 :]) == unbroken[k + 1 :], (
            f"resumed after day {k}"
        )

        unrestored = _fresh_heston_module(monkeypatch).HestonVolSurfaceAnalyst()
        unrestored_diverged |= _signals(unrestored, market, _SESSIONS[k + 1 :]) != unbroken[k + 1 :]

    assert unrestored_diverged
