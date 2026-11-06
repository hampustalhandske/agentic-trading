# test_fama_macbeth_cross_section_cache.py

"""`FamaMacbethAnalyst` runs its cross-sectional stage (every kept ticker's factor loadings and
the Fama-MacBeth premia) once per decision day and reference index, and reuses it for every
asset that shares both: each signal equals, field for field, the one a fresh analyst computes
from scratch; the cache holds only the current day; and a checkpointed analyst carries none of
it, so a restored one gives the same signals."""

from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import pytest

from agentic_trading.agents._market_math import RefitCache
from agentic_trading.agents.analysts.factor_pod import fama_macbeth_analyst as mod
from agentic_trading.agents.analysts.factor_pod.fama_macbeth_analyst import FamaMacbethAnalyst
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.data.loaders.lakehouse_loader import IndexMember

_FACTORS = ["Mkt-RF", "SMB", "HML"]
_N_TICKERS = 40
_DATES = pd.bdate_range("2021-01-04", periods=200)
_DAY_ONE, _DAY_TWO = _DATES[-2], _DATES[-1]
# T001 and T002 each miss one recent date: their own reference index differs from every other
# member's, and they drop out of every other member's balanced panel.
_GAPS = {"T001": -30, "T002": -10}
# ZZZ is not an S&P member: it is read by ticker and appended to the cross-section, with a gap
# of its own.
_OUTSIDER = "ZZZ"
_ASSETS = ["T000", "T003", "T004", "T001", "T005", "T002", _OUTSIDER, "T006", "T000"]


def _synthetic() -> tuple[dict[str, pd.Series], pd.DataFrame]:
    """Member and outsider log returns, gaps dropped, and a factor table on `_DATES`."""
    rng = np.random.default_rng(42)
    factors = pd.DataFrame(rng.normal(0.0, 0.01, (_DATES.size, 3)), index=_DATES, columns=_FACTORS)
    factors["RF"] = rng.uniform(0.00005, 0.00015, _DATES.size)
    returns: dict[str, pd.Series] = {}
    for ticker in [*(f"T{i:03d}" for i in range(_N_TICKERS)), _OUTSIDER]:
        betas = rng.normal(1.0, 0.5, 3)
        values = factors[_FACTORS].to_numpy() @ betas + rng.normal(0.0, 0.004, _DATES.size)
        series = pd.Series(values, index=_DATES)
        if ticker in _GAPS:
            series.iloc[_GAPS[ticker]] = np.nan
        if ticker == _OUTSIDER:
            series.iloc[-50] = np.nan
        returns[ticker] = series.dropna()
    return returns, factors


@pytest.fixture
def lake(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serves the synthetic data through the analyst module's loaders, each cut at its end date."""
    returns, factors = _synthetic()
    members = [IndexMember(security_id=i, ticker=f"T{i:03d}") for i in range(_N_TICKERS)]

    def security_returns(ids: list[int], start: date, end: date) -> dict[int, pd.Series]:
        return {
            m.security_id: returns[m.ticker].loc[pd.Timestamp(start) : pd.Timestamp(end)]
            for m in members
            if m.security_id in ids
        }

    def price_history(tickers: list[str], start: date, end: date) -> pd.DataFrame:
        series = returns[_OUTSIDER].loc[pd.Timestamp(start) : pd.Timestamp(end)]
        return pd.concat({_OUTSIDER: pd.DataFrame({"log_return": series})}, axis=1)

    monkeypatch.setattr(mod, "load_sp500_member_securities_asof", lambda *_a: members)
    monkeypatch.setattr(mod, "load_security_log_returns", security_returns)
    monkeypatch.setattr(mod, "load_price_history_from_lakehouse", price_history)
    monkeypatch.setattr(
        mod,
        "load_fama_french_factors_from_lakehouse",
        lambda as_of, model="3factor": factors.loc[: pd.Timestamp(as_of)],
    )


def _counter(monkeypatch: pytest.MonkeyPatch, name: str) -> list[int]:
    """Wraps `mod.<name>` so every call appends to the returned list."""
    real = getattr(mod, name)
    calls: list[int] = []

    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(mod, name, _wrapped)
    return calls


def _uncached(analyst: FamaMacbethAnalyst, asset: str, day: pd.Timestamp) -> AgentSignal:
    """The signal a fresh analyst computes from scratch, with an empty cross-section cache, on the
    broad universe `analyst` holds for `day`."""
    fresh = FamaMacbethAnalyst()
    fresh._universe_refit_cache = analyst._universe_refit_cache
    fresh._universe_tickers_by_security = analyst._universe_tickers_by_security
    return fresh.generate_signal(asset, pd.DataFrame(), day)


@pytest.mark.usefixtures("lake")
@pytest.mark.parametrize("refit_interval_days", [1, 5])
def test_cached_signals_equal_fresh_signals_field_for_field(refit_interval_days: int) -> None:
    """Over two consecutive days, with the broad universe either reloaded or reused on the second,
    every asset's signal equals a fresh analyst's, including the assets whose reference index
    differs from the others' and the non-member read by ticker."""
    analyst = FamaMacbethAnalyst()
    analyst._universe_refit_cache = RefitCache(refit_interval_days=refit_interval_days)

    for day in (_DAY_ONE, _DAY_TWO):
        for asset in _ASSETS:
            cached = analyst.generate_signal(asset, pd.DataFrame(), day)
            reference = _uncached(analyst, asset, day)

            # On a day the reused universe ends before, the outsider's own returns reach a date
            # no member has, which leaves too few names: unavailable, for both analysts.
            assert cached.status == "active" or asset == _OUTSIDER
            assert cached.model_dump() == reference.model_dump()
            assert cached == reference


@pytest.mark.usefixtures("lake")
def test_stage_runs_once_per_day_and_reference_index(monkeypatch: pytest.MonkeyPatch) -> None:
    """Members sharing a reference index share one stage; a member with its own reference index
    runs its own; a new day runs it again."""
    loadings = _counter(monkeypatch, "factor_loadings")
    regressions = _counter(monkeypatch, "fama_macbeth_regression")
    analyst = FamaMacbethAnalyst()
    # The full-history members keep every reference date, T001 and T002 miss one each.
    shared_panel = _N_TICKERS - len(_GAPS)

    for asset in ("T000", "T003", "T004", "T010"):
        analyst.generate_signal(asset, pd.DataFrame(), _DAY_ONE)
    assert (len(loadings), len(regressions)) == (shared_panel, 1)

    # T001's own reference dates leave out its gap, so only T002 misses one of them.
    analyst.generate_signal("T001", pd.DataFrame(), _DAY_ONE)
    analyst.generate_signal("T001", pd.DataFrame(), _DAY_ONE)
    analyst.generate_signal("T000", pd.DataFrame(), _DAY_ONE)
    assert (len(loadings), len(regressions)) == (shared_panel + _N_TICKERS - 1, 2)

    analyst.generate_signal("T000", pd.DataFrame(), _DAY_TWO)
    analyst.generate_signal("T003", pd.DataFrame(), _DAY_TWO)
    assert (len(loadings), len(regressions)) == (2 * shared_panel + _N_TICKERS - 1, 3)


@pytest.mark.usefixtures("lake")
def test_cache_holds_only_the_current_day() -> None:
    """A new decision day drops every stage of the day before."""
    analyst = FamaMacbethAnalyst()
    cache = analyst._cross_section_cache

    analyst.generate_signal("T000", pd.DataFrame(), _DAY_ONE)
    analyst.generate_signal("T001", pd.DataFrame(), _DAY_ONE)
    assert (cache.day, len(cache)) == (_DAY_ONE, 2)

    analyst.generate_signal("T000", pd.DataFrame(), _DAY_TWO)
    assert (cache.day, len(cache)) == (_DAY_TWO, 1)


@pytest.mark.usefixtures("lake")
def test_checkpoint_carries_no_stage_and_a_restored_analyst_gives_the_same_signals() -> None:
    """`carried_state` snapshots the broad-universe cache but not the cross-section cache: the
    restored analyst starts with an empty one and gives the signals the original goes on to give."""
    original = FamaMacbethAnalyst()
    for asset in ("T000", "T001"):
        original.generate_signal(asset, pd.DataFrame(), _DAY_ONE)
    assert len(original._cross_section_cache) == 2

    restored = FamaMacbethAnalyst()
    restored.restore_carried_state(original.carried_state())

    assert len(restored._cross_section_cache) == 0
    assert restored._cross_section_cache.day is None
    for day in (_DAY_ONE, _DAY_TWO):
        for asset in _ASSETS:
            assert restored.generate_signal(asset, pd.DataFrame(), day) == (
                original.generate_signal(asset, pd.DataFrame(), day)
            )
