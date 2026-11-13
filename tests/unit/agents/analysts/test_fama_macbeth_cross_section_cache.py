# test_fama_macbeth_cross_section_cache.py

"""`FamaMacbethAnalyst` runs its cross-sectional stage (every kept ticker's factor loadings and
the Fama-MacBeth premia) once per decision day, universe objects, reference index and factor
rows, and reuses it for every asset that shares all of them. Only a stage built on the universe
objects `_universe_refit_cache` holds is stored; an asset that extends the universe gets a stage
of its own. Each signal equals, field for field, the one a fresh analyst computes from scratch;
the cache holds only the current day; and a checkpointed analyst carries it empty, so a restored
one gives the same signals. A weakened cache key fails these tests."""

from collections.abc import Callable, Hashable
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
# FULL is not an S&P member either and has every date: its reference index and factor rows are
# T000's, its universe holds one more name.
_FULL = "FULL"
_ASSETS = ["T000", "T003", "T004", "T001", "T005", "T002", _OUTSIDER, _FULL, "T006", "T000"]


class _Lake:
    """The synthetic lake the analyst module's loaders read at call time: member and non-member
    log returns (gaps dropped), the members, and the factor table."""

    def __init__(self) -> None:
        rng = np.random.default_rng(42)
        self.factors = pd.DataFrame(
            rng.normal(0.0, 0.01, (_DATES.size, 3)), index=_DATES, columns=_FACTORS
        )
        self.factors["RF"] = rng.uniform(0.00005, 0.00015, _DATES.size)
        self.returns: dict[str, pd.Series] = {}
        for ticker in [*(f"T{i:03d}" for i in range(_N_TICKERS)), _OUTSIDER, _FULL]:
            betas = rng.normal(1.0, 0.5, 3)
            values = self.factors[_FACTORS].to_numpy() @ betas
            series = pd.Series(values + rng.normal(0.0, 0.004, _DATES.size), index=_DATES)
            if ticker in _GAPS:
                series.iloc[_GAPS[ticker]] = np.nan
            if ticker == _OUTSIDER:
                series.iloc[-50] = np.nan
            self.returns[ticker] = series.dropna()
        self.members = [IndexMember(security_id=i, ticker=f"T{i:03d}") for i in range(_N_TICKERS)]

    def security_returns(self, ids: list[int], start: date, end: date) -> dict[int, pd.Series]:
        return {
            m.security_id: self.returns[m.ticker].loc[pd.Timestamp(start) : pd.Timestamp(end)]
            for m in self.members
            if m.security_id in ids
        }

    def price_history(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
        return pd.concat(
            {
                ticker: pd.DataFrame(
                    {
                        "log_return": self.returns[ticker].loc[
                            pd.Timestamp(start) : pd.Timestamp(end)
                        ]
                    }
                )
                for ticker in tickers
            },
            axis=1,
        )

    def factor_table(self, as_of: date, model: str = "3factor") -> pd.DataFrame:
        return self.factors.loc[: pd.Timestamp(as_of)]


@pytest.fixture
def lake(monkeypatch: pytest.MonkeyPatch) -> _Lake:
    """Serves a fresh `_Lake` through the analyst module's loaders, each cut at its end date."""
    served = _Lake()
    monkeypatch.setattr(mod, "load_sp500_member_securities_asof", lambda *_a: list(served.members))
    monkeypatch.setattr(mod, "load_security_log_returns", served.security_returns)
    monkeypatch.setattr(mod, "load_price_history_from_lakehouse", served.price_history)
    monkeypatch.setattr(mod, "load_fama_french_factors_from_lakehouse", served.factor_table)
    return served


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


def _assert_parity(analyst: FamaMacbethAnalyst, asset: str, day: pd.Timestamp) -> AgentSignal:
    """Asserts `analyst`'s signal equals a fresh analyst's field for field and returns it."""
    cached = analyst.generate_signal(asset, pd.DataFrame(), day)
    reference = _uncached(analyst, asset, day)
    assert cached.model_dump() == reference.model_dump()
    assert cached == reference
    return cached


@pytest.mark.usefixtures("lake")
@pytest.mark.parametrize("refit_interval_days", [1, 5])
def test_cached_signals_equal_fresh_signals_field_for_field(refit_interval_days: int) -> None:
    """Over two consecutive days, with the broad universe either reloaded or reused on the second,
    every asset's signal equals a fresh analyst's, including the assets whose reference index
    differs from the others' and the non-members read by ticker."""
    analyst = FamaMacbethAnalyst()
    analyst._universe_refit_cache = RefitCache(refit_interval_days=refit_interval_days)

    for day in (_DAY_ONE, _DAY_TWO):
        for asset in _ASSETS:
            signal = _assert_parity(analyst, asset, day)
            # On a day the reused universe ends before, a non-member's own returns reach a date
            # no member has, which leaves too few names: unavailable, for both analysts.
            assert signal.status == "active" or asset in (_OUTSIDER, _FULL)


def _scenario_non_member_on_a_members_panel(lake: _Lake, order: tuple[str, str]) -> None:
    """FULL shares T000's reference index and factor rows but extends the universe by one name;
    in either order, on both days, each gets its own cross-section."""
    analyst = FamaMacbethAnalyst()
    analyst._universe_refit_cache = RefitCache(refit_interval_days=1)
    for day in (_DAY_ONE, _DAY_TWO):
        signals = {asset: _assert_parity(analyst, asset, day) for asset in order}
        assert f"{_N_TICKERS - len(_GAPS)}-name" in signals["T000"].rationale
        assert f"{_N_TICKERS - len(_GAPS) + 1}-name" in signals[_FULL].rationale


def _scenario_universe_changes_within_a_day(lake: _Lake) -> None:
    """A universe refit during the day that adds a member gives new universe objects, and the
    next asset's stage is built on them, not on the stage of the earlier objects."""
    analyst = FamaMacbethAnalyst()
    analyst._universe_refit_cache = RefitCache(refit_interval_days=0)
    first = _assert_parity(analyst, "T000", _DAY_ONE)
    lake.members.append(IndexMember(security_id=_N_TICKERS, ticker=_FULL))
    second = _assert_parity(analyst, "T003", _DAY_ONE)
    assert f"{_N_TICKERS - len(_GAPS)}-name" in first.rationale
    assert f"{_N_TICKERS - len(_GAPS) + 1}-name" in second.rationale


def _scenario_factors_change_within_a_day(lake: _Lake) -> None:
    """Factor values that change on the same dates between two calls of one day give the second
    asset a stage of its own."""
    analyst = FamaMacbethAnalyst()
    first = _assert_parity(analyst, "T000", _DAY_ONE)
    lake.factors = lake.factors.assign(
        **{column: lake.factors[column] * 1.5 for column in _FACTORS}
    )
    _assert_parity(analyst, "T003", _DAY_ONE)
    again = _assert_parity(analyst, "T000", _DAY_ONE)
    assert again != first


def _scenario_same_factor_rows_on_other_dates(lake: _Lake) -> None:
    """T039 has no return on the decision day, so its reference dates end a session before
    T000's; the factor table of the second call holds, on those dates, the very rows T000's
    reference held. T039 still gets a stage of its own."""
    lake.returns["T039"] = lake.returns["T039"].loc[:_DAY_ONE]
    analyst = FamaMacbethAnalyst()
    _assert_parity(analyst, "T000", _DAY_TWO)
    lake.factors = lake.factors.shift(-1)
    signal = _assert_parity(analyst, "T039", _DAY_TWO)
    assert signal.status == "active"


@pytest.mark.parametrize(
    "order", [("T000", _FULL), (_FULL, "T000")], ids=["member_first", "full_first"]
)
def test_non_member_on_a_members_panel_matches_a_fresh_analyst(
    lake: _Lake, order: tuple[str, str]
) -> None:
    _scenario_non_member_on_a_members_panel(lake, order)


def test_universe_change_within_a_day_misses_the_cache(lake: _Lake) -> None:
    _scenario_universe_changes_within_a_day(lake)


def test_factor_change_within_a_day_misses_the_cache(lake: _Lake) -> None:
    _scenario_factors_change_within_a_day(lake)


def test_same_factor_rows_on_other_dates_miss_the_cache(lake: _Lake) -> None:
    _scenario_same_factor_rows_on_other_dates(lake)


# Positions in `_cross_section_key`'s tuple: universe id, returns id, index dtype, index, factor
# columns, reference factor columns, reference factor bytes.
_KEY_LENGTH = 7
_WEAKENED_KEYS: dict[str, tuple[frozenset[int], Callable[[_Lake], None]]] = {
    "without_universe_ids": (frozenset({0, 1}), _scenario_universe_changes_within_a_day),
    "without_factor_bytes": (frozenset({6}), _scenario_factors_change_within_a_day),
    "without_index_and_dtype": (frozenset({2, 3}), _scenario_same_factor_rows_on_other_dates),
}


@pytest.mark.parametrize("weakening", list(_WEAKENED_KEYS))
def test_each_weakened_key_fails_its_scenario(
    lake: _Lake, monkeypatch: pytest.MonkeyPatch, weakening: str
) -> None:
    """With the key stripped of the components a scenario relies on, a wrong stage is reused and
    the scenario fails, so the tests above pin every one of those components."""
    dropped, scenario = _WEAKENED_KEYS[weakening]
    real_key = mod._cross_section_key

    def weakened(*args: Any) -> Hashable:
        key = real_key(*args)
        if not isinstance(key, tuple) or len(key) != _KEY_LENGTH:
            raise TypeError(f"the key layout changed: {key!r}")
        return tuple(part for position, part in enumerate(key) if position not in dropped)

    monkeypatch.setattr(mod, "_cross_section_key", weakened)
    with pytest.raises((AssertionError, KeyError)):
        scenario(lake)


@pytest.mark.usefixtures("lake")
def test_only_the_cached_universe_objects_store_a_stage() -> None:
    """A non-member's call, which extends the universe, computes its stage without storing it."""
    analyst = FamaMacbethAnalyst()
    for asset in (_OUTSIDER, _FULL, _FULL):
        analyst.generate_signal(asset, pd.DataFrame(), _DAY_ONE)
    assert len(analyst._cross_section_cache) == 0

    analyst.generate_signal("T000", pd.DataFrame(), _DAY_ONE)
    assert len(analyst._cross_section_cache) == 1


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
def test_checkpoint_carries_an_empty_cache_and_a_restored_analyst_gives_the_same_signals() -> None:
    """`carried_state` holds the cross-section cache as an empty cache with no stages, beside the
    broad-universe cache it snapshots in full: the restored analyst starts with no stages and gives
    the signals the original goes on to give."""
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
