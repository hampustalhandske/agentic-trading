# test_carried_state.py

"""`carried_state` / `restore_carried_state` on every default-roster analyst and manager: the
snapshot is opaque bytes holding every pydantic private attribute as it stood when it was taken,
and restoring it onto a freshly built instance with the same `agent_id` gives that instance the
same private attributes, so a restored `RefitCache` reuses the snapshotted fit until its refit
interval elapses."""

import copy
from typing import Any
from unittest.mock import Mock

import pandas as pd
import pytest

from agentic_trading.agents._market_math import RefitCache
from agentic_trading.agents.base import AnalystAgent, ManagerAgent
from agentic_trading.agents.roster import build_default_analysts, build_default_managers

_DAY = pd.Timestamp("2024-03-01")

_Agent = AnalystAgent | ManagerAgent


def _roster() -> dict[str, _Agent]:
    """A freshly built default roster: every analyst and every manager, keyed by agent_id."""
    analysts = build_default_analysts()
    agents: list[_Agent] = [*analysts, *build_default_managers(analysts)]
    return {agent.agent_id: agent for agent in agents}


_AGENT_IDS = list(_roster())
_REFIT_CACHES = [
    (agent_id, name)
    for agent_id, agent in _roster().items()
    for name, value in (agent.__pydantic_private__ or {}).items()
    if isinstance(value, RefitCache)
]


def _contents(agent: _Agent) -> dict[str, Any] | None:
    """The agent's private attributes, each `RefitCache` as its interval and cached entries."""
    private = agent.__pydantic_private__
    if private is None:
        return None
    return {
        name: (value.refit_interval_days, dict(value._cache))
        if isinstance(value, RefitCache)
        else value
        for name, value in private.items()
    }


def _plant_sentinels(agent: _Agent) -> list[Any]:
    """Sets every private attribute to a distinct sentinel and returns the sentinels."""
    sentinels: list[Any] = []
    for index, name in enumerate(agent.__pydantic_private__ or {}):
        value: Any
        if isinstance(getattr(agent, name), RefitCache):
            value = RefitCache(refit_interval_days=11 + index)
            value.get_or_refit(
                f"{agent.agent_id}:{name}",
                _DAY + pd.Timedelta(days=index),
                Mock(return_value={"fit": [index, name]}),
            )
        else:
            value = {"agent": agent.agent_id, "attribute": name, "values": [index, str(index)]}
        setattr(agent, name, value)
        sentinels.append(value)
    return sentinels


@pytest.mark.parametrize("agent_id", _AGENT_IDS)
def test_carried_state_round_trips_every_private_attribute(agent_id: str) -> None:
    """A fresh instance restored from the snapshot holds the private attributes the snapshotted
    instance held when the snapshot was taken."""
    original = _roster()[agent_id]
    sentinels = _plant_sentinels(original)
    expected = copy.deepcopy(_contents(original))

    state = original.carried_state()
    for sentinel in sentinels:
        if isinstance(sentinel, RefitCache):
            sentinel.get_or_refit("after snapshot", _DAY, Mock(return_value=0))
        else:
            sentinel["values"].append("after snapshot")

    fresh = _roster()[agent_id]
    assert type(fresh) is type(original)
    if sentinels:
        assert _contents(fresh) != expected
    fresh.restore_carried_state(state)

    assert isinstance(state, bytes)
    assert _contents(fresh) == expected
    assert set(fresh.__pydantic_private__ or {}) == set(original.__pydantic_private__ or {})


@pytest.mark.parametrize(
    ("agent_id", "attribute"), _REFIT_CACHES, ids=[f"{a}.{n}" for a, n in _REFIT_CACHES]
)
def test_restored_refit_cache_reuses_the_snapshotted_fit(agent_id: str, attribute: str) -> None:
    """A fit cached on day D is reused without refitting inside the refit interval after a restore,
    and refit once the interval has elapsed; an unrestored instance refits at once."""
    original = _roster()[agent_id]
    fit = {"fit": agent_id}
    getattr(original, attribute).get_or_refit("AAPL", _DAY, Mock(return_value=fit))
    interval = getattr(original, attribute).refit_interval_days
    inside = _DAY + pd.Timedelta(days=interval - 1)

    fresh = _roster()[agent_id]
    fresh.restore_carried_state(original.carried_state())
    cache = getattr(fresh, attribute)
    refit = Mock(return_value={"fit": "refit"})

    assert isinstance(cache, RefitCache)
    assert cache.refit_interval_days == interval
    assert cache.get_or_refit("AAPL", inside, refit) == fit
    refit.assert_not_called()
    assert cache.get_or_refit("AAPL", _DAY + pd.Timedelta(days=interval), refit) == {"fit": "refit"}
    refit.assert_called_once()

    unrestored = Mock(return_value={"fit": "unrestored"})
    assert getattr(_roster()[agent_id], attribute).get_or_refit("AAPL", inside, unrestored) == {
        "fit": "unrestored"
    }
    unrestored.assert_called_once()
