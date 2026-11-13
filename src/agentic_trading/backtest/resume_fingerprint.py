# resume_fingerprint.py

"""What a checkpoint records about the run that wrote it, beyond its dates
and assets, so a resume under other strategy settings or another roster is
refused: the settings fingerprint, the roster fingerprint, and the agents
whose carried state the checkpoint holds."""

import hashlib
from typing import Any

from agentic_trading.agents.base import Agent, AnalystAgent, ManagerAgent
from agentic_trading.stage_a.spec import RUNTIME_SETTINGS_FIELDS, canonical_json, hashed_settings


def _sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def settings_fingerprint() -> str:
    """The sha256 hex digest of the canonical JSON of `settings`, read now,
    minus `RUNTIME_SETTINGS_FIELDS`: changing a strategy constant changes
    it, changing a runtime-only setting does not."""
    return _sha256(hashed_settings(RUNTIME_SETTINGS_FIELDS))


def roster_agents(commander: object) -> list[Agent]:
    """The commander's managers, in order, each followed by its managed
    analysts in order; an analyst two managers share appears under each.
    Empty for a commander without `managers` (a baseline or a scripted
    stub)."""
    managers = getattr(commander, "managers", None)
    if not isinstance(managers, list | tuple):
        return []
    agents: list[Agent] = []
    for manager in managers:
        if not isinstance(manager, ManagerAgent):
            continue
        agents.append(manager)
        agents.extend(
            analyst for analyst in manager.managed_analysts if isinstance(analyst, AnalystAgent)
        )
    return agents


def roster_fingerprint(commander: object) -> str:
    """The sha256 hex digest of the canonical JSON of each of
    `roster_agents(commander)`'s `built_parameters()`, in roster order: its
    `agent_id`, class and constructor parameters as built. Any change of
    agent, order, class or parameter changes it."""
    return _sha256([agent.built_parameters() for agent in roster_agents(commander)])


def agents_by_id(commander: object) -> dict[str, Agent]:
    """`roster_agents(commander)` keyed by `agent_id`, each instance once.
    Raises `ValueError` when two distinct instances share an `agent_id`,
    since carried state is keyed by it."""
    agents: dict[str, Agent] = {}
    for agent in roster_agents(commander):
        known = agents.setdefault(agent.agent_id, agent)
        if known is not agent:
            raise ValueError(
                f"two distinct agents share agent_id {agent.agent_id!r}; "
                "a checkpoint keys carried state by agent_id"
            )
    return agents
