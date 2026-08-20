"""Unit tests for agentic_trading.agents.roster.

Covers the invariant the module's own docstring claims but never enforced:
every default analyst is owned by exactly one of the two managers in
build_default_managers()'s "Dual stack" split — no analyst silently falls
through uncovered, and none is double-counted under both managers (which
would double-weight its signal in Commander.decide()).
"""

from agentic_trading.agents.managers import CVaRManager, RiskManager
from agentic_trading.agents.roster import (
    CVAR_MANAGER_ANALYST_IDS,
    RISK_MANAGER_ANALYST_IDS,
    build_default_analysts,
    build_default_managers,
)


def test_analyst_id_sets_are_disjoint() -> None:
    """No analyst id may be claimed by both managers — see roster.py's
    module docstring on why (double-counting a signal in Commander.decide())."""
    assert RISK_MANAGER_ANALYST_IDS.isdisjoint(CVAR_MANAGER_ANALYST_IDS)


def test_analyst_id_sets_cover_every_default_analyst() -> None:
    """Every analyst build_default_analysts() produces must be claimed by one
    of the two sets — an analyst that isn't ends up silently excluded from
    Dual stack mode (collect_signals() just returns fewer signals; nothing
    raises), even though it's still listed as an available analyst elsewhere
    (e.g. the GUI's Analysts tab)."""
    default_ids = {analyst.agent_id for analyst in build_default_analysts()}
    claimed_ids = RISK_MANAGER_ANALYST_IDS | CVAR_MANAGER_ANALYST_IDS

    assert default_ids == claimed_ids


def test_build_default_managers_splits_analysts_by_the_id_sets() -> None:
    """The actual manager construction must match what the id sets claim —
    not just the id sets being internally consistent with each other."""
    analysts = build_default_analysts()

    managers = build_default_managers(analysts)

    assert len(managers) == 2
    risk_manager = next(m for m in managers if isinstance(m, RiskManager))
    cvar_manager = next(m for m in managers if isinstance(m, CVaRManager))
    risk_owned_ids = {a.agent_id for a in risk_manager.managed_analysts}
    cvar_owned_ids = {a.agent_id for a in cvar_manager.managed_analysts}

    assert risk_owned_ids == RISK_MANAGER_ANALYST_IDS
    assert cvar_owned_ids == CVAR_MANAGER_ANALYST_IDS
    assert risk_owned_ids.isdisjoint(cvar_owned_ids)


def test_build_default_managers_owns_no_analyst_twice_across_managers() -> None:
    """Every managed analyst instance must appear under exactly one manager."""
    managers = build_default_managers()

    all_owned_ids = [
        analyst.agent_id for manager in managers for analyst in manager.managed_analysts
    ]

    assert len(all_owned_ids) == len(set(all_owned_ids))


def test_build_default_managers_defaults_to_build_default_analysts() -> None:
    """Calling with no argument must use the same roster as an explicit call."""
    implicit = build_default_managers()
    explicit = build_default_managers(build_default_analysts())

    implicit_ids = {a.agent_id for m in implicit for a in m.managed_analysts}
    explicit_ids = {a.agent_id for m in explicit for a in m.managed_analysts}
    assert implicit_ids == explicit_ids


def test_build_default_analysts_returns_five_uniquely_identified_analysts() -> None:
    analysts = build_default_analysts()

    assert len(analysts) == 5
    assert len({a.agent_id for a in analysts}) == 5
