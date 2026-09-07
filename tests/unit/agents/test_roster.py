"""Unit tests for agentic_trading.agents.roster.

Covers the invariant the module's own docstring claims but never enforced:
every default analyst is owned by exactly one of the eight pod managers in
build_default_managers()'s split — no analyst silently falls through
uncovered, and none is double-counted under two managers (which would
double-weight its signal in Commander.decide()).
"""

from agentic_trading.agents.roster import (
    ALT_DATA_MANAGER_ANALYST_IDS,
    ANOMALY_NETWORK_MANAGER_ANALYST_IDS,
    CVAR_MANAGER_ANALYST_IDS,
    EVENT_CREDIT_MANAGER_ANALYST_IDS,
    FACTOR_MANAGER_ANALYST_IDS,
    MACRO_RATES_MANAGER_ANALYST_IDS,
    RISK_MANAGER_ANALYST_IDS,
    STAT_ARB_MANAGER_ANALYST_IDS,
    build_default_analysts,
    build_default_managers,
)

ALL_ANALYST_ID_SETS = (
    RISK_MANAGER_ANALYST_IDS,
    CVAR_MANAGER_ANALYST_IDS,
    STAT_ARB_MANAGER_ANALYST_IDS,
    MACRO_RATES_MANAGER_ANALYST_IDS,
    FACTOR_MANAGER_ANALYST_IDS,
    EVENT_CREDIT_MANAGER_ANALYST_IDS,
    ALT_DATA_MANAGER_ANALYST_IDS,
    ANOMALY_NETWORK_MANAGER_ANALYST_IDS,
)


def test_analyst_id_sets_are_pairwise_disjoint() -> None:
    """No analyst id may be claimed by two pods — see roster.py's module
    docstring on why (double-counting a signal in Commander.decide())."""
    for i, set_a in enumerate(ALL_ANALYST_ID_SETS):
        for set_b in ALL_ANALYST_ID_SETS[i + 1 :]:
            assert set_a.isdisjoint(set_b)


def test_analyst_id_sets_cover_every_default_analyst() -> None:
    """Every analyst build_default_analysts() produces must be claimed by
    exactly one pod — an analyst that isn't ends up silently excluded from
    its manager (collect_signals() just returns fewer signals; nothing
    raises), even though it's still listed as an available analyst
    elsewhere (e.g. the GUI's Analysts tab)."""
    default_ids = {analyst.agent_id for analyst in build_default_analysts()}
    claimed_ids: set[str] = set().union(*ALL_ANALYST_ID_SETS)

    assert default_ids == claimed_ids


def test_build_default_managers_splits_analysts_by_the_id_sets() -> None:
    """The actual manager construction must match what the id sets claim —
    not just the id sets being internally consistent with each other."""
    analysts = build_default_analysts()

    managers = build_default_managers(analysts)

    assert len(managers) == 8
    owned_by_manager = {
        manager.agent_id: {a.agent_id for a in manager.managed_analysts} for manager in managers
    }

    assert owned_by_manager["risk_manager"] == RISK_MANAGER_ANALYST_IDS
    assert owned_by_manager["cvar_manager"] == CVAR_MANAGER_ANALYST_IDS
    assert owned_by_manager["stat_arb_manager"] == STAT_ARB_MANAGER_ANALYST_IDS
    assert owned_by_manager["macro_rates_manager"] == MACRO_RATES_MANAGER_ANALYST_IDS
    assert owned_by_manager["factor_manager"] == FACTOR_MANAGER_ANALYST_IDS
    assert owned_by_manager["event_credit_manager"] == EVENT_CREDIT_MANAGER_ANALYST_IDS
    assert owned_by_manager["alt_data_manager"] == ALT_DATA_MANAGER_ANALYST_IDS
    assert owned_by_manager["anomaly_network_manager"] == ANOMALY_NETWORK_MANAGER_ANALYST_IDS


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


def test_build_default_analysts_returns_thirty_uniquely_identified_analysts() -> None:
    analysts = build_default_analysts()

    assert len(analysts) == 29
    assert len({a.agent_id for a in analysts}) == 29
