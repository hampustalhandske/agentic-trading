# test_decisions.py

"""AT-110 precompute spec item 3: `decision_weights` turns one day's signals into a gross-1.0
book over the active instruments. The look-ahead audit classifies every roster source by whether
it can publish after 16:00 ET and refuses a `source_as_of` the decision date cannot know."""

from datetime import date, timedelta
from typing import Any

import pytest

from agentic_trading.agents.analysts import CIRRateRegimeAnalyst, GBMMomentumAnalyst
from agentic_trading.agents.base import DataRequirement
from agentic_trading.agents.roster import build_default_analysts
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.stage_a.decisions import (
    SOURCES,
    LookAheadError,
    check_source_as_of,
    decision_weights,
    row_source,
)

_DAY = date(2015, 6, 3)
_AFTER_CLOSE = {
    "earnings",
    "fred",
    "sec_edgar_facts",
    "sec_insider_transactions",
    "finra_short_interest",
}
_BEFORE_CLOSE = {
    "cftc_cot",
    "ken_french_factors",
    "sp500_membership",
    "options_chain",
    "dividends",
    "sec_sic",
}


def _signal(asset: str, direction: str, confidence: float, status: str = "active") -> AgentSignal:
    fields: dict[str, Any] = {"direction": direction, "confidence": confidence, "status": status}
    return AgentSignal(agent_id="a", asset=asset, as_of_date=_DAY, rationale="", **fields)


def _requires(*sources: str) -> CIRRateRegimeAnalyst:
    """A CIR analyst whose data requirements read `sources`, one requirement each."""
    requirements = tuple(
        DataRequirement(source=source, identifier=f"id{n}", asset_scoped=False, description="")
        for n, source in enumerate(sources)
    )
    return CIRRateRegimeAnalyst(data_requirements=requirements)


def _has_active_view(signals: list[AgentSignal]) -> bool:
    """Whether any active signal has a non-zero signed confidence."""
    signed = [s.confidence for s in signals if s.status == "active" and s.direction != "flat"]
    return any(signed)


def test_decision_weights() -> None:
    """Signed confidence over the active gross; inactive and zero-sum days weight 0."""
    mixed = [
        _signal("A", "long", 0.6),
        _signal("B", "short", 0.2),
        _signal("C", "flat", 0.9),
        _signal("D", "long", 0.8, status="unavailable"),
        _signal("E", "short", 0.5, status="error"),
    ]
    lone_short = [_signal("A", "short", 0.3), _signal("B", "long", 0.9, status="error")]
    zero_sum = [
        [_signal("A", "flat", 0.7), _signal("B", "long", 0.0)],
        [
            _signal("A", "long", 0.7, status="unavailable"),
            _signal("B", "short", 0.4, status="error"),
        ],
    ]

    assert decision_weights(mixed) == pytest.approx(
        {"A": 0.75, "B": -0.25, "C": 0.0, "D": 0.0, "E": 0.0}
    )
    assert decision_weights(lone_short) == pytest.approx({"A": -1.0, "B": 0.0})
    assert decision_weights([]) == {}
    for day in zero_sum:
        assert decision_weights(day) == {"A": 0.0, "B": 0.0}
    for day in [mixed, lone_short, *zero_sum]:
        weights = decision_weights(day)
        gross = sum(abs(weight) for weight in weights.values())
        assert set(weights) == {signal.asset for signal in day}
        assert all(isinstance(weight, float) for weight in weights.values())
        assert gross == pytest.approx(1.0 if _has_active_view(day) else 0.0)


def test_sources_classify_every_roster_source() -> None:
    """SOURCES marks exactly the five after-close sources and covers every roster source."""
    roster_sources = {
        requirement.source
        for analyst in build_default_analysts()
        for requirement in analyst.data_requirements
    }

    assert {name for name, after_close in SOURCES.items() if after_close} == _AFTER_CLOSE
    assert {name for name, after_close in SOURCES.items() if not after_close} == _BEFORE_CLOSE
    assert all(isinstance(after_close, bool) for after_close in SOURCES.values())
    assert roster_sources <= set(SOURCES)


def test_row_source() -> None:
    """None without requirements, the shared source, else the first after-close, else first."""
    assert row_source(GBMMomentumAnalyst()) is None
    assert row_source(_requires()) is None
    assert row_source(CIRRateRegimeAnalyst()) == "fred"
    assert row_source(_requires("fred", "fred")) == "fred"
    assert row_source(_requires("fred", "earnings")) == "earnings"
    assert row_source(_requires("sec_edgar_facts", "cftc_cot", "finra_short_interest")) == (
        "finra_short_interest"
    )
    assert row_source(_requires("sp500_membership", "ken_french_factors")) == "ken_french_factors"
    assert row_source(_requires("fred", "cftc_cot", "fred")) == "fred"


def test_check_source_as_of() -> None:
    """Refuses a half-filled pair, a later date or a same-day after-close one; passes the rest."""
    day_before, day_after = _DAY - timedelta(days=1), _DAY + timedelta(days=1)
    refused = [
        (_DAY, None),
        (None, "fred"),
        (None, "earnings"),
        (day_after, "fred"),
        (day_after, "earnings"),
        (_DAY, "earnings"),
        (_DAY, "fred"),
        (_DAY, "sec_edgar_facts"),
        (_DAY, "sec_insider_transactions"),
        (_DAY, "finra_short_interest"),
    ]
    passed = [
        (None, None),
        (_DAY, "cftc_cot"),
        (day_before, "fred"),
        (day_before, "earnings"),
        (day_before, "finra_short_interest"),
        (date(2010, 1, 4), "sec_edgar_facts"),
    ]

    for source_as_of, source in refused:
        with pytest.raises(LookAheadError):
            check_source_as_of(source_as_of, _DAY, source)
    for source_as_of, source in passed:
        assert check_source_as_of(source_as_of, _DAY, source) is None
