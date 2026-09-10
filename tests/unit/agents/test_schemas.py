"""Unit tests for agentic_trading.agents.schemas.

AgentSignal and ManagerSignal are exercised only as building blocks for
PortfolioDecision below; this file focuses on PortfolioDecision (including
its still-unimplemented agent_weights property), PerformanceRecord,
AgentWeight, and BacktestResult.

PortfolioDecision.agent_weights's docstring says it derives
"{source_agent_id: weight applied}" from contributing_signals. ManagerSignal
has two candidate fields for "weight applied": `confidence` (the final,
post-scale signal strength) and `scale_applied` (the multiplicative factor
the manager applied on top of the raw analyst confidence). The field named
`scale_applied` is the more literal match for "weight applied", so that is
what this file tests against -- flagged here as a genuine reading of an
otherwise ambiguous docstring, not a certainty.
"""

from datetime import date

import pytest
from pydantic import ValidationError

from agentic_trading.agents.schemas import (
    AgentSignal,
    AgentWeight,
    BacktestResult,
    ManagerSignal,
    PerformanceRecord,
    PortfolioDecision,
)

_DATE = date(2024, 1, 5)


def test_agent_signal_classifies_unusable_data_without_calling_it_a_real_flat_signal() -> None:
    signal = AgentSignal(
        agent_id="macro_var_spillover_analyst",
        asset="MSFT",
        as_of_date=_DATE,
        direction="flat",
        confidence=0.0,
        rationale="VAR spillover fit failed on the available finite observations",
    )

    assert signal.status == "error"


def test_agent_signal_keeps_a_legitimate_flat_signal_active() -> None:
    signal = AgentSignal(
        agent_id="pairs_cointegration_analyst",
        asset="MSFT",
        as_of_date=_DATE,
        direction="flat",
        confidence=0.0,
        rationale="no cointegration detected (Engle-Granger p=0.42)",
    )

    assert signal.status == "active"


def _manager_signal(
    source_agent_id: str, scale_applied: float, confidence: float = 0.5
) -> ManagerSignal:
    return ManagerSignal(
        manager_id="risk_manager",
        source_agent_id=source_agent_id,
        asset="AAPL",
        as_of_date=_DATE,
        direction="long",
        confidence=confidence,
        scale_applied=scale_applied,
        rationale="test rationale",
    )


def test_target_weight_within_bounds_is_accepted() -> None:
    """The boundary values -1.0 and 1.0 are valid target weights, not exclusive bounds."""
    for boundary in (-1.0, 1.0):
        decision = PortfolioDecision(
            asset="AAPL", as_of_date=_DATE, target_weight=boundary, contributing_signals=[]
        )
        assert decision.target_weight == boundary


def test_target_weight_outside_bounds_is_rejected() -> None:
    """A target_weight beyond +/-1.0 is not a valid portfolio allocation."""
    with pytest.raises(ValidationError):
        PortfolioDecision(
            asset="AAPL", as_of_date=_DATE, target_weight=1.5, contributing_signals=[]
        )


def test_agent_weights_derives_scale_applied_per_source_agent() -> None:
    """agent_weights maps each contributing signal's source_agent_id to its scale_applied."""
    signals = [
        _manager_signal("risk_minimizing_analyst", scale_applied=0.8),
        _manager_signal("risk_maximizing_analyst", scale_applied=1.2),
    ]
    decision = PortfolioDecision(
        asset="AAPL", as_of_date=_DATE, target_weight=0.1, contributing_signals=signals
    )

    assert decision.agent_weights == {
        "risk_minimizing_analyst": 0.8,
        "risk_maximizing_analyst": 1.2,
    }


def test_agent_weights_is_empty_for_a_decision_with_no_contributing_signals() -> None:
    """A decision traceable to zero signals yields an empty attribution mapping, not an error."""
    decision = PortfolioDecision(
        asset="AAPL", as_of_date=_DATE, target_weight=0.0, contributing_signals=[]
    )

    assert decision.agent_weights == {}


def test_performance_record_rejects_unknown_source_type() -> None:
    """source_type is constrained to the two known kinds of scored sources."""
    with pytest.raises(ValidationError):
        PerformanceRecord(
            source_id="risk_manager",
            source_type="portfolio",  # type: ignore[arg-type]
            asset="AAPL",
            as_of_date=_DATE,
            attributable_return=0.01,
        )


def test_performance_record_allows_negative_attributable_return() -> None:
    """A losing day is a valid, unbounded-below attributable return, not an error condition."""
    record = PerformanceRecord(
        source_id="risk_manager",
        source_type="manager",
        asset="AAPL",
        as_of_date=_DATE,
        attributable_return=-0.05,
    )

    assert record.attributable_return == -0.05


def test_agent_weight_rejects_weight_outside_zero_one() -> None:
    """A WeightingStrategy weight is a normalized influence figure, bounded to [0, 1]."""
    with pytest.raises(ValidationError):
        AgentWeight(agent_id="risk_minimizing_analyst", as_of_date=_DATE, weight=1.5)


def test_agent_weight_accepts_zero_and_one_boundaries() -> None:
    """Zero influence and full influence are both valid, inclusive boundary weights."""
    for boundary in (0.0, 1.0):
        weight = AgentWeight(agent_id="risk_minimizing_analyst", as_of_date=_DATE, weight=boundary)
        assert weight.weight == boundary


def test_backtest_result_allows_negative_sharpe_and_drawdown() -> None:
    """A losing backtest has a negative Sharpe ratio and a real drawdown"""
    result = BacktestResult(
        start_date=date(2024, 1, 1),
        end_date=date(2024, 12, 31),
        portfolio_value_history=[(date(2024, 1, 1), 100.0), (date(2024, 12, 31), 80.0)],
        final_sharpe_ratio=-0.5,
        final_max_drawdown=-0.2,
        final_sortino_ratio=-0.4,
        final_calmar_ratio=-0.3,
        final_hit_rate=0.4,
        final_profit_factor=0.8,
    )

    assert result.final_sharpe_ratio == -0.5
    assert result.final_max_drawdown == -0.2
    assert result.portfolio_value_history[0] == (date(2024, 1, 1), 100.0)
