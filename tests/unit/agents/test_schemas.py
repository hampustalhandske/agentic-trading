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
    MAX_LEVERAGE,
    AgentSignal,
    AgentWeight,
    BacktestResult,
    ManagerSignal,
    OptionContract,
    PerformanceRecord,
    PortfolioDecision,
    TradeRecord,
)

_DATE = date(2024, 1, 5)


def test_agent_signal_keeps_the_status_the_analyst_set() -> None:
    signal = AgentSignal(
        agent_id="macro_var_spillover_analyst",
        asset="MSFT",
        as_of_date=_DATE,
        direction="flat",
        confidence=0.0,
        rationale="VAR spillover fit failed on the available finite observations",
        status="error",
    )

    assert signal.status == "error"


def test_agent_signal_status_defaults_to_active_regardless_of_rationale_wording() -> None:
    signal = AgentSignal(
        agent_id="pairs_cointegration_analyst",
        asset="MSFT",
        as_of_date=_DATE,
        direction="flat",
        confidence=0.0,
        rationale="insufficient history — fit failed",
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
        raw_confidence=confidence,
        scale_applied=scale_applied,
        rationale="test rationale",
    )


def test_target_weight_within_bounds_is_accepted() -> None:
    """The boundary values -MAX_LEVERAGE and MAX_LEVERAGE are valid target
    weights, not exclusive bounds."""
    for boundary in (-MAX_LEVERAGE, MAX_LEVERAGE):
        decision = PortfolioDecision(
            asset="AAPL", as_of_date=_DATE, target_weight=boundary, contributing_signals=[]
        )
        assert decision.target_weight == boundary


def test_target_weight_outside_bounds_is_rejected() -> None:
    """A target_weight beyond +/-MAX_LEVERAGE is not a valid portfolio allocation."""
    with pytest.raises(ValidationError):
        PortfolioDecision(
            asset="AAPL",
            as_of_date=_DATE,
            target_weight=MAX_LEVERAGE + 0.5,
            contributing_signals=[],
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
            raw_attributable_return=0.01,
        )


def test_performance_record_allows_negative_attributable_return() -> None:
    """A losing day is a valid, unbounded-below attributable return, not an error condition."""
    record = PerformanceRecord(
        source_id="risk_manager",
        source_type="manager",
        asset="AAPL",
        as_of_date=_DATE,
        attributable_return=-0.05,
        raw_attributable_return=-0.05,
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


# ---------------------------------------------------------------------------
# OptionContract and option-aware decisions/trades
# ---------------------------------------------------------------------------


def _contract() -> OptionContract:
    return OptionContract(
        underlying="AAPL", expiry=date(2024, 3, 15), strike=185.0, option_type="call"
    )


def test_option_contract_id_round_trips_through_parse() -> None:
    contract = _contract()

    assert contract.contract_id == "AAPL 2024-03-15 C 185.00"
    assert OptionContract.parse(contract.contract_id) == contract
    put = OptionContract(
        underlying="SPY", expiry=date(2025, 1, 17), strike=450.5, option_type="put"
    )
    assert put.contract_id == "SPY 2025-01-17 P 450.50"
    assert OptionContract.parse(put.contract_id) == put


def test_option_contract_is_contract_id_distinguishes_options_from_tickers() -> None:
    assert OptionContract.is_contract_id("AAPL 2024-03-15 C 185.00")
    assert OptionContract.is_contract_id("SPY 2025-01-17 P 450.5")
    assert not OptionContract.is_contract_id("AAPL")
    assert not OptionContract.is_contract_id("AAPL 2024-03-15 X 185.00")
    assert not OptionContract.is_contract_id("AAPL 2024-3-15 C 185.00")


def test_option_contract_parse_rejects_malformed_ids_and_bad_strikes() -> None:
    with pytest.raises(ValueError):
        OptionContract.parse("AAPL")
    with pytest.raises(ValidationError):
        OptionContract(underlying="AAPL", expiry=_DATE, strike=0.0, option_type="call")


def test_option_contract_is_frozen_and_hashable() -> None:
    contract = _contract()
    with pytest.raises(ValidationError):
        contract.strike = 190.0  # type: ignore[misc]
    assert {contract: 1}[OptionContract.parse(contract.contract_id)] == 1


def test_portfolio_decision_asset_must_match_its_instrument_contract_id() -> None:
    contract = _contract()
    decision = PortfolioDecision(
        asset=contract.contract_id,
        as_of_date=_DATE,
        target_weight=0.02,
        contributing_signals=[],
        instrument=contract,
    )
    assert decision.instrument == contract

    with pytest.raises(ValidationError):
        PortfolioDecision(
            asset="AAPL",
            as_of_date=_DATE,
            target_weight=0.02,
            contributing_signals=[],
            instrument=contract,
        )


def test_trade_record_asset_must_match_its_instrument_and_allows_expire() -> None:
    contract = _contract()
    record = TradeRecord(
        as_of_date=_DATE,
        asset=contract.contract_id,
        action="expire",
        target_weight=0.0,
        previous_weight=0.0,
        target_shares=0.0,
        delta_shares=-2.0,
        price=0.0,
        notional=0.0,
        instrument=contract,
        contracts=-2.0,
        premium=0.0,
    )
    assert record.action == "expire"
    assert record.contracts == -2.0

    with pytest.raises(ValidationError):
        TradeRecord(
            as_of_date=_DATE,
            asset="AAPL",
            action="buy",
            target_weight=0.0,
            previous_weight=0.0,
            target_shares=0.0,
            delta_shares=0.0,
            price=1.0,
            notional=0.0,
            instrument=contract,
        )


def test_trade_record_defaults_to_an_equity_with_no_option_fields() -> None:
    record = TradeRecord(
        as_of_date=_DATE,
        asset="AAPL",
        action="hold",
        target_weight=0.1,
        previous_weight=0.1,
        target_shares=10.0,
        delta_shares=0.0,
        price=100.0,
        notional=0.0,
    )
    assert record.instrument is None
    assert record.contracts == 0.0
    assert record.premium == 0.0
