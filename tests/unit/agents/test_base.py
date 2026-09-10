"""Unit tests for agentic_trading.agents.base — specifically the exception
safety net in ManagerAgent.collect_signals()/evaluate(): a crashing analyst
or manager must never take down the whole backtest, and must default to a
flat/0-confidence ("hold") signal for that asset/day instead.
"""

import pandas as pd
import pytest

from agentic_trading.agents.base import (
    SAFE_FALLBACK_CONFIDENCE,
    SAFE_FALLBACK_DIRECTION,
    AnalystAgent,
    ManagerAgent,
)
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.scoring.scoreboard import PerformanceScoreboard
from agentic_trading.scoring.track_record import TrackRecord
from agentic_trading.scoring.weighting import Hedge

AS_OF = pd.Timestamp("2024-01-02")
MARKET_DATA = pd.DataFrame({"close": [100.0]}, index=pd.to_datetime(["2024-01-02"]))


class _WorkingAnalyst(AnalystAgent):
    agent_id: str = "working_analyst"

    def generate_signal(
        self, asset: str, market_data: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> AgentSignal:
        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction="long",
            confidence=0.8,
            rationale="working fine",
        )


class _CrashingAnalyst(AnalystAgent):
    agent_id: str = "crashing_analyst"

    def generate_signal(
        self, asset: str, market_data: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> AgentSignal:
        raise RuntimeError("simulated analyst failure")


class _TestManager(ManagerAgent):
    """A concrete ManagerAgent whose scale_factor is controlled per-test via
    a plain (non-Pydantic-field) callable, so tests can make it crash."""

    agent_id: str = "test_manager"
    should_crash: bool = False

    def scale_factor(
        self, signal: AgentSignal, market_data: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> float:
        if self.should_crash:
            raise ValueError("simulated manager scale_factor failure")
        return 1.0


def _scoreboard() -> PerformanceScoreboard:
    return PerformanceScoreboard(TrackRecord())


def test_collect_signals_defaults_a_crashing_analyst_to_a_hold_signal() -> None:
    manager = _TestManager(
        managed_analysts=[_WorkingAnalyst(), _CrashingAnalyst()],
        weighting_strategy=Hedge(),
    )

    signals = manager.collect_signals("AAPL", MARKET_DATA, AS_OF)

    assert len(signals) == 2
    crashed_signal = next(s for s in signals if s.agent_id == "crashing_analyst")
    assert crashed_signal.direction == SAFE_FALLBACK_DIRECTION
    assert crashed_signal.confidence == SAFE_FALLBACK_CONFIDENCE
    assert crashed_signal.status == "error"
    assert crashed_signal.asset == "AAPL"
    assert crashed_signal.as_of_date == AS_OF.date()


def test_collect_signals_does_not_affect_other_analysts_when_one_crashes() -> None:
    manager = _TestManager(
        managed_analysts=[_WorkingAnalyst(), _CrashingAnalyst()],
        weighting_strategy=Hedge(),
    )

    signals = manager.collect_signals("AAPL", MARKET_DATA, AS_OF)

    working_signal = next(s for s in signals if s.agent_id == "working_analyst")
    assert working_signal.direction == "long"
    assert working_signal.confidence == pytest.approx(0.8)


def test_collect_signals_never_raises_even_if_every_analyst_crashes() -> None:
    manager = _TestManager(
        managed_analysts=[_CrashingAnalyst()],
        weighting_strategy=Hedge(),
    )

    signals = manager.collect_signals("AAPL", MARKET_DATA, AS_OF)

    assert len(signals) == 1
    assert signals[0].direction == SAFE_FALLBACK_DIRECTION


def test_evaluate_defaults_a_crashing_scale_factor_to_zero_contribution() -> None:
    manager = _TestManager(
        managed_analysts=[_WorkingAnalyst()],
        weighting_strategy=Hedge(),
        should_crash=True,
    )
    signal = AgentSignal(
        agent_id="working_analyst",
        asset="AAPL",
        as_of_date=AS_OF.date(),
        direction="long",
        confidence=0.8,
        rationale="test",
    )

    scaled = manager.evaluate([signal], MARKET_DATA, AS_OF, _scoreboard())

    assert len(scaled) == 1
    assert scaled[0].scale_applied == pytest.approx(0.0)
    assert scaled[0].confidence == pytest.approx(0.0)
    assert scaled[0].status == "error"
    # direction is preserved (a manager dampens, never flips) even though
    # this manager's own scale_factor crashed and contributed nothing.
    assert scaled[0].direction == "long"


def test_evaluate_does_not_raise_when_scale_factor_crashes() -> None:
    manager = _TestManager(
        managed_analysts=[_WorkingAnalyst()],
        weighting_strategy=Hedge(),
        should_crash=True,
    )
    signal = AgentSignal(
        agent_id="working_analyst",
        asset="AAPL",
        as_of_date=AS_OF.date(),
        direction="short",
        confidence=0.5,
        rationale="test",
    )

    # Must not raise.
    scaled = manager.evaluate([signal], MARKET_DATA, AS_OF, _scoreboard())

    assert scaled[0].asset == "AAPL"


def test_safe_fallback_direction_and_confidence_contribute_nothing_to_target_weight() -> None:
    """The whole point of the "flat"/0.0 fallback: direction_sign("flat")=0
    already zeroes out any contribution regardless of confidence, and
    confidence=0.0 makes that doubly explicit rather than implying a
    (unearned) 50/50 view — see base.py's SAFE_FALLBACK_* docstring."""
    direction_sign = {"long": 1.0, "short": -1.0, "flat": 0.0}[SAFE_FALLBACK_DIRECTION]
    assert direction_sign * SAFE_FALLBACK_CONFIDENCE == 0.0
