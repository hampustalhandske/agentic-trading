"""
# base.py
-----------
Contains the abstract base classes for all agents in the system, both analysts and managers.
"""

from abc import ABC, abstractmethod
from typing import Literal

import pandas as pd
from pydantic import BaseModel

from agentic_trading.agents.schemas import AgentSignal, ManagerSignal
from agentic_trading.logging_config import get_logger
from agentic_trading.scoring.scoreboard import PerformanceScoreboard
from agentic_trading.scoring.weighting import WeightingStrategy

logger = get_logger(__name__)

# The safe fallback for any agent that fails to produce a real decision
SAFE_FALLBACK_DIRECTION: Literal["flat"] = "flat"
SAFE_FALLBACK_CONFIDENCE = 0.0


class Agent(BaseModel, ABC):
    """Shared base for every agent in the system — analysts and managers alike"""

    agent_id: str

    model_config = {"arbitrary_types_allowed": True}
    # Needed for concrete subclasses that will eventually hold non-Pydantic


class AnalystAgent(Agent):
    """Layer-1 analyst agent scaffold, every concrete analyst implements this"""

    @abstractmethod
    def generate_signal(
        self,
        asset: str,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> "AgentSignal":
        """Produce one directional view for one asset, using data available as of `as_of_date`"""


class ManagerAgent(Agent):
    """Layer-2 manager agent scaffold, every concrete manager implements this"""

    # The list of analysts this manager owns and scales for
    managed_analysts: list[AnalystAgent]
    # The weighting strategy this manager uses to scale each analyst's signal confidence
    weighting_strategy: WeightingStrategy

    def collect_signals(
        self,
        asset: str,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> list["AgentSignal"]:
        """Gathers one AgentSignal from every analyst this manager owns"""
        signals = []
        for analyst in self.managed_analysts:
            try:
                signals.append(analyst.generate_signal(asset, market_data, as_of_date))
            except Exception:
                logger.warning(
                    "Analyst %s failed to generate a signal for %s on %s — defaulting to "
                    "flat/0-confidence (hold) for this asset/day.",
                    analyst.agent_id,
                    asset,
                    as_of_date.date(),
                    exc_info=True,
                )
                signals.append(
                    AgentSignal(
                        agent_id=analyst.agent_id,
                        asset=asset,
                        as_of_date=as_of_date.date(),
                        direction=SAFE_FALLBACK_DIRECTION,
                        confidence=SAFE_FALLBACK_CONFIDENCE,
                        rationale="generate_signal() raised an exception — signal unavailable.",
                        status="error",
                    )
                )
        return signals

    @abstractmethod
    def scale_factor(
        self,
        signal: "AgentSignal",
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> float:
        """The multiplier this manager applies to one analyst's signal confidence"""

    def evaluate(
        self,
        signals: list["AgentSignal"],
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
        scoreboard: "PerformanceScoreboard",
    ) -> list["ManagerSignal"]:
        """Scales a batch of already-collected analyst signals (typically
        produced by a prior call to collect_signals) into ManagerSignals.

        Applies two distinct, multiplicative adjustments to each signal's
        confidence: scale_factor() (this manager's own domain judgment,
        implemented per concrete subclass) and this manager's
        weighting_strategy, evaluated against `scoreboard` to reflect each
        source analyst's trailing track record. Concrete managers
        implement scale_factor, not evaluate — this method is the same
        mechanical wiring for every manager type.
        """
        # WeightingStrategy/PerformanceScoreboard/TrackRecord are all keyed by
        # plain `date` (see scoring/), while `as_of_date` here is a
        # pd.Timestamp used for market-data slicing — normalize at this
        # boundary rather than leaking a Timestamp into date-typed comparisons.
        weights = self.weighting_strategy.get_weights(
            scoreboard, [signal.agent_id for signal in signals], as_of_date.date()
        )
        scaled = []
        for signal in signals:
            status = signal.status
            try:
                factor = self.scale_factor(signal, market_data, as_of_date)
            except Exception:
                logger.warning(
                    "Manager %s's scale_factor() failed for %s's signal on %s (%s) — "
                    "defaulting to a 0 factor, excluding this signal's contribution "
                    "(hold) rather than crashing the run.",
                    self.agent_id,
                    signal.agent_id,
                    signal.asset,
                    as_of_date.date(),
                    exc_info=True,
                )
                factor = 0.0
                status = "error"
            combined = factor * weights[signal.agent_id]
            scaled.append(
                ManagerSignal(
                    manager_id=self.agent_id,
                    source_agent_id=signal.agent_id,
                    asset=signal.asset,
                    as_of_date=signal.as_of_date,
                    direction=signal.direction,
                    confidence=min(signal.confidence * combined, 1.0),
                    scale_applied=combined,
                    rationale=f"{signal.rationale} [scaled x{combined:.2f} by {self.agent_id}]",
                    status=status,
                )
            )
        return scaled
