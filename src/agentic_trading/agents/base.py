# base.py

from abc import ABC, abstractmethod
from typing import Literal

import pandas as pd
from pydantic import BaseModel

from agentic_trading.agents.schemas import AgentSignal, ManagerSignal
from agentic_trading.logging_config import get_logger
from agentic_trading.scoring.scoreboard import PerformanceScoreboard
from agentic_trading.scoring.weighting import WeightingStrategy

logger = get_logger(__name__)

# The safe fallback for any agent that fails to produce a real decision —
# whether because it crashed (see collect_signals()/evaluate() below) or
# because it's genuinely unimplemented (e.g. NewsSentimentAnalyst, which has
# no real news feed wired in yet). "flat" + confidence 0.0 means this
# source contributes exactly zero to Commander.decide()'s target_weight —
# neither buying nor selling — which is the trading-safe default: an agent
# with no real view should abstain, not force a stance it can't back.
SAFE_FALLBACK_DIRECTION: Literal["flat"] = "flat"
SAFE_FALLBACK_CONFIDENCE = 0.0


class Agent(BaseModel, ABC):
    """Shared base for every agent in the system — analysts and managers alike.

    Making this a BaseModel (not just an ABC) is what lets ManagerAgent
    hold `managed_analysts: list[AnalystAgent]` as a normal, validated
    Pydantic field, since AnalystAgent instances are themselves BaseModel
    instances. This is the same principle as the schemas: uniform data
    contracts, applied here to the agents themselves.
    """

    agent_id: str

    model_config = {"arbitrary_types_allowed": True}
    # Needed for concrete subclasses that will eventually hold non-Pydantic
    # objects as attributes (e.g. a fitted GARCH model, an sklearn object)
    # — without this, Pydantic rejects any field type it doesn't recognize.


class AnalystAgent(Agent):
    """Every Layer-1 analyst implements this. Internally can be anything —
    GARCH-driven, valuation-driven, eventually news/text-driven — as long
    as it honors this one contract."""

    @abstractmethod
    def generate_signal(
        self,
        asset: str,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> "AgentSignal":
        """Produce one directional view for one asset, using only data
        available as of `as_of_date` — the caller (a Manager) is
        responsible for ensuring `market_data` has already been
        point-in-time sliced (see data/point_in_time.py); this method must
        not assume it can safely see rows beyond `as_of_date`.
        """


class ManagerAgent(Agent):
    """Every Layer-2 manager implements this. A manager owns a specific
    group of analysts and scales their raw signals — it does not decide
    direction itself, only how much weight a proposed direction deserves.

    scale_factor() and weighting_strategy are two distinct, multiplicative
    adjustments applied to the same signal, not redundant with each other:
    scale_factor() is this manager's own domain judgment (e.g. a
    RiskManager dampening confidence for an asset with elevated GARCH
    volatility), while weighting_strategy reflects each analyst's trailing
    track record (see scoring/). The same WeightingStrategy abstraction is
    used one level up by Commander, to weight its managers rather than
    their analysts.
    """

    managed_analysts: list[AnalystAgent]
    weighting_strategy: WeightingStrategy

    def collect_signals(
        self,
        asset: str,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> list["AgentSignal"]:
        """Gathers one AgentSignal from every analyst this manager owns.
        Concrete managers should not need to override this — it's the same
        mechanism regardless of what the manager is scaling for.

        An analyst that raises is not allowed to take down the whole
        backtest over one bad day: its exception is caught, logged as a
        WARNING (never silent), and replaced with a safe flat/0-confidence
        signal for that asset/day — the trading-safe default of abstaining
        rather than forcing a stance from a component that just failed."""
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
                        rationale="generate_signal() raised an exception — defaulting to hold.",
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
        """The multiplier this manager applies to one analyst's signal
        confidence. A RiskManager might return a low factor (e.g. 0.2) for
        a signal on an asset with elevated GARCH volatility, and closer to
        1.0 for a calm asset. Must be >= 0 — a manager can dampen a
        signal toward zero confidence, but does not flip its direction.
        """

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
                )
            )
        return scaled
