# weighting.py

import math
from abc import ABC, abstractmethod
from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentic_trading.scoring.scoreboard import PerformanceScoreboard


class WeightingStrategy(ABC):
    """How to convert trailing track record into per-source portfolio
    influence, generic over what the sources are.

    This same abstraction is used in two places: inside
    ManagerAgent.evaluate() (weighting the analysts it owns) and inside
    Commander.decide() (weighting the managers it owns). `source_ids` is
    deliberately untyped as either analyst_ids or manager_ids — the
    weighting logic itself does not care which.

    The eventual concrete implementation will follow Freund & Schapire
    (1997), "A Decision-Theoretic Generalization of On-Line Learning and
    an Application to Boosting," Journal of Computer and System Sciences,
    55(1), 119-139 — see REFERENCES.md.
    """

    @abstractmethod
    def get_weights(
        self,
        scoreboard: "PerformanceScoreboard",
        source_ids: list[str],
        as_of_date: date,
    ) -> dict[str, float]:
        """Return a mapping of source_id -> weight, derived from each
        source's trailing performance as recorded in `scoreboard` as of
        `as_of_date`. Weights need not sum to 1.0 here — the caller
        (ManagerAgent.evaluate or Commander.decide) is responsible for how
        the result is combined with other multiplicative factors."""
        ...


class Hedge(WeightingStrategy):
    """Exponentiated trailing-return weighting normalized to average 1.0."""

    def __init__(self, eta: float = 1.0, lookback_days: int = 30) -> None:
        self.eta = eta
        self.lookback_days = lookback_days

    def get_weights(
        self,
        scoreboard: "PerformanceScoreboard",
        source_ids: list[str],
        as_of_date: date,
    ) -> dict[str, float]:
        if not source_ids:
            return {}

        cumulative_returns = {
            source_id: scoreboard.cumulative_return(
                source_id, as_of_date, self.lookback_days
            )
            for source_id in source_ids
        }
        raw_weights = {
            source_id: math.exp(self.eta * cumulative_return)
            for source_id, cumulative_return in cumulative_returns.items()
        }
        average_weight = sum(raw_weights.values()) / len(raw_weights)
        return {
            source_id: weight / average_weight for source_id, weight in raw_weights.items()
        }
