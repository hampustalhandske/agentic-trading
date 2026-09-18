# scoreboard.py

import math
import statistics
from datetime import date

import numpy as np
from quantcore.performance.metrics import calmar_ratio as quantcore_calmar_ratio
from quantcore.performance.metrics import sortino_ratio as quantcore_sortino_ratio

from agentic_trading.agents.schemas import PerformanceRecord
from agentic_trading.config import settings
from agentic_trading.scoring.track_record import TrackRecord

DEFAULT_LOOKBACK_DAYS = 30
# Fewer paired (forecast, outcome) observations than this and a sample
# correlation is noise, not a measured information coefficient.
MIN_IC_OBSERVATIONS = 20


def shrink_information_coefficient(measured: float | None, observations: int) -> float:
    """Bayesian shrinkage of a measured information coefficient toward the
    firm-wide prior `settings.portfolio.information_coefficient`, weighting
    the prior as `ic_shrinkage_observations` pseudo-observations:
    `(n * measured + N0 * prior) / (n + N0)`. A source with no measurable
    IC (`measured` None) gets the prior back unchanged. The result is
    clipped to `[-max_measured_ic, max_measured_ic]`, so a long run of
    lucky calls can't inflate the view beyond a credible skill level."""
    cfg = settings.portfolio
    if measured is None:
        return cfg.information_coefficient
    n0 = cfg.ic_shrinkage_observations
    shrunk = (observations * measured + n0 * cfg.information_coefficient) / (observations + n0)
    return float(np.clip(shrunk, -cfg.max_measured_ic, cfg.max_measured_ic))


class PerformanceScoreboard:
    """Single place to see how any analyst or manager is currently
    performing, derived from a TrackRecord's history.

    References
    ----------
    Sharpe, W. F. (1966). Mutual Fund Performance. Journal of Business,
    39(1), 119-138.
    Sharpe, W. F. (1994). The Sharpe Ratio. Journal of Portfolio
    Management, 21(1), 49-58.
    See REFERENCES.md.
    """

    def __init__(self, track_record: TrackRecord) -> None:
        self.track_record = track_record

    def sharpe_ratio(
        self,
        source_id: str,
        as_of_date: date,
        lookback_days: int,
    ) -> float:
        """Trailing Sharpe ratio for `source_id` over the lookback window
        ending at `as_of_date`, computed from attributable_return in the
        underlying TrackRecord history."""
        history = self.track_record.history_for(source_id, as_of_date, lookback_days)
        returns = [entry.attributable_return for entry in history]
        if len(returns) < 2:
            raise ValueError("sharpe_ratio requires at least two observations")
        stdev = statistics.stdev(returns)
        if stdev == 0:
            return 0.0
        mean = statistics.mean(returns)
        return mean / stdev * math.sqrt(252)

    def sortino_ratio(
        self,
        source_id: str,
        as_of_date: date,
        lookback_days: int,
    ) -> float:
        """Trailing Sortino ratio for `source_id` over the lookback window
        ending at `as_of_date`, via quantcore.performance.metrics — same
        undefined-on-fewer-than-two-observations behavior as sharpe_ratio."""
        history = self.track_record.history_for(source_id, as_of_date, lookback_days)
        returns = [entry.attributable_return for entry in history]
        if len(returns) < 2:
            raise ValueError("sortino_ratio requires at least two observations")
        return float(quantcore_sortino_ratio(np.array(returns, dtype=np.float64)))

    def calmar_ratio(
        self,
        source_id: str,
        as_of_date: date,
        lookback_days: int,
    ) -> float:
        """Trailing Calmar ratio for `source_id` over the lookback window
        ending at `as_of_date`, via quantcore.performance.metrics — same
        undefined-on-fewer-than-two-observations behavior as sharpe_ratio."""
        history = self.track_record.history_for(source_id, as_of_date, lookback_days)
        returns = [entry.attributable_return for entry in history]
        if len(returns) < 2:
            raise ValueError("calmar_ratio requires at least two observations")
        return float(quantcore_calmar_ratio(np.array(returns, dtype=np.float64)))

    def max_drawdown(
        self,
        source_id: str,
        as_of_date: date,
        lookback_days: int,
    ) -> float:
        """Trailing max drawdown for `source_id` over the lookback window
        ending at `as_of_date`, computed from attributable_return in the
        underlying TrackRecord history."""
        history = self.track_record.history_for(source_id, as_of_date, lookback_days)
        if not history:
            return 0.0

        equity = 0.0
        running_peak = 0.0
        worst_drawdown = 0.0
        for entry in history:
            equity += entry.attributable_return
            running_peak = max(running_peak, equity)
            worst_drawdown = min(worst_drawdown, equity - running_peak)
        return worst_drawdown

    def cumulative_return(
        self,
        source_id: str,
        as_of_date: date,
        lookback_days: int,
    ) -> float:
        """Trailing cumulative attributable return for `source_id` over
        the lookback window ending at `as_of_date`."""
        history = self.track_record.history_for(source_id, as_of_date, lookback_days)
        return sum(entry.attributable_return for entry in history)

    def _analyst_history(
        self, source_id: str, as_of_date: date, lookback_days: int
    ) -> list[PerformanceRecord]:
        return self.track_record.history_for(
            source_id, as_of_date, lookback_days, source_type="analyst"
        )

    @staticmethod
    def _correlation(history: list[PerformanceRecord]) -> float | None:
        if len(history) < MIN_IC_OBSERVATIONS:
            return None
        scores = np.array([entry.signal_score for entry in history], dtype=np.float64)
        realized = np.array([entry.realized_return for entry in history], dtype=np.float64)
        if np.std(scores) < 1e-12 or np.std(realized) < 1e-12:
            return None
        return float(np.corrcoef(scores, realized)[0, 1])

    def information_coefficient(
        self,
        source_id: str,
        as_of_date: date,
        lookback_days: int,
    ) -> float | None:
        """Trailing information coefficient for `source_id`: the Pearson
        correlation between `signal_score` (direction x raw confidence)
        and `realized_return` over its analyst-level records in the
        lookback window ending at `as_of_date`. None with fewer than
        `MIN_IC_OBSERVATIONS` records or when either series has zero
        variance, since a correlation is undefined or meaningless there."""
        return self._correlation(self._analyst_history(source_id, as_of_date, lookback_days))

    def shrunk_information_coefficient(self, source_id: str, as_of_date: date) -> float:
        """`information_coefficient` over `settings.portfolio.ic_lookback_days`,
        shrunk toward the firm-wide prior via `shrink_information_coefficient`
        — always a usable number, the prior itself when nothing is measurable."""
        history = self._analyst_history(source_id, as_of_date, settings.portfolio.ic_lookback_days)
        return shrink_information_coefficient(self._correlation(history), len(history))

    def summary(self, as_of_date: date) -> dict[str, dict[str, float]]:
        """Full snapshot of every known analyst's and manager's current
        metrics (sharpe_ratio, max_drawdown, cumulative_return) as of
        `as_of_date`, keyed by source_id."""
        source_ids = {
            entry.source_id
            for entry in self.track_record._entries
            if entry.as_of_date <= as_of_date
        }
        result: dict[str, dict[str, float]] = {}
        for source_id in sorted(source_ids):
            try:
                sharpe = self.sharpe_ratio(source_id, as_of_date, DEFAULT_LOOKBACK_DAYS)
            except ValueError:
                # sharpe_ratio() deliberately raises on fewer than two
                # observations (undefined sample stdev) — expected on a
                # source's first day or two, not an error for a snapshot
                # covering every known source regardless of how long each
                # has been tracked.
                sharpe = 0.0
            result[source_id] = {
                "sharpe_ratio": sharpe,
                "max_drawdown": self.max_drawdown(source_id, as_of_date, DEFAULT_LOOKBACK_DAYS),
                "cumulative_return": self.cumulative_return(
                    source_id, as_of_date, DEFAULT_LOOKBACK_DAYS
                ),
            }
        return result
