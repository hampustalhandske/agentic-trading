# scoreboard.py

import math
import statistics
from datetime import date

import numpy as np
from quantcore.performance.metrics import calmar_ratio as quantcore_calmar_ratio
from quantcore.performance.metrics import sortino_ratio as quantcore_sortino_ratio

from agentic_trading.scoring.track_record import TrackRecord

DEFAULT_LOOKBACK_DAYS = 30


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
