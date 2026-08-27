"""Unit tests for agentic_trading.scoring.scoreboard.

No mocking needed — PerformanceScoreboard operates on an in-memory TrackRecord populated
with PerformanceRecord instances constructed directly in these tests.
"""

import math
import statistics
from datetime import date, timedelta

import numpy as np
import pytest
from quantcore.performance.metrics import calmar_ratio, sortino_ratio

from agentic_trading.agents.schemas import PerformanceRecord
from agentic_trading.scoring.scoreboard import PerformanceScoreboard
from agentic_trading.scoring.track_record import TrackRecord

TRADING_DAYS_PER_YEAR = 252


def _record(source_id: str, as_of_date: date, attributable_return: float) -> PerformanceRecord:
    return PerformanceRecord(
        source_id=source_id,
        source_type="analyst",
        asset="AAPL",
        as_of_date=as_of_date,
        attributable_return=attributable_return,
    )


def _seed(source_id: str, start: date, returns: list[float]) -> PerformanceScoreboard:
    track_record = TrackRecord()
    for offset, attributable_return in enumerate(returns):
        entry_date = start + timedelta(days=offset)
        track_record.record(_record(source_id, entry_date, attributable_return))
    return PerformanceScoreboard(track_record)


def test_cumulative_return_sums_attributable_returns_additively() -> None:
    """cumulative_return sums attributable_return over the window, additive not compounded."""
    start = date(2024, 1, 1)
    returns = [0.01, -0.02, 0.03]
    scoreboard = _seed("analyst-a", start, returns)

    result = scoreboard.cumulative_return("analyst-a", start + timedelta(days=2), lookback_days=10)

    assert result == pytest.approx(sum(returns))


def test_sharpe_ratio_matches_hand_computed_annualized_formula() -> None:
    """sharpe_ratio equals mean(returns) / sample_stdev(returns) * sqrt(252) for a known series."""
    start = date(2024, 1, 1)
    returns = [0.01, 0.03, -0.01, 0.02, 0.0]
    scoreboard = _seed("analyst-a", start, returns)
    mean = statistics.mean(returns)
    stdev = statistics.stdev(returns)
    expected = mean / stdev * math.sqrt(TRADING_DAYS_PER_YEAR)

    result = scoreboard.sharpe_ratio("analyst-a", start + timedelta(days=4), lookback_days=10)

    assert result == pytest.approx(expected)


def test_sharpe_ratio_raises_on_a_single_data_point() -> None:
    """A single-observation window has undefined sample stdev, so sharpe_ratio raises ValueError."""
    start = date(2024, 1, 1)
    scoreboard = _seed("analyst-a", start, [0.01])

    # scoreboard.py's docstring does not specify this edge case; ValueError is chosen here over
    # silently returning 0.0/NaN, since a NaN would otherwise propagate silently into Hedge weights.
    with pytest.raises(ValueError):
        scoreboard.sharpe_ratio("analyst-a", start, lookback_days=10)


def test_sortino_ratio_matches_quantcore_directly() -> None:
    start = date(2024, 1, 1)
    returns = [0.01, 0.03, -0.01, 0.02, 0.0]
    scoreboard = _seed("analyst-a", start, returns)
    expected = float(sortino_ratio(np.array(returns)))

    result = scoreboard.sortino_ratio("analyst-a", start + timedelta(days=4), lookback_days=10)

    assert result == pytest.approx(expected)


def test_sortino_ratio_raises_on_a_single_data_point() -> None:
    start = date(2024, 1, 1)
    scoreboard = _seed("analyst-a", start, [0.01])

    with pytest.raises(ValueError):
        scoreboard.sortino_ratio("analyst-a", start, lookback_days=10)


def test_calmar_ratio_matches_quantcore_directly() -> None:
    start = date(2024, 1, 1)
    returns = [0.05, 0.03, -0.10, -0.02, 0.04]
    scoreboard = _seed("analyst-a", start, returns)
    expected = float(calmar_ratio(np.array(returns)))

    result = scoreboard.calmar_ratio("analyst-a", start + timedelta(days=4), lookback_days=10)

    assert result == pytest.approx(expected)


def test_calmar_ratio_raises_on_a_single_data_point() -> None:
    start = date(2024, 1, 1)
    scoreboard = _seed("analyst-a", start, [0.01])

    with pytest.raises(ValueError):
        scoreboard.calmar_ratio("analyst-a", start, lookback_days=10)


def test_max_drawdown_matches_known_peak_to_trough_decline() -> None:
    """max_drawdown returns the most negative (value - running_peak) over the equity curve."""
    start = date(2024, 1, 1)
    returns = [0.05, 0.03, -0.10, -0.02, 0.04]
    scoreboard = _seed("analyst-a", start, returns)

    # Running-sum equity curve: [0.05, 0.08, -0.02, -0.04, 0.00]; running peak: [0.05, 0.08, 0.08,
    # 0.08, 0.08]. value - peak bottoms out at -0.04 - 0.08 = -0.12, the worst decline observed.
    result = scoreboard.max_drawdown("analyst-a", start + timedelta(days=4), lookback_days=10)

    assert result == pytest.approx(-0.12)
    assert result <= 0.0


def test_max_drawdown_is_zero_for_a_monotonically_increasing_series() -> None:
    """A curve that only ever sets new highs never declines from its running peak."""
    start = date(2024, 1, 1)
    returns = [0.01, 0.02, 0.01, 0.03]
    scoreboard = _seed("analyst-a", start, returns)

    result = scoreboard.max_drawdown("analyst-a", start + timedelta(days=3), lookback_days=10)

    assert result == 0.0


def test_summary_matches_direct_calls_for_every_known_source_id_up_to_as_of_date() -> None:
    """summary() reports every known source_id, matching direct calls with the same lookback."""
    track_record = TrackRecord()
    day1, day2, day3, day5 = (date(2024, 1, d) for d in (1, 2, 3, 5))
    track_record.record(_record("analyst-a", day1, 0.01))
    track_record.record(_record("analyst-a", day2, -0.01))
    track_record.record(_record("analyst-a", day3, 0.02))
    track_record.record(_record("analyst-b", day2, 0.005))
    track_record.record(_record("analyst-b", day3, 0.01))
    track_record.record(_record("analyst-c", day5, 0.02))
    scoreboard = PerformanceScoreboard(track_record)
    full_lookback = 30

    result = scoreboard.summary(day3)

    assert set(result.keys()) == {"analyst-a", "analyst-b"}
    for source_id, metrics in result.items():
        expected = {
            "sharpe_ratio": scoreboard.sharpe_ratio(source_id, day3, full_lookback),
            "max_drawdown": scoreboard.max_drawdown(source_id, day3, full_lookback),
            "cumulative_return": scoreboard.cumulative_return(source_id, day3, full_lookback),
        }
        assert metrics == pytest.approx(expected)
