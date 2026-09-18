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
from agentic_trading.config import settings
from agentic_trading.scoring.scoreboard import (
    PerformanceScoreboard,
    shrink_information_coefficient,
)
from agentic_trading.scoring.track_record import TrackRecord

TRADING_DAYS_PER_YEAR = 252


def _record(source_id: str, as_of_date: date, attributable_return: float) -> PerformanceRecord:
    return PerformanceRecord(
        source_id=source_id,
        source_type="analyst",
        asset="AAPL",
        as_of_date=as_of_date,
        attributable_return=attributable_return,
        raw_attributable_return=attributable_return,
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


def _ic_record(
    source_id: str,
    as_of_date: date,
    signal_score: float,
    realized_return: float,
    source_type: str = "analyst",
) -> PerformanceRecord:
    return PerformanceRecord(
        source_id=source_id,
        source_type=source_type,  # type: ignore[arg-type]
        asset="AAPL",
        as_of_date=as_of_date,
        attributable_return=signal_score * realized_return,
        raw_attributable_return=signal_score * realized_return,
        signal_score=signal_score,
        realized_return=realized_return,
    )


def _seed_ic(
    scores: list[float], realized: list[float], source_type: str = "analyst"
) -> tuple[PerformanceScoreboard, date]:
    start = date(2024, 1, 1)
    track_record = TrackRecord()
    for offset, (score, ret) in enumerate(zip(scores, realized, strict=True)):
        track_record.record(
            _ic_record("analyst-a", start + timedelta(days=offset), score, ret, source_type)
        )
    return PerformanceScoreboard(track_record), start + timedelta(days=len(scores) - 1)


def test_information_coefficient_is_pearson_correlation_of_score_and_realized() -> None:
    rng = np.random.default_rng(1)
    scores = rng.normal(size=40).tolist()
    realized = [
        0.3 * s + float(noise) for s, noise in zip(scores, rng.normal(size=40), strict=True)
    ]
    scoreboard, as_of = _seed_ic(scores, realized)

    result = scoreboard.information_coefficient("analyst-a", as_of, lookback_days=60)

    assert result == pytest.approx(float(np.corrcoef(scores, realized)[0, 1]))


def test_information_coefficient_is_none_below_twenty_records() -> None:
    scores = [float(i) for i in range(19)]
    scoreboard, as_of = _seed_ic(scores, scores)

    assert scoreboard.information_coefficient("analyst-a", as_of, lookback_days=60) is None


def test_information_coefficient_is_none_with_zero_variance() -> None:
    scores = [0.5] * 25
    realized = [float(i) for i in range(25)]
    scoreboard, as_of = _seed_ic(scores, realized)

    assert scoreboard.information_coefficient("analyst-a", as_of, lookback_days=60) is None


def test_information_coefficient_ignores_manager_level_records() -> None:
    scores = [float(i) for i in range(25)]
    scoreboard, as_of = _seed_ic(scores, scores, source_type="manager")

    assert scoreboard.information_coefficient("analyst-a", as_of, lookback_days=60) is None


def test_information_coefficient_respects_the_lookback_window() -> None:
    scores = [float(i) for i in range(30)]
    scoreboard, as_of = _seed_ic(scores, scores)

    # Only 10 of the 30 records fall inside a 9-day window.
    assert scoreboard.information_coefficient("analyst-a", as_of, lookback_days=9) is None
    assert scoreboard.information_coefficient("analyst-a", as_of, lookback_days=60) == (
        pytest.approx(1.0)
    )


def test_shrink_information_coefficient_blends_toward_the_prior() -> None:
    cfg = settings.portfolio
    n = 40
    measured = 0.10
    expected = (n * measured + cfg.ic_shrinkage_observations * cfg.information_coefficient) / (
        n + cfg.ic_shrinkage_observations
    )

    assert shrink_information_coefficient(measured, n) == pytest.approx(expected)


def test_shrink_information_coefficient_returns_prior_when_unmeasurable() -> None:
    assert shrink_information_coefficient(None, 0) == settings.portfolio.information_coefficient
    assert shrink_information_coefficient(None, 500) == settings.portfolio.information_coefficient


def test_shrink_information_coefficient_clips_to_max_measured_ic() -> None:
    cap = settings.portfolio.max_measured_ic

    assert shrink_information_coefficient(1.0, 100_000) == pytest.approx(cap)
    assert shrink_information_coefficient(-1.0, 100_000) == pytest.approx(-cap)


def test_shrink_information_coefficient_lets_a_negative_measurement_pull_below_prior() -> None:
    prior = settings.portfolio.information_coefficient

    assert shrink_information_coefficient(-0.2, 60) < prior


def test_shrunk_information_coefficient_uses_configured_lookback_and_prior() -> None:
    scores = [float(i) for i in range(30)]
    scoreboard, as_of = _seed_ic(scores, scores)

    result = scoreboard.shrunk_information_coefficient("analyst-a", as_of)

    assert result == pytest.approx(shrink_information_coefficient(1.0, 30))
    assert result <= settings.portfolio.max_measured_ic
