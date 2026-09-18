"""Unit tests for the Hedge weighting strategy in agentic_trading.scoring.weighting.

Hedge is exercised against a minimal fake scoreboard rather than a real TrackRecord/
PerformanceScoreboard, so the algorithm is pinned down independent of those classes'
own (separately tested) behavior. No mocking library is used — this is a plain
in-memory stand-in implementing the same cumulative_return(source_id, as_of_date,
lookback_days) -> float signature PerformanceScoreboard exposes.
"""

import math
from datetime import date

import pytest

from agentic_trading.scoring.weighting import EqualWeight, Hedge, ShrinkageWeight


class _FixedScoreboard:
    def __init__(self, cumulative_returns: dict[str, float]) -> None:
        self._cumulative_returns = cumulative_returns

    def cumulative_return(self, source_id: str, as_of_date: date, lookback_days: int) -> float:
        return self._cumulative_returns.get(source_id, 0.0)


AS_OF_DATE = date(2024, 6, 1)


def _expected_weights(cumulative_returns: dict[str, float], eta: float) -> dict[str, float]:
    raw = {source_id: math.exp(eta * value) for source_id, value in cumulative_returns.items()}
    average = sum(raw.values()) / len(raw)
    return {source_id: value / average for source_id, value in raw.items()}


@pytest.mark.parametrize("source_count", [1, 2, 5])
def test_get_weights_averages_to_one_across_source_ids(source_count: int) -> None:
    """Weights sum to len(source_ids) — normalized by average, not to a probability simplex."""
    cumulative_returns = {f"analyst-{i}": 0.01 * i for i in range(source_count)}
    scoreboard = _FixedScoreboard(cumulative_returns)
    hedge = Hedge(eta=1.0, lookback_days=30)

    weights = hedge.get_weights(scoreboard, list(cumulative_returns), AS_OF_DATE)

    assert sum(weights.values()) == pytest.approx(source_count)


def test_get_weights_matches_hand_computed_multiplicative_formula() -> None:
    """Weights equal exp(eta * cumulative_return), normalized so they average to 1.0."""
    cumulative_returns = {"analyst-a": 0.05, "analyst-b": -0.02, "analyst-c": 0.0}
    scoreboard = _FixedScoreboard(cumulative_returns)
    hedge = Hedge(eta=2.0, lookback_days=30)
    expected = _expected_weights(cumulative_returns, eta=2.0)

    weights = hedge.get_weights(scoreboard, list(cumulative_returns), AS_OF_DATE)

    assert weights == pytest.approx(expected)


def test_get_weights_gives_strictly_higher_weight_to_better_performer() -> None:
    """A strictly higher trailing cumulative_return must translate into a strictly higher weight."""
    scoreboard = _FixedScoreboard({"analyst-a": 0.10, "analyst-b": 0.01})
    hedge = Hedge(eta=1.0, lookback_days=30)

    weights = hedge.get_weights(scoreboard, ["analyst-a", "analyst-b"], AS_OF_DATE)

    assert weights["analyst-a"] > weights["analyst-b"]


def test_get_weights_are_equal_when_performance_is_identical() -> None:
    """Identical cumulative_return across all sources means each gets exactly the average weight."""
    scoreboard = _FixedScoreboard({"analyst-a": 0.03, "analyst-b": 0.03, "analyst-c": 0.03})
    hedge = Hedge(eta=1.0, lookback_days=30)

    weights = hedge.get_weights(scoreboard, ["analyst-a", "analyst-b", "analyst-c"], AS_OF_DATE)

    assert weights == pytest.approx({"analyst-a": 1.0, "analyst-b": 1.0, "analyst-c": 1.0})


def test_get_weights_treats_unrecorded_source_id_as_neutral() -> None:
    """A source_id absent from the scoreboard is treated identically to an explicit zero entry."""
    scoreboard = _FixedScoreboard({"analyst-a": 0.05, "explicit-zero": 0.0})
    hedge = Hedge(eta=1.0, lookback_days=30)

    with_explicit_zero = hedge.get_weights(scoreboard, ["analyst-a", "explicit-zero"], AS_OF_DATE)
    with_unrecorded = hedge.get_weights(scoreboard, ["analyst-a", "new-analyst"], AS_OF_DATE)

    assert with_unrecorded["new-analyst"] == pytest.approx(with_explicit_zero["explicit-zero"])
    assert with_unrecorded["new-analyst"] < with_unrecorded["analyst-a"]


def test_get_weights_with_zero_eta_ignores_performance_entirely() -> None:
    """eta=0.0 is a zero learning rate: weights collapse to 1.0 regardless of performance."""
    scoreboard = _FixedScoreboard({"analyst-a": 5.0, "analyst-b": -3.0, "analyst-c": 0.0})
    hedge = Hedge(eta=0.0, lookback_days=30)

    weights = hedge.get_weights(scoreboard, ["analyst-a", "analyst-b", "analyst-c"], AS_OF_DATE)

    assert weights == pytest.approx({"analyst-a": 1.0, "analyst-b": 1.0, "analyst-c": 1.0})


def test_get_weights_returns_exactly_the_requested_source_ids() -> None:
    """The result's keys are exactly source_ids, regardless of what the scoreboard also contains."""
    scoreboard = _FixedScoreboard({"analyst-a": 0.02, "analyst-b": 0.01, "analyst-untracked": 0.9})
    hedge = Hedge(eta=1.0, lookback_days=30)

    weights = hedge.get_weights(scoreboard, ["analyst-a", "analyst-b"], AS_OF_DATE)

    assert set(weights.keys()) == {"analyst-a", "analyst-b"}


def test_hedge_constructs_with_sensible_defaults() -> None:
    """Hedge is usable with no explicit arguments, exposing a positive eta and lookback_days."""
    hedge = Hedge()

    assert hedge.eta > 0
    assert hedge.lookback_days > 0


class _FakeTrackRecord:
    def __init__(self, observation_counts: dict[str, int]) -> None:
        self._observation_counts = observation_counts

    def history_for(self, source_id: str, as_of_date: date, lookback_days: int) -> list[object]:
        return [object()] * self._observation_counts.get(source_id, 0)


class _ScoreboardWithTrackRecord(_FixedScoreboard):
    def __init__(
        self, cumulative_returns: dict[str, float], observation_counts: dict[str, int]
    ) -> None:
        super().__init__(cumulative_returns)
        self.track_record = _FakeTrackRecord(observation_counts)


def test_shrinkage_weight_keeps_a_dense_source_almost_unchanged() -> None:
    """A source with observations >= full_trust_observations keeps (almost) its
    raw weight — shrinkage_factor is capped at 1.0."""
    scoreboard = _ScoreboardWithTrackRecord(
        {"analyst-a": 0.10, "analyst-b": 0.01},
        {"analyst-a": 30, "analyst-b": 30},
    )
    base = Hedge(eta=1.0, lookback_days=30)
    shrinkage = ShrinkageWeight(base=base, full_trust_observations=20)

    raw = base.get_weights(scoreboard, ["analyst-a", "analyst-b"], AS_OF_DATE)
    shrunk = shrinkage.get_weights(scoreboard, ["analyst-a", "analyst-b"], AS_OF_DATE)

    assert shrunk == pytest.approx(raw)


def test_shrinkage_weight_pulls_a_thin_source_toward_the_average() -> None:
    """A source with zero real observations gets pulled all the way to the
    cross-sectional average, regardless of how extreme its raw weight is."""
    scoreboard = _ScoreboardWithTrackRecord(
        {"analyst-a": 0.10, "analyst-b": 0.01},
        {"analyst-a": 30, "analyst-b": 0},
    )
    base = Hedge(eta=1.0, lookback_days=30)
    shrinkage = ShrinkageWeight(base=base, full_trust_observations=20)

    raw = base.get_weights(scoreboard, ["analyst-a", "analyst-b"], AS_OF_DATE)
    shrunk = shrinkage.get_weights(scoreboard, ["analyst-a", "analyst-b"], AS_OF_DATE)
    average = sum(raw.values()) / len(raw)

    assert shrunk["analyst-b"] == pytest.approx(average)
    assert shrunk["analyst-a"] == pytest.approx(raw["analyst-a"])


def test_shrinkage_weight_interpolates_linearly_between_raw_and_average() -> None:
    scoreboard = _ScoreboardWithTrackRecord(
        {"analyst-a": 0.10, "analyst-b": 0.01},
        {"analyst-a": 10, "analyst-b": 10},
    )
    base = Hedge(eta=1.0, lookback_days=30)
    shrinkage = ShrinkageWeight(base=base, full_trust_observations=20)

    raw = base.get_weights(scoreboard, ["analyst-a", "analyst-b"], AS_OF_DATE)
    shrunk = shrinkage.get_weights(scoreboard, ["analyst-a", "analyst-b"], AS_OF_DATE)
    average = sum(raw.values()) / len(raw)

    # 10/20 observations -> shrinkage_factor 0.5
    expected = {source_id: 0.5 * value + 0.5 * average for source_id, value in raw.items()}
    assert shrunk == pytest.approx(expected)


def test_shrinkage_weight_wraps_equal_weight_without_error() -> None:
    """EqualWeight has no lookback_days attribute — ShrinkageWeight must fall
    back to a default lookback rather than raising."""
    scoreboard = _ScoreboardWithTrackRecord(
        {"analyst-a": 0.0, "analyst-b": 0.0}, {"analyst-a": 5, "analyst-b": 5}
    )
    shrinkage = ShrinkageWeight(base=EqualWeight())

    weights = shrinkage.get_weights(scoreboard, ["analyst-a", "analyst-b"], AS_OF_DATE)

    assert weights == pytest.approx({"analyst-a": 1.0, "analyst-b": 1.0})


def test_shrinkage_weight_handles_empty_source_ids() -> None:
    scoreboard = _ScoreboardWithTrackRecord({}, {})
    shrinkage = ShrinkageWeight(base=Hedge())

    assert shrinkage.get_weights(scoreboard, [], AS_OF_DATE) == {}
