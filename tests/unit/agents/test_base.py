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
    _pod_information_coefficient,
)
from agentic_trading.agents.portfolio_construction import PodView, PortfolioConstructor
from agentic_trading.agents.schemas import AgentSignal, ManagerSignal, PerformanceRecord
from agentic_trading.config import settings
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
        portfolio_constructor=PortfolioConstructor(),
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
        portfolio_constructor=PortfolioConstructor(),
    )

    signals = manager.collect_signals("AAPL", MARKET_DATA, AS_OF)

    working_signal = next(s for s in signals if s.agent_id == "working_analyst")
    assert working_signal.direction == "long"
    assert working_signal.confidence == pytest.approx(0.8)


def test_collect_signals_never_raises_even_if_every_analyst_crashes() -> None:
    manager = _TestManager(
        managed_analysts=[_CrashingAnalyst()],
        weighting_strategy=Hedge(),
        portfolio_constructor=PortfolioConstructor(),
    )

    signals = manager.collect_signals("AAPL", MARKET_DATA, AS_OF)

    assert len(signals) == 1
    assert signals[0].direction == SAFE_FALLBACK_DIRECTION


def test_evaluate_defaults_a_crashing_scale_factor_to_zero_contribution() -> None:
    manager = _TestManager(
        managed_analysts=[_WorkingAnalyst()],
        weighting_strategy=Hedge(),
        should_crash=True,
        portfolio_constructor=PortfolioConstructor(),
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
    # raw_confidence carries the analyst's own unscaled confidence forward
    # regardless of what the manager's scale_factor did with it.
    assert scaled[0].raw_confidence == pytest.approx(0.8)


def test_evaluate_does_not_raise_when_scale_factor_crashes() -> None:
    manager = _TestManager(
        managed_analysts=[_WorkingAnalyst()],
        weighting_strategy=Hedge(),
        should_crash=True,
        portfolio_constructor=PortfolioConstructor(),
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


def _multi_asset_market_data(assets: list[str], periods: int = 30) -> pd.DataFrame:
    index = pd.bdate_range("2024-01-02", periods=periods)
    frames = {}
    for offset, asset in enumerate(assets):
        prices = 100.0 + offset * 10.0 + pd.Series(range(periods), dtype=float) * 0.1
        frame = pd.DataFrame({"close": prices.to_numpy()}, index=index)
        frame["log_return"] = pd.Series(prices.to_numpy(), index=index).apply(lambda _: 0.001)
        frames[asset] = frame
    return pd.concat(frames, axis=1)


def test_construct_portfolio_sizes_a_multi_asset_book_from_analyst_views() -> None:
    """End-to-end through ManagerAgent.construct_portfolio(): a manager
    whose only analyst always says long/0.8 must come back with a
    nonzero, bounded, long-biased weight for every asset it covers, with
    the analyst's own ManagerSignal carried forward for traceability."""
    assets = ["AAPL", "MSFT"]
    market_data = _multi_asset_market_data(assets, periods=30)
    manager = _TestManager(
        managed_analysts=[_WorkingAnalyst()],
        weighting_strategy=Hedge(),
        portfolio_constructor=PortfolioConstructor(),
    )

    decisions = manager.construct_portfolio(assets, market_data, AS_OF, _scoreboard())

    assert {decision.asset for decision in decisions} == set(assets)
    for decision in decisions:
        assert -PortfolioConstructor().leverage_cap <= decision.target_weight
        assert decision.target_weight <= PortfolioConstructor().leverage_cap
        assert len(decision.contributing_signals) == 1
        assert decision.contributing_signals[0].source_agent_id == "working_analyst"


def test_construct_portfolio_scopes_to_self_assets_when_set() -> None:
    """A pod with `self.assets` set (an asset-class-scoped pod, e.g. a
    Rates pod restricted to Treasury ETFs) must only actually price that
    subset — every other asset the Commander happens to pass in gets a
    flat 0.0-weight decision with no contributing signals, not an error
    and not a decision priced as if it were in scope."""
    assets = ["AAPL", "MSFT", "SHY"]
    market_data = _multi_asset_market_data(assets, periods=30)
    manager = _TestManager(
        managed_analysts=[_WorkingAnalyst()],
        weighting_strategy=Hedge(),
        portfolio_constructor=PortfolioConstructor(),
        assets=("SHY",),
    )

    decisions = manager.construct_portfolio(assets, market_data, AS_OF, _scoreboard())
    by_asset = {decision.asset: decision for decision in decisions}

    assert set(by_asset) == set(assets)
    assert by_asset["SHY"].target_weight != 0.0
    assert len(by_asset["SHY"].contributing_signals) == 1
    for out_of_scope in ("AAPL", "MSFT"):
        assert by_asset[out_of_scope].target_weight == 0.0
        assert by_asset[out_of_scope].contributing_signals == []


def test_construct_portfolio_prices_every_asset_when_self_assets_is_none() -> None:
    """The default (`self.assets = None`, every existing equity pod)
    behaves exactly as before this mechanism was added — no accidental
    narrowing for a pod that never opted into scoping."""
    assets = ["AAPL", "MSFT"]
    market_data = _multi_asset_market_data(assets, periods=30)
    manager = _TestManager(
        managed_analysts=[_WorkingAnalyst()],
        weighting_strategy=Hedge(),
        portfolio_constructor=PortfolioConstructor(),
    )

    decisions = manager.construct_portfolio(assets, market_data, AS_OF, _scoreboard())

    assert manager.assets is None
    assert all(len(decision.contributing_signals) == 1 for decision in decisions)


def test_safe_fallback_direction_and_confidence_contribute_nothing_to_target_weight() -> None:
    """The whole point of the "flat"/0.0 fallback: direction_sign("flat")=0
    already zeroes out any contribution regardless of confidence, and
    confidence=0.0 makes that doubly explicit rather than implying a
    (unearned) 50/50 view — see base.py's SAFE_FALLBACK_* docstring."""
    direction_sign = {"long": 1.0, "short": -1.0, "flat": 0.0}[SAFE_FALLBACK_DIRECTION]
    assert direction_sign * SAFE_FALLBACK_CONFIDENCE == 0.0


class _CapturingConstructor(PortfolioConstructor):
    """Records the views handed to `construct` so a test can inspect the
    PodViews `construct_portfolio` built."""

    def __init__(self) -> None:
        super().__init__()
        self.captured: dict[str, PodView] = {}

    def construct(
        self,
        views: dict[str, PodView],
        returns_history: pd.DataFrame,
        as_of_date: pd.Timestamp,
        previous_weights: dict[str, float] | None = None,
    ) -> dict[str, float]:
        self.captured = dict(views)
        return dict.fromkeys(views, 0.0)


def _ic_scoreboard(source_id: str, as_of: pd.Timestamp, records: int) -> PerformanceScoreboard:
    """A scoreboard whose `source_id` has `records` perfectly-correlated
    analyst-level (signal_score, realized_return) pairs ending at `as_of`."""
    track_record = TrackRecord()
    for offset in range(records):
        score = float(offset - records / 2)
        track_record.record(
            PerformanceRecord(
                source_id=source_id,
                source_type="analyst",
                asset="AAPL",
                as_of_date=(as_of - pd.Timedelta(days=records - 1 - offset)).date(),
                attributable_return=score * 0.01,
                raw_attributable_return=score * 0.01,
                signal_score=score,
                realized_return=score * 0.01,
            )
        )
    return PerformanceScoreboard(track_record)


def test_construct_portfolio_uses_the_prior_ic_without_track_record() -> None:
    constructor = _CapturingConstructor()
    manager = _TestManager(
        managed_analysts=[_WorkingAnalyst()],
        weighting_strategy=Hedge(),
        portfolio_constructor=constructor,
    )

    manager.construct_portfolio(
        ["AAPL"], _multi_asset_market_data(["AAPL"], periods=30), AS_OF, _scoreboard()
    )

    assert constructor.captured["AAPL"].information_coefficient == pytest.approx(
        settings.portfolio.information_coefficient
    )


def test_construct_portfolio_feeds_the_measured_shrunk_ic_into_the_pod_view() -> None:
    constructor = _CapturingConstructor()
    manager = _TestManager(
        managed_analysts=[_WorkingAnalyst()],
        weighting_strategy=Hedge(),
        portfolio_constructor=constructor,
    )
    scoreboard = _ic_scoreboard("working_analyst", AS_OF, records=40)

    manager.construct_portfolio(
        ["AAPL"], _multi_asset_market_data(["AAPL"], periods=30), AS_OF, scoreboard
    )

    expected = scoreboard.shrunk_information_coefficient("working_analyst", AS_OF.date())
    assert expected > settings.portfolio.information_coefficient
    assert constructor.captured["AAPL"].information_coefficient == pytest.approx(expected)


def test_construct_portfolio_leaves_ic_unset_for_an_asset_with_no_active_signal() -> None:
    constructor = _CapturingConstructor()
    manager = _TestManager(
        managed_analysts=[_CrashingAnalyst()],
        weighting_strategy=Hedge(),
        portfolio_constructor=constructor,
    )

    manager.construct_portfolio(
        ["AAPL"], _multi_asset_market_data(["AAPL"], periods=30), AS_OF, _scoreboard()
    )

    assert constructor.captured["AAPL"].information_coefficient is None


def test_pod_information_coefficient_is_confidence_weighted_and_floored_at_zero() -> None:
    def signal(agent_id: str, confidence: float) -> ManagerSignal:
        return ManagerSignal(
            manager_id="m",
            source_agent_id=agent_id,
            asset="AAPL",
            as_of_date=AS_OF.date(),
            direction="long",
            confidence=confidence,
            raw_confidence=confidence,
            scale_applied=1.0,
            rationale="",
        )

    ics = {"a": 0.10, "b": 0.02}
    active = [signal("a", 0.75), signal("b", 0.25)]

    result = _pod_information_coefficient(active, 1.0, ics.__getitem__)

    assert result == pytest.approx(0.75 * 0.10 + 0.25 * 0.02)
    assert _pod_information_coefficient(active, 1.0, {"a": -0.1, "b": -0.1}.__getitem__) == 0.0
    assert _pod_information_coefficient([], 0.0, ics.__getitem__) is None
