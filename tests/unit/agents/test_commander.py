"""Unit tests for agentic_trading.agents.commander.Commander — now a pure
capital allocator across pods, not a per-asset signal combiner. Each fake
manager below overrides construct_portfolio() directly so these tests pin
down Commander's own allocation/combination math independent of any real
analyst/covariance machinery (separately tested elsewhere)."""

from collections.abc import Iterator
from datetime import date

import numpy as np
import pandas as pd
import pytest
from pydantic import Field
from quantcore.portfolio.covariance import ledoit_wolf_shrinkage
from quantcore.risk.var import value_at_risk
from scipy.stats import norm

from agentic_trading.agents.base import ManagerAgent
from agentic_trading.agents.commander import (
    TAIL_HEDGE_MANAGER_ID,
    Commander,
    CommanderRiskState,
    _apply_firm_limits,
    _DrawdownBreaker,
    _PodReturnTracker,
    _risk_budget_weights,
)
from agentic_trading.agents.portfolio_construction import PortfolioConstructor
from agentic_trading.agents.schemas import MAX_LEVERAGE, OptionContract, PortfolioDecision
from agentic_trading.config import settings, temporary_overrides
from agentic_trading.data.loaders import option_marks
from agentic_trading.data.loaders.option_marks import OptionMark
from agentic_trading.scoring.scoreboard import PerformanceScoreboard
from agentic_trading.scoring.track_record import TrackRecord
from agentic_trading.scoring.weighting import WeightingStrategy

AS_OF = pd.Timestamp("2024-06-01")
MARKET_DATA = pd.concat(
    {
        "AAPL": pd.DataFrame({"close": [100.0], "log_return": [0.0]}, index=[AS_OF]),
        "MSFT": pd.DataFrame({"close": [200.0], "log_return": [0.0]}, index=[AS_OF]),
    },
    axis=1,
)


@pytest.fixture(autouse=True)
def _lift_firm_limits() -> Iterator[None]:
    """The allocation tests below pin Commander's cross-pod math with
    stub pods at full weight, well above the firm's single-name/gross
    lines; the firm layer is exercised on its own with explicit tight
    limits in the `_apply_firm_limits` tests further down."""
    with temporary_overrides(
        {
            "firm.max_single_name_weight": 100.0,
            "firm.max_gross_exposure": 1000.0,
            "firm.max_daily_var": 1000.0,
        }
    ):
        yield


class _FixedPodWeights(WeightingStrategy):
    def __init__(self, weights: dict[str, float]) -> None:
        self.weights = weights

    def get_weights(self, scoreboard, source_ids, as_of_date):
        return {source_id: self.weights.get(source_id, 0.0) for source_id in source_ids}


class _StubManager(ManagerAgent):
    """Bypasses real signal collection/sizing entirely — returns a
    caller-fixed set of PortfolioDecisions from construct_portfolio()."""

    fixed_decisions: dict[str, float] = Field(default_factory=dict)

    def scale_factor(self, signal, market_data, as_of_date) -> float:
        return 1.0

    def construct_portfolio(self, assets, market_data, as_of_date, scoreboard):
        return [
            PortfolioDecision(
                asset=asset,
                as_of_date=as_of_date.date(),
                target_weight=self.fixed_decisions.get(asset, 0.0),
                contributing_signals=[],
            )
            for asset in assets
        ]


def _stub_manager(agent_id: str, fixed_decisions: dict[str, float]) -> _StubManager:
    return _StubManager(
        agent_id=agent_id,
        managed_analysts=[],
        weighting_strategy=_FixedPodWeights({}),
        portfolio_constructor=PortfolioConstructor(),
        fixed_decisions=fixed_decisions,
    )


def _scoreboard() -> PerformanceScoreboard:
    return PerformanceScoreboard(TrackRecord())


def test_decide_returns_empty_for_empty_market_data() -> None:
    commander = Commander(managers=[], weighting_strategy=_FixedPodWeights({}))

    assert commander.decide(pd.DataFrame(), AS_OF, _scoreboard()) == []


def test_decide_allocates_capital_proportionally_across_pods() -> None:
    """Two pods, one weighted 3x the other — the combined decision should
    reflect capital allocated in that 3:1 ratio, not an equal split."""
    pod_a = _stub_manager("pod_a", {"AAPL": 1.0, "MSFT": 0.0})
    pod_b = _stub_manager("pod_b", {"AAPL": 0.0, "MSFT": 1.0})
    commander = Commander(
        managers=[pod_a, pod_b],
        weighting_strategy=_FixedPodWeights({"pod_a": 3.0, "pod_b": 1.0}),
    )

    decisions = commander.decide(MARKET_DATA, AS_OF, _scoreboard())
    by_asset = {decision.asset: decision.target_weight for decision in decisions}

    assert by_asset["AAPL"] == pytest.approx(0.75)
    assert by_asset["MSFT"] == pytest.approx(0.25)


def test_decide_normalizes_allocations_that_do_not_sum_to_one() -> None:
    """Raw pod weights need not sum to 1.0 (Hedge-style strategies average
    to 1.0 across sources, summing to N) — Commander must normalize so the
    fund never commits more than its own capital."""
    pod_a = _stub_manager("pod_a", {"AAPL": 1.0})
    pod_b = _stub_manager("pod_b", {"AAPL": 1.0})
    commander = Commander(
        managers=[pod_a, pod_b],
        weighting_strategy=_FixedPodWeights({"pod_a": 1.0, "pod_b": 1.0}),
    )

    decisions = commander.decide(MARKET_DATA, AS_OF, _scoreboard())
    aapl_decision = next(d for d in decisions if d.asset == "AAPL")

    # Each pod wants full (1.0) weight on AAPL; normalized 50/50 allocation
    # means the combined result is 1.0, not 2.0.
    assert aapl_decision.target_weight == pytest.approx(1.0)


def test_decide_clamps_combined_weight_to_max_leverage() -> None:
    """Even if every pod pushes a single asset toward its own leverage_cap
    and gets 100% allocation, the final combined decision cannot exceed
    MAX_LEVERAGE — the schema-level safety backstop."""
    pod_a = _stub_manager("pod_a", {"AAPL": MAX_LEVERAGE})
    commander = Commander(
        managers=[pod_a],
        weighting_strategy=_FixedPodWeights({"pod_a": 1.0}),
    )

    decisions = commander.decide(MARKET_DATA, AS_OF, _scoreboard())
    aapl_decision = next(d for d in decisions if d.asset == "AAPL")

    assert aapl_decision.target_weight == pytest.approx(MAX_LEVERAGE)


def test_decide_carries_contributing_signals_forward_from_every_pod() -> None:
    class _SignalfulManager(_StubManager):
        def construct_portfolio(self, assets, market_data, as_of_date, scoreboard):
            from agentic_trading.agents.schemas import ManagerSignal

            return [
                PortfolioDecision(
                    asset=asset,
                    as_of_date=as_of_date.date(),
                    target_weight=0.1,
                    contributing_signals=[
                        ManagerSignal(
                            manager_id=self.agent_id,
                            source_agent_id="some_analyst",
                            asset=asset,
                            as_of_date=as_of_date.date(),
                            direction="long",
                            confidence=0.5,
                            raw_confidence=0.5,
                            scale_applied=1.0,
                            rationale="test",
                        )
                    ],
                )
                for asset in assets
            ]

    pod = _SignalfulManager(
        agent_id="pod_a",
        managed_analysts=[],
        weighting_strategy=_FixedPodWeights({}),
        portfolio_constructor=PortfolioConstructor(),
    )
    commander = Commander(managers=[pod], weighting_strategy=_FixedPodWeights({"pod_a": 1.0}))

    decisions = commander.decide(MARKET_DATA, AS_OF, _scoreboard())
    aapl_decision = next(d for d in decisions if d.asset == "AAPL")

    assert len(aapl_decision.contributing_signals) == 1
    assert aapl_decision.contributing_signals[0].source_agent_id == "some_analyst"


def test_decide_falls_back_to_weighting_strategy_before_any_pod_history_exists() -> None:
    """Day one of a run: `_PodReturnTracker` has no prior weight vector for
    either pod yet, so no pod can have a real recorded return — Commander
    must fall back to `weighting_strategy` alone, exactly like the
    existing 3:1 allocation test, not silently produce a zero/empty
    decision."""
    pod_a = _stub_manager("pod_a", {"AAPL": 1.0, "MSFT": 0.0})
    pod_b = _stub_manager("pod_b", {"AAPL": 0.0, "MSFT": 1.0})
    commander = Commander(
        managers=[pod_a, pod_b],
        weighting_strategy=_FixedPodWeights({"pod_a": 1.0, "pod_b": 1.0}),
    )

    decisions = commander.decide(MARKET_DATA, AS_OF, _scoreboard())
    by_asset = {decision.asset: decision.target_weight for decision in decisions}

    assert by_asset["AAPL"] == pytest.approx(0.5)
    assert by_asset["MSFT"] == pytest.approx(0.5)


def test_decide_uses_real_pod_return_risk_budgeting_once_history_is_sufficient() -> None:
    """Once `_PodReturnTracker` has accumulated enough real daily
    pod-return observations (built purely from repeated `decide()` calls
    — each day's realized return is yesterday's weight vector times
    today's actual asset log_return, never an attribution score), the
    allocation should tilt away from a high-volatility pod toward a
    low-volatility one, even with an equal trailing-Sharpe tilt from
    weighting_strategy — proving the inverse-vol risk-budgeting is doing
    real work, not just echoing the fallback."""
    dates = pd.bdate_range("2024-01-02", periods=8)
    # AAPL: alternating +5%/-5% swings (high vol, ~0 mean).
    # MSFT: a steady, small +0.1% every day (low vol).
    aapl_returns = [0.05 if i % 2 == 0 else -0.05 for i in range(len(dates))]
    msft_returns = [0.001] * len(dates)
    market_data = pd.concat(
        {
            "AAPL": pd.DataFrame(
                {"close": [100.0] * len(dates), "log_return": aapl_returns}, index=dates
            ),
            "MSFT": pd.DataFrame(
                {"close": [200.0] * len(dates), "log_return": msft_returns}, index=dates
            ),
        },
        axis=1,
    )
    pod_a = _stub_manager("pod_a", {"AAPL": 1.0, "MSFT": 0.0})
    pod_b = _stub_manager("pod_b", {"AAPL": 0.0, "MSFT": 1.0})
    commander = Commander(
        managers=[pod_a, pod_b],
        weighting_strategy=_FixedPodWeights({"pod_a": 1.0, "pod_b": 1.0}),
    )
    scoreboard = _scoreboard()

    decisions: list[PortfolioDecision] = []
    for as_of in dates:
        decisions = commander.decide(market_data, as_of, scoreboard)

    by_asset = {decision.asset: decision.target_weight for decision in decisions}

    # Equal trailing-Sharpe tilt (_FixedPodWeights gives both pods 1.0), so
    # any departure from a 50/50 split comes purely from inverse-vol
    # risk-budgeting reacting to AAPL's real, much higher realized
    # volatility relative to MSFT's steady series.
    assert by_asset["MSFT"] > by_asset["AAPL"]
    assert by_asset["AAPL"] != pytest.approx(0.5)
    assert abs(by_asset["AAPL"]) <= MAX_LEVERAGE
    assert abs(by_asset["MSFT"]) <= MAX_LEVERAGE


def test_pod_return_tracker_does_not_record_a_fake_zero_for_an_empty_decision_day() -> None:
    """A pod that produces zero decisions on some day (e.g. an
    asset-class-scoped pod outside that day's traded universe) must not
    have that day recorded as a genuine 0.0 return — a real fabricated
    zero would understate its trailing volatility once it does start
    trading again."""
    tracker = _PodReturnTracker(lookback_days=90)
    dates = pd.bdate_range("2024-01-02", periods=3)
    market_data = pd.concat(
        {
            "AAPL": pd.DataFrame(
                {"close": [100.0] * 3, "log_return": [0.02, 0.0, -0.02]}, index=dates
            )
        },
        axis=1,
    )

    # Day 1: pod_a trades AAPL.
    tracker.record_and_get_history(
        [("pod_a", [_decision("AAPL", 1.0, dates[0])])], market_data, dates[0]
    )
    # Day 2: pod_a produces no decisions at all (empty list).
    tracker.record_and_get_history([("pod_a", [])], market_data, dates[1])
    # Day 3: pod_a trades AAPL again.
    history = tracker.record_and_get_history(
        [("pod_a", [_decision("AAPL", 1.0, dates[2])])], market_data, dates[2]
    )

    # Only 2 real observations recorded (day 2 -> day 3's realized return
    # from day 1's weights, and day 3 -> day 4's, if it existed) — the
    # empty-decision day must not have contributed a fabricated 0.0 entry
    # on top of the real ones.
    assert len(history["pod_a"]) == 1
    assert history["pod_a"].iloc[0] == pytest.approx(0.0)  # day1->day2 realized on flat log_return


def _decision(asset: str, weight: float, as_of: pd.Timestamp) -> PortfolioDecision:
    return PortfolioDecision(
        asset=asset, as_of_date=as_of.date(), target_weight=weight, contributing_signals=[]
    )


def test_risk_budget_weights_gives_a_non_qualifying_pod_a_bootstrap_share_not_zero() -> None:
    """A pod below settings.commander.min_observations_for_risk_budget must still get a
    nonzero allocation (its weighting_strategy bootstrap weight) rather
    than being silently zeroed out just because it hasn't accumulated
    enough trailing-return history yet to size a vol estimate from."""
    qualifying_history = pd.Series(
        [0.01, -0.01, 0.01, -0.01, 0.01, -0.01],
        index=pd.bdate_range("2024-01-02", periods=6).date,
    )
    non_qualifying_history = pd.Series([0.02], index=[pd.Timestamp("2024-01-08").date()])
    assert len(non_qualifying_history) < settings.commander.min_observations_for_risk_budget

    weights = _risk_budget_weights(
        {"pod_a": qualifying_history, "pod_b": non_qualifying_history},
        ["pod_a", "pod_b"],
        _scoreboard(),
        pd.Timestamp("2024-01-09").date(),
        _FixedPodWeights({"pod_a": 1.0, "pod_b": 1.0}),
    )

    assert weights["pod_b"] > 0.0
    assert set(weights) == {"pod_a", "pod_b"}


def test_risk_budget_weights_floors_an_outlier_low_vol_pod_relative_to_peers() -> None:
    """A pod whose trailing return history happens to be far lower-vol
    than every other qualifying pod's must not get to dominate the
    whole allocation via an unbounded 1/vol multiplier — naive inverse-
    vol risk-budgeting is a known failure mode here (a real but
    abnormally tiny vol estimate produces a huge weight that crowds out
    every other pod to near-zero), which `_MIN_RELATIVE_VOL_FLOOR`
    guards against. The two pods' histories deliberately don't share any
    trading days (disjoint date ranges), so `_cross_pod_risk_parity_weights`
    finds no overlapping observations and this test exercises the
    peer-floor fallback specifically, not real cross-pod risk parity —
    see `test_risk_budget_weights_uses_real_cross_pod_risk_parity_once_history_overlaps`
    for that path."""
    normal_vol_history = pd.Series(
        [0.01, -0.012, 0.011, -0.009, 0.01, -0.011],
        index=pd.bdate_range("2024-01-02", periods=6).date,
    )
    tiny_vol_history = pd.Series(
        [1e-9, -1e-9, 1e-9, -1e-9, 1e-9, -1e-9],
        index=pd.bdate_range("2024-03-01", periods=6).date,
    )

    weights = _risk_budget_weights(
        {"normal_pod": normal_vol_history, "outlier_pod": tiny_vol_history},
        ["normal_pod", "outlier_pod"],
        _scoreboard(),
        pd.Timestamp("2024-03-10").date(),
        _FixedPodWeights({"normal_pod": 1.0, "outlier_pod": 1.0}),
    )

    # Without the peer-relative floor, outlier_pod's near-zero vol would
    # produce a 1/vol many orders of magnitude larger than normal_pod's,
    # collapsing normal_pod's normalized share to near-zero. With the
    # floor, both pods stay within the same order of magnitude.
    assert weights["normal_pod"] > 0.01
    ratio = weights["outlier_pod"] / weights["normal_pod"]
    assert ratio < 10.0


def test_risk_budget_weights_uses_real_cross_pod_risk_parity_once_history_overlaps() -> None:
    """Once >=2 qualifying pods share enough overlapping trailing-return
    history (T > k), `_risk_budget_weights` sizes them with real
    cross-pod risk parity (Ledoit-Wolf-shrunk covariance +
    `risk_parity_weights`) instead of the simpler per-pod inverse-vol
    fallback — proving the upgrade is actually wired in, not just present
    as dead code."""
    dates = pd.bdate_range("2024-01-02", periods=6).date
    high_vol = pd.Series([0.05, -0.05, 0.05, -0.05, 0.05, -0.05], index=dates)
    low_vol = pd.Series([0.005, -0.004, 0.005, -0.004, 0.005, -0.004], index=dates)

    weights = _risk_budget_weights(
        {"high_vol_pod": high_vol, "low_vol_pod": low_vol},
        ["high_vol_pod", "low_vol_pod"],
        _scoreboard(),
        pd.Timestamp("2024-01-09").date(),
        _FixedPodWeights({"high_vol_pod": 1.0, "low_vol_pod": 1.0}),
    )

    # Real risk parity equalizes risk contribution, not raw capital, so
    # the lower-vol pod gets materially more capital than the high-vol
    # one, and every pod still ends up with a real, nonzero share.
    assert weights["low_vol_pod"] > weights["high_vol_pod"]
    assert weights["high_vol_pod"] > 0.0
    assert sum(weights.values()) == pytest.approx(1.0)


def test_drawdown_breaker_cuts_a_tripped_pod_and_leaves_others_untouched() -> None:
    breaker = _DrawdownBreaker()
    tripped_series = pd.Series([0.0, -0.11])
    healthy_series = pd.Series([0.0, 0.02])

    multipliers = breaker.multipliers(
        {"tripped_pod": tripped_series, "healthy_pod": healthy_series},
        ["tripped_pod", "healthy_pod"],
    )

    assert multipliers["tripped_pod"] == pytest.approx(settings.commander.drawdown_cut_multiplier)
    assert multipliers["healthy_pod"] == pytest.approx(1.0)


def test_drawdown_breaker_hysteresis_stays_tripped_until_recovery_threshold() -> None:
    """A pod that dips past the trip threshold then only partially
    recovers (still worse than the recovery threshold) must stay
    tripped — the whole point of hysteresis is not resetting the moment
    a pod merely climbs back above the trip line itself."""
    breaker = _DrawdownBreaker()

    multipliers = breaker.multipliers({"pod": pd.Series([0.0, -0.11])}, ["pod"])
    assert multipliers["pod"] == pytest.approx(settings.commander.drawdown_cut_multiplier)

    # Equity climbs from -0.11 to -0.06 — still worse than the -0.03
    # recovery threshold, so it must remain tripped.
    multipliers = breaker.multipliers({"pod": pd.Series([0.0, -0.11, 0.05])}, ["pod"])
    assert multipliers["pod"] == pytest.approx(settings.commander.drawdown_cut_multiplier)

    # Equity climbs on to -0.02 — inside the recovery threshold, resets.
    multipliers = breaker.multipliers({"pod": pd.Series([0.0, -0.11, 0.05, 0.04])}, ["pod"])
    assert multipliers["pod"] == pytest.approx(1.0)


def test_drawdown_breaker_multiplier_is_never_zero() -> None:
    with pytest.raises(ValueError):
        _DrawdownBreaker(cut_multiplier=0.0)


def test_decide_gives_tail_hedge_pod_a_fixed_budget_regardless_of_history() -> None:
    """The tail-hedge pod's allocation must always be exactly
    `TAIL_HEDGE_BUDGET_FRACTION`, carved out before the (now
    risk-parity-upgraded) allocation across the other pods runs, not a
    trailing-performance-sized share like every other pod gets."""
    pod_a = _stub_manager("pod_a", {"AAPL": 1.0, "MSFT": 0.0})
    pod_b = _stub_manager("pod_b", {"AAPL": 0.0, "MSFT": 1.0})
    tail_hedge = _stub_manager(TAIL_HEDGE_MANAGER_ID, {"AAPL": -1.0, "MSFT": 0.0})
    commander = Commander(
        managers=[pod_a, pod_b, tail_hedge],
        weighting_strategy=_FixedPodWeights(
            {"pod_a": 1.0, "pod_b": 1.0, TAIL_HEDGE_MANAGER_ID: 1.0}
        ),
    )

    decisions = commander.decide(MARKET_DATA, AS_OF, _scoreboard())

    by_asset = {decision.asset: decision.target_weight for decision in decisions}
    # pod_a/pod_b bootstrap to an equal 0.5/0.5 split (no history yet),
    # then renormalized to (1 - settings.commander.tail_hedge_budget_fraction); tail_hedge
    # gets exactly settings.commander.tail_hedge_budget_fraction regardless of its own
    # (bearish, -1.0) position.
    remaining_gross = 1.0 - settings.commander.tail_hedge_budget_fraction
    expected_aapl = 0.5 * remaining_gross - settings.commander.tail_hedge_budget_fraction
    assert by_asset["AAPL"] == pytest.approx(expected_aapl, abs=1e-6)


def _returns_frame(returns: dict[str, list[float]], start: str = "2023-01-02") -> pd.DataFrame:
    periods = len(next(iter(returns.values())))
    dates = pd.bdate_range(start, periods=periods)
    return pd.concat(
        {
            asset: pd.DataFrame({"close": [100.0] * periods, "log_return": series}, index=dates)
            for asset, series in returns.items()
        },
        axis=1,
    )


def test_apply_firm_limits_clips_each_single_name() -> None:
    with temporary_overrides({"firm.max_single_name_weight": 0.2}):
        limited = _apply_firm_limits({"AAPL": 0.5, "MSFT": -0.1}, MARKET_DATA, AS_OF)

    assert limited == pytest.approx({"AAPL": 0.2, "MSFT": -0.1})


def test_apply_firm_limits_scales_the_book_down_to_the_gross_cap() -> None:
    with temporary_overrides({"firm.max_gross_exposure": 1.0}):
        limited = _apply_firm_limits({"AAPL": 1.5, "MSFT": -0.5}, MARKET_DATA, AS_OF)

    assert sum(abs(w) for w in limited.values()) == pytest.approx(1.0)
    # Proportional: the long/short ratio survives the scale-down.
    assert limited["AAPL"] / limited["MSFT"] == pytest.approx(-3.0)


def test_apply_firm_limits_scales_the_book_down_to_the_var_limit() -> None:
    """Two assets with a known, deterministic return history: the firm
    VaR is the normal quantile times the Ledoit-Wolf portfolio std, and
    a breach scales the whole book by exactly max_daily_var / VaR."""
    rng = np.random.default_rng(seed=7)
    periods = 300
    returns = {
        "AAPL": list(rng.normal(0.0, 0.02, periods)),
        "MSFT": list(rng.normal(0.0, 0.01, periods)),
    }
    market_data = _returns_frame(returns)
    as_of = market_data.index[-1]
    weights = {"AAPL": 1.0, "MSFT": 1.0}

    trailing = market_data.tail(250)
    cov = ledoit_wolf_shrinkage(
        np.column_stack([trailing["AAPL"]["log_return"], trailing["MSFT"]["log_return"]])
    )
    w = np.array([1.0, 1.0])
    expected_var = -norm.ppf(1 - 0.95) * float(np.sqrt(w @ cov @ w))
    assert expected_var > 0.01  # the limit below must actually bind

    with temporary_overrides({"firm.max_daily_var": 0.01, "firm.var_confidence_level": 0.95}):
        limited = _apply_firm_limits(weights, market_data, as_of)

    expected_scale = 0.01 / expected_var
    assert limited["AAPL"] == pytest.approx(expected_scale)
    assert limited["MSFT"] == pytest.approx(expected_scale)


def test_apply_firm_limits_skips_var_when_history_is_thin() -> None:
    """Fewer than min_observations_for_covariance + n_assets observations:
    the VaR step is skipped rather than acting on a degenerate covariance,
    so a book that would otherwise breach the limit passes through."""
    periods = settings.portfolio.min_observations_for_covariance + 2 - 1
    market_data = _returns_frame({"AAPL": [0.05, -0.05] * periods, "MSFT": [0.05, -0.05] * periods})
    market_data = market_data.iloc[:periods]

    with temporary_overrides({"firm.max_daily_var": 1e-6}):
        limited = _apply_firm_limits({"AAPL": 1.0, "MSFT": 1.0}, market_data, market_data.index[-1])

    assert limited == pytest.approx({"AAPL": 1.0, "MSFT": 1.0})


def test_apply_firm_limits_scales_the_book_down_to_the_beta_limit() -> None:
    """AAPL is built with beta ~2x SPY, MSFT ~0.5x SPY: the book beta at
    equal weights comes out well above the default 1.0 cap, and the whole
    book must be scaled down proportionally by cap / |book beta|."""
    rng = np.random.default_rng(seed=3)
    periods = 300
    spy = rng.normal(0.0, 0.01, periods)
    aapl = 2.0 * spy + rng.normal(0.0, 0.0005, periods)
    msft = 0.5 * spy + rng.normal(0.0, 0.0005, periods)
    market_data = _returns_frame_with_spy({"AAPL": list(aapl), "MSFT": list(msft)}, list(spy))
    as_of = market_data.index[-1]
    weights = {"AAPL": 1.0, "MSFT": 1.0}

    trailing = market_data.tail(250)
    merged = np.column_stack(
        [
            trailing[("AAPL", "log_return")],
            trailing[("MSFT", "log_return")],
            trailing[("SPY", "log_return")],
        ]
    )
    cov = ledoit_wolf_shrinkage(merged)
    beta_aapl = cov[0, 2] / cov[2, 2]
    beta_msft = cov[1, 2] / cov[2, 2]
    book_beta = weights["AAPL"] * beta_aapl + weights["MSFT"] * beta_msft
    assert abs(book_beta) > 1.0  # the default cap below must actually bind

    with temporary_overrides(
        {
            "firm.max_single_name_weight": 100.0,
            "firm.max_gross_exposure": 1000.0,
            "firm.max_daily_var": 1000.0,
            "firm.max_stress_loss": 1000.0,
        }
    ):
        limited = _apply_firm_limits(weights, market_data, as_of)

    expected_scale = 1.0 / abs(book_beta)
    assert limited["AAPL"] == pytest.approx(expected_scale)
    assert limited["MSFT"] == pytest.approx(expected_scale)
    scaled_book_beta = limited["AAPL"] * beta_aapl + limited["MSFT"] * beta_msft
    assert abs(scaled_book_beta) == pytest.approx(1.0)


def test_apply_firm_limits_skips_beta_check_when_spy_is_absent() -> None:
    """No SPY series in this run's frame at all: the beta check must be
    skipped outright, with no crash, even with a cap tight enough that it
    would otherwise certainly bind."""
    rng = np.random.default_rng(seed=3)
    periods = 300
    returns = {
        "AAPL": list(rng.normal(0.0, 0.02, periods)),
        "MSFT": list(rng.normal(0.0, 0.01, periods)),
    }
    market_data = _returns_frame(returns)
    as_of = market_data.index[-1]
    weights = {"AAPL": 1.0, "MSFT": 1.0}

    with temporary_overrides(
        {
            "firm.max_single_name_weight": 100.0,
            "firm.max_gross_exposure": 1000.0,
            "firm.max_daily_var": 1000.0,
            "firm.max_stress_loss": 1000.0,
            "firm.max_portfolio_beta": 1e-9,
        }
    ):
        limited = _apply_firm_limits(weights, market_data, as_of)

    # No SPY column in `market_data`, so the tiny beta cap above never
    # gets a chance to bind at all.
    assert limited == pytest.approx(weights)


def test_apply_firm_limits_scales_the_book_down_to_the_stress_limit() -> None:
    """A correlation-breakdown stress test with a tight cap must scale the
    whole book down by cap / stressed_loss, independently recomputed from
    the same Ledoit-Wolf covariance the VaR check already built."""
    rng = np.random.default_rng(seed=9)
    periods = 300
    returns = {
        "AAPL": list(rng.normal(0.0, 0.015, periods)),
        "MSFT": list(rng.normal(0.0, 0.015, periods)),
    }
    market_data = _returns_frame(returns)
    as_of = market_data.index[-1]
    weights = {"AAPL": 1.0, "MSFT": 1.0}

    trailing = market_data.tail(250)
    merged = np.column_stack([trailing[("AAPL", "log_return")], trailing[("MSFT", "log_return")]])
    cov = ledoit_wolf_shrinkage(merged)
    w = np.array([1.0, 1.0])
    variances = np.diag(cov)
    corr = cov / np.sqrt(np.outer(variances, variances))
    shocked_corr = np.sign(corr) * np.maximum(np.abs(corr), 0.90)
    np.fill_diagonal(shocked_corr, 1.0)
    stressed_variances = variances * 2.0**2
    stressed_cov = shocked_corr * np.sqrt(np.outer(stressed_variances, stressed_variances))
    np.fill_diagonal(stressed_cov, stressed_variances)
    stressed_std = float(np.sqrt(w @ stressed_cov @ w))
    expected_stressed_loss = value_at_risk(merged @ w, 0.95, mean=0.0, volatility=stressed_std)
    assert expected_stressed_loss > 0.03  # the cap below must actually bind

    with temporary_overrides(
        {
            "firm.max_single_name_weight": 100.0,
            "firm.max_gross_exposure": 1000.0,
            "firm.max_daily_var": 1000.0,
            "firm.max_stress_loss": 0.03,
        }
    ):
        limited = _apply_firm_limits(weights, market_data, as_of)

    expected_scale = 0.03 / expected_stressed_loss
    assert limited["AAPL"] == pytest.approx(expected_scale)
    assert limited["MSFT"] == pytest.approx(expected_scale)


def test_apply_firm_limits_stress_test_does_not_bind_for_a_low_vol_low_correlation_book() -> None:
    """A low-vol, low-correlation book stays well under even a 2x-vol,
    0.90-correlation stress scenario, so the stress test must not touch
    the book at all."""
    rng = np.random.default_rng(seed=21)
    periods = 300
    returns = {
        "AAPL": list(rng.normal(0.0, 0.001, periods)),
        "MSFT": list(rng.normal(0.0, 0.001, periods)),
    }
    market_data = _returns_frame(returns)
    as_of = market_data.index[-1]
    weights = {"AAPL": 0.2, "MSFT": 0.2}

    limited = _apply_firm_limits(weights, market_data, as_of)

    assert limited == pytest.approx(weights)


def test_apply_firm_limits_skips_beta_and_stress_when_history_is_thin() -> None:
    """Fewer than min_observations_for_covariance + n_assets observations:
    both the new beta and stress checks must be skipped together, exactly
    like the existing VaR check, even with caps tight enough to certainly
    bind given real history."""
    periods = settings.portfolio.min_observations_for_covariance + 2 - 1
    spy_returns = [0.05, -0.05] * periods
    market_data = _returns_frame_with_spy(
        {"AAPL": [0.05, -0.05] * periods, "MSFT": [0.05, -0.05] * periods}, spy_returns
    )
    market_data = market_data.iloc[:periods]

    with temporary_overrides({"firm.max_portfolio_beta": 1e-9, "firm.max_stress_loss": 1e-9}):
        limited = _apply_firm_limits({"AAPL": 1.0, "MSFT": 1.0}, market_data, market_data.index[-1])

    assert limited == pytest.approx({"AAPL": 1.0, "MSFT": 1.0})


def _returns_frame_with_spy(
    returns: dict[str, list[float]], spy_returns: list[float], start: str = "2023-01-02"
) -> pd.DataFrame:
    periods = len(spy_returns)
    dates = pd.bdate_range(start, periods=periods)
    frames = {
        asset: pd.DataFrame({"close": [100.0] * periods, "log_return": series}, index=dates)
        for asset, series in returns.items()
    }
    frames["SPY"] = pd.DataFrame(
        {"close": [400.0] * periods, "log_return": spy_returns}, index=dates
    )
    return pd.concat(frames, axis=1)


def test_decide_leaves_the_tail_hedge_budget_untouched_by_firm_scaling() -> None:
    """The gross limit binds on the core book, which is scaled down; the
    tail-hedge pod's fixed budget is added back afterwards at full size."""
    pod_a = _stub_manager("pod_a", {"AAPL": 1.0, "MSFT": 1.0})
    tail_hedge = _stub_manager(TAIL_HEDGE_MANAGER_ID, {"AAPL": -1.0, "MSFT": 0.0})
    commander = Commander(
        managers=[pod_a, tail_hedge],
        weighting_strategy=_FixedPodWeights({"pod_a": 1.0, TAIL_HEDGE_MANAGER_ID: 1.0}),
    )
    hedge_budget = settings.commander.tail_hedge_budget_fraction
    core_gross = 2.0 * (1.0 - hedge_budget)

    with temporary_overrides({"firm.max_gross_exposure": core_gross / 2.0}):
        decisions = commander.decide(MARKET_DATA, AS_OF, _scoreboard())

    by_asset = {decision.asset: decision.target_weight for decision in decisions}
    core_each = (1.0 - hedge_budget) / 2.0
    assert by_asset["MSFT"] == pytest.approx(core_each)
    # AAPL carries the full, unscaled hedge on top of its halved core weight.
    assert by_asset["AAPL"] == pytest.approx(core_each - hedge_budget)


def test_risk_state_round_trips_through_json_and_restores_breaker_and_history() -> None:
    dates = pd.bdate_range("2024-01-02", periods=3)
    market_data = _returns_frame(
        {"AAPL": [0.0, 0.01, -0.12], "MSFT": [0.0, 0.0, 0.0]}, "2024-01-02"
    )
    pod_a = _stub_manager("pod_a", {"AAPL": 1.0, "MSFT": 0.0})
    pod_b = _stub_manager("pod_b", {"AAPL": 0.0, "MSFT": 1.0})
    commander = Commander(
        managers=[pod_a, pod_b],
        weighting_strategy=_FixedPodWeights({"pod_a": 1.0, "pod_b": 1.0}),
    )
    for as_of in dates:
        commander.decide(market_data, as_of, _scoreboard())

    state = commander.risk_state()
    assert state.tripped_pods == ["pod_a"]
    assert [d for d, _ in state.pod_return_history["pod_a"]] == [d.date() for d in dates[1:]]
    assert state.last_pod_weights["pod_a"] == {"AAPL": 1.0, "MSFT": 0.0}

    restored_state = CommanderRiskState.model_validate(state.model_dump(mode="json"))
    assert restored_state == state

    fresh = Commander(
        managers=[pod_a, pod_b],
        weighting_strategy=_FixedPodWeights({"pod_a": 1.0, "pod_b": 1.0}),
    )
    fresh.restore_risk_state(restored_state)
    assert fresh.risk_state() == state

    # The restored breaker still has pod_a cut: an unchanged drawdown
    # keeps the multiplier at the cut level, exactly as the original would.
    multipliers = fresh.drawdown_breaker.multipliers(
        {"pod_a": pd.Series([r for _, r in state.pod_return_history["pod_a"]])}, ["pod_a"]
    )
    assert multipliers["pod_a"] == pytest.approx(settings.commander.drawdown_cut_multiplier)


# ---------------------------------------------------------------------------
# Option rows: kept as their own decisions, delta-adjusted for the firm
# limits, short premium capped, the tail hedge's puts passed through.
# ---------------------------------------------------------------------------

CALL_ID = "AAPL 2024-07-19 C 100.00"
PUT_ID = "AAPL 2024-07-19 P 100.00"
HEDGE_PUT_ID = "AAPL 2024-07-19 P 95.00"
MID = 5.0
DELTA = 0.5


class _OptionStubManager(_StubManager):
    """A stub pod that also emits caller-fixed option rows (contract id
    to premium weight) alongside its equity decisions."""

    fixed_option_decisions: dict[str, float] = Field(default_factory=dict)

    def construct_portfolio(self, assets, market_data, as_of_date, scoreboard):
        return super().construct_portfolio(assets, market_data, as_of_date, scoreboard) + [
            PortfolioDecision(
                asset=contract_id,
                as_of_date=as_of_date.date(),
                target_weight=weight,
                contributing_signals=[],
                instrument=OptionContract.parse(contract_id),
            )
            for contract_id, weight in self.fixed_option_decisions.items()
        ]


def _option_stub_manager(
    agent_id: str, fixed_decisions: dict[str, float], option_decisions: dict[str, float]
) -> _OptionStubManager:
    return _OptionStubManager(
        agent_id=agent_id,
        managed_analysts=[],
        weighting_strategy=_FixedPodWeights({}),
        portfolio_constructor=PortfolioConstructor(),
        fixed_decisions=fixed_decisions,
        fixed_option_decisions=option_decisions,
    )


def _stub_mark(contract: OptionContract, as_of_date, spot: float) -> OptionMark:
    return OptionMark(
        contract=contract,
        as_of_date=as_of_date,
        snapshot_date=as_of_date,
        mid=MID,
        bid=MID - 0.1,
        ask=MID + 0.1,
        implied_volatility=0.2,
        delta=DELTA if contract.option_type == "call" else -DELTA,
        gamma=0.0,
        vega=0.0,
        theta=0.0,
        is_model_mark=False,
        is_expired=False,
        tradeable=True,
    )


def test_decide_keeps_option_rows_with_their_instrument(monkeypatch) -> None:
    monkeypatch.setattr(option_marks, "load_option_mark", lambda *_: None)
    pod = _option_stub_manager("pod_a", {"AAPL": 0.5, "MSFT": 0.0}, {CALL_ID: 0.1})
    commander = Commander(managers=[pod], weighting_strategy=_FixedPodWeights({"pod_a": 1.0}))

    decisions = commander.decide(MARKET_DATA, AS_OF, _scoreboard())

    by_asset = {decision.asset: decision for decision in decisions}
    assert set(by_asset) == {"AAPL", "MSFT", CALL_ID}
    assert by_asset["AAPL"].target_weight == pytest.approx(0.5)
    assert by_asset["AAPL"].instrument is None
    assert by_asset[CALL_ID].instrument == OptionContract.parse(CALL_ID)
    assert by_asset[CALL_ID].target_weight == pytest.approx(0.1)


def test_decide_delta_adjusts_option_rows_for_the_firm_limits(monkeypatch) -> None:
    """The single-name line is checked against equity plus the option's
    delta-equivalent (`weight x delta x spot / mid`), and the scale-down
    it forces applies to the option row as well as the equity row."""
    monkeypatch.setattr(option_marks, "load_option_mark", _stub_mark)
    pod = _option_stub_manager("pod_a", {"AAPL": 0.4, "MSFT": 0.0}, {CALL_ID: 0.1})
    commander = Commander(managers=[pod], weighting_strategy=_FixedPodWeights({"pod_a": 1.0}))
    spot = float(MARKET_DATA[("AAPL", "close")].iloc[-1])
    equivalent = 0.4 + 0.1 * DELTA * spot / MID
    limit = equivalent / 2.0

    with temporary_overrides({"firm.max_single_name_weight": limit}):
        decisions = commander.decide(MARKET_DATA, AS_OF, _scoreboard())

    by_asset = {decision.asset: decision.target_weight for decision in decisions}
    assert by_asset["AAPL"] == pytest.approx(0.4 / 2.0)
    assert by_asset[CALL_ID] == pytest.approx(0.1 / 2.0)


def test_decide_scales_short_option_rows_down_to_the_premium_cap(monkeypatch) -> None:
    monkeypatch.setattr(option_marks, "load_option_mark", lambda *_: None)
    pod = _option_stub_manager(
        "pod_a", {"AAPL": 0.0, "MSFT": 0.0}, {CALL_ID: -0.04, PUT_ID: -0.04, HEDGE_PUT_ID: 0.1}
    )
    commander = Commander(managers=[pod], weighting_strategy=_FixedPodWeights({"pod_a": 1.0}))

    with temporary_overrides({"options.max_short_premium_fraction": 0.05}):
        decisions = commander.decide(MARKET_DATA, AS_OF, _scoreboard())

    by_asset = {decision.asset: decision.target_weight for decision in decisions}
    assert by_asset[CALL_ID] == pytest.approx(-0.025)
    assert by_asset[PUT_ID] == pytest.approx(-0.025)
    assert by_asset[HEDGE_PUT_ID] == pytest.approx(0.1)


def test_decide_passes_the_tail_hedge_put_through_untouched(monkeypatch) -> None:
    """The hedge pod sizes its put premium against firm NAV, so its
    option row is neither scaled by the pod's budget allocation nor by a
    gross breach the core book caused."""
    monkeypatch.setattr(option_marks, "load_option_mark", _stub_mark)
    pod_a = _stub_manager("pod_a", {"AAPL": 1.0, "MSFT": 1.0})
    tail_hedge = _option_stub_manager(
        TAIL_HEDGE_MANAGER_ID, {"AAPL": 0.0, "MSFT": 0.0}, {HEDGE_PUT_ID: 0.01}
    )
    commander = Commander(
        managers=[pod_a, tail_hedge],
        weighting_strategy=_FixedPodWeights({"pod_a": 1.0, TAIL_HEDGE_MANAGER_ID: 1.0}),
    )
    hedge_budget = settings.commander.tail_hedge_budget_fraction
    core_gross = 2.0 * (1.0 - hedge_budget)

    with temporary_overrides({"firm.max_gross_exposure": core_gross / 2.0}):
        decisions = commander.decide(MARKET_DATA, AS_OF, _scoreboard())

    by_asset = {decision.asset: decision for decision in decisions}
    assert by_asset["AAPL"].target_weight == pytest.approx((1.0 - hedge_budget) / 2.0)
    assert by_asset[HEDGE_PUT_ID].target_weight == pytest.approx(0.01)
    assert by_asset[HEDGE_PUT_ID].instrument == OptionContract.parse(HEDGE_PUT_ID)


def test_pod_return_tracker_counts_option_mark_returns() -> None:
    """A pod holding an option contract earns that contract's mark-to-mark
    return, supplied by the engine, weighted by yesterday's premium weight."""
    from agentic_trading.agents.commander import _PodReturnTracker
    from agentic_trading.agents.schemas import OptionContract

    contract = OptionContract(
        underlying="AAPL", expiry=date(2024, 3, 15), strike=185.0, option_type="call"
    )
    dates = pd.bdate_range("2024-01-02", periods=2)
    market_data = pd.concat(
        {"AAPL": pd.DataFrame({"close": [100.0, 100.0], "log_return": [0.0, 0.0]}, index=dates)},
        axis=1,
    )
    tracker = _PodReturnTracker(lookback_days=30)
    day_one = [
        PortfolioDecision(
            asset=contract.contract_id,
            as_of_date=dates[0].date(),
            target_weight=0.02,
            instrument=contract,
            contributing_signals=[],
        )
    ]
    tracker.record_and_get_history([("cvar_manager", day_one)], market_data, dates[0])
    history = tracker.record_and_get_history(
        [("cvar_manager", day_one)],
        market_data,
        dates[1],
        instrument_returns={contract.contract_id: 0.5},
    )
    assert history["cvar_manager"].iloc[-1] == pytest.approx(0.02 * 0.5)
