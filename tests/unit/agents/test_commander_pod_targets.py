# test_commander_pod_targets.py

"""AT-125: `Commander.pod_targets()` splits each final target weight into one row per pod
whose rows sum to the firm's target after firm limits. Firm limits scale every core pod's
contribution to an instrument by the factor they applied to that instrument; the tail hedge's
rows pass at face value, as the Commander books them; the leverage clamp scales all of an
instrument's rows alike. Option rows carry their delta-equivalent weight.

An instrument without a close on the decision day that has a decided target keeps each pod's
last decided target (edgelab's rule (c)): no analyst is asked about it, the other instruments
are sized as if it were present, the firm limits count it, and both trackers keep it. One never
decided gets no decision. An analyst asked about an instrument sees only the rows on which it
has a close."""

from datetime import date

import numpy as np
import pandas as pd
import pytest
from pydantic import Field, PrivateAttr

from agentic_trading.agents.base import AnalystAgent, ManagerAgent
from agentic_trading.agents.commander import Commander
from agentic_trading.agents.portfolio_construction import PortfolioConstructor
from agentic_trading.agents.schemas import (
    MAX_LEVERAGE,
    AgentSignal,
    ManagerSignal,
    OptionContract,
    PortfolioDecision,
)
from agentic_trading.config import settings, temporary_overrides
from agentic_trading.data.loaders import option_marks
from agentic_trading.data.loaders.option_marks import OptionMark
from agentic_trading.scoring.scoreboard import PerformanceScoreboard
from agentic_trading.scoring.track_record import TrackRecord
from agentic_trading.scoring.weighting import EqualWeight, WeightingStrategy

_INDEX = pd.bdate_range("2016-01-04", periods=6)
_DAY = _INDEX[-1]
_CALL = OptionContract(
    underlying="AAPL", expiry=date(2016, 2, 19), strike=100.0, option_type="call"
)
_PUT = OptionContract(underlying="SPY", expiry=date(2016, 3, 18), strike=180.0, option_type="put")


class _FixedPodWeights(WeightingStrategy):
    def __init__(self, weights: dict[str, float]) -> None:
        self.weights = weights

    def get_weights(self, scoreboard, source_ids, as_of_date):
        return {source_id: self.weights.get(source_id, 0.0) for source_id in source_ids}


class _ZeroConstructor(PortfolioConstructor):
    def construct(self, views, returns_history, as_of_date, previous_weights=None):
        return dict.fromkeys(views, 0.0)


class _ScriptedManager(ManagerAgent):
    """Returns `weights` for the assets it is given, each with one active signal, plus option
    rows of `option_weights` (contract id to premium weight)."""

    weights: dict[str, float] = Field(default_factory=dict)
    option_weights: dict[str, float] = Field(default_factory=dict)

    def scale_factor(self, signal, market_data, as_of_date) -> float:
        return 1.0

    def construct_portfolio(self, assets, market_data, as_of_date, scoreboard):
        day = as_of_date.date()

        def signals(asset: str) -> list[ManagerSignal]:
            return [
                ManagerSignal(
                    manager_id=self.agent_id,
                    source_agent_id=f"{self.agent_id}_analyst",
                    asset=asset,
                    as_of_date=day,
                    direction="long",
                    confidence=0.5,
                    raw_confidence=0.5,
                    scale_applied=1.0,
                    rationale="stub",
                )
            ]

        decisions = [
            PortfolioDecision(
                asset=asset,
                as_of_date=day,
                target_weight=self.weights.get(asset, 0.0),
                contributing_signals=signals(asset),
            )
            for asset in assets
        ]
        for contract_id, weight in self.option_weights.items():
            decisions.append(
                PortfolioDecision(
                    asset=contract_id,
                    as_of_date=day,
                    target_weight=weight,
                    contributing_signals=signals(contract_id),
                    instrument=OptionContract.parse(contract_id),
                )
            )
        return decisions


def _manager(agent_id: str, **fields: object) -> _ScriptedManager:
    return _ScriptedManager(
        agent_id=agent_id,
        managed_analysts=[],
        weighting_strategy=EqualWeight(),
        portfolio_constructor=_ZeroConstructor(),
        **fields,
    )


def _frame(assets: tuple[str, ...] = ("AAPL", "SPY")) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    columns = {}
    for asset in assets:
        close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, len(_INDEX))))
        columns[(asset, "close")] = close
        columns[(asset, "open")] = close
        columns[(asset, "log_return")] = np.log(close / np.roll(close, 1))
    frame = pd.DataFrame(columns, index=_INDEX)
    for asset in assets:
        frame.loc[_INDEX[0], (asset, "log_return")] = np.nan
    return frame


def _scoreboard() -> PerformanceScoreboard:
    return PerformanceScoreboard(TrackRecord())


def _commander(*managers: ManagerAgent) -> Commander:
    return Commander(
        managers=list(managers),
        weighting_strategy=_FixedPodWeights({manager.agent_id: 1.0 for manager in managers}),
    )


def _by_pod(commander: Commander) -> dict[tuple[str, str], float]:
    return {
        (target.pod_id, target.asset): target.target_weight for target in commander.pod_targets()
    }


def _net(commander: Commander) -> dict[str, float]:
    net: dict[str, float] = {}
    for target in commander.pod_targets():
        net[target.asset] = net.get(target.asset, 0.0) + target.target_weight
    return net


def test_pod_rows_sum_to_the_target_after_firm_limits() -> None:
    """With the single-name and gross limits binding, every core pod's row is its contribution
    times the instrument's limit factor, and the rows sum to the firm's target."""
    commander = _commander(
        _manager("a_manager", weights={"AAPL": 0.6, "SPY": 0.4}),
        _manager("b_manager", weights={"AAPL": -0.2, "SPY": 0.8}),
    )
    with temporary_overrides({"firm.max_gross_exposure": 0.4}):
        decisions = commander.decide(_frame(), _DAY, _scoreboard())
    targets = {decision.asset: decision.target_weight for decision in decisions}
    rows = _by_pod(commander)
    contributions = commander.pod_contributions()

    assert targets["SPY"] == pytest.approx(0.25 * 0.4 / 0.45)
    for asset, target in targets.items():
        assert _net(commander)[asset] == pytest.approx(target, abs=1e-12)
        factors = {
            rows[(pod, asset)] / contributions[pod][asset] for pod in ("a_manager", "b_manager")
        }
        assert max(factors) == pytest.approx(min(factors), rel=1e-12)
    assert rows[("a_manager", "AAPL")] == pytest.approx(0.3 * 0.4 / 0.45)


def test_tail_hedge_rows_pass_at_face_value() -> None:
    """The tail hedge's SPY row is its contribution, unscaled; the core pod's row takes the
    whole limit, and the two sum to the firm's target."""
    commander = _commander(
        _manager("a_manager", weights={"SPY": 0.6}),
        _manager("tail_hedge_manager", weights={"SPY": -0.5}),
    )
    with temporary_overrides({"firm.max_single_name_weight": 0.1}):
        decisions = commander.decide(_frame(("SPY",)), _DAY, _scoreboard())
    rows = _by_pod(commander)
    budget = settings.commander.tail_hedge_budget_fraction

    assert rows[("tail_hedge_manager", "SPY")] == pytest.approx(-0.5 * budget)
    assert rows[("a_manager", "SPY")] == pytest.approx(0.1)
    assert decisions[0].target_weight == pytest.approx(0.1 - 0.5 * budget, abs=1e-12)
    assert _net(commander)["SPY"] == pytest.approx(decisions[0].target_weight, abs=1e-12)


def test_option_rows_carry_their_delta_equivalent_weight(monkeypatch: pytest.MonkeyPatch) -> None:
    """An option pod row is the scaled premium weight; its delta_equiv_weight is that weight x
    delta x spot / mid, negative for a long put; an equity row has none."""

    def mark(contract: OptionContract, as_of: date, spot: float) -> OptionMark:
        delta = 0.5 if contract.option_type == "call" else -0.4
        return OptionMark(
            contract=contract,
            as_of_date=as_of,
            snapshot_date=as_of,
            mid=2.0,
            bid=1.9,
            ask=2.1,
            implied_volatility=0.2,
            delta=delta,
            gamma=0.0,
            vega=0.0,
            theta=0.0,
            is_model_mark=False,
            is_expired=False,
            tradeable=True,
        )

    monkeypatch.setattr(option_marks, "load_option_mark", mark)
    commander = _commander(
        _manager("a_manager", weights={"AAPL": 0.1}, option_weights={_CALL.contract_id: 0.02}),
        _manager("tail_hedge_manager", option_weights={_PUT.contract_id: 0.004}),
    )
    frame = _frame()
    decisions = commander.decide(frame, _DAY, _scoreboard())
    targets = {target.asset: target for target in commander.pod_targets()}
    final = {decision.asset: decision.target_weight for decision in decisions}
    aapl_spot = float(frame.loc[_DAY, ("AAPL", "close")])
    spy_spot = float(frame.loc[_DAY, ("SPY", "close")])

    call = targets[_CALL.contract_id]
    put = targets[_PUT.contract_id]
    assert call.instrument == _CALL and call.pod_id == "a_manager"
    assert call.target_weight == pytest.approx(final[_CALL.contract_id])
    assert call.delta_equiv_weight == pytest.approx(call.target_weight * 0.5 * aapl_spot / 2.0)
    assert put.target_weight == pytest.approx(0.004)
    assert put.delta_equiv_weight == pytest.approx(0.004 * -0.4 * spy_spot / 2.0)
    assert put.delta_equiv_weight is not None and put.delta_equiv_weight < 0.0
    assert targets["AAPL"].delta_equiv_weight is None


def test_option_row_without_a_mark_has_no_delta(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no mark for the contract, the option row's delta_equiv_weight is None."""
    monkeypatch.setattr(option_marks, "load_option_mark", lambda *args, **kwargs: None)
    commander = _commander(_manager("a_manager", option_weights={_CALL.contract_id: 0.02}))
    commander.decide(_frame(), _DAY, _scoreboard())
    call = next(t for t in commander.pod_targets() if t.instrument is not None)

    assert call.target_weight == pytest.approx(0.02)
    assert call.delta_equiv_weight is None


def test_pod_rows_carry_each_pods_own_signals() -> None:
    """A pod row carries the signals of its own pod only."""
    commander = _commander(
        _manager("a_manager", weights={"SPY": 0.2}), _manager("b_manager", weights={"SPY": 0.1})
    )
    commander.decide(_frame(("SPY",)), _DAY, _scoreboard())

    for target in commander.pod_targets():
        assert {signal.manager_id for signal in target.contributing_signals} == {target.pod_id}
        assert target.as_of_date == _DAY.date()


class _RecordingAnalyst(AnalystAgent):
    """Long at 0.5 on every asset; records the assets it is asked about and whether the frame
    it saw had a missing close for that asset."""

    agent_id: str = "recording_analyst"
    _asked: list[tuple[str, bool]] = PrivateAttr(default_factory=list)

    def generate_signal(
        self, asset: str, market_data: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> AgentSignal:
        closes = market_data[(asset, "close")]
        self._asked.append((asset, bool(closes.loc[closes.index <= as_of_date].isna().any())))
        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction="long",
            confidence=0.5,
            rationale="stub",
        )


class _HalfViewConstructor(PortfolioConstructor):
    def construct(self, views, returns_history, as_of_date, previous_weights=None):
        return {asset: 0.5 * view.net_signal for asset, view in views.items()}


def _real_manager(analyst: AnalystAgent) -> ManagerAgent:
    """A pod running the base `ManagerAgent.construct_portfolio` over `analyst`."""

    class _PassThroughManager(ManagerAgent):
        def scale_factor(self, signal, market_data, as_of_date) -> float:
            return 1.0

    return _PassThroughManager(
        agent_id="pass_through_manager",
        managed_analysts=[analyst],
        weighting_strategy=EqualWeight(),
        portfolio_constructor=_HalfViewConstructor(),
    )


def test_no_bar_instrument_carries_its_last_targets() -> None:
    """BTC-USD has no close before the last day and ETH-USD misses the last day. BTC-USD, never
    decided, gets no decision or pod row while it has no close; ETH-USD on its missing day keeps
    each pod's last decided target, carried and held, and no analyst is asked about it. On the
    last day BTC-USD is asked about, on a frame holding only its closes."""
    frame = _frame(("SPY", "BTC-USD", "ETH-USD"))
    frame.loc[_INDEX[:-1], ("BTC-USD", ["close", "open", "log_return"])] = np.nan
    frame.loc[_INDEX[-1], ("BTC-USD", "log_return")] = np.nan
    frame.loc[_INDEX[-1], ("ETH-USD", ["close", "open", "log_return"])] = np.nan
    analyst = _RecordingAnalyst()
    commander = Commander(managers=[_real_manager(analyst)], weighting_strategy=EqualWeight())

    earlier = commander.decide(frame.loc[: _INDEX[-2]], _INDEX[-2], _scoreboard())
    earlier_eth = [t for t in commander.pod_targets() if t.asset == "ETH-USD"]
    asked_before = list(analyst._asked)
    analyst._asked.clear()
    last = commander.decide(frame, _INDEX[-1], _scoreboard())
    last_eth = [t for t in commander.pod_targets() if t.asset == "ETH-USD"]

    assert [decision.asset for decision in earlier] == ["SPY", "ETH-USD"]
    assert {asset for asset, _ in asked_before} == {"SPY", "ETH-USD"}
    assert [decision.asset for decision in last] == ["SPY", "BTC-USD", "ETH-USD"]
    assert sorted(analyst._asked) == [("BTC-USD", False), ("SPY", False)]
    assert [(t.pod_id, t.target_weight) for t in last_eth] == [
        (t.pod_id, t.target_weight) for t in earlier_eth
    ]
    assert earlier_eth[0].target_weight != 0.0
    assert all(t.carried and t.held for t in last_eth)
    assert last[-1].target_weight == pytest.approx(earlier[-1].target_weight)
    assert [s.as_of_date for s in last_eth[0].contributing_signals] == [_INDEX[-2].date()]


class _NormalisingConstructor(PortfolioConstructor):
    """Splits a gross of 0.8 across the views in proportion to their net signal, so dropping
    one view would inflate the others."""

    def construct(self, views, returns_history, as_of_date, previous_weights=None):
        total = sum(abs(view.net_signal) for view in views.values())
        if total == 0.0:
            return dict.fromkeys(views, 0.0)
        return {asset: 0.8 * view.net_signal / total for asset, view in views.items()}


class _ConfidenceAnalyst(AnalystAgent):
    """Long at a fixed confidence per asset."""

    agent_id: str = "confidence_analyst"
    confidence: dict[str, float] = Field(default_factory=dict)

    def generate_signal(
        self, asset: str, market_data: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> AgentSignal:
        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction="long",
            confidence=self.confidence.get(asset, 0.5),
            rationale="stub",
        )


def _normalising_commander() -> Commander:
    class _Pod(ManagerAgent):
        def scale_factor(self, signal, market_data, as_of_date) -> float:
            return 1.0

    pod = _Pod(
        agent_id="normalising_manager",
        managed_analysts=[_ConfidenceAnalyst(confidence={"SPY": 0.6, "AAPL": 0.3, "GLD": 0.9})],
        weighting_strategy=EqualWeight(),
        portfolio_constructor=_NormalisingConstructor(),
    )
    return Commander(managers=[pod], weighting_strategy=EqualWeight())


def _run_days(commander: Commander, frame: pd.DataFrame, days: int) -> list[PortfolioDecision]:
    decisions: list[PortfolioDecision] = []
    for day in frame.index[-days:]:
        decisions = commander.decide(frame.loc[:day], day, _scoreboard())
    return decisions


def test_a_gap_leaves_the_other_targets_unchanged() -> None:
    """On GLD's missing day, with no limit binding, every other instrument's decision and pod
    row equals the one of the same firm on a frame where GLD has its bar."""
    full = _frame(("SPY", "AAPL", "GLD"))
    gapped = full.copy()
    gapped.loc[_DAY, ("GLD", ["close", "open", "log_return"])] = np.nan
    with_bar, without_bar = _normalising_commander(), _normalising_commander()

    reference = _run_days(with_bar, full, 3)
    carried = _run_days(without_bar, gapped, 3)

    assert [d.asset for d in carried] == [d.asset for d in reference]
    for ours, theirs in zip(carried, reference, strict=True):
        assert ours.target_weight == pytest.approx(theirs.target_weight, abs=1e-15)
    assert _by_pod(without_bar) == pytest.approx(_by_pod(with_bar), abs=1e-15)


def test_the_firm_limits_count_a_carried_instrument() -> None:
    """With the gross limit binding on GLD's missing day, the book's gross, GLD's carried
    target included, stays within the limit and GLD keeps its share of it."""
    gapped = _frame(("SPY", "AAPL", "GLD"))
    gapped.loc[_DAY, ("GLD", ["close", "open", "log_return"])] = np.nan
    commander = _normalising_commander()
    _run_days(commander, gapped.loc[: _INDEX[-2]], 2)
    with temporary_overrides({"firm.max_gross_exposure": 0.2}):
        decisions = commander.decide(gapped, _DAY, _scoreboard())
    weights = {decision.asset: decision.target_weight for decision in decisions}

    assert sum(abs(weight) for weight in weights.values()) == pytest.approx(0.2)
    assert weights["GLD"] > 0.0
    assert _net(commander) == pytest.approx(weights, abs=1e-12)


def test_trackers_keep_a_no_bar_instrument() -> None:
    """On GLD's missing day the pod's previous weights and the pod-return tracker keep GLD at
    the pod's weight of the day before, so its return over the gap is counted next day."""
    gapped = _frame(("SPY", "AAPL", "GLD"))
    gapped.loc[_DAY, ("GLD", ["close", "open", "log_return"])] = np.nan
    commander = _normalising_commander()
    _run_days(commander, gapped.loc[: _INDEX[-2]], 2)
    before = commander.risk_state()
    commander.decide(gapped, _DAY, _scoreboard())
    after = commander.risk_state()
    pod = "normalising_manager"

    assert before.manager_states[pod].previous_weights["GLD"] != 0.0
    assert (
        after.manager_states[pod].previous_weights["GLD"]
        == (before.manager_states[pod].previous_weights["GLD"])
    )
    assert after.last_pod_weights[pod]["GLD"] == before.last_pod_weights[pod]["GLD"]
    assert {t.asset for t in after.last_targets["GLD"]} == {"GLD"}


def test_single_name_clip_scales_each_instrument_by_its_own_factor() -> None:
    """With only AAPL over the single-name limit, AAPL's rows take the clip's factor and SPY's
    rows stay at their contributions."""
    commander = _commander(
        _manager("a_manager", weights={"AAPL": 0.6, "SPY": 0.1}),
        _manager("b_manager", weights={"AAPL": 0.2, "SPY": 0.05}),
    )
    commander.decide(_frame(), _DAY, _scoreboard())
    rows = _by_pod(commander)
    contributions = commander.pod_contributions()
    aapl = sum(contributions[pod]["AAPL"] for pod in ("a_manager", "b_manager"))

    assert aapl > settings.firm.max_single_name_weight
    for pod in ("a_manager", "b_manager"):
        assert rows[(pod, "AAPL")] / contributions[pod]["AAPL"] == pytest.approx(
            settings.firm.max_single_name_weight / aapl
        )
        assert rows[(pod, "SPY")] == pytest.approx(contributions[pod]["SPY"])


def test_short_premium_cap_scales_the_short_option_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """A short option row beyond the short-premium cap is scaled to the cap, and the pod rows
    still sum to the firm's target."""
    monkeypatch.setattr(option_marks, "load_option_mark", lambda *args, **kwargs: None)
    commander = _commander(_manager("a_manager", option_weights={_CALL.contract_id: -0.4}))
    decisions = commander.decide(_frame(), _DAY, _scoreboard())
    final = {decision.asset: decision.target_weight for decision in decisions}
    (call,) = [t for t in commander.pod_targets() if t.instrument is not None]

    cap = settings.options.max_short_premium_fraction
    assert commander.pod_contributions()["a_manager"][_CALL.contract_id] < -cap
    assert call.target_weight == pytest.approx(-cap)
    assert final[_CALL.contract_id] == pytest.approx(call.target_weight, abs=1e-12)


class _RepeatedPutManager(_ScriptedManager):
    """A tail hedge emitting two rows of one put, each at `put_weight`."""

    put_weight: float = 0.0

    def construct_portfolio(self, assets, market_data, as_of_date, scoreboard):
        decisions = super().construct_portfolio(assets, market_data, as_of_date, scoreboard)
        row = PortfolioDecision(
            asset=_PUT.contract_id,
            as_of_date=as_of_date.date(),
            target_weight=self.put_weight,
            contributing_signals=[],
            instrument=_PUT,
        )
        return [*decisions, row, row.model_copy()]


def test_leverage_clamp_scales_every_row_of_a_clamped_instrument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A put the hedge holds beyond MAX_LEVERAGE is clamped, and its pod row with it."""
    monkeypatch.setattr(option_marks, "load_option_mark", lambda *args, **kwargs: None)
    hedge = _RepeatedPutManager(
        agent_id="tail_hedge_manager",
        managed_analysts=[],
        weighting_strategy=EqualWeight(),
        portfolio_constructor=_ZeroConstructor(),
        put_weight=0.6 * MAX_LEVERAGE,
    )
    commander = _commander(_manager("a_manager", weights={"SPY": 0.1}), hedge)
    decisions = commander.decide(_frame(("SPY",)), _DAY, _scoreboard())
    final = {decision.asset: decision.target_weight for decision in decisions}
    (put,) = [t for t in commander.pod_targets() if t.instrument is not None]

    assert commander.pod_contributions()["tail_hedge_manager"][_PUT.contract_id] == (
        pytest.approx(1.2 * MAX_LEVERAGE)
    )
    assert final[_PUT.contract_id] == pytest.approx(MAX_LEVERAGE)
    assert put.target_weight == pytest.approx(MAX_LEVERAGE)


def test_core_row_of_a_contract_the_hedge_holds_is_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """When a core pod and the tail hedge hold one put, the firm books it at the hedge's row:
    the core row is 0 and the hedge row its face value."""
    monkeypatch.setattr(option_marks, "load_option_mark", lambda *args, **kwargs: None)
    commander = _commander(
        _manager("a_manager", option_weights={_PUT.contract_id: 0.01}),
        _manager("tail_hedge_manager", option_weights={_PUT.contract_id: 0.004}),
    )
    decisions = commander.decide(_frame(("SPY",)), _DAY, _scoreboard())
    final = {decision.asset: decision.target_weight for decision in decisions}
    rows = _by_pod(commander)

    assert rows[("a_manager", _PUT.contract_id)] == 0.0
    assert rows[("tail_hedge_manager", _PUT.contract_id)] == pytest.approx(0.004)
    assert final[_PUT.contract_id] == pytest.approx(0.004)


def _golden_mark(contract: OptionContract, as_of: date, spot: float) -> OptionMark:
    return OptionMark(
        contract=contract,
        as_of_date=as_of,
        snapshot_date=as_of,
        mid=2.0,
        bid=1.9,
        ask=2.1,
        implied_volatility=0.2,
        delta=0.5 if contract.option_type == "call" else -0.4,
        gamma=0.0,
        vega=0.0,
        theta=0.0,
        is_model_mark=False,
        is_expired=False,
        tradeable=True,
    )


def golden_scenario(commander_class: type) -> list[list[tuple[str, float]]]:
    """Each day's decisions, as (asset, weight), of a firm with an analyst-driven pod, two
    scripted pods with options and a tail hedge, over four days of an all-finite frame with
    the gross limit binding."""
    commander = commander_class(
        managers=[
            _real_manager(_RecordingAnalyst()),
            _manager(
                "a_manager",
                weights={"AAPL": 0.5, "SPY": -0.3},
                option_weights={_CALL.contract_id: -0.08},
            ),
            _manager(
                "tail_hedge_manager",
                weights={"SPY": -0.4},
                option_weights={_PUT.contract_id: 0.004},
            ),
        ],
        weighting_strategy=_FixedPodWeights(
            {"pass_through_manager": 1.0, "a_manager": 2.0, "tail_hedge_manager": 1.0}
        ),
    )
    frame = _frame()
    days: list[list[tuple[str, float]]] = []
    with temporary_overrides({"firm.max_gross_exposure": 0.3}):
        for day in _INDEX[-4:]:
            decisions = commander.decide(frame.loc[:day], day, _scoreboard())
            days.append([(decision.asset, decision.target_weight) for decision in decisions])
    return days


# The decisions `golden_scenario` gives on the all-finite frame, pinned so that any change to
# the Commander's path for instruments that all have a bar shows here.
_GOLDEN: list[list[tuple[str, float]]] = [
    [
        ("AAPL", 0.09009369537874283),
        ("SPY", -0.1021445562671546),
        ("AAPL 2016-02-19 C 100.00", -0.011531993008479083),
        ("SPY 2016-03-18 P 180.00", 0.004),
    ],
    [
        ("AAPL", 0.09075784024789098),
        ("SPY", -0.1021445562671546),
        ("AAPL 2016-02-19 C 100.00", -0.011617003551730046),
        ("SPY 2016-03-18 P 180.00", 0.004),
    ],
    [
        ("AAPL", 0.09114859401313256),
        ("SPY", -0.1021445562671546),
        ("AAPL 2016-02-19 C 100.00", -0.01166702003368097),
        ("SPY 2016-03-18 P 180.00", 0.004),
    ],
    [
        ("AAPL", 0.09184560688351394),
        ("SPY", -0.1021445562671546),
        ("AAPL 2016-02-19 C 100.00", -0.011756237681089785),
        ("SPY 2016-03-18 P 180.00", 0.004),
    ],
]


def test_all_finite_decisions_are_pinned(monkeypatch: pytest.MonkeyPatch) -> None:
    """On a frame where every instrument has every bar, the decisions are the pinned ones."""
    monkeypatch.setattr(option_marks, "load_option_mark", _golden_mark)
    days = golden_scenario(Commander)

    assert [[asset for asset, _ in day] for day in days] == [
        [asset for asset, _ in day] for day in _GOLDEN
    ]
    for day, golden in zip(days, _GOLDEN, strict=True):
        assert [weight for _, weight in day] == pytest.approx(
            [weight for _, weight in golden], rel=1e-12, abs=1e-15
        )


def test_a_carried_target_is_held_only_while_no_limit_scales_it() -> None:
    """On GLD's missing day its carried row is held at the carried weight; when the gross limit
    scales it, the row is the scaled weight the book holds and is no longer held."""
    gapped = _frame(("SPY", "AAPL", "GLD"))
    gapped.loc[_DAY, ("GLD", ["close", "open", "log_return"])] = np.nan

    def gld_on_gap_day(overrides: dict[str, object]) -> tuple[float, float, bool]:
        commander = _normalising_commander()
        _run_days(commander, gapped.loc[: _INDEX[-2]], 2)
        (before,) = [t for t in commander.pod_targets() if t.asset == "GLD"]
        with temporary_overrides(overrides):
            commander.decide(gapped, _DAY, _scoreboard())
        (after,) = [t for t in commander.pod_targets() if t.asset == "GLD"]
        assert after.carried
        return before.target_weight, after.target_weight, after.held

    before, unscaled, held = gld_on_gap_day({})
    assert unscaled == before and held
    before, scaled, held = gld_on_gap_day({"firm.max_gross_exposure": 0.2})
    assert 0.0 < scaled < before and not held


class _OptionPod(_ScriptedManager):
    """A scripted pod that, like the CVaR pod, emits no option row on an underlying without a
    bar today."""

    def construct_portfolio(self, assets, market_data, as_of_date, scoreboard, *, carried=None):
        decisions = super().construct_portfolio(assets, market_data, as_of_date, scoreboard)
        return [
            decision
            for decision in decisions
            if decision.instrument is None or decision.instrument.underlying not in (carried or {})
        ]


def test_option_legs_over_a_gap_stay_in_the_pod_tracker(monkeypatch: pytest.MonkeyPatch) -> None:
    """On AAPL's missing day the pod's AAPL call stays in the pod-return tracker at the pod's
    weight of the day before, and in the firm book at its carried target."""
    monkeypatch.setattr(option_marks, "load_option_mark", lambda *args, **kwargs: None)
    frame = _frame()
    frame.loc[_DAY, ("AAPL", ["close", "open", "log_return"])] = np.nan
    pod = _OptionPod(
        agent_id="a_manager",
        managed_analysts=[],
        weighting_strategy=EqualWeight(),
        portfolio_constructor=_ZeroConstructor(),
        weights={"AAPL": 0.1, "SPY": 0.1},
        option_weights={_CALL.contract_id: 0.02},
    )
    commander = _commander(pod)
    commander.decide(frame.loc[: _INDEX[-2]], _INDEX[-2], _scoreboard())
    before = commander.risk_state().last_pod_weights["a_manager"][_CALL.contract_id]
    (carried_call,) = [t for t in commander.pod_targets() if t.asset == _CALL.contract_id]
    commander.decide(frame, _DAY, _scoreboard())
    after = commander.risk_state().last_pod_weights["a_manager"]
    (call,) = [t for t in commander.pod_targets() if t.asset == _CALL.contract_id]

    assert before == pytest.approx(0.02)
    assert after[_CALL.contract_id] == before
    assert call.carried and call.held
    assert call.target_weight == carried_call.target_weight
