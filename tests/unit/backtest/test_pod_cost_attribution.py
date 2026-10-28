# test_pod_cost_attribution.py

"""Per-pod attribution of fill costs, option premium and turnover: `Commander.pod_contributions`,
`BacktestEngine` and `BacktestResult.pod_costs`."""

from collections import defaultdict
from collections.abc import Iterator
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agentic_trading.agents.base import ManagerAgent
from agentic_trading.agents.commander import Commander
from agentic_trading.agents.portfolio_construction import PortfolioConstructor
from agentic_trading.agents.schemas import (
    BacktestResult,
    OptionContract,
    PodCostAttribution,
    PortfolioDecision,
    SizingDiagnostics,
)
from agentic_trading.backtest.engine import BacktestEngine
from agentic_trading.backtest.portfolio import Portfolio
from agentic_trading.config import settings, temporary_overrides
from agentic_trading.data.loaders.option_marks import OptionMark
from agentic_trading.scoring.scoreboard import PerformanceScoreboard
from agentic_trading.scoring.track_record import TrackRecord
from agentic_trading.scoring.weighting import WeightingStrategy

_INDEX = pd.bdate_range("2024-01-02", periods=6)
_DAYS = [day.date() for day in _INDEX]
_COMMISSION = 0.001


class _FixedPodWeights(WeightingStrategy):
    def __init__(self, weights: dict[str, float]) -> None:
        self.weights = weights

    def get_weights(self, scoreboard, source_ids, as_of_date):
        return {source_id: self.weights.get(source_id, 0.0) for source_id in source_ids}


class _StubConstructor(PortfolioConstructor):
    """Holds 0.1 in every asset."""

    def construct(self, views, returns_history, as_of_date, previous_weights=None):
        return dict.fromkeys(views, 0.1)

    def last_diagnostics(self) -> SizingDiagnostics | None:
        return None


class _Manager(ManagerAgent):
    def scale_factor(self, signal, market_data, as_of_date) -> float:
        return 1.0


def _manager(agent_id: str) -> _Manager:
    return _Manager(
        agent_id=agent_id,
        managed_analysts=[],
        weighting_strategy=_FixedPodWeights({}),
        portfolio_constructor=_StubConstructor(),
    )


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    columns = {}
    for asset in ("AAPL", "MSFT"):
        close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, len(_INDEX)))), _INDEX)
        columns[(asset, "open")] = close.shift(1).fillna(close.iloc[0])
        columns[(asset, "close")] = close
        columns[(asset, "log_return")] = np.log(close / close.shift(1))
    return pd.DataFrame(columns)


@pytest.fixture(autouse=True)
def _lift_firm_limits() -> Iterator[None]:
    """Firm limits are lifted so the stub pods' weights reach the book."""
    with temporary_overrides(
        {
            "firm.max_single_name_weight": 100.0,
            "firm.max_gross_exposure": 1000.0,
            "firm.max_daily_var": 1000.0,
            "options.max_notional_fraction": 1_000.0,
        }
    ):
        yield


def _days(series: dict[str, dict[str, list[float]]]) -> list[dict[str, dict[str, float]]]:
    """One {pod: {asset: contribution}} mapping per day of `_INDEX`."""
    return [
        {pod: {a: values[i] for a, values in assets.items()} for pod, assets in series.items()}
        for i in range(len(_INDEX))
    ]


class _Plain:
    """Targets the sum of the day's contributions (or `targets[day]`); no `pod_contributions`."""

    def __init__(self, script, targets=None, assets=("AAPL",)) -> None:
        self.script = dict(zip(_DAYS, script, strict=True))
        self.targets = targets or {}
        self.assets = assets
        self.by_day: dict[date, dict[str, dict[str, float]]] = {}
        self.current: dict[str, dict[str, float]] = {}

    def decide(self, market_data, as_of_date, scoreboard, instrument_returns=None):
        day = pd.Timestamp(as_of_date).date()
        self.current = self.by_day[day] = self.script[day]
        return [
            PortfolioDecision(
                asset=asset,
                as_of_date=day,
                target_weight=self.targets.get(
                    day, sum(pod.get(asset, 0.0) for pod in self.current.values())
                ),
                contributing_signals=[],
            )
            for asset in self.assets
        ]


class _Scripted(_Plain):
    def pod_contributions(self) -> dict[str, dict[str, float]]:
        return {pod: dict(weights) for pod, weights in self.current.items()}


def _engine(commander, db_path: Path | None = None, resume: bool = False, assets=("AAPL",)):
    return BacktestEngine(
        commander=commander,
        portfolio=Portfolio(
            cash=100_000.0, positions={}, commission_rate=_COMMISSION, slippage_bps=0.0
        ),
        track_record=TrackRecord(),
        data_loader=lambda _assets, _start, _end: _frame(),
        start_date=_INDEX[0].date(),
        end_date=_INDEX[-1].date(),
        assets=list(assets),
        execution_lag_days=1,
        checkpoint_path=db_path,
        resume=resume,
    )


def _delta(commander: _Plain, pod: str, asset: str, decide_day: date) -> float:
    """|contribution change| of `pod` in `asset` on `decide_day` versus the prior decide."""
    days = sorted(commander.by_day)
    index = days.index(decide_day)
    now = commander.by_day[decide_day].get(pod, {}).get(asset, 0.0)
    before = commander.by_day[days[index - 1]].get(pod, {}).get(asset, 0.0) if index else 0.0
    return abs(now - before)


def _expected_costs(result: BacktestResult, commander: _Plain, pods: list[str]) -> dict[str, float]:
    """Pro-rata split of each AAPL fill over the decide that produced it."""
    expected: dict[str, float] = defaultdict(float)
    days = sorted(commander.by_day)
    for trade in result.trade_log:
        if trade.asset != "AAPL" or trade.action not in ("buy", "sell"):
            continue
        decide_day = trade.as_of_date
        deltas = {pod: _delta(commander, pod, "AAPL", decide_day) for pod in pods}
        total = sum(deltas.values())
        if total == 0.0:
            expected["commander"] += trade.transaction_cost
            continue
        for pod, delta in deltas.items():
            expected[pod] += trade.transaction_cost * delta / total
    return expected


def _costs(result: BacktestResult) -> dict[str, float]:
    return {pod: cost.transaction_cost for pod, cost in result.pod_costs.items()}


_TWO_PODS = {
    "pod_a": {"AAPL": [0.3, 0.3, 0.1, 0.1, 0.4, 0.4]},
    "pod_b": {"AAPL": [-0.1, 0.2, 0.2, -0.2, -0.2, 0.0]},
}


def test_pod_costs_sum_to_firm_totals_and_match_the_pro_rata_split() -> None:
    """Costs over keys equal `total_transaction_cost` and each pod gets its pro-rata share."""
    commander = _Scripted(_days(_TWO_PODS))
    result = _engine(commander).run()

    assert result.total_transaction_cost > 0.0
    assert all(isinstance(cost, PodCostAttribution) for cost in result.pod_costs.values())
    assert {"pod_a", "pod_b"} <= set(result.pod_costs)
    total = sum(cost.transaction_cost for cost in result.pod_costs.values())
    assert total == pytest.approx(result.total_transaction_cost, rel=1e-9)
    premium = sum(cost.option_premium_paid for cost in result.pod_costs.values())
    assert premium == pytest.approx(result.total_option_premium_paid, abs=1e-9)

    expected = _expected_costs(result, commander, ["pod_a", "pod_b"])
    for pod, cost in _costs(result).items():
        assert cost == pytest.approx(expected.get(pod, 0.0), rel=1e-9, abs=1e-12)


def test_first_fill_splits_three_to_one_for_plus_point_three_and_minus_point_one() -> None:
    """pod_a +0.3 and pod_b -0.1 in AAPL split the fill's cost 0.75 / 0.25."""
    constant = {"pod_a": {"AAPL": [0.3] * 6}, "pod_b": {"AAPL": [-0.1] * 6}}
    result = _engine(_Scripted(_days(constant))).run()

    first = next(trade for trade in result.trade_log if trade.transaction_cost > 0.0)
    costs = _costs(result)
    assert costs["pod_a"] == pytest.approx(0.75 * first.transaction_cost, rel=1e-9)
    assert costs["pod_b"] == pytest.approx(0.25 * first.transaction_cost, rel=1e-9)


def test_fill_with_no_contribution_change_goes_to_commander() -> None:
    """Decisions that change while contributions stay put are charged to "commander"."""
    constant = {"pod_a": {"AAPL": [0.2] * 6}}
    targets = dict(zip(_DAYS, [0.2, 0.2, 0.5, 0.3, 0.4, 0.4], strict=True))
    result = _engine(_Scripted(_days(constant), targets=targets)).run()

    fills = [t for t in result.trade_log if t.action in ("buy", "sell") and t.transaction_cost > 0]
    assert len(fills) >= 2
    costs = _costs(result)
    assert costs["pod_a"] == pytest.approx(fills[0].transaction_cost, rel=1e-9)
    assert costs["commander"] == pytest.approx(sum(t.transaction_cost for t in fills[1:]), rel=1e-9)


def test_commander_without_pod_contributions_gets_all_cost() -> None:
    """A commander lacking `pod_contributions` yields only a "commander" entry."""
    result = _engine(_Plain(_days(_TWO_PODS))).run()

    assert result.total_transaction_cost > 0.0
    assert set(result.pod_costs) == {"commander"}
    assert result.pod_costs["commander"].transaction_cost == pytest.approx(
        result.total_transaction_cost, rel=1e-9
    )


def test_turnover_trading_days_and_sign_changes_follow_the_contributions() -> None:
    """AAPL +0.2, -0.2, -0.2, +0.1 gives 2 sign changes (a pass through 0 counts once)."""
    series = {
        "pod_a": {"AAPL": [0.2, -0.2, -0.2, 0.1, 0.1, 0.1]},
        "pod_b": {"AAPL": [0.0] * 6, "MSFT": [0.1, 0.0, -0.1, -0.1, -0.1, -0.1]},
    }
    result = _engine(_Scripted(_days(series)), assets=("AAPL", "MSFT")).run()

    pod_a = result.pod_costs["pod_a"]
    assert pod_a.sign_changes == 2
    assert pod_a.trading_days == 3
    sum_delta = 0.2 + 0.4 + 0.0 + 0.3
    observed = len(result.portfolio_value_history)
    assert pod_a.turnover == pytest.approx(
        sum_delta * settings.firm.trading_days_per_year / observed, rel=1e-9
    )
    assert result.pod_costs["pod_b"].sign_changes == 1
    assert result.pod_costs["pod_b"].trading_days == 3


def test_resumed_run_has_same_pod_costs_as_uninterrupted_run(tmp_path: Path) -> None:
    """Resuming from a checkpoint reproduces the uninterrupted run's `pod_costs`."""
    db_path = tmp_path / "checkpoint.db"
    fresh = _engine(_Scripted(_days(_TWO_PODS)), db_path).run()
    assert fresh.pod_costs

    resumed = _engine(_Scripted(_days(_TWO_PODS)), db_path, resume=True).run()

    assert resumed.pod_costs == fresh.pod_costs


_CONTRACT = OptionContract(
    underlying="AAPL", expiry=date(2024, 1, 5), strike=100.0, option_type="call"
)


def _fake_mark(contract: OptionContract, as_of_date: date, spot: float) -> OptionMark:
    """Flat 1.0 time value with a 0.1 half spread; intrinsic on or after expiry."""
    expired = as_of_date >= contract.expiry
    mid = max(spot - contract.strike, 0.0) + (0.0 if expired else 1.0)
    half = 0.0 if expired else 0.1
    return OptionMark(
        contract=contract,
        as_of_date=as_of_date,
        snapshot_date=None if expired else as_of_date,
        mid=mid,
        bid=mid - half,
        ask=mid + half,
        implied_volatility=0.25,
        delta=0.5,
        gamma=0.02,
        vega=0.1,
        theta=-0.05,
        is_model_mark=False,
        is_expired=expired,
        tradeable=not expired,
    )


class _OptionScripted:
    """Buys the AAPL call on the first day (pods a and b), holds on the second."""

    def __init__(self) -> None:
        self.script = {
            date(2024, 1, 2): {"pod_a": 0.03, "pod_b": 0.02},
            date(2024, 1, 3): {"pod_a": 0.02, "pod_b": 0.02},
        }
        self.current: dict[str, float] = {}

    def decide(self, market_data, as_of_date, scoreboard, instrument_returns=None):
        day = pd.Timestamp(as_of_date).date()
        self.current = self.script.get(day, {})
        if not self.current:
            return []
        return [
            PortfolioDecision(
                asset=_CONTRACT.contract_id,
                as_of_date=day,
                target_weight=sum(self.current.values()),
                contributing_signals=[],
                instrument=_CONTRACT,
            )
        ]

    def pod_contributions(self) -> dict[str, dict[str, float]]:
        return {pod: {_CONTRACT.contract_id: w} for pod, w in self.current.items()}


def test_option_premium_and_commission_are_split_and_expiry_is_excluded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 0.03 / 0.02 option buy splits premium and commission 60 / 40; the expire fill adds none."""
    monkeypatch.setattr("agentic_trading.backtest.engine.load_option_mark", _fake_mark)
    monkeypatch.setattr(
        "agentic_trading.backtest.engine.lakehouse_loader.split_adjustment_factor",
        lambda *_args: 1.0,
    )
    dates = pd.to_datetime(
        ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]
    )
    close = pd.Series([100.0, 102.0, 101.0, 104.0, 103.0, 110.0], index=dates)
    frame = pd.DataFrame({"close": close, "log_return": np.log(close / close.shift(1))})
    with temporary_overrides({"options.max_snapshot_age_days": 10}):
        result = BacktestEngine(
            commander=_OptionScripted(),  # type: ignore[arg-type]
            portfolio=Portfolio(cash=10_000.0, positions={}, commission_rate=0.0, slippage_bps=0.0),
            track_record=TrackRecord(),
            data_loader=lambda _assets, _start, _end: frame,
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 8),
        ).run()

    assert result.total_option_premium_paid > 0.0
    assert any(trade.action == "expire" for trade in result.trade_log)
    premium = {pod: cost.option_premium_paid for pod, cost in result.pod_costs.items()}
    assert sum(premium.values()) == pytest.approx(result.total_option_premium_paid, rel=1e-9)
    assert premium["pod_a"] == pytest.approx(0.6 * result.total_option_premium_paid, rel=1e-9)
    assert premium["pod_b"] == pytest.approx(0.4 * result.total_option_premium_paid, rel=1e-9)
    costs = _costs(result)
    assert sum(costs.values()) == pytest.approx(result.total_transaction_cost, rel=1e-9)
    assert costs["pod_a"] == pytest.approx(0.6 * result.total_transaction_cost, rel=1e-9)


def test_real_commander_contributions_are_allocation_times_weight() -> None:
    """`pod_contributions` is {} before decide, then allocation x weight per pod and asset."""
    commander = Commander(
        managers=[_manager("pod_a"), _manager("pod_b")],
        weighting_strategy=_FixedPodWeights({"pod_a": 3.0, "pod_b": 1.0}),
    )
    assert commander.pod_contributions() == {}
    data = _frame()
    day = _INDEX[3]

    decisions = commander.decide(data.loc[:day], day, PerformanceScoreboard(TrackRecord()))

    contributions = commander.pod_contributions()
    assert decisions and set(contributions) == {"pod_a", "pod_b"}
    for decision in decisions:
        a = contributions["pod_a"][decision.asset]
        b = contributions["pod_b"][decision.asset]
        assert a == pytest.approx(3.0 * b)
        assert a + b == pytest.approx(decision.target_weight)
        assert a > 0.0
    commander.close()
