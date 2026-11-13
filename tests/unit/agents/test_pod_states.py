# test_pod_states.py

"""Pod states: `Commander.pod_states`, the unavailable / no-position pod shares,
`CapitalUsage` and the matching `extract_run_summary` fields."""

from collections.abc import Iterator
from datetime import date

import numpy as np
import pandas as pd
import pytest
from pydantic import Field

from agentic_trading.agents.base import ManagerAgent
from agentic_trading.agents.commander import Commander
from agentic_trading.agents.portfolio_construction import PortfolioConstructor
from agentic_trading.agents.schemas import (
    CapitalUsage,
    ManagerSignal,
    OptionContract,
    PortfolioDecision,
)
from agentic_trading.backtest.engine import BacktestEngine
from agentic_trading.backtest.portfolio import Portfolio
from agentic_trading.backtest.run_summary import extract_run_summary
from agentic_trading.config import temporary_overrides
from agentic_trading.data.loaders import option_marks
from agentic_trading.scoring.scoreboard import PerformanceScoreboard
from agentic_trading.scoring.track_record import TrackRecord
from agentic_trading.scoring.weighting import WeightingStrategy

_INDEX = pd.bdate_range("2024-01-02", periods=6)
_OPTION = OptionContract(
    underlying="AAPL", expiry=date(2024, 3, 15), strike=185.0, option_type="call"
)


class _FixedPodWeights(WeightingStrategy):
    def __init__(self, weights: dict[str, float]) -> None:
        self.weights = weights

    def get_weights(self, scoreboard, source_ids, as_of_date):
        return {source_id: self.weights.get(source_id, 0.0) for source_id in source_ids}


class _StubConstructor(PortfolioConstructor):
    """Holds 0.1 in every asset."""

    def construct(self, views, returns_history, as_of_date, previous_weights=None):
        return dict.fromkeys(views, 0.1)


class _SignalManager(ManagerAgent):
    """Returns hand-built decisions whose signals carry the configured statuses."""

    statuses: list[str] = Field(default_factory=list)
    weights: dict[str, float] = Field(default_factory=dict)
    option_weight: float = 0.0
    empty: bool = False

    def scale_factor(self, signal, market_data, as_of_date) -> float:
        return 1.0

    def construct_portfolio(self, assets, market_data, as_of_date, scoreboard):
        if self.empty:
            return []
        day = as_of_date.date()

        def signals(asset: str) -> list[ManagerSignal]:
            return [
                ManagerSignal(
                    manager_id=self.agent_id,
                    source_agent_id=f"analyst_{index}",
                    asset=asset,
                    as_of_date=day,
                    direction="long",
                    confidence=0.5,
                    raw_confidence=0.5,
                    scale_applied=1.0,
                    rationale="stub",
                    status=status,
                )
                for index, status in enumerate(self.statuses)
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
        if self.option_weight:
            decisions.append(
                PortfolioDecision(
                    asset=_OPTION.contract_id,
                    as_of_date=day,
                    target_weight=self.option_weight,
                    contributing_signals=signals(_OPTION.contract_id),
                    instrument=_OPTION,
                )
            )
        return decisions


def _manager(agent_id: str, **fields: object) -> _SignalManager:
    return _SignalManager(
        agent_id=agent_id,
        managed_analysts=[],
        weighting_strategy=_FixedPodWeights({}),
        portfolio_constructor=_StubConstructor(),
        **fields,
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
        }
    ):
        yield


def _scoreboard() -> PerformanceScoreboard:
    return PerformanceScoreboard(TrackRecord())


def test_commander_classifies_pods_and_splits_idle_share() -> None:
    """Pods map to unavailable, no_position and holding; the shares sum to the idle share."""
    commander = Commander(
        managers=[
            _manager("a", statuses=["unavailable", "error"]),
            _manager("b", statuses=["active", "unavailable"]),
            _manager("c", statuses=["active"], weights={"AAPL": 0.5}),
        ],
        weighting_strategy=_FixedPodWeights({"a": 0.5, "b": 0.3, "c": 0.2}),
    )
    assert commander.pod_states() == {}
    assert commander.unavailable_pod_share() is None
    assert commander.no_position_pod_share() is None

    data = _frame()
    for day in _INDEX[2:5]:
        commander.decide(data.loc[:day], day, _scoreboard())

        states = commander.pod_states()
        assert states == {"a": "unavailable", "b": "no_position", "c": "holding"}
        assert list(states) == ["a", "b", "c"]
        unavailable = commander.unavailable_pod_share()
        no_position = commander.no_position_pod_share()
        assert unavailable is not None and no_position is not None
        assert unavailable > 0.0 and no_position > 0.0
        assert unavailable / no_position == pytest.approx(0.5 / 0.3)
        assert commander.idle_pod_share() == pytest.approx(unavailable + no_position)
        assert unavailable + no_position < 1.0


def test_pod_state_edge_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    """No decisions or no signals is unavailable; an option row with weight is holding."""
    monkeypatch.setattr(option_marks, "load_option_mark", lambda *args, **kwargs: None)
    commander = Commander(
        managers=[
            _manager("empty", empty=True),
            _manager("no_signals", statuses=[]),
            _manager("option", statuses=["active"], option_weight=0.1),
            _manager("zero_option", statuses=["active"]),
        ],
        weighting_strategy=_FixedPodWeights(
            {"empty": 1.0, "no_signals": 1.0, "option": 1.0, "zero_option": 1.0}
        ),
    )
    data = _frame()
    commander.decide(data.loc[: _INDEX[3]], _INDEX[3], _scoreboard())

    assert commander.pod_states() == {
        "empty": "unavailable",
        "no_signals": "unavailable",
        "option": "holding",
        "zero_option": "no_position",
    }


class _PlainCommander:
    """Targets a 0.5 AAPL weight and reports nothing else."""

    def decide(self, market_data, as_of_date, scoreboard, instrument_returns=None):
        day = pd.Timestamp(as_of_date).date()
        return [
            PortfolioDecision(
                asset="AAPL", as_of_date=day, target_weight=0.5, contributing_signals=[]
            )
        ]


class _StateCommander(_PlainCommander):
    """Also reports fixed shares and pod states."""

    def idle_pod_share(self) -> float:
        return 0.3

    def unavailable_pod_share(self) -> float:
        return 0.2

    def no_position_pod_share(self) -> float:
        return 0.1

    def pod_states(self) -> dict[str, str]:
        return {"a": "unavailable", "b": "no_position", "c": "holding"}


def _run(commander: object):
    return BacktestEngine(
        commander=commander,  # type: ignore[arg-type]
        portfolio=Portfolio(cash=100_000.0, positions={}, commission_rate=0.0, slippage_bps=0.0),
        track_record=TrackRecord(),
        data_loader=lambda _assets, _start, _end: _frame(),
        start_date=_INDEX[0].date(),
        end_date=_INDEX[-1].date(),
        assets=["AAPL"],
        execution_lag_days=1,
    ).run()


def test_engine_records_pod_states_and_shares() -> None:
    """Each CapitalUsage carries the commander's shares and states; absent methods give None."""
    usage = _run(_StateCommander()).capital_usage

    assert usage and all(isinstance(entry, CapitalUsage) for entry in usage)
    for entry in usage:
        assert entry.idle_pod_share == 0.3
        assert entry.unavailable_pod_share == 0.2
        assert entry.no_position_pod_share == 0.1
        assert entry.pod_states == {"a": "unavailable", "b": "no_position", "c": "holding"}

    plain = _run(_PlainCommander()).capital_usage
    assert plain
    for entry in plain:
        assert entry.unavailable_pod_share is None
        assert entry.no_position_pod_share is None
        assert entry.pod_states == {}


def _block(**overrides: object) -> dict:
    block = {
        "start_date": "2018-04-02",
        "end_date": "2018-04-03",
        "portfolio_value_history": [["2018-04-02", 10_000.0], ["2018-04-03", 10_050.0]],
        "final_sharpe_ratio": 0.5,
        "final_max_drawdown": -0.01,
        "final_sortino_ratio": 0.6,
        "final_calmar_ratio": 0.4,
        "final_information_ratio": None,
        "final_hit_rate": 0.55,
        "final_profit_factor": 1.1,
        "signal_health": {},
        "source_performance": {},
    }
    block.update(overrides)
    return block


def _usage(unavailable: float | None, no_position: float | None, states: dict) -> dict:
    return {
        "idle_pod_share": None,
        "cash_share": 0.5,
        "unavailable_pod_share": unavailable,
        "no_position_pod_share": no_position,
        "pod_states": states,
    }


def test_run_summary_reports_mean_shares_and_pod_state_days() -> None:
    """Means skip None days, and pod_state_days counts days per state per pod."""
    usage = [
        _usage(0.2, 0.1, {"a": "unavailable", "b": "holding"}),
        _usage(0.4, None, {"a": "unavailable", "b": "no_position"}),
        _usage(None, 0.3, {"a": "holding", "b": "no_position"}),
        _usage(None, None, {}),
    ]
    result = {
        "experiment": {},
        "market_data_source": "lakehouse data",
        "calibration": _block(),
        "test": _block(capital_usage=usage),
        "baselines": {"empty": _block(capital_usage=[]), "absent": _block()},
    }

    summary = extract_run_summary(result)

    test = summary["test"]
    assert test["mean_unavailable_pod_share"] == pytest.approx(0.3)
    assert test["mean_no_position_pod_share"] == pytest.approx(0.2)
    assert test["pod_state_days"]["a"] == {"unavailable": 2, "holding": 1}
    assert test["pod_state_days"]["b"] == {"holding": 1, "no_position": 2}
    for block in (summary["calibration"], *summary["baselines"].values()):
        assert block["mean_unavailable_pod_share"] is None
        assert block["mean_no_position_pod_share"] is None
        assert block["pod_state_days"] == {}
