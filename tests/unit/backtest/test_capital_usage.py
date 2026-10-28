# test_capital_usage.py

"""Capital usage reporting: `Commander.idle_pod_share`, `BacktestResult.capital_usage`
and the `mean_idle_pod_share` / `mean_cash_share` fields of `extract_run_summary`."""

import numpy as np
import pandas as pd
import pytest
from pydantic import Field

from agentic_trading.agents.base import ManagerAgent
from agentic_trading.agents.commander import Commander
from agentic_trading.agents.portfolio_construction import PortfolioConstructor
from agentic_trading.agents.schemas import CapitalUsage, PortfolioDecision
from agentic_trading.backtest.engine import BacktestEngine
from agentic_trading.backtest.portfolio import Portfolio
from agentic_trading.backtest.run_summary import extract_run_summary
from agentic_trading.config import temporary_overrides
from agentic_trading.scoring.scoreboard import PerformanceScoreboard
from agentic_trading.scoring.track_record import TrackRecord
from agentic_trading.scoring.weighting import WeightingStrategy

_DATES = pd.bdate_range("2024-01-02", periods=8)


class _FixedPodWeights(WeightingStrategy):
    def __init__(self, weights: dict[str, float]) -> None:
        self.weights = weights

    def get_weights(self, scoreboard, source_ids, as_of_date):
        return {source_id: self.weights.get(source_id, 0.0) for source_id in source_ids}


class _StubManager(ManagerAgent):
    """Returns caller-fixed target weights from construct_portfolio()."""

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


def _market_data() -> pd.DataFrame:
    aapl = [0.01 if i % 2 == 0 else -0.012 for i in range(len(_DATES))]
    msft = [0.004, -0.002, 0.003, -0.001, 0.002, -0.003, 0.001, 0.002]
    return pd.concat(
        {
            "AAPL": pd.DataFrame({"close": [100.0] * 8, "log_return": aapl}, index=_DATES),
            "MSFT": pd.DataFrame({"close": [200.0] * 8, "log_return": msft}, index=_DATES),
        },
        axis=1,
    )


def test_commander_idle_pod_share_tracks_flat_pods() -> None:
    """The share is None before decide, in (0, 1) with one flat pod, and 0.0 with none."""
    weights = _FixedPodWeights({"a": 1.0, "b": 1.0, "c": 1.0})
    overrides = {
        "firm.max_single_name_weight": 100.0,
        "firm.max_gross_exposure": 1000.0,
        "firm.max_daily_var": 1000.0,
    }
    data = _market_data()
    with temporary_overrides(overrides):
        flat = Commander(
            managers=[
                _stub_manager("a", {"AAPL": 0.5}),
                _stub_manager("b", {"MSFT": 0.5}),
                _stub_manager("c", {}),
            ],
            weighting_strategy=weights,
        )
        assert flat.idle_pod_share() is None
        for day in _DATES:
            flat.decide(data.loc[:day], day, _scoreboard())
            share = flat.idle_pod_share()
            assert share is not None and 0.0 < share < 1.0

        active = Commander(
            managers=[
                _stub_manager("a", {"AAPL": 0.5}),
                _stub_manager("b", {"MSFT": 0.5}),
                _stub_manager("c", {"AAPL": 0.2}),
            ],
            weighting_strategy=weights,
        )
        for day in _DATES:
            active.decide(data.loc[:day], day, _scoreboard())
            assert active.idle_pod_share() == 0.0


_INDEX = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])


def _frame() -> pd.DataFrame:
    close = pd.Series([100.0, 101.0, 102.0, 103.0], index=_INDEX)
    return pd.DataFrame(
        {
            ("AAPL", "open"): close,
            ("AAPL", "close"): close,
            ("AAPL", "log_return"): np.log(close / close.shift(1)),
        }
    )


class _NoIdleCommander:
    """Targets a 0.5 weight in AAPL every day and reports no idle share."""

    def decide(self, market_data, as_of_date, scoreboard, instrument_returns=None):
        day = pd.Timestamp(as_of_date)
        return [
            PortfolioDecision(
                asset="AAPL", as_of_date=day.date(), target_weight=0.5, contributing_signals=[]
            )
        ]


class _IdleCommander(_NoIdleCommander):
    def idle_pod_share(self) -> float:
        return 0.3


def _run(commander: object) -> tuple[Portfolio, object]:
    portfolio = Portfolio(cash=100_000.0, positions={}, commission_rate=0.0, slippage_bps=0.0)
    result = BacktestEngine(
        commander=commander,  # type: ignore[arg-type]
        portfolio=portfolio,
        track_record=TrackRecord(),
        data_loader=lambda _assets, _start, _end: _frame(),
        start_date=_INDEX[0].date(),
        end_date=_INDEX[-1].date(),
        assets=["AAPL"],
        execution_lag_days=1,
    ).run()
    return portfolio, result


def test_engine_records_capital_usage_per_valuation_day() -> None:
    """One entry per valuation date carries the commander's idle share and the cash share."""
    portfolio, result = _run(_IdleCommander())

    usage = result.capital_usage
    assert all(isinstance(entry, CapitalUsage) for entry in usage)
    assert [entry.as_of_date for entry in usage] == [
        pd.Timestamp(day).date() for day, _ in result.portfolio_value_history
    ]
    assert [entry.as_of_date for entry in usage] == sorted(entry.as_of_date for entry in usage)
    assert all(entry.idle_pod_share == 0.3 for entry in usage)
    nav = result.portfolio_value_history[-1][1]
    assert usage[-1].cash_share == pytest.approx(portfolio.cash / nav)
    assert usage[-1].cash_share == pytest.approx(0.5, abs=1e-2)


def test_engine_stores_none_idle_share_when_commander_lacks_it() -> None:
    """A commander without `idle_pod_share` yields None entries."""
    _, result = _run(_NoIdleCommander())

    assert len(result.capital_usage) == len(_INDEX)
    assert all(entry.idle_pod_share is None for entry in result.capital_usage)


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


def test_run_summary_reports_mean_capital_usage() -> None:
    """Period and baseline blocks report mean idle and cash shares; missing data gives None."""
    usage = [
        {"idle_pod_share": 0.2, "cash_share": 0.9},
        {"idle_pod_share": 0.4, "cash_share": 0.7},
    ]
    none_idle = [{"idle_pod_share": None, "cash_share": 0.5}]
    result = {
        "experiment": {},
        "market_data_source": "lakehouse data",
        "calibration": _block(),
        "test": _block(capital_usage=usage),
        "baselines": {
            "with": _block(capital_usage=usage),
            "none_idle": _block(capital_usage=none_idle),
            "empty": _block(capital_usage=[]),
            "absent": _block(),
        },
    }

    summary = extract_run_summary(result)

    assert summary["test"]["mean_idle_pod_share"] == pytest.approx(0.3)
    assert summary["test"]["mean_cash_share"] == pytest.approx(0.8)
    assert summary["baselines"]["with"]["mean_idle_pod_share"] == pytest.approx(0.3)
    assert summary["baselines"]["with"]["mean_cash_share"] == pytest.approx(0.8)
    assert summary["baselines"]["none_idle"]["mean_idle_pod_share"] is None
    assert summary["baselines"]["none_idle"]["mean_cash_share"] == pytest.approx(0.5)
    assert summary["baselines"]["empty"]["mean_idle_pod_share"] is None
    for block in (summary["calibration"], summary["baselines"]["absent"]):
        assert block["mean_idle_pod_share"] is None
        assert block["mean_cash_share"] is None
