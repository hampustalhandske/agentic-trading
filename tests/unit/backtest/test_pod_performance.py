# test_pod_performance.py

"""`BacktestEngine` reports `pod_performance` and a whole-run `pod_return_history`;
`extract_run_summary` carries `pod_performance` into each period block."""

from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest
from pydantic import Field
from quantcore.performance.metrics import maximum_drawdown, sharpe_ratio

from agentic_trading.agents.base import ManagerAgent
from agentic_trading.agents.commander import Commander
from agentic_trading.agents.portfolio_construction import PortfolioConstructor
from agentic_trading.agents.schemas import BacktestResult, PodPerformance, PortfolioDecision
from agentic_trading.backtest.engine import BacktestEngine
from agentic_trading.backtest.portfolio import Portfolio
from agentic_trading.backtest.run_summary import extract_run_summary
from agentic_trading.config import settings, temporary_overrides
from agentic_trading.scoring.track_record import TrackRecord
from agentic_trading.scoring.weighting import WeightingStrategy

_INDEX = pd.bdate_range("2023-01-02", periods=160)
_ASSETS = ["AAPL", "MSFT"]


class _FixedPodWeights(WeightingStrategy):
    def __init__(self, weights: dict[str, float]) -> None:
        self.weights = weights

    def get_weights(self, scoreboard, source_ids, as_of_date):
        return {source_id: self.weights.get(source_id, 0.0) for source_id in source_ids}


class _StubManager(ManagerAgent):
    """Returns a caller-fixed set of PortfolioDecisions from construct_portfolio()."""

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


@pytest.fixture(autouse=True)
def _lift_firm_limits() -> Iterator[None]:
    """Firm limits are lifted so the stub pods' fixed weights reach the book."""
    with temporary_overrides(
        {
            "firm.max_single_name_weight": 100.0,
            "firm.max_gross_exposure": 1000.0,
            "firm.max_daily_var": 1000.0,
        }
    ):
        yield


def _frame() -> pd.DataFrame:
    """Random-walk AAPL and MSFT bars over `_INDEX`."""
    rng = np.random.default_rng(7)
    columns = {}
    for asset in _ASSETS:
        close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, len(_INDEX)))), _INDEX)
        columns[(asset, "open")] = close.shift(1).fillna(close.iloc[0])
        columns[(asset, "close")] = close
        columns[(asset, "log_return")] = np.log(close / close.shift(1))
    return pd.DataFrame(columns)


def _run(commander: object) -> BacktestResult:
    portfolio = Portfolio(cash=100_000.0, positions={}, commission_rate=0.0, slippage_bps=0.0)
    return BacktestEngine(
        commander=commander,  # type: ignore[arg-type]
        portfolio=portfolio,
        track_record=TrackRecord(),
        data_loader=lambda _assets, _start, _end: _frame(),
        start_date=_INDEX[0].date(),
        end_date=_INDEX[-1].date(),
        assets=_ASSETS,
        execution_lag_days=1,
    ).run()


class _NoRiskStateCommander:
    """Holds AAPL at a fixed weight and exposes no risk state."""

    def decide(self, market_data, as_of_date, scoreboard, instrument_returns=None):
        return [
            PortfolioDecision(
                asset="AAPL",
                as_of_date=pd.Timestamp(as_of_date).date(),
                target_weight=0.5,
                contributing_signals=[],
            )
        ]


def test_engine_reports_whole_run_pod_history_and_matching_performance() -> None:
    """Pod history spans the full run and `pod_performance` follows the spec formulas."""
    commander = Commander(
        managers=[
            _stub_manager("pod_a", {"AAPL": 0.6, "MSFT": 0.2}),
            _stub_manager("pod_b", {"AAPL": -0.3, "MSFT": 0.5}),
        ],
        weighting_strategy=_FixedPodWeights({"pod_a": 1.0, "pod_b": 1.0}),
    )

    result = _run(commander)

    end = _INDEX[-1]
    window_start = end - pd.Timedelta(days=settings.commander.pod_returns_lookback_days)
    trailing = len(pd.bdate_range(window_start, end))
    assert set(result.pod_return_history) == {"pod_a", "pod_b"}
    assert set(result.pod_performance) == {"pod_a", "pod_b"}
    for pod, history in result.pod_return_history.items():
        returns = np.array([r for _, r in history])
        assert len(history) > trailing
        assert (history[0][0] - _INDEX[0].date()).days <= 7

        perf = result.pod_performance[pod]
        assert isinstance(perf, PodPerformance)
        assert perf.observations == len(history)
        assert perf.cumulative_return == pytest.approx(np.prod(1 + returns) - 1, abs=1e-9)
        assert perf.annualized_vol == pytest.approx(
            np.std(returns, ddof=1) * np.sqrt(settings.firm.trading_days_per_year), abs=1e-9
        )
        assert perf.sharpe_ratio == pytest.approx(float(sharpe_ratio(returns)), abs=1e-9)
        assert perf.max_drawdown == pytest.approx(float(maximum_drawdown(returns)), abs=1e-9)


def test_engine_without_risk_state_has_empty_pod_outputs() -> None:
    """A commander with no risk state yields empty `pod_performance` and `pod_return_history`."""
    result = _run(_NoRiskStateCommander())

    assert result.pod_performance == {}
    assert result.pod_return_history == {}


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


def test_run_summary_passes_pod_performance_through_or_defaults_empty() -> None:
    """Each period carries its block's `pod_performance`, or {} when absent."""
    perf = {
        "pod_a": {
            "observations": 10,
            "cumulative_return": 0.02,
            "annualized_vol": 0.05,
            "sharpe_ratio": 0.8,
            "max_drawdown": -0.01,
        }
    }
    result = {
        "experiment": {"assets": ["AAPL"]},
        "market_data_source": "lakehouse data",
        "calibration": _block(),
        "test": _block(pod_performance=perf),
    }

    summary = extract_run_summary(result)

    assert summary["test"]["pod_performance"] == perf
    assert summary["calibration"]["pod_performance"] == {}
