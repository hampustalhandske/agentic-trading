# test_sizing_diagnostics_propagation.py

"""Sizing diagnostics flow from `ManagerAgent` through `Commander`, `BacktestEngine` and the
checkpoint store into `BacktestResult` and `extract_run_summary`."""

from collections.abc import Iterator
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agentic_trading.agents.base import ManagerAgent
from agentic_trading.agents.commander import Commander
from agentic_trading.agents.portfolio_construction import PortfolioConstructor
from agentic_trading.agents.schemas import PortfolioDecision, SizingDiagnostics
from agentic_trading.backtest import checkpoint_store
from agentic_trading.backtest.engine import BacktestEngine
from agentic_trading.backtest.portfolio import Portfolio
from agentic_trading.backtest.run_summary import extract_run_summary
from agentic_trading.config import temporary_overrides
from agentic_trading.scoring.scoreboard import PerformanceScoreboard
from agentic_trading.scoring.track_record import TrackRecord
from agentic_trading.scoring.weighting import WeightingStrategy

_INDEX = pd.bdate_range("2024-01-02", periods=6)


class _FixedPodWeights(WeightingStrategy):
    def __init__(self, weights: dict[str, float]) -> None:
        self.weights = weights

    def get_weights(self, scoreboard, source_ids, as_of_date):
        return {source_id: self.weights.get(source_id, 0.0) for source_id in source_ids}


def _diag(day: date, final_vol: float = 0.08, **fields: object) -> SizingDiagnostics:
    values = {
        "as_of_date": day,
        "path": "full",
        "target_vol": 0.1,
        "unlevered_vol": 0.05,
        "vol_after": {"vol_target_cap": 0.09, "kelly": final_vol},
        "final_vol": final_vol,
        "binding_stage": "kelly",
    }
    values.update(fields)
    return SizingDiagnostics(**values)  # type: ignore[arg-type]


class _StubConstructor(PortfolioConstructor):
    """Holds 0.1 in every asset and reports diagnostics whose final vol is `final_vol`."""

    def __init__(self, final_vol: float) -> None:
        super().__init__()
        self.final_vol = final_vol
        self.diagnostics: SizingDiagnostics | None = None

    def construct(self, views, returns_history, as_of_date, previous_weights=None):
        self.diagnostics = _diag(as_of_date.date(), self.final_vol)
        return dict.fromkeys(views, 0.1)

    def last_diagnostics(self) -> SizingDiagnostics | None:
        return self.diagnostics


class _Manager(ManagerAgent):
    def scale_factor(self, signal, market_data, as_of_date) -> float:
        return 1.0


def _manager(agent_id: str, final_vol: float) -> _Manager:
    return _Manager(
        agent_id=agent_id,
        managed_analysts=[],
        weighting_strategy=_FixedPodWeights({}),
        portfolio_constructor=_StubConstructor(final_vol),
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


def test_manager_exposes_constructor_diagnostics_with_its_id() -> None:
    """The manager returns the constructor's latest diagnostics tagged with its own id."""
    manager = _manager("pod_a", 0.07)
    data = _frame()
    assert manager.last_sizing_diagnostics() is None

    for day in _INDEX[2:4]:
        manager.construct_portfolio(["AAPL", "MSFT"], data.loc[:day], day, _scoreboard())
        expected = _diag(day.date(), 0.07, manager_id="pod_a")
        assert manager.last_sizing_diagnostics() == expected


@pytest.mark.parametrize("workers", [1, 2])
def test_commander_collects_one_diagnostic_per_pod_in_manager_order(workers: int) -> None:
    """After each decide, diagnostics hold one entry per pod, in manager order, with ids set."""
    commander = Commander(
        managers=[_manager("pod_a", 0.05), _manager("pod_b", 0.06), _manager("pod_c", 0.07)],
        weighting_strategy=_FixedPodWeights({"pod_a": 1.0, "pod_b": 1.0, "pod_c": 1.0}),
        parallel_workers=workers,
    )
    assert commander.sizing_diagnostics() == []
    data = _frame()
    try:
        for day in _INDEX[2:5]:
            commander.decide(data.loc[:day], day, _scoreboard())
            assert commander.sizing_diagnostics() == [
                _diag(day.date(), vol, manager_id=pod_id)
                for pod_id, vol in (("pod_a", 0.05), ("pod_b", 0.06), ("pod_c", 0.07))
            ]
    finally:
        commander.close()


class _DiagnosticCommander:
    """Targets a 0.5 AAPL weight and reports two hand-built diagnostics per decide."""

    def __init__(self) -> None:
        self.emitted: list[SizingDiagnostics] = []
        self._current: list[SizingDiagnostics] = []

    def decide(self, market_data, as_of_date, scoreboard, instrument_returns=None):
        day = pd.Timestamp(as_of_date).date()
        self._current = [
            _diag(day, 0.04, manager_id="pod_a"),
            _diag(day, 0.09, manager_id="pod_b", binding_stage=None),
        ]
        return [
            PortfolioDecision(
                asset="AAPL", as_of_date=day, target_weight=0.5, contributing_signals=[]
            )
        ]

    def sizing_diagnostics(self) -> list[SizingDiagnostics]:
        self.emitted.extend(self._current)
        return list(self._current)


class _PlainCommander:
    def decide(self, market_data, as_of_date, scoreboard, instrument_returns=None):
        return [
            PortfolioDecision(
                asset="AAPL",
                as_of_date=pd.Timestamp(as_of_date).date(),
                target_weight=0.5,
                contributing_signals=[],
            )
        ]


def _engine(commander: object, db_path: Path | None = None, resume: bool = False):
    return BacktestEngine(
        commander=commander,  # type: ignore[arg-type]
        portfolio=Portfolio(cash=100_000.0, positions={}, commission_rate=0.0, slippage_bps=0.0),
        track_record=TrackRecord(),
        data_loader=lambda _assets, _start, _end: _frame(),
        start_date=_INDEX[0].date(),
        end_date=_INDEX[-1].date(),
        assets=["AAPL"],
        execution_lag_days=1,
        checkpoint_path=db_path,
        resume=resume,
    )


def test_engine_accumulates_commander_diagnostics_each_day() -> None:
    """Each day's diagnostics are appended in order; a commander without them adds none."""
    commander = _DiagnosticCommander()
    result = _engine(commander).run()

    assert len(result.sizing_diagnostics) == 2 * len(result.portfolio_value_history)
    assert result.sizing_diagnostics == commander.emitted
    assert [entry.manager_id for entry in result.sizing_diagnostics[:2]] == ["pod_a", "pod_b"]

    assert _engine(_PlainCommander()).run().sizing_diagnostics == []


def test_store_round_trips_sizing_diagnostics(tmp_path: Path) -> None:
    """The checkpoint store returns the diagnostics it was given, in order."""
    connection = checkpoint_store.connect(tmp_path / "checkpoint.db")
    days = [date(2024, 1, 2), date(2024, 1, 3)]
    expected = []
    for index, day in enumerate(days):
        entries = [
            _diag(day, 0.04, manager_id="pod_a"),
            _diag(day, 0.09, manager_id="pod_b", path="view_direction", binding_stage=None),
        ]
        expected.extend(entries)
        checkpoint_store.append_day(
            connection,
            as_of_date=day,
            portfolio_value=100_000.0,
            new_track_record=[],
            decisions=[],
            new_trades=[],
            next_day_index=index + 1,
            start_date=days[0],
            end_date=days[-1],
            assets=["AAPL"],
            portfolio={"cash": 100_000.0},
            signal_health={},
            pending_decisions=[],
            sizing_diagnostics=entries,
        )

    payload = checkpoint_store.read_all(connection)

    assert payload is not None
    restored = [SizingDiagnostics.model_validate(e) for e in payload["sizing_diagnostics"]]
    assert restored == expected


def test_resumed_run_has_same_sizing_diagnostics_as_uninterrupted_run(tmp_path: Path) -> None:
    """Resuming from a checkpoint reproduces the uninterrupted run's diagnostics."""
    db_path = tmp_path / "checkpoint.db"
    fresh = _engine(_DiagnosticCommander(), db_path).run()
    assert fresh.sizing_diagnostics

    resumed = _engine(_DiagnosticCommander(), db_path, resume=True).run()

    assert resumed.sizing_diagnostics == fresh.sizing_diagnostics


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


def _entry(day: int, pod: str, final_vol: float, stage: str | None, path: str = "full") -> dict:
    return _diag(
        date(2024, 1, day),
        final_vol,
        manager_id=pod,
        binding_stage=stage,
        path=path,
        target_vol=0.1,
    ).model_dump(mode="json")


def test_run_summary_reports_binding_shares_and_vol_ratio() -> None:
    """Shares cover full-path days incl. a "none" bucket; view_direction days are excluded."""
    entries = [
        _entry(2, "pod_a", 0.05, "kelly"),
        _entry(3, "pod_a", 0.10, "kelly"),
        _entry(4, "pod_a", 0.10, None),
        _entry(5, "pod_a", 0.05, "gross_cap"),
        _entry(8, "pod_a", 0.50, "net_trim", path="view_direction"),
        _entry(2, "pod_b", 0.10, None),
        _entry(3, "pod_b", 0.20, "final_clip"),
        _entry(2, "pod_c", 0.30, "kelly", path="view_direction"),
    ]
    result = {
        "experiment": {},
        "market_data_source": "lakehouse data",
        "calibration": _block(),
        "test": _block(sizing_diagnostics=entries),
    }

    summary = extract_run_summary(result)["test"]

    shares = summary["sizing_binding_shares"]
    assert set(shares) == {"pod_a", "pod_b"}
    assert shares["pod_a"] == pytest.approx({"kelly": 0.5, "none": 0.25, "gross_cap": 0.25})
    assert shares["pod_b"] == pytest.approx({"none": 0.5, "final_clip": 0.5})
    assert all(sum(pod.values()) == pytest.approx(1.0) for pod in shares.values())
    assert summary["sizing_vol_ratio"] == pytest.approx({"pod_a": 0.75, "pod_b": 1.5})

    empty = extract_run_summary(result)["calibration"]
    assert empty["sizing_binding_shares"] == {}
    assert empty["sizing_vol_ratio"] == {}
