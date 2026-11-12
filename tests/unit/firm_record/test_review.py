# test_review.py

"""`firm_record.review`: a firm record's checkpoint read back, without changing it, into the
`BacktestResult` its engine run returned and into the dashboard's run-summary shape with one
`record` period and the SPY total-return baseline."""

import hashlib
import json
from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agentic_trading.agents.base import AnalystAgent, ManagerAgent
from agentic_trading.agents.commander import Commander
from agentic_trading.agents.portfolio_construction import PortfolioConstructor
from agentic_trading.agents.roster import FIRM_V1_EXCLUDED_ANALYST_IDS
from agentic_trading.agents.schemas import AgentSignal, BacktestResult
from agentic_trading.backtest.engine import BacktestEngine, longest_track_record_lookback
from agentic_trading.backtest.portfolio import Portfolio
from agentic_trading.backtest.run_summary import PERIOD_METRIC_KEYS, summarize_period
from agentic_trading.config import temporary_overrides
from agentic_trading.data.loaders import lakehouse_loader
from agentic_trading.firm_record import review
from agentic_trading.firm_record.submission import META_FILE, AgenticMeta
from agentic_trading.provenance import runtime_environment
from agentic_trading.scoring.track_record import TrackRecord
from agentic_trading.scoring.weighting import EqualWeight
from agentic_trading.stage_a.spec import FirmSpec, FirmSpecRecord, firm_spec_hash

_WARMUP = 10
_INDEX = pd.bdate_range("2024-01-01", periods=_WARMUP + 45)
_DAYS = [day.date() for day in _INDEX[_WARMUP:]]
_ASSETS = ["AAPL", "MSFT"]
_LAKE_TIME = datetime(2026, 1, 5, 12, 0, tzinfo=UTC)
_LAKE_VERSION = "abcdef123456.0123456789ab"
_STARTING_CASH = 1_000_000.0
# Fields of `BacktestResult` a checkpoint does not reproduce: the decisions the engine held in
# memory, its horizon setting and the split-move registry of its own reads.
_ENGINE_ONLY_FIELDS = {"decisions_by_day", "history_horizon_days", "unfactored_split_moves"}


class _MomentumAnalyst(AnalystAgent):
    """Long after an up close and short after a down one (the reverse when `contrarian`)."""

    agent_id: str = "momentum_analyst"
    contrarian: bool = False
    strength: float = 0.4

    def generate_signal(
        self, asset: str, market_data: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> AgentSignal:
        up = float(market_data[(asset, "log_return")].iloc[-1]) >= 0.0
        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction="long" if up != self.contrarian else "short",
            confidence=self.strength,
            rationale="stub",
        )


class _ViewConstructor(PortfolioConstructor):
    """Sizes each asset at half its pod view's net signal."""

    def construct(self, views, returns_history, as_of_date, previous_weights=None):
        return {asset: 0.5 * view.net_signal for asset, view in views.items()}


class _PassThroughManager(ManagerAgent):
    def scale_factor(self, signal, market_data, as_of_date) -> float:
        return 1.0


def _commander() -> Commander:
    """Two pods over both assets, one following momentum and one fading it more weakly, so the
    book holds a net position."""
    managers = [
        _PassThroughManager(
            agent_id="momentum_manager",
            managed_analysts=[_MomentumAnalyst()],
            weighting_strategy=EqualWeight(),
            portfolio_constructor=_ViewConstructor(),
        ),
        _PassThroughManager(
            agent_id="contrarian_manager",
            managed_analysts=[
                _MomentumAnalyst(agent_id="contrarian_analyst", contrarian=True, strength=0.15)
            ],
            weighting_strategy=EqualWeight(),
            portfolio_constructor=_ViewConstructor(),
        ),
    ]
    return Commander(managers=managers, weighting_strategy=EqualWeight())


def _random_walk(seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.012, len(_INDEX)))), _INDEX)


def _frame() -> pd.DataFrame:
    columns = {}
    for seed, asset in enumerate(_ASSETS, start=11):
        close = _random_walk(seed)
        columns[(asset, "open")] = close.shift(1).fillna(close.iloc[0])
        columns[(asset, "close")] = close
        columns[(asset, "log_return")] = np.log(close / close.shift(1)).fillna(0.0)
    return pd.DataFrame(columns)


def _risk_free() -> pd.Series:
    """A daily rate stepping from 2% to 4% a year halfway through the window."""
    calendar = pd.date_range(_DAYS[0], _DAYS[-1], freq="D")
    annual = np.where(np.arange(len(calendar)) < len(calendar) // 2, 0.02, 0.04)
    return pd.Series(annual / 252.0, index=calendar)


def _spy_closes() -> pd.Series:
    return _random_walk(29).loc[pd.Timestamp(_DAYS[0]) :]


@pytest.fixture(autouse=True)
def _lift_firm_limits() -> Iterator[None]:
    with temporary_overrides(
        {
            "firm.max_single_name_weight": 100.0,
            "firm.max_gross_exposure": 1000.0,
            "firm.max_daily_var": 1000.0,
        }
    ):
        yield


def _spec_record() -> FirmSpecRecord:
    spec = FirmSpec(
        commander={"class": "agentic_trading.agents.commander.Commander"},
        roster=[
            {"agent_id": "momentum_manager", "class": "agentic_trading.agents.managers.x.M"},
            {"agent_id": "momentum_analyst", "class": "agentic_trading.agents.analysts.x.A"},
            {"agent_id": "contrarian_manager", "class": "agentic_trading.agents.managers.x.M"},
            {"agent_id": "contrarian_analyst", "class": "agentic_trading.agents.analysts.x.A"},
        ],
        settings={"stage_a": {"data_end": "2024-12-31"}},
        module_hashes={},
        universe=list(_ASSETS),
        window_start=_DAYS[0],
        window_end=_DAYS[-1],
        decision_rule="test rule",
        quantcore_sha="0" * 40,
        lake_time=_LAKE_TIME,
    )
    return FirmSpecRecord(
        spec=spec,
        spec_hash=firm_spec_hash(spec),
        lake_time=_LAKE_TIME,
        lake_version=_LAKE_VERSION,
        quantcore_version="0.0.0",
    )


class _Record:
    """A finished firm record on disk: the store's trial folder and the engine checkpoint beside
    it, and the `BacktestResult` the engine run that wrote them returned."""

    def __init__(self, store: Path) -> None:
        self.spec = _spec_record()
        self.store = store
        self.trial_dir = store / self.spec.spec_hash
        self.trial_dir.mkdir(parents=True)
        (self.trial_dir / "spec.json").write_text(self.spec.model_dump_json(indent=2))
        meta = AgenticMeta(
            counted=False,
            git_sha="1" * 40,
            src_tree_clean=True,
            declared_n_from="uncounted_default",
            environment=runtime_environment(),
        )
        (self.trial_dir / META_FILE).write_text(meta.model_dump_json(indent=2))
        self.checkpoint = store / f"{self.spec.spec_hash}.checkpoint.sqlite"
        commander = _commander()
        horizon = longest_track_record_lookback(commander)
        engine = BacktestEngine(
            commander=commander,
            portfolio=Portfolio(
                cash=_STARTING_CASH, positions={}, commission_rate=5e-4, slippage_bps=2.0
            ),
            track_record=TrackRecord(horizon_days=horizon),
            data_loader=lambda _assets, _start, _end: _frame(),
            start_date=_DAYS[0],
            end_date=_DAYS[-1],
            assets=list(_ASSETS),
            market_data_source="lakehouse data",
            execution_lag_days=1,
            checkpoint_path=self.checkpoint,
            risk_free_rate_series=_risk_free(),
            benchmark_returns=_spy_closes().pct_change().dropna(),
            history_horizon_days=horizon,
        )
        with lakehouse_loader.lake_session(_LAKE_TIME):
            self.result: BacktestResult = engine.run()


class _CatalogueOnlyLake:
    """A lake whose catalogue answers every table as an empty current one, so a `lake_session`
    opens; any other call fails. The record fixture is module-scoped, so the suite's per-test
    offline lake is not in place while it runs."""

    last_version: str | None = None

    def describe(self, table: str) -> dict[str, object]:
        return {"kind": "current", "key_columns": [], "rebuilt_whole": False, "last_write": None}


@pytest.fixture(scope="module")
def record(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Record]:
    lake = _CatalogueOnlyLake()
    with (
        pytest.MonkeyPatch.context() as patch,
        temporary_overrides(
            {
                "firm.max_single_name_weight": 100.0,
                "firm.max_gross_exposure": 1000.0,
                "firm.max_daily_var": 1000.0,
            }
        ),
    ):
        patch.setattr(lakehouse_loader, "_lake", lambda: lake)
        written = _Record(tmp_path_factory.mktemp("firm_store"))
    yield written


def _benchmarks() -> review.RecordBenchmarks:
    return review.RecordBenchmarks(
        risk_free_rate_series=_risk_free(), benchmark_closes=_spy_closes()
    )


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_record_run_is_one_worth_reading(record: _Record) -> None:
    """The fixture run has pods, pod costs, attributed sources and defined ratios, so the
    comparisons below compare numbers rather than empty containers."""
    assert set(record.result.pod_return_history) == {"momentum_manager", "contrarian_manager"}
    assert record.result.pod_costs and record.result.trade_log
    assert {"momentum_analyst", "contrarian_analyst"} <= set(record.result.source_performance)
    assert record.result.final_sharpe_ratio is not None
    assert record.result.final_information_ratio is not None


def test_record_result_is_the_engine_result(record: _Record) -> None:
    """Read back from the checkpoint with the run's own risk-free and benchmark series, every
    field of the engine's `BacktestResult` is reproduced exactly, but for what only the engine's
    memory held."""
    rebuilt = review.record_result(
        record.checkpoint,
        risk_free_rate_series=_risk_free(),
        benchmark_returns=_spy_closes().pct_change().dropna(),
        market_data_source="lakehouse data",
    )

    assert rebuilt.model_dump(exclude=_ENGINE_ONLY_FIELDS) == record.result.model_dump(
        exclude=_ENGINE_ONLY_FIELDS
    )
    assert rebuilt.unfactored_split_moves is None


def test_the_record_period_is_the_engine_results_summary(record: _Record) -> None:
    """The `record` block is `summarize_period` of the engine's own result, its information
    ratio against SPY's total return."""
    summary = review.firm_record_summary(
        record.checkpoint, record.trial_dir, benchmarks=_benchmarks()
    )

    # A checkpoint carries no split-move registry: the record's count is None, as for a run
    # resumed from its checkpoint.
    engine_result = record.result.model_copy(update={"unfactored_split_moves": None})
    expected = summarize_period(engine_result.model_dump(mode="json"))
    assert json.loads(json.dumps(summary["record"])) == json.loads(json.dumps(expected))
    for key in PERIOD_METRIC_KEYS:
        assert key in summary["record"]
    assert "calibration" not in summary and "test" not in summary
    assert summary["record"]["pod_performance"] and summary["record"]["pod_costs"]


def test_the_summary_names_the_record_and_its_lake(record: _Record) -> None:
    summary = review.firm_record_summary(
        record.checkpoint, record.trial_dir, benchmarks=_benchmarks()
    )

    experiment = summary["experiment"]
    assert experiment["spec_hash"] == record.spec.spec_hash
    assert experiment["window_start"] == _DAYS[0].isoformat()
    assert experiment["window_end"] == _DAYS[-1].isoformat()
    assert experiment["assets"] == _ASSETS
    assert experiment["managers"] == ["momentum_manager", "contrarian_manager"]
    assert experiment["analysts"] == ["momentum_analyst", "contrarian_analyst"]
    assert experiment["excluded_analysts"] == sorted(FIRM_V1_EXCLUDED_ANALYST_IDS)
    assert experiment["git_sha"] == "1" * 40
    assert experiment["counted"] is False
    assert summary["lake"] == {
        "lake_time": _LAKE_TIME.isoformat(),
        "lake_version": _LAKE_VERSION,
    }
    assert summary["market_data_source"]


def test_the_spy_baseline_starts_at_the_firms_value_and_follows_spy(record: _Record) -> None:
    summary = review.firm_record_summary(
        record.checkpoint, record.trial_dir, benchmarks=_benchmarks()
    )

    baseline = summary["baselines"][review.BENCHMARK_BASELINE]
    firm_history = summary["record"]["portfolio_value_history"]
    history = baseline["portfolio_value_history"]
    assert [day for day, _ in history] == [day for day, _ in firm_history]
    assert history[0][1] == firm_history[0][1]
    closes = _spy_closes()
    first = closes.loc[pd.Timestamp(_DAYS[0])]
    for day, value in history:
        assert value == pytest.approx(firm_history[0][1] * closes.loc[pd.Timestamp(day)] / first)
    assert baseline["final_sharpe_ratio"] is not None
    assert baseline["final_information_ratio"] is None


def test_reading_the_record_changes_nothing_on_disk(record: _Record) -> None:
    before = {path: _digest(path) for path in record.store.rglob("*") if path.is_file()}

    review.firm_record_summary(record.checkpoint, record.trial_dir, benchmarks=_benchmarks())

    after = {path: _digest(path) for path in record.store.rglob("*") if path.is_file()}
    assert after == before


def test_the_trial_folder_is_found_beside_the_checkpoint(record: _Record) -> None:
    assert review.checkpoint_trial_dir(record.checkpoint) == record.trial_dir
    summary = review.firm_record_summary(record.checkpoint, benchmarks=_benchmarks())
    assert summary["experiment"]["spec_hash"] == record.spec.spec_hash


def test_a_file_not_named_as_a_checkpoint_has_no_trial_folder(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"\.checkpoint\.sqlite"):
        review.checkpoint_trial_dir(tmp_path / "run.sqlite")


def test_a_trial_folder_of_another_record_is_refused(record: _Record, tmp_path: Path) -> None:
    other = tmp_path / "other"
    other.mkdir()
    spec = _spec_record().spec.model_copy(update={"decision_rule": "another rule"})
    other_record = FirmSpecRecord(
        spec=spec,
        spec_hash=firm_spec_hash(spec),
        lake_time=_LAKE_TIME,
        lake_version=_LAKE_VERSION,
        quantcore_version="0.0.0",
    )
    (other / "spec.json").write_text(other_record.model_dump_json())

    with pytest.raises(ValueError, match="spec hash"):
        review.firm_record_summary(record.checkpoint, other, benchmarks=_benchmarks())


def test_a_checkpoint_at_another_lake_time_is_refused(record: _Record, tmp_path: Path) -> None:
    spec = _spec_record().spec.model_copy(update={"lake_time": datetime(2026, 2, 1, tzinfo=UTC)})
    moved = FirmSpecRecord(
        spec=spec,
        spec_hash=firm_spec_hash(spec),
        lake_time=spec.lake_time,
        lake_version=_LAKE_VERSION,
        quantcore_version="0.0.0",
    )
    trial_dir = tmp_path / moved.spec_hash
    trial_dir.mkdir()
    (trial_dir / "spec.json").write_text(moved.model_dump_json())
    checkpoint = tmp_path / f"{moved.spec_hash}.checkpoint.sqlite"
    checkpoint.write_bytes(record.checkpoint.read_bytes())

    with pytest.raises(ValueError, match="lake time"):
        review.firm_record_summary(checkpoint, trial_dir, benchmarks=_benchmarks())


def test_a_lake_cache_inside_the_store_is_refused(record: _Record) -> None:
    with pytest.raises(ValueError, match="store"):
        review.firm_record_summary(
            record.checkpoint, record.trial_dir, lake_cache_dir=record.store / "lake_cache"
        )


def test_the_benchmarks_are_read_at_the_records_lake_time_and_version(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The risk-free series and SPY's closes are read over the record's window inside one lake
    session pinned to the record's lake time, lake version and data_end, through a lake answer
    cache at the given directory."""
    reads: list[tuple[str, date, date, datetime | None, str | None, date | None, bool]] = []

    def _session_state() -> tuple[datetime | None, str | None, date | None, bool]:
        session = lakehouse_loader._current_session()
        assert session is not None
        return (
            session.lake_time,
            session.lake_version,
            session.data_end,
            lakehouse_loader.lake_answer_cache_counts() is not None,
        )

    def _risk_free_reader(start: date, end: date) -> pd.Series:
        reads.append(("risk_free", start, end, *_session_state()))
        return _risk_free()

    def _closes_reader(ticker: str, start: date, end: date) -> pd.Series:
        reads.append((ticker, start, end, *_session_state()))
        return _spy_closes()

    monkeypatch.setattr(review, "load_risk_free_rate_series", _risk_free_reader)
    monkeypatch.setattr(review, "load_benchmark_closes", _closes_reader)

    benchmarks = review.load_record_benchmarks(_spec_record(), lake_cache_dir=tmp_path / "cache")

    pinned = (_LAKE_TIME, _LAKE_VERSION, date(2024, 12, 31), True)
    assert reads == [
        ("risk_free", _DAYS[0], _DAYS[-1], *pinned),
        (review.BENCHMARK_TICKER, _DAYS[0], _DAYS[-1], *pinned),
    ]
    pd.testing.assert_series_equal(benchmarks.benchmark_closes, _spy_closes())
    assert review.BENCHMARK_TICKER == "SPY"
