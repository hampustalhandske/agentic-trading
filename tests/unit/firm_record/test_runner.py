# test_runner.py

"""AT-125: `run_firm_record` runs a firm through the backtest engine on every XNYS session of
its window and writes the Commander's daily pod targets as a decision_file.v1 trial through the
stage (a) store, with the firm spec beside it. An instrument without a bar on a session gets an
unavailable, weight-0 row per pod in its scope; a run stopped after `max_days` resumes from its
engine checkpoint and ends with the rows of an unbroken run; with the agentic-firm-v1 registry
file present the run is registered before its first day and records one trial row; no
performance figure leaves the run."""

import json
import logging
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from pydantic import PrivateAttr

from agentic_trading.agents.base import AnalystAgent, DataRequirement, ManagerAgent
from agentic_trading.agents.commander import Commander
from agentic_trading.agents.portfolio_construction import PortfolioConstructor
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.backtest import checkpoint_store
from agentic_trading.data.loaders import lakehouse_loader
from agentic_trading.data.loaders.lakehouse_loader import HoldoutReadError
from agentic_trading.data.trading_calendar import sessions_between
from agentic_trading.firm_record import runner as runner_module
from agentic_trading.firm_record.runner import FirmRecordResult, run_firm_record
from agentic_trading.scoring.weighting import EqualWeight
from agentic_trading.stage_a.decisions import check_source_as_of
from agentic_trading.stage_a.store import DecisionRow
from agentic_trading.trials.ledger import TrialLedger
from agentic_trading.trials.rows import FIRM_HYPOTHESIS_ID

_T = datetime(2026, 11, 6, tzinfo=UTC)
_WARMUP = pd.DatetimeIndex(sessions_between(date(2015, 12, 1), date(2015, 12, 31)))
_START, _END = date(2016, 1, 4), date(2016, 1, 22)
_SESSIONS = pd.DatetimeIndex(sessions_between(_START, _END))
_LATE_START = _SESSIONS[4]
_GAP_DAY = _SESSIONS[6]
# A session the frame holds no row for at all: every instrument lacks a bar on it.
_HOLE_DAY = _SESSIONS[9]
_UNIVERSE = ["SPY", "LATE-USD", "GAP"]
_REGISTRATION = """id: agentic-firm-v1
family: agentic-firm
registered_at: 2026-11-07T09:00:00Z   # UTC, set when the master approves
kind: firm
metric_kind: return
"""


class _FredAnalyst(AnalystAgent):
    """Long at a confidence that alternates by day, resting on a FRED value published two
    calendar days before the decision day."""

    agent_id: str = "fred_stub_analyst"
    data_requirements: tuple[DataRequirement, ...] = (
        DataRequirement(source="fred", identifier="DGS3MO", asset_scoped=False, description="x"),
    )
    _calls: int = PrivateAttr(default=0)

    def generate_signal(
        self, asset: str, market_data: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> AgentSignal:
        self._calls += 1
        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction="long" if self._calls % 3 else "short",
            confidence=0.3 + 0.1 * (self._calls % 4),
            rationale="stub",
            source_as_of=(as_of_date - pd.Timedelta(days=2)).date(),
        )


class _PriceAnalyst(AnalystAgent):
    agent_id: str = "price_stub_analyst"

    def generate_signal(
        self, asset: str, market_data: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> AgentSignal:
        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction="long",
            confidence=0.5,
            rationale="stub",
        )


class _HoldoutPeekingAnalyst(_PriceAnalyst):
    """Asks for data past the session's data_end and swallows the refusal."""

    agent_id: str = "peeking_stub_analyst"

    def generate_signal(
        self, asset: str, market_data: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> AgentSignal:
        try:
            lakehouse_loader._guard_data_end(date(2025, 6, 2), "peek")
        except HoldoutReadError:
            pass
        return super().generate_signal(asset, market_data, as_of_date)


class _ViewConstructor(PortfolioConstructor):
    def construct(self, views, returns_history, as_of_date, previous_weights=None):
        return {asset: 0.3 * view.net_signal for asset, view in views.items()}


class _PassThroughManager(ManagerAgent):
    def scale_factor(self, signal, market_data, as_of_date) -> float:
        return 1.0


def _commander(peek: bool = False) -> Commander:
    def manager(agent_id: str, analyst: AnalystAgent, assets: tuple[str, ...] | None) -> Any:
        return _PassThroughManager(
            agent_id=agent_id,
            managed_analysts=[analyst],
            weighting_strategy=EqualWeight(),
            portfolio_constructor=_ViewConstructor(),
            assets=assets,
        )

    first: AnalystAgent = _HoldoutPeekingAnalyst() if peek else _FredAnalyst()
    return Commander(
        managers=[
            manager("broad_manager", first, None),
            manager("crypto_manager", _PriceAnalyst(), ("LATE-USD",)),
        ],
        weighting_strategy=EqualWeight(),
    )


def _frame() -> pd.DataFrame:
    """Warm-up then the window; LATE-USD from `_LATE_START`, GAP without `_GAP_DAY`, and no
    row at all on `_HOLE_DAY`."""
    index = _WARMUP.append(_SESSIONS)
    rng = np.random.default_rng(3)
    columns: dict[tuple[str, str], pd.Series] = {}
    for asset in _UNIVERSE:
        close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, len(index)))), index)
        if asset == "LATE-USD":
            close[close.index < _LATE_START] = np.nan
        if asset == "GAP":
            close[_GAP_DAY] = np.nan
        present = close.dropna()
        columns[(asset, "open")] = close
        columns[(asset, "close")] = close
        columns[(asset, "log_return")] = np.log(present / present.shift(1)).reindex(index)
    return pd.DataFrame(columns).drop(index=_HOLE_DAY)


def _run(store_dir: Path, **kwargs: Any) -> FirmRecordResult:
    options: dict[str, Any] = {
        "lake_time": _T,
        "lake_version": "lake",
        "quantcore_version": "0.3.1",
    } | kwargs
    commander = options.pop("commander", None) or _commander()
    return run_firm_record(commander, _UNIVERSE, _START, _END, store_dir, _frame(), **options)


def _rows(store_dir: Path, result: FirmRecordResult) -> list[dict[str, Any]]:
    return pq.read_table(store_dir / result.spec_hash / "decisions.parquet").to_pylist()


def test_record_decides_on_every_session(tmp_path: Path) -> None:
    """Every XNYS session of the window has rows, the hole day included; the result is complete
    and uncounted without a ledger."""
    result = _run(tmp_path)
    days = sorted({row["decision_date"] for row in _rows(tmp_path, result)})

    assert days == [session.date() for session in _SESSIONS]
    assert result.days_written == len(_SESSIONS)
    assert result.last_completed == _END
    assert result.complete and not result.counted


def test_no_bar_sessions_carry_the_held_targets(tmp_path: Path) -> None:
    """Before its first bar LATE-USD, never decided, has one unavailable weight-0 row per pod in
    its scope. On GAP's missing session, and on the hole day for every instrument, each pod's
    row is active at its target of the session before, with that row's source and
    source_as_of. No non-active row carries a weight."""
    result = _run(tmp_path)
    rows = _rows(tmp_path, result)

    def at(day: pd.Timestamp, instrument: str) -> list[tuple[Any, ...]]:
        return sorted(
            (row["pod_id"], row["status"], row["target_weight"], row["source"], row["source_as_of"])
            for row in rows
            if row["decision_date"] == day.date() and row["instrument_id"] == instrument
        )

    def carried(day: pd.Timestamp, instrument: str) -> list[tuple[Any, ...]]:
        before = _SESSIONS[_SESSIONS.get_loc(day) - 1]
        return [("active", *row[2:]) for row in at(before, instrument)]

    unavailable = [
        ("broad_manager", "unavailable", 0.0, None, None),
        ("crypto_manager", "unavailable", 0.0, None, None),
    ]
    assert at(_SESSIONS[0], "LATE-USD") == unavailable
    assert [row[1:] for row in at(_GAP_DAY, "GAP")] == carried(_GAP_DAY, "GAP")
    assert at(_GAP_DAY, "GAP")[0][2] != 0.0
    for instrument in _UNIVERSE:
        assert [row[1:] for row in at(_HOLE_DAY, instrument)] == carried(_HOLE_DAY, instrument)
    assert all(row["target_weight"] == 0.0 for row in rows if row["status"] != "active")
    late = [row for row in rows if row["instrument_id"] == "LATE-USD"]
    assert {row["instrument_kind"] for row in late} == {"crypto"}
    assert not any(
        row["target_weight"] != 0.0 for row in late if row["decision_date"] < _LATE_START.date()
    )


def test_carried_row_keeps_the_setting_days_source(tmp_path: Path) -> None:
    """GAP's broad row on its missing session names the source_as_of of the session before,
    which set the target, and the row passed the stage (a) audit as it was written."""
    result = _run(tmp_path)
    rows = _rows(tmp_path, result)
    before = _SESSIONS[_SESSIONS.get_loc(_GAP_DAY) - 1].date()
    (gap_row,) = [
        r for r in rows if r["decision_date"] == _GAP_DAY.date() and r["instrument_id"] == "GAP"
    ]

    assert gap_row["source"] == "fred"
    assert gap_row["source_as_of"] == before - timedelta(days=2)
    check_source_as_of(gap_row["source_as_of"], _GAP_DAY.date(), gap_row["source"])


def test_rows_carry_source_and_pod(tmp_path: Path) -> None:
    """An active row of the FRED pod names fred and its publication date; the price-only pod's
    rows name no source; every row has a pod_id."""
    result = _run(tmp_path)
    rows = _rows(tmp_path, result)
    broad = [r for r in rows if r["pod_id"] == "broad_manager" and r["status"] == "active"]
    crypto = [r for r in rows if r["pod_id"] == "crypto_manager"]

    assert broad and all(r["source"] == "fred" for r in broad)
    assert all(
        (r["decision_date"] - r["source_as_of"]).days >= 2 for r in broad if r["source_as_of"]
    )
    assert any((r["decision_date"] - r["source_as_of"]).days == 2 for r in broad)
    assert crypto and all(r["source"] is None and r["source_as_of"] is None for r in crypto)
    assert all(row["pod_id"] for row in rows)
    sources = json.loads((tmp_path / result.spec_hash / "sources.json").read_text())
    assert sources == [{"name": "fred", "can_publish_after_close": True}]


def test_paused_record_resumes_to_the_unbroken_rows(tmp_path: Path) -> None:
    """A record stopped after 3 days, then resumed by a fresh firm, ends with exactly the rows
    of an unbroken record."""
    paused = _run(tmp_path / "paused", max_days=3)
    assert (paused.days_written, paused.complete) == (3, False)
    assert paused.last_completed == _SESSIONS[2].date()
    resumed = _run(tmp_path / "paused")
    unbroken = _run(tmp_path / "unbroken")
    again = _run(tmp_path / "unbroken")

    assert resumed.spec_hash == unbroken.spec_hash
    assert resumed.days_written == len(_SESSIONS) - 3 and resumed.complete
    assert _rows(tmp_path / "paused", resumed) == _rows(tmp_path / "unbroken", unbroken)
    assert again.days_written == 0 and again.complete


def test_trial_dir_holds_the_submission_only(tmp_path: Path) -> None:
    """The trial directory holds spec, sources, decisions and progress; the engine checkpoint,
    which carries the book's value, sits outside it; the result carries no performance."""
    result = _run(tmp_path)
    trial = tmp_path / result.spec_hash

    assert sorted(path.name for path in trial.iterdir()) == [
        "decisions.parquet",
        "progress.json",
        "sources.json",
        "spec.json",
    ]
    assert set(FirmRecordResult.model_fields) == {
        "spec_hash",
        "days_written",
        "last_completed",
        "complete",
        "counted",
    }
    spec = json.loads((trial / "spec.json").read_text())
    assert spec["spec_hash"] == result.spec_hash
    assert spec["spec"]["universe"] == _UNIVERSE


def test_no_performance_figure_is_logged(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """The engine's per-day value and final Sharpe lines are not logged during a record, while
    the run's other log lines still are."""
    package_logger = logging.getLogger("agentic_trading")
    previous_level = package_logger.level
    package_logger.addHandler(caplog.handler)
    package_logger.setLevel(logging.DEBUG)
    try:
        _run(tmp_path)
    finally:
        package_logger.setLevel(previous_level)
        package_logger.removeHandler(caplog.handler)
    text = caplog.text

    assert "Firm record" in text
    assert "Sharpe" not in text and "portfolio value" not in text


def _ledger(tmp_path: Path, registered: bool) -> tuple[TrialLedger, Path]:
    repo = tmp_path / "repo"
    registry = repo / "research" / "registry"
    registry.mkdir(parents=True)
    if registered:
        (registry / f"{FIRM_HYPOTHESIS_ID}.yaml").write_text(_REGISTRATION)
    return TrialLedger(repo / "research" / "ledger.duckdb", repo), repo


def test_registered_record_writes_one_trial_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the registry file present the hypothesis is registered and one trial row is
    recorded under it, keyed by the spec hash, with n the sessions written; spec.json records
    the commit the record ran from."""
    monkeypatch.setattr(runner_module, "source_tree_state", lambda: ("c" * 40, False))
    ledger, repo = _ledger(tmp_path, registered=True)
    result = _run(tmp_path / "store", ledger=ledger, repo_root=repo)
    spec = json.loads((tmp_path / "store" / result.spec_hash / "spec.json").read_text())
    assert spec["git_sha"] == "c" * 40
    with duckdb.connect(str(ledger.path), read_only=True) as con:
        trials = con.execute(
            "SELECT hypothesis_id, family, config_hash, n, metric_name, metric_value FROM trials"
        ).fetchall()
        registrations = con.execute("SELECT id, family FROM registrations").fetchall()

    assert result.counted
    assert registrations == [(FIRM_HYPOTHESIS_ID, "agentic-firm")]
    assert trials == [
        (FIRM_HYPOTHESIS_ID, "agentic-firm", result.spec_hash, len(_SESSIONS), "submitted", None)
    ]


@pytest.mark.parametrize("state", [("c" * 40, True), (None, None)], ids=["dirty", "no-git"])
def test_counted_record_needs_a_clean_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: tuple[str | None, bool | None]
) -> None:
    """A record that would be counted is refused, before anything is written or registered,
    when src/ has changes or no known commit."""
    monkeypatch.setattr(runner_module, "source_tree_state", lambda: state)
    ledger, repo = _ledger(tmp_path, registered=True)

    with pytest.raises(ValueError, match="clean src/ tree"):
        _run(tmp_path / "store", ledger=ledger, repo_root=repo)
    assert not (tmp_path / "store").exists() and not ledger.path.exists()


def test_a_kill_between_the_rows_and_the_checkpoint_redoes_the_day(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A record killed after a day's rows are written but before its checkpoint holds the day
    resumes by dropping that day's rows and computing it again, ending with the rows of an
    unbroken record."""
    unbroken = _run(tmp_path / "unbroken")
    append_day = checkpoint_store.append_day
    calls = {"n": 0}

    def killed(*args: Any, **kwargs: Any) -> None:
        calls["n"] += 1
        if calls["n"] == 4:
            raise KeyboardInterrupt
        append_day(*args, **kwargs)

    monkeypatch.setattr(checkpoint_store, "append_day", killed)
    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path / "killed")
    monkeypatch.setattr(checkpoint_store, "append_day", append_day)
    trial = tmp_path / "killed" / unbroken.spec_hash
    progress = json.loads((trial / "progress.json").read_text())
    assert progress == {"last_completed": _SESSIONS[3].date().isoformat()}

    resumed = _run(tmp_path / "killed", commander=_commander())

    assert resumed.complete and resumed.days_written == len(_SESSIONS) - 3
    assert _rows(tmp_path / "killed", resumed) == _rows(tmp_path / "unbroken", unbroken)


def test_a_store_refusal_stops_before_the_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rows the store refuses (a net weight beyond 6) stop the record with the store and the
    engine checkpoint both at the day before."""
    real_rows = runner_module.firm_day_rows

    def refused(day: date, *args: Any, **kwargs: Any) -> list[DecisionRow]:
        rows = real_rows(day, *args, **kwargs)
        if day == _SESSIONS[2].date():
            rows = [
                *rows,
                *(
                    DecisionRow(
                        decision_date=day,
                        instrument_kind="equity",
                        instrument_id="SPY",
                        target_weight=4.0,
                        pod_id=pod,
                    )
                    for pod in ("x_pod", "y_pod")
                ),
            ]
        return rows

    monkeypatch.setattr(runner_module, "firm_day_rows", refused)
    with pytest.raises(ValueError, match="nets"):
        _run(tmp_path)
    spec_hash = next(path.name for path in tmp_path.iterdir() if path.is_dir())
    progress = json.loads((tmp_path / spec_hash / "progress.json").read_text())

    assert progress == {"last_completed": _SESSIONS[1].date().isoformat()}
    assert (
        runner_module._checkpoint_last_day(tmp_path / f"{spec_hash}.checkpoint.sqlite")
        == _SESSIONS[1].date()
    )


def test_unregistered_record_writes_no_trial_row(tmp_path: Path) -> None:
    """Without the registry file the record is uncounted and the ledger is left untouched."""
    ledger, repo = _ledger(tmp_path, registered=False)
    result = _run(tmp_path / "store", ledger=ledger, repo_root=repo)

    assert not result.counted and result.complete
    assert not ledger.path.exists()


def test_refusals(tmp_path: Path) -> None:
    """A window ending after stage_a.data_end, or a ledger without a repo root, is refused
    before anything is written."""
    frame = _frame()
    with pytest.raises(ValueError, match="data_end"):
        run_firm_record(
            _commander(), _UNIVERSE, _START, date(2025, 1, 2), tmp_path, frame,
            lake_time=_T, lake_version="v", quantcore_version="q",
        )  # fmt: skip
    with pytest.raises(ValueError, match="repo_root"):
        _run(tmp_path, ledger=TrialLedger(tmp_path / "l.duckdb", tmp_path))
    assert list(tmp_path.iterdir()) == []


def test_holdout_read_refuses_the_day(tmp_path: Path) -> None:
    """A read past data_end refused and caught inside a day stops the record before the day
    is written."""
    with pytest.raises(HoldoutReadError):
        _run(tmp_path, commander=_commander(peek=True))
    progress = list(tmp_path.glob("*/progress.json"))
    assert progress == []
