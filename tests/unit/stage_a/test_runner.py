# test_runner.py

"""AT-110 precompute spec item 5: `run_batch` runs a list of analysts over the NYSE sessions of a
window inside a `lake_session` cut at `settings.stage_a.data_end`, hands each `generate_signal`
point-in-time prices, stores every day through `SignalStore`, resumes from progress.json, checks
resources every `stage_a.check_every_days` days and stops cleanly when they run low."""

import contextlib
import inspect
import json
from datetime import UTC, date, datetime
from itertools import product
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from pydantic import PrivateAttr

from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.config import StageAConfig, settings, temporary_overrides
from agentic_trading.data.loaders import lakehouse_loader as ll
from agentic_trading.data.point_in_time import get_point_in_time_data
from agentic_trading.stage_a.runner import BatchResult, run_batch, system_resources_ok
from agentic_trading.stage_a.spec import RUNTIME_SETTINGS_FIELDS, SpecRecord, build_spec, spec_hash
from agentic_trading.stage_a.store import SignalStore

_UNIVERSE = ("SHY", "IEF", "TLT")
_SIGNALS = {"SHY": ("long", 0.6), "IEF": ("short", 0.2), "TLT": ("flat", 0.0)}
_START, _END = date(2010, 1, 14), date(2010, 1, 21)
# The NYSE sessions from _START to _END: 16-17 January is a weekend, 18 January Martin Luther
# King Day, which the synthetic prices nonetheless carry as a business day.
_SESSIONS = [date(2010, 1, 14), date(2010, 1, 15), date(2010, 1, 19), date(2010, 1, 20), _END]
_DATA_END = date(2024, 12, 31)
_PAST_END = date(2025, 1, 2)
_T = datetime(2026, 10, 29, tzinfo=UTC)
_META: dict[str, Any] = {
    "lake_time": _T,
    "lake_version": "f89aef2f428a",
    "quantcore_version": "0.4.0",
}
_RUNTIME_FIELDS = {
    "stage_a.min_free_memory_pct",
    "stage_a.min_free_disk_gb",
    "stage_a.check_every_days",
    "firm.history_horizon_days",
    "trials.ledger_path",
    "trials.registry_dir",
    "trials.firm_family",
}
_ERRORS: dict[str, type[Exception]] = {
    "value": ValueError,
    "holdout": ll.HoldoutReadError,
    "lake_time": ll.LakeTimeError,
}


def _signal(agent_id: str, asset: str, day: date) -> AgentSignal:
    direction, confidence = _SIGNALS[asset]
    return AgentSignal(
        agent_id=agent_id,
        asset=asset,
        as_of_date=day,
        direction=direction,
        confidence=confidence,
        rationale="",
    )


class _Recording(AnalystAgent):
    """Returns `_SIGNALS` and records each call's asset, date and frame, and the open lake
    session's time and data_end."""

    agent_id: str = "fake_recording"
    _calls: list[tuple[str, pd.Timestamp, pd.DataFrame]] = PrivateAttr(default_factory=list)
    _sessions: list[tuple[datetime | None, date | None]] = PrivateAttr(default_factory=list)

    def generate_signal(
        self, asset: str, market_data: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> AgentSignal:
        self._calls.append((asset, as_of_date, market_data))
        lake_time = ll.current_lake_time()
        with ll.lake_session() as session:
            self._sessions.append((lake_time, session.data_end))
        return _signal(self.agent_id, asset, as_of_date.date())


class _Failing(_Recording):
    """`_Recording` that raises the `_ERRORS[error]` exception for `fail_asset` on `fail_day`."""

    agent_id: str = "fake_failing"
    fail_asset: str = "IEF"
    fail_day: date = _SESSIONS[1]
    error: Literal["value", "holdout", "lake_time"] = "value"

    def generate_signal(
        self, asset: str, market_data: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> AgentSignal:
        if (asset, as_of_date.date()) == (self.fail_asset, self.fail_day):
            self._calls.append((asset, as_of_date, market_data))
            raise _ERRORS[self.error](f"{self.error} raised for {asset}")
        return super().generate_signal(asset, market_data, as_of_date)


class _SwallowsHoldout(_Recording):
    """`_Recording` that first asks FRED for a date past data_end and swallows the refusal."""

    agent_id: str = "fake_swallows_holdout"

    def generate_signal(
        self, asset: str, market_data: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> AgentSignal:
        with contextlib.suppress(ll.HoldoutReadError):
            ll.load_fred_series_from_lakehouse("DGS10", _PAST_END)
        return super().generate_signal(asset, market_data, as_of_date)


class _Resources:
    """A fake resource check: records each store_dir it is asked about and answers `answers` in
    order, then True."""

    def __init__(self, *answers: bool) -> None:
        self.calls: list[Path] = []
        self._answers = list(answers)

    def __call__(self, store_dir: Path) -> bool:
        self.calls.append(Path(store_dir))
        return self._answers.pop(0) if self._answers else True


def _market_data() -> pd.DataFrame:
    """Prices in the (asset, field) column shape of `load_price_history_from_lakehouse`, on every
    business day from before the window to after it."""
    index = pd.bdate_range("2009-12-01", "2010-02-26", name="date")
    close = pd.Series(np.linspace(100.0, 110.0, len(index)), index=index)
    fields = {
        "open": close,
        "close": close,
        "raw_close": close,
        "volume": pd.Series(1e6, index=index),
        "log_return": np.log(close / close.shift(1)),
    }
    return pd.DataFrame({(a, field): v for a in _UNIVERSE for field, v in fields.items()})


def _run(
    analysts: list[AnalystAgent],
    store_dir: Path,
    *,
    start: date = _START,
    end: date = _END,
    market: pd.DataFrame | None = None,
    resources: _Resources | None = None,
) -> BatchResult:
    """`run_batch` over `_UNIVERSE` as etf, with `_META` and a resource check that answers True."""
    return run_batch(
        analysts,
        list(_UNIVERSE),
        "etf",
        start,
        end,
        store_dir,
        _market_data() if market is None else market,
        resources_ok=_Resources() if resources is None else resources,
        **_META,
    )


def _record(analyst: AnalystAgent, start: date = _START, end: date = _END) -> SpecRecord:
    spec = build_spec(analyst, list(_UNIVERSE), start, end)
    return SpecRecord(spec=spec, spec_hash=spec_hash(spec), **_META)


def _entry(result: BatchResult, agent_id: str) -> Any:
    """The per-analyst part of `result` for `agent_id`."""
    return result.analysts[agent_id]


def _rows(path: Path) -> list[dict[str, Any]]:
    """The Parquet file's rows sorted by date, then instrument_id."""
    table = pq.read_table(path)
    day = "date" if "date" in table.schema.names else "decision_date"
    return sorted(table.to_pylist(), key=lambda row: (row[day], row["instrument_id"]))


def _days(analyst: _Recording) -> list[date]:
    """The sorted distinct decision dates `analyst` was asked about."""
    return sorted({stamp.date() for _, stamp, _ in analyst._calls})


def _progress(trial: Path) -> Any:
    return json.loads((trial / "progress.json").read_text())


def test_stage_a_settings() -> None:
    """settings.stage_a holds the stage (a) constants; only its runtime fields leave the hash."""
    analyst = _Recording()
    base = spec_hash(build_spec(analyst, list(_UNIVERSE), _START, _END))
    runtime = {
        "stage_a.min_free_memory_pct": 35.0,
        "stage_a.min_free_disk_gb": 50.0,
        "stage_a.check_every_days": 5,
    }
    unchanged = []
    for key, value in runtime.items():
        with temporary_overrides({key: value}):
            unchanged.append(spec_hash(build_spec(analyst, list(_UNIVERSE), _START, _END)))
    with temporary_overrides({"stage_a.data_end": date(2024, 12, 30)}):
        changed = spec_hash(build_spec(analyst, list(_UNIVERSE), _START, _END))
    defaults = StageAConfig()

    assert isinstance(settings.stage_a, StageAConfig)
    assert defaults.data_end == _DATA_END
    assert defaults.window_start == date(2010, 1, 4)
    assert defaults.min_free_memory_pct == 20.0
    assert defaults.min_free_disk_gb == 10.0
    assert defaults.check_every_days == 20
    assert set(RUNTIME_SETTINGS_FIELDS) == _RUNTIME_FIELDS
    assert len(RUNTIME_SETTINGS_FIELDS) == len(_RUNTIME_FIELDS)
    assert build_spec(analyst, list(_UNIVERSE), _START, _END).settings["stage_a"] == {
        "data_end": "2024-12-31",
        "window_start": "2010-01-04",
        "session_calendar": "XNYS",
        "warmup_days": 1100,
    }
    assert unchanged == [base] * len(runtime)
    assert changed != base


def test_run_batch_rejects_bad_windows(tmp_path: Path) -> None:
    """A window past settings.stage_a.data_end, read at call time, or reversed raises first."""
    store = tmp_path / "store"
    analyst, resources = _Recording(), _Resources()
    for start, end in [(_START, _PAST_END), (_END, _START)]:
        with pytest.raises(ValueError):
            _run([analyst], store, start=start, end=end, resources=resources)
    with temporary_overrides({"stage_a.data_end": date(2010, 1, 20)}), pytest.raises(ValueError):
        _run([analyst], store, resources=resources)

    assert analyst._calls == [] and resources.calls == []
    assert not store.exists() or not any(store.iterdir())

    with temporary_overrides({"stage_a.data_end": _END}):
        result = _run([analyst], store)
    assert _entry(result, analyst.agent_id).last_completed == _END
    assert set(analyst._sessions) == {(_T, _END)}


def test_run_batch_writes_the_trial(tmp_path: Path) -> None:
    """One run writes spec.json, every session's rows of the given kind, progress and counts."""
    analyst = _Recording()
    result = _run([analyst], tmp_path)
    record = _record(analyst)
    trial = tmp_path / record.spec_hash
    entry = _entry(result, analyst.agent_id)
    decisions = _rows(trial / "decisions.parquet")
    keys = [(r["decision_date"], r["instrument_kind"], r["instrument_id"]) for r in decisions]
    signals = _rows(trial / "signals.parquet")
    counts = json.loads((trial / "counts.json").read_text())

    assert result.stopped_for_resources is False
    assert (entry.spec_hash, entry.days_written, entry.last_completed) == (
        record.spec_hash,
        len(_SESSIONS),
        _END,
    )
    assert SpecRecord.model_validate_json((trial / "spec.json").read_text()) == record
    assert keys == [(day, "etf", asset) for day in _SESSIONS for asset in sorted(_UNIVERSE)]
    assert [row["target_weight"] for row in decisions] == pytest.approx(
        [-0.25, 0.75, 0.0] * len(_SESSIONS)
    )
    assert len(signals) == len(_SESSIONS) * len(_UNIVERSE)
    assert {row["status"] for row in signals} == {"active"}
    assert _progress(trial) == {"last_completed": _END.isoformat()}
    assert counts["instruments"] == {
        asset: {"active": len(_SESSIONS), "unavailable": 0, "error": 0} for asset in _UNIVERSE
    }


def test_run_batch_calls_each_asset_on_each_session(tmp_path: Path) -> None:
    """Each asset is asked once per NYSE session, at a midnight Timestamp, with point-in-time
    prices, inside a lake session at the given time cut at settings.stage_a.data_end."""
    market = _market_data()
    analyst, weekend = _Recording(), _Recording()
    _run([analyst], tmp_path / "window", market=market)
    _run([weekend], tmp_path / "weekend", start=date(2010, 1, 16), end=date(2010, 1, 19))

    calls = sorted((asset, stamp.date()) for asset, stamp, _ in analyst._calls)
    assert calls == sorted(product(_UNIVERSE, _SESSIONS))
    for _, stamp, frame in analyst._calls:
        assert isinstance(stamp, pd.Timestamp)
        assert stamp == pd.Timestamp(stamp.date())
        assert frame.index.max() <= stamp
        assert frame.equals(get_point_in_time_data(market, stamp))
    assert _days(weekend) == [date(2010, 1, 19)]
    assert set(analyst._sessions) == {(_T, _DATA_END)}
    assert ll.current_lake_time() is None


def test_run_batch_records_an_analyst_exception_as_an_error(tmp_path: Path) -> None:
    """An exception from generate_signal is that asset's error row that day; the batch goes on."""
    analyst = _Failing()
    result = _run([analyst], tmp_path)
    trial = tmp_path / _record(analyst).spec_hash
    statuses = {
        (row["date"], row["instrument_id"]): row["status"]
        for row in _rows(trial / "signals.parquet")
    }
    weights = {
        (row["decision_date"], row["instrument_id"]): row["target_weight"]
        for row in _rows(trial / "decisions.parquet")
    }
    failed = (_SESSIONS[1], "IEF")

    assert _entry(result, analyst.agent_id).days_written == len(_SESSIONS)
    assert statuses.pop(failed) == "error"
    assert set(statuses.values()) == {"active"}
    assert len(statuses) == len(_SESSIONS) * len(_UNIVERSE) - 1
    assert weights[failed] == 0.0
    assert weights[(_SESSIONS[1], "SHY")] == pytest.approx(1.0)


def test_run_batch_propagates_lake_errors(tmp_path: Path) -> None:
    """HoldoutReadError and LakeTimeError from generate_signal stop the batch on that day."""
    for kind in ("holdout", "lake_time"):
        analyst = _Failing(error=kind)
        with pytest.raises(_ERRORS[kind]):
            _run([analyst], tmp_path / kind)
        trial = tmp_path / kind / _record(analyst).spec_hash

        assert _days(analyst) == _SESSIONS[:2], kind
        assert _progress(trial) == {"last_completed": _SESSIONS[0].isoformat()}, kind
        assert ll.current_lake_time() is None


def test_run_batch_fails_on_a_swallowed_holdout_read(tmp_path: Path, fake_lake: Any) -> None:
    """An analyst that catches its own read past data_end still ends the batch in
    HoldoutReadError, the session's end-of-block check, with no query sent."""
    fake_lake.error = AssertionError("queried the lake past data_end")
    analyst = _SwallowsHoldout()

    with pytest.raises(ll.HoldoutReadError, match="load_fred_series_from_lakehouse"):
        _run([analyst], tmp_path)

    assert analyst._calls
    assert fake_lake.calls == []
    assert ll.current_lake_time() is None


def test_run_batch_resumes_and_a_rerun_writes_nothing(tmp_path: Path) -> None:
    """A batch resumes after the store's last completed day; a rerun writes nothing new."""
    record = _record(_Recording())
    store = SignalStore(tmp_path, record)
    store.open()
    first_day = [_signal(record.spec.agent_id, asset, _SESSIONS[0]) for asset in _UNIVERSE]
    store.write_day(_SESSIONS[0], first_day, {asset: asset for asset in _UNIVERSE}, "etf")
    trial = tmp_path / record.spec_hash

    resumed, rerun = _Recording(), _Recording()
    first = _entry(_run([resumed], tmp_path), resumed.agent_id)
    written = (_rows(trial / "signals.parquet"), _rows(trial / "decisions.parquet"))
    second = _entry(_run([rerun], tmp_path), rerun.agent_id)

    assert _days(resumed) == _SESSIONS[1:]
    assert (first.spec_hash, first.days_written, first.last_completed) == (
        record.spec_hash,
        len(_SESSIONS) - 1,
        _END,
    )
    assert len(written[1]) == len(_SESSIONS) * len(_UNIVERSE)
    assert rerun._calls == []
    assert (second.days_written, second.last_completed) == (0, _END)
    assert (_rows(trial / "signals.parquet"), _rows(trial / "decisions.parquet")) == written


def test_run_batch_checks_resources(tmp_path: Path) -> None:
    """resources_ok(store_dir) runs after every check_every_days-th day; on False the batch stops
    after that day, and a rerun resumes from the next."""
    signature = inspect.signature(run_batch)
    counting, stopping, quiet = _Resources(), _Resources(False), _Resources()
    first, second = _Recording(agent_id="fake_first"), _Recording(agent_id="fake_second")
    with temporary_overrides({"stage_a.check_every_days": 2}):
        _run([_Recording()], tmp_path / "count", resources=counting)
        stopped = _run([first, second], tmp_path / "stop", resources=stopping)
    stopped_first = _entry(stopped, "fake_first")
    stop_progress = _progress(tmp_path / "stop" / stopped_first.spec_hash)
    again_first = _Recording(agent_id="fake_first")
    again_second = _Recording(agent_id="fake_second")
    resumed = _run([again_first, again_second], tmp_path / "stop", resources=quiet)

    assert signature.parameters["resources_ok"].default is system_resources_ok
    assert list(inspect.signature(system_resources_ok).parameters) == ["store_dir"]
    assert counting.calls == [tmp_path / "count"] * (len(_SESSIONS) // 2)
    assert stopping.calls == [tmp_path / "stop"]
    assert stopped.stopped_for_resources is True
    assert (stopped_first.days_written, stopped_first.last_completed) == (2, _SESSIONS[1])
    assert stop_progress == {"last_completed": _SESSIONS[1].isoformat()}
    assert _days(first) == _SESSIONS[:2]
    assert second._calls == []
    assert resumed.stopped_for_resources is False
    assert _days(again_first) == _SESSIONS[2:]
    assert _entry(resumed, "fake_first").days_written == len(_SESSIONS) - 2
    assert _entry(resumed, "fake_second").days_written == len(_SESSIONS)
    assert quiet.calls == []
