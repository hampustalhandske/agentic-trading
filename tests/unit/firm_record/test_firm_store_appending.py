"""A trial store written inside `appending()` ends with the files of a store that rewrote every
Parquet file each day, byte for byte: after a whole pass, after a pass that rewrites its last
day or resumes another's, after one the block's code stops with an error, and, once reopened,
after one killed inside the block. Inside the block each day adds only a part file under
`.parts/` and progress.json."""

import shutil
from datetime import date, timedelta
from pathlib import Path

import pytest

from agentic_trading.agents.analysts import CIRRateRegimeAnalyst
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.stage_a.spec import FirmSpecRecord
from agentic_trading.stage_a.store import PARTS_DIR, DecisionRow, FirmDecisionStore, SignalStore
from unit.firm_record.test_firm_store import _SOURCES, _day_rows, _record, _row
from unit.stage_a.test_store import _RATE_IDS
from unit.stage_a.test_store import _open as _open_signal_store

_DAYS = [date(2016, 1, 4) + timedelta(days=offset) for offset in range(8)]
_RECORD = _record(window_end=_DAYS[-1])


def _store(store_dir: Path, record: FirmSpecRecord = _RECORD) -> FirmDecisionStore:
    store = FirmDecisionStore(store_dir, record, sources=_SOURCES)
    store.open()
    return store


def _rows(day: date, version: int = 0) -> list[DecisionRow]:
    """`day`'s rows; another `version` changes a weight, and an odd day has a row more."""
    rows = _day_rows(day)
    rows[0] = rows[0].model_copy(update={"target_weight": 0.3 + 0.01 * version})
    if day.toordinal() % 2:
        rows.append(_row(day, "QQQ", None, -0.05))
    return rows


def _files(store_dir: Path) -> dict[str, bytes]:
    """Every file under `store_dir`, by relative path, with its bytes."""
    return {
        path.relative_to(store_dir).as_posix(): path.read_bytes()
        for path in sorted(store_dir.rglob("*"))
        if path.is_file()
    }


def _day_by_day(store_dir: Path, days: list[date], rewrite_last: bool = False) -> dict[str, bytes]:
    """The files of a store writing `days` one rewrite at a time, the last day written twice
    with `rewrite_last`."""
    store = _store(store_dir)
    for day in days:
        store.write_day(day, _rows(day))
    if rewrite_last:
        store.write_day(days[-1], _rows(days[-1], version=1))
    return _files(store_dir)


def test_a_pass_writes_the_bytes_of_day_by_day_rewrites(tmp_path: Path) -> None:
    """A pass of every day inside `appending` ends with the same files and bytes as rewriting
    the Parquet file every day; inside the block the day only adds its part file and moves
    progress.json, and `day_rows` reads it."""
    expected = _day_by_day(tmp_path / "rewrites", _DAYS)
    store = _store(tmp_path / "appending")
    trial = tmp_path / "appending" / _RECORD.spec_hash

    with store.appending():
        for day in _DAYS:
            store.write_day(day, _rows(day))
        assert not (trial / "decisions.parquet").exists()
        assert sorted(
            path.name for path in (trial / PARTS_DIR / "decisions.parquet").iterdir()
        ) == [f"{day.isoformat()}.parquet" for day in _DAYS]
        assert store.last_completed() == _DAYS[-1]
        assert store.day_rows(_DAYS[3]) == sorted(
            _rows(_DAYS[3]), key=lambda row: (row.instrument_id, row.pod_id or "")
        )

    assert _files(tmp_path / "appending") == expected
    assert not (trial / PARTS_DIR).exists()


def test_a_resumed_pass_and_a_rewritten_day_write_the_same_bytes(tmp_path: Path) -> None:
    """A second pass resuming a first one, rewriting the first's last day inside its block and
    then rewriting its own last day, ends with the bytes of the same writes made one rewrite at a
    time."""
    expected_dir = tmp_path / "rewrites"
    store = _store(expected_dir)
    for day in _DAYS[:4]:
        store.write_day(day, _rows(day))
    store.write_day(_DAYS[3], _rows(_DAYS[3], version=1))
    for day in _DAYS[4:]:
        store.write_day(day, _rows(day))
    store.write_day(_DAYS[-1], _rows(_DAYS[-1], version=1))
    expected = _files(expected_dir)

    first = _store(tmp_path / "appending")
    with first.appending():
        for day in _DAYS[:4]:
            first.write_day(day, _rows(day))
    second = _store(tmp_path / "appending")
    with second.appending():
        second.write_day(_DAYS[3], _rows(_DAYS[3], version=1))
        for day in _DAYS[4:]:
            second.write_day(day, _rows(day))
        second.write_day(_DAYS[-1], _rows(_DAYS[-1], version=1))

    assert _files(tmp_path / "appending") == expected


def test_an_error_inside_the_block_still_writes_the_days_before_it(tmp_path: Path) -> None:
    """When the block's code raises, the days written before it are merged into the Parquet file
    as day-by-day rewrites would have left them, and the error propagates."""
    expected = _day_by_day(tmp_path / "rewrites", _DAYS[:3])
    store = _store(tmp_path / "appending")

    with pytest.raises(RuntimeError, match="stopped"), store.appending():
        for day in _DAYS[:3]:
            store.write_day(day, _rows(day))
        raise RuntimeError("stopped")

    assert _files(tmp_path / "appending") == expected


@pytest.mark.parametrize("unfinished_day", [False, True])
def test_a_pass_killed_inside_the_block_is_merged_when_reopened(
    tmp_path: Path, unfinished_day: bool
) -> None:
    """A pass killed inside the block leaves its part files; opening the store again merges
    those of the days progress.json counts, drops a part of a day it does not count (written
    before the kill could record the day) and removes `.parts/`, leaving the files of day-by-day
    rewrites of the counted days, from which a further pass continues. The killed pass's folder
    is a copy taken inside a running block."""
    running_dir, store_dir = tmp_path / "running", tmp_path / "appending"
    running = _store(running_dir)
    with running.appending():
        for day in _DAYS[:5]:
            running.write_day(day, _rows(day))
        shutil.copytree(running_dir, store_dir)
    trial = store_dir / _RECORD.spec_hash
    assert (trial / PARTS_DIR).exists()
    if unfinished_day:
        parts = trial / PARTS_DIR / "decisions.parquet"
        (parts / f"{_DAYS[5].isoformat()}.parquet").write_bytes(
            (parts / f"{_DAYS[4].isoformat()}.parquet").read_bytes()
        )

    reopened = _store(store_dir)

    assert reopened.last_completed() == _DAYS[4]
    assert not (trial / PARTS_DIR).exists()
    assert _files(store_dir) == _day_by_day(tmp_path / "rewrites", _DAYS[:5])
    with reopened.appending():
        for day in _DAYS[5:]:
            reopened.write_day(day, _rows(day))
    assert _files(store_dir) == _day_by_day(tmp_path / "all", _DAYS)


def test_appending_refuses_nesting_and_a_rewind(tmp_path: Path) -> None:
    """A second `appending` inside one, and a `rewind` inside one, raise `RuntimeError`."""
    store = _store(tmp_path)
    with store.appending():
        store.write_day(_DAYS[0], _rows(_DAYS[0]))
        with pytest.raises(RuntimeError, match="already appending"), store.appending():
            pass
        with pytest.raises(RuntimeError, match="rewind"):
            store.rewind(None)


def _signals(day: date) -> list[AgentSignal]:
    return [
        AgentSignal(
            agent_id="test",
            asset=asset,
            as_of_date=day,
            direction="long" if index % 2 else "short",
            confidence=0.1 * (index + 1),
            rationale="",
            source_as_of=day - timedelta(days=1),
            sample_count=10 + index,
        )
        for index, asset in enumerate(_RATE_IDS)
    ]


def test_a_stage_a_pass_writes_the_bytes_of_day_by_day_rewrites(tmp_path: Path) -> None:
    """A stage (a) store, with its two Parquet files, ends a pass inside `appending` with the
    bytes of day-by-day rewrites, and its counts are those of the same rows."""
    days = [date(2010, 1, 4) + timedelta(days=offset) for offset in range(5)]
    rewrites, _ = _open_signal_store(tmp_path / "rewrites", CIRRateRegimeAnalyst())
    for day in days:
        rewrites.write_day(day, _signals(day), _RATE_IDS, "etf")
    appending: SignalStore
    appending, _ = _open_signal_store(tmp_path / "appending", CIRRateRegimeAnalyst())

    with appending.appending():
        for day in days:
            appending.write_day(day, _signals(day), _RATE_IDS, "etf")
        assert appending.counts() == rewrites.counts()

    assert _files(tmp_path / "appending") == _files(tmp_path / "rewrites")
