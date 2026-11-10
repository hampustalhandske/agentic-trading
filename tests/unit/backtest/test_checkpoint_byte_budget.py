# test_checkpoint_byte_budget.py

"""The checkpoint's bytes per session stay within a budget.

The determinism tests' firm (`tests/unit/determinism/_firm.py`: four analysts in three pods over
four instruments) runs through `BacktestEngine` for `_SESSIONS` sessions with a checkpoint and
the firm's history horizon, against a `FakeLake` holding seeded DGS3MO vintages. After sessions
10, 20 and 30 each table's bytes are read: the payload bytes `dbstat` reports, or, in an SQLite
built without `dbstat`, the bytes of the pages the table takes copied alone into a database of
512-byte pages.

The fixture decides on 16 signals a session. Each budget is a number of bytes per signal per
session, applied to the growth from session 10 to 30, and was set a fifth to a third above what
the fixture measured with `dbstat` when it was set (in brackets, then without `dbstat`):

- decisions: 80 B (63 B, 67 B; firm-v1's days, whose 921 signals repeat one another far more
  than the fixture's 16, compress to 40 B a signal);
- track_record: 45 B (34 B, 42 B), at most 40 KB a session for firm-v1's 921 signals;
- every table together: 500 B (419 B, 477 B), most of it the trades, sizing diagnostics, pod
  costs and capital usage the fixture's instruments and pods write every session whatever
  their signal count.

The growth from session 20 to 30 of decisions, track_record, signal_tally and every table
together stays within 1.1 times their growth from session 10 to 20. The cursor, the Commander's
risk state aside (it holds every session's pod returns), and carried_state do not grow with the
sessions, and the cursor holds no pending decisions. A negative control replays the run's
checkpoint writes with each day's decisions stored uncompressed, and requires the budgets to
break."""

import copy
import sqlite3
from collections.abc import Iterator
from contextlib import closing
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from agentic_trading.backtest import checkpoint_store
from agentic_trading.backtest.engine import BacktestEngine
from agentic_trading.backtest.engine_inputs import EngineInputs
from agentic_trading.config import settings
from agentic_trading.data.loaders import lakehouse_loader
from agentic_trading.data.trading_calendar import sessions_between
from agentic_trading.scoring.track_record import TrackRecord
from conftest import FakeLake
from unit.determinism import _firm

_SESSIONS = 30
_SNAPSHOTS = (10, 20, 30)
# Bytes per signal per session.
_DECISIONS_BUDGET = 80.0
_TRACK_RECORD_BUDGET = 45.0
_TOTAL_BUDGET = 500.0
# The most the growth of sessions 20 to 30 may exceed that of sessions 10 to 20, as a ratio.
_GROWTH_RATIO = 1.1
# The most a table that does not grow with the sessions may change from session 10 to 30.
_FLAT_SLACK_BYTES = 256

_WARMUP = pd.DatetimeIndex(sessions_between(date(2015, 3, 2), date(2015, 12, 31)))
_WINDOW = pd.DatetimeIndex(sessions_between(date(2016, 1, 4), date(2016, 3, 31)))[:_SESSIONS]


def _market_frame() -> pd.DataFrame:
    """`_firm.market_frame`'s seeded columns over the warm-up and `_SESSIONS` sessions."""
    index = _WARMUP.append(_WINDOW)
    rng = np.random.default_rng(11)
    columns: dict[tuple[str, str], pd.Series] = {}
    for asset in _firm.UNIVERSE:
        close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, len(index)))), index)
        columns[(asset, "open")] = close.shift(1).fillna(close.iloc[0])
        columns[(asset, "close")] = close
        columns[(asset, "log_return")] = np.log(close / close.shift(1))
    return pd.DataFrame(columns)


def _lake() -> FakeLake:
    """A `FakeLake` holding a DGS3MO vintage per weekday through the window, published the next
    day."""
    days = pd.bdate_range("2015-10-01", _WINDOW[-1])
    rng = np.random.default_rng(5)
    lake = FakeLake()
    lake.add(
        "fred_vintages",
        pd.DataFrame(
            {
                "series_id": "DGS3MO",
                "date": days.date,
                "realtime_start": (days + pd.Timedelta(days=1)).date,
                "value": 0.2 + np.cumsum(rng.normal(0.0, 0.01, len(days))),
            }
        ),
        ("date", "realtime_start"),
        key_columns=["series_id", "date", "realtime_start"],
    )
    return lake


def _has_dbstat() -> bool:
    with closing(sqlite3.connect(":memory:")) as connection:
        try:
            connection.execute("SELECT COUNT(*) FROM dbstat").fetchone()
        except sqlite3.OperationalError:
            return False
    return True


def table_bytes(path: Path) -> dict[str, int]:
    """Each table's bytes in the checkpoint at `path`, its indexes included: `dbstat`'s payload
    bytes, or without `dbstat` the page bytes of the table copied alone into a database of
    512-byte pages."""
    with closing(sqlite3.connect(path)) as connection:
        owners = dict(connection.execute("SELECT name, tbl_name FROM sqlite_master"))
        tables = [
            name
            for (name,) in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
            )
        ]
        sizes = dict.fromkeys(tables, 0)
        if _has_dbstat():
            for name, payload in connection.execute(
                "SELECT name, SUM(payload) FROM dbstat GROUP BY name"
            ):
                owner = owners.get(name, name)
                if owner in sizes:
                    sizes[owner] += int(payload)
            return sizes
        for table in tables:
            create = connection.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
            ).fetchone()[0]
            with closing(sqlite3.connect(":memory:")) as alone:
                alone.execute("PRAGMA page_size = 512")
                alone.execute(create)
                columns = len(connection.execute(f"SELECT * FROM {table} LIMIT 0").description)
                alone.executemany(
                    f"INSERT INTO {table} VALUES ({', '.join('?' * columns)})",
                    connection.execute(f"SELECT * FROM {table}"),
                )
                (pages,) = alone.execute("PRAGMA page_count").fetchone()
                sizes[table] = int(pages) * 512
        return sizes


def _cursor_lengths(path: Path) -> dict[str, int | None]:
    """The byte length of each cursor column but the Commander's risk state."""
    with closing(sqlite3.connect(path)) as connection:
        columns = [
            row[1]
            for row in connection.execute("PRAGMA table_info(cursor)")
            if row[1] != "commander_risk_state"
        ]
        values = connection.execute(
            f"SELECT {', '.join(f'length({column})' for column in columns)} FROM cursor"
        ).fetchone()
    return dict(zip(columns, values, strict=True))


class _Run:
    """What the fixture's run left: table bytes and cursor column lengths after each snapshot
    session, the signals decided per session, and every `append_day` call's arguments."""

    def __init__(self) -> None:
        self.sizes: dict[int, dict[str, int]] = {}
        self.cursor: dict[int, dict[str, int | None]] = {}
        self.signals = 0
        self.appends: list[tuple[tuple[Any, ...], dict[str, Any]]] = []


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Run]:
    """The fixture's firm run for `_SESSIONS` sessions with a checkpoint."""
    path = tmp_path_factory.mktemp("budget") / "checkpoint.sqlite"
    inputs = EngineInputs(
        assets=_firm.UNIVERSE, market_data=_market_frame(), market_data_source="fixture"
    )
    horizon = settings.firm.history_horizon_days
    engine = BacktestEngine(
        commander=_firm.build_commander(),
        portfolio=inputs.portfolio(),
        track_record=TrackRecord(horizon_days=horizon),
        data_loader=inputs.data_loader,
        start_date=_WINDOW[0].date(),
        end_date=_WINDOW[-1].date(),
        assets=inputs.assets,
        market_data_source=inputs.market_data_source,
        execution_lag_days=settings.firm.execution_lag_days,
        checkpoint_path=path,
        history_horizon_days=horizon,
    )
    result = _Run()
    append_day = checkpoint_store.append_day

    def recording(*args: Any, **kwargs: Any) -> None:
        # A copy: the engine hands over some of its dicts, which it changes on later days.
        result.appends.append(copy.deepcopy((args[1:], kwargs)))
        result.signals += sum(len(d.contributing_signals) for d in kwargs["decisions"])
        append_day(*args, **kwargs)

    def snapshot(index: int, *_args: object) -> None:
        if index + 1 in _SNAPSHOTS:
            result.sizes[index + 1] = table_bytes(path)
            result.cursor[index + 1] = _cursor_lengths(path)

    with pytest.MonkeyPatch.context() as patch:
        lake = _lake()
        patch.setattr(lakehouse_loader, "_lake", lambda: lake)
        patch.setattr(lakehouse_loader, "EMPTY_READ_RETRY_SECONDS", 0.0)
        patch.setattr(checkpoint_store, "append_day", recording)
        with lakehouse_loader.lake_session(_firm.LAKE_TIME):
            engine.run(on_day_complete=snapshot)
    yield result


def _replay(run: _Run, path: Path) -> dict[int, dict[str, int]]:
    """Table bytes after each snapshot session of `run`'s checkpoint writes made again at
    `path`."""
    connection = checkpoint_store.connect(path)
    sizes: dict[int, dict[str, int]] = {}
    for session, (args, kwargs) in enumerate(run.appends, start=1):
        checkpoint_store.append_day(connection, *args, **kwargs)
        if session in _SNAPSHOTS:
            sizes[session] = table_bytes(path)
    connection.close()
    return sizes


def _per_signal(run: _Run, sizes: dict[int, dict[str, int]], tables: list[str] | None) -> float:
    """The growth of `tables` (every table for `None`) from session 10 to 30, per signal per
    session."""
    names = list(sizes[30]) if tables is None else tables
    growth = sum(sizes[30].get(name, 0) - sizes[10].get(name, 0) for name in names)
    per_session = run.signals / _SESSIONS
    return growth / (_SNAPSHOTS[2] - _SNAPSHOTS[0]) / per_session


def _breaches(run: _Run, sizes: dict[int, dict[str, int]]) -> list[str]:
    """Each budget `sizes` breaks."""
    found = []
    for tables, budget in (
        (["decisions"], _DECISIONS_BUDGET),
        (["track_record"], _TRACK_RECORD_BUDGET),
        (None, _TOTAL_BUDGET),
    ):
        value = _per_signal(run, sizes, tables)
        if value > budget:
            found.append(f"{tables or 'every table'}: {value:.1f} B > {budget} B per signal")
    return found


def test_the_fixture_decides_sixteen_signals_a_session(run: _Run) -> None:
    """The budgets are per signal of a fixture deciding 16 signals each session."""
    assert len(run.appends) == _SESSIONS
    assert run.signals == 16 * _SESSIONS


def test_bytes_per_signal_stay_within_budget(run: _Run) -> None:
    """Decisions, the track record and every table together grow within their budgets."""
    assert _breaches(run, run.sizes) == []


@pytest.mark.parametrize("tables", [["decisions"], ["track_record"], ["signal_tally"], None])
def test_growth_does_not_speed_up(run: _Run, tables: list[str] | None) -> None:
    """The growth from session 20 to 30 is at most 1.1 times the growth from session 10 to
    20."""
    names = list(run.sizes[30]) if tables is None else tables

    def growth(first: int, last: int) -> int:
        return sum(run.sizes[last].get(name, 0) - run.sizes[first].get(name, 0) for name in names)

    assert growth(20, 30) <= _GROWTH_RATIO * growth(10, 20)


def test_cursor_and_carried_state_do_not_grow(run: _Run) -> None:
    """carried_state and every cursor column but the risk state keep their size from session 10
    to 30, and the cursor stores no pending decisions."""
    sizes = run.sizes
    assert abs(sizes[30]["carried_state"] - sizes[10]["carried_state"]) <= _FLAT_SLACK_BYTES
    for session in _SNAPSHOTS:
        assert run.cursor[session]["pending_decisions"] is None

    def lengths(session: int) -> int:
        return sum(length or 0 for length in run.cursor[session].values())

    assert abs(lengths(30) - lengths(10)) <= _FLAT_SLACK_BYTES


def test_uncompressed_decisions_break_the_budget(
    run: _Run, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Negative control: the run's checkpoint writes made again with each day's decisions stored
    as their uncompressed JSON break the decisions budget and the budget of every table."""
    monkeypatch.setattr(checkpoint_store, "_compressed", lambda text: text.encode("utf-8"))

    breaches = _breaches(run, _replay(run, tmp_path / "uncompressed.sqlite"))

    assert any("decisions" in breach for breach in breaches)
    assert any("every table" in breach for breach in breaches)


def test_the_replay_measures_what_the_run_wrote(run: _Run, tmp_path: Path) -> None:
    """The negative control's replay, made without its change, writes the bytes the run did."""
    assert _replay(run, tmp_path / "replay.sqlite") == run.sizes
