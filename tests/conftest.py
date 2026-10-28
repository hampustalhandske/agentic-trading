# conftest.py

"""Project-wide pytest configuration.

Disables agentic_trading's logging for the whole test session — at import
time, before any test module's own imports can trigger
`agentic_trading.logging_config.get_logger()`'s self-configuration.
`configure_logging()` only does real work on its *first* call per process
(see its docstring); since pytest imports this file before collecting any
test module, this call wins that race deterministically. Without it,
running the suite would append hundreds of test-run lines to the real
`log/agentic_trading.log` on every single `pytest` invocation — a file
meant to be an actual user's run history, not test noise.

`tests/unit/test_logging_config.py` tests the logging module itself; it
resets the module's internal state per-test (see that file) to configure
fresh instances despite this session-wide disabling.
"""

import json
import math
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pytest

from agentic_trading.logging_config import configure_logging

configure_logging(disable=True)

from agentic_trading import provenance  # noqa: E402
from agentic_trading.data.loaders import lakehouse_loader  # noqa: E402
from agentic_trading.data.loaders.lakehouse_loader import LakeProvenance  # noqa: E402

_DEFAULT_SCRAPED_AT = "2020-01-01T00:00:00Z"


def _utc_iso(value: Any) -> str:
    """`value` (a string, `datetime` or `Timestamp`; naive means UTC) as an ISO 8601 UTC string."""
    stamp = pd.Timestamp(value)
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    return str(stamp.isoformat())


class FakeLake:
    """In-memory stand-in for `Lake`.

    `add(name, frame, date_columns=...)` registers a table (the named
    columns become `DATE`, as the real lake serves them) and a `name_log`
    table holding the same rows plus a `scraped_at TIMESTAMPTZ` column:
    `scraped_at` (default 2020-01-01T00:00:00Z) for every row, or the
    frame's own `scraped_at` values when the frame has that column and no
    `scraped_at` argument is given. A `rebuilt_whole` table's current rows
    carry that `scraped_at` too, its build time, as the lake's current
    views do. `add_log_rows` appends further versions
    to the log only. A table added with `pin_column` (an append-only table
    with no `scraped_at`) gets no log; its pin is its own row-time column.
    `add_view` registers a view with no log. `describe`, `tables`, `status`
    and `health` answer from that metadata (`kind`, `key_columns`,
    `rebuilt_whole`); `last_write_overrides` replaces a table's `last_write`.
    `sql` runs the query against the tables and exposes `version` as
    `last_version`. Setting `fixed_result` makes every query return that
    frame regardless of the SQL (for testing a loader's own guards against
    a result its query should never produce); setting `error` makes every
    call raise it."""

    def __init__(self) -> None:
        self._connection = duckdb.connect(database=":memory:")
        self.fixed_result: pd.DataFrame | None = None
        self.error: Exception | None = None
        self.calls: list[tuple[str, list[Any]]] = []
        self.version = "fake-lake-1"
        self.last_version: str | None = None
        self.last_write_overrides: dict[str, str] = {}
        self._meta: dict[str, dict[str, Any]] = {}
        self._date_columns: dict[str, tuple[str, ...]] = {}

    def _columns(self, name: str) -> list[str]:
        description = self._connection.execute(f"SELECT * FROM {name} LIMIT 0").description
        return [column[0] for column in description]

    def _register(
        self,
        name: str,
        kind: str,
        key_columns: list[str] | None,
        rebuilt_whole: bool,
        pin_column: str | None = None,
        has_log: bool = True,
    ) -> None:
        columns = self._columns(name)
        keys = key_columns if key_columns is not None else [c for c in columns if c != "scraped_at"]
        self._meta[name] = {
            "kind": kind,
            "key_columns": list(keys),
            "rebuilt_whole": rebuilt_whole,
            "pin_column": pin_column,
            "has_log": has_log and pin_column is None,
        }

    def _build_log(self, name: str, scraped_at: Any) -> None:
        """Creates `name_log` from `name`'s rows plus a `scraped_at` column."""
        if not self._meta[name]["has_log"]:
            return
        has_column = "scraped_at" in self._columns(name)
        if scraped_at is None and has_column:
            self._connection.execute(f"CREATE OR REPLACE TABLE {name}_log AS SELECT * FROM {name}")
            return
        selected = "* EXCLUDE (scraped_at)" if has_column else "*"
        stamp = _utc_iso(_DEFAULT_SCRAPED_AT if scraped_at is None else scraped_at)
        self._connection.execute(
            f"CREATE OR REPLACE TABLE {name}_log AS "
            f"SELECT {selected}, CAST(? AS TIMESTAMPTZ) AS scraped_at FROM {name}",
            [stamp],
        )

    def add(
        self,
        name: str,
        frame: pd.DataFrame,
        date_columns: tuple[str, ...] = (),
        *,
        scraped_at: Any = None,
        kind: str = "current",
        key_columns: list[str] | None = None,
        rebuilt_whole: bool = False,
        pin_column: str | None = None,
    ) -> None:
        frame = frame.copy()
        for column in date_columns:
            frame[column] = pd.to_datetime(frame[column]).dt.date
        self._connection.register(f"_staged_{name}", frame)
        self._connection.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _staged_{name}")
        self._connection.unregister(f"_staged_{name}")
        self._date_columns[name] = tuple(date_columns)
        self._register(name, kind, key_columns, rebuilt_whole, pin_column)
        self._build_log(name, scraped_at)
        if rebuilt_whole and "scraped_at" not in self._columns(name):
            self._connection.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM {name}_log")

    def add_empty(
        self,
        name: str,
        columns: dict[str, str],
        *,
        kind: str = "current",
        key_columns: list[str] | None = None,
        rebuilt_whole: bool = False,
        pin_column: str | None = None,
    ) -> None:
        """Registers a table with no rows; `columns` maps names to DuckDB types."""
        definition = ", ".join(f"{column} {kind_}" for column, kind_ in columns.items())
        self._connection.execute(f"CREATE OR REPLACE TABLE {name} ({definition})")
        self._date_columns[name] = ()
        self._register(name, kind, key_columns, rebuilt_whole, pin_column)
        self._build_log(name, None)

    def add_view(self, name: str, select_sql: str, kind: str = "compatibility view") -> None:
        """Registers view `name` over `select_sql`; it has no log and no key columns."""
        self._connection.execute(f"CREATE OR REPLACE VIEW {name} AS {select_sql}")
        self._date_columns[name] = ()
        self._register(name, kind, [], False, has_log=False)

    def add_log_rows(
        self,
        name: str,
        frame: pd.DataFrame,
        scraped_at: Any,
        date_columns: tuple[str, ...] | None = None,
    ) -> None:
        """Appends `frame` to `name_log` as versions scraped at `scraped_at`; the current
        table `name` is left as it is. `date_columns` defaults to the ones `add` was given."""
        frame = frame.copy()
        for column in self._date_columns.get(name, ()) if date_columns is None else date_columns:
            frame[column] = pd.to_datetime(frame[column]).dt.date
        self._connection.register("_staged_log_rows", frame)
        self._connection.execute(
            f"INSERT INTO {name}_log BY NAME "
            "SELECT *, CAST(? AS TIMESTAMPTZ) AS scraped_at FROM _staged_log_rows",
            [_utc_iso(scraped_at)],
        )
        self._connection.unregister("_staged_log_rows")

    def describe(self, table: str) -> dict[str, Any]:
        """The lake's table metadata; a table never added reads as an empty current table."""
        meta = self._meta.get(table)
        if meta is None:
            return {
                "kind": "current",
                "key_columns": [],
                "rebuilt_whole": False,
                "last_write": None,
            }
        last_write = self.last_write_overrides.get(table)
        source = f"max({meta['pin_column']})" if meta["pin_column"] else "max(scraped_at)"
        relation = table if meta["pin_column"] else f"{table}_log"
        if last_write is None and (meta["has_log"] or meta["pin_column"]):
            newest = self._connection.execute(f"SELECT epoch({source}) FROM {relation}").fetchone()
            if newest is not None and newest[0] is not None:
                rounded = datetime.fromtimestamp(math.ceil(newest[0]), UTC)
                last_write = rounded.isoformat().replace("+00:00", "Z")
        return {
            "kind": meta["kind"],
            "key_columns": list(meta["key_columns"]),
            "rebuilt_whole": meta["rebuilt_whole"],
            "last_write": last_write,
        }

    def tables(self, include_logs: bool = False) -> list[dict[str, Any]]:
        return [{"table": name, "kind": meta["kind"]} for name, meta in self._meta.items()]

    def status(self) -> dict[str, Any]:
        return {"generated_at": None, "tables": self.tables()}

    def health(self) -> dict[str, Any]:
        return {"version": self.version}

    def sql(self, query: str, params: list[Any] | None = None) -> pd.DataFrame:
        self.calls.append((query, list(params or [])))
        self.last_version = self.version
        if self.error is not None:
            raise self.error
        if self.fixed_result is not None:
            return self.fixed_result
        return self._connection.execute(query, params or []).df()


_REAL_LAKE_ACCESSOR = lakehouse_loader._lake


_NO_NETWORK_MESSAGE = "unit tests must not reach the lakehouse API; use the fake_lake fixture"
# The `last_write` the offline catalogue and `preflight_lake` report for every table.
_CATALOGUE_LAST_WRITE = "2020-01-01T00:00:00Z"


class _OfflineLake:
    """The lake a test without `fake_lake` sees: its catalogue (`describe`)
    answers every table as an empty current table last written at
    `_CATALOGUE_LAST_WRITE`, so a `lake_session` can open, and every other
    call (any SQL read, `status`, `health`, ...) raises."""

    last_version: str | None = None

    def describe(self, table: str) -> dict[str, Any]:
        return {
            "kind": "current",
            "key_columns": [],
            "rebuilt_whole": False,
            "last_write": _CATALOGUE_LAST_WRITE,
        }

    def __getattr__(self, name: str) -> Any:
        raise RuntimeError(_NO_NETWORK_MESSAGE)


_OFFLINE_LAKE = _OfflineLake()


def _no_network_lake() -> Any:
    return _OFFLINE_LAKE


@pytest.fixture(autouse=True)
def _block_real_lake(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Replaces the loader's lake accessor with `_OfflineLake`, whose every
    data read raises, so no unit test reaches the network. Tests that need
    lake data request `fake_lake`, which overrides this. The accessor cache
    is cleared on teardown so a client built or closed inside one test
    never leaks."""
    monkeypatch.setattr(lakehouse_loader, "_lake", _no_network_lake)
    yield
    monkeypatch.undo()
    _REAL_LAKE_ACCESSOR.cache_clear()


@pytest.fixture(autouse=True)
def _stub_run_provenance_lake(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gives `build_run_provenance` a fixed lake build so CLI tests that
    don't care about provenance never reach the lake's status endpoint."""
    monkeypatch.setattr(
        provenance,
        "lake_provenance",
        lambda: LakeProvenance(
            lake_version="test-lake",
            read_at=datetime(2024, 1, 1, tzinfo=UTC),
            status_generated_at=None,
        ),
    )


@pytest.fixture
def real_lake_accessor(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Restores the loader's real, cached lake accessor (for tests of the
    accessor itself) with an empty cache."""
    monkeypatch.setattr(lakehouse_loader, "_lake", _REAL_LAKE_ACCESSOR)
    _REAL_LAKE_ACCESSOR.cache_clear()
    return _REAL_LAKE_ACCESSOR


@pytest.fixture
def fake_lake(monkeypatch: pytest.MonkeyPatch) -> FakeLake:
    lake = FakeLake()
    monkeypatch.setattr(lakehouse_loader, "_lake", lambda: lake)
    return lake


_CONTRACT_PATH = Path(__file__).parent / "fixtures" / "lakehouse_contract_v1.json"


@pytest.fixture
def preflight_lake(fake_lake: FakeLake) -> FakeLake:
    """A `fake_lake` holding every view of the frozen lake contract
    (`tests/fixtures/lakehouse_contract_v1.json`) as an empty table with
    the contract's column types, so the CLI and dashboard run paths (the
    data-availability preflight and the price history) bind their SQL
    against the real schema. Every requirement reports no coverage. Each
    view reports a `last_write` of `_CATALOGUE_LAST_WRITE`, so a
    `lake_session` opened on it has a default lake time."""
    contract = json.loads(_CONTRACT_PATH.read_text())
    for view, columns in contract["views"].items():
        fake_lake.add_empty(
            view,
            {
                f'"{name}"': ("VARCHAR" if kind == '"NULL"' else kind)
                for name, kind in columns.items()
            },
        )
        fake_lake.last_write_overrides[view] = _CATALOGUE_LAST_WRITE
    return fake_lake
