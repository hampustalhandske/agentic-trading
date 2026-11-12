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
import sys
import threading
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pyarrow as pa
import pytest

from agentic_trading.logging_config import configure_logging

configure_logging(disable=True)

from agentic_trading import provenance  # noqa: E402
from agentic_trading.data import lakehouse_client  # noqa: E402
from agentic_trading.data.lakehouse_client import LakeAnswer  # noqa: E402
from agentic_trading.data.loaders import lakehouse_loader  # noqa: E402
from agentic_trading.data.loaders.lakehouse_loader import LakeProvenance  # noqa: E402
from edgelab_live import configure as configure_edgelab_live  # noqa: E402

_UNPINNED_LAKE_READS_MARKER = "unpinned_lake_reads"


def pytest_configure(config: pytest.Config) -> None:
    """Registers the `edgelab_live` marker and the skip guard of `tests/edgelab_live.py`: a run
    with any skip other than a live edgelab check skipped for an absent checkout fails. Also
    registers the `unpinned_lake_reads` marker (see `_unpinned_lake_reads`)."""
    configure_edgelab_live(config)
    config.addinivalue_line(
        "markers",
        f"{_UNPINNED_LAKE_READS_MARKER}: the test reads the fake lake's current tables with no "
        "lake session open, inside lakehouse_loader.unpinned_lake_reads",
    )


_DEFAULT_SCRAPED_AT = "2020-01-01T00:00:00Z"
# A log table's bookkeeping columns, never part of the tie order the lake serves.
_LOG_BOOKKEEPING_COLUMNS = ("scraped_at", "dt", "table")


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
    and `health` answer from that metadata (`kind`, `key_columns` and the
    same list as `keys`, `rebuilt_whole`); `last_write_overrides` replaces a
    table's `last_write`. A "current" table's `describe` serves the lake's
    `tie_order`, its non-key columns in column order without `scraped_at`,
    `dt` and `table`; `tie_order_overrides` replaces it, and an override of
    `None` serves none.
    `health` answers HTTP 200's body: `version`, `health_status` as `status`
    ("ok" by default) and `write_times` (table to entry, empty by default)
    under `write_times.tables`, so a test plants a degraded answer by setting
    those two. `sql` runs the query against the tables and exposes `version` as
    `last_version`; `sql_answer` gives the same rows as an Arrow table with
    `version`. Setting `fixed_result` makes every query return that
    frame regardless of the SQL (for testing a loader's own guards against
    a result its query should never produce); setting `error` makes every
    call raise it."""

    def __init__(self) -> None:
        self._connection = duckdb.connect(database=":memory:")
        # One connection holds one pending result, so concurrent reads (pool
        # workers inside a lake session) take turns.
        self._sql_lock = threading.Lock()
        self.fixed_result: pd.DataFrame | None = None
        self.error: Exception | None = None
        self.calls: list[tuple[str, list[Any]]] = []
        self.version = "fake-lake-1"
        self.last_version: str | None = None
        self.last_write_overrides: dict[str, str] = {}
        self.tie_order_overrides: dict[str, list[str] | None] = {}
        self.health_status: str | None = "ok"
        self.write_times: dict[str, dict[str, Any]] = {}
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
                "keys": [],
                "rebuilt_whole": False,
                "last_write": None,
                "tie_order": [],
            }
        last_write = self.last_write_overrides.get(table)
        source = f"max({meta['pin_column']})" if meta["pin_column"] else "max(scraped_at)"
        relation = table if meta["pin_column"] else f"{table}_log"
        if last_write is None and (meta["has_log"] or meta["pin_column"]):
            newest = self._connection.execute(f"SELECT epoch({source}) FROM {relation}").fetchone()
            if newest is not None and newest[0] is not None:
                rounded = datetime.fromtimestamp(math.ceil(newest[0]), UTC)
                last_write = rounded.isoformat().replace("+00:00", "Z")
        described: dict[str, Any] = {
            "kind": meta["kind"],
            "key_columns": list(meta["key_columns"]),
            "keys": list(meta["key_columns"]),
            "rebuilt_whole": meta["rebuilt_whole"],
            "last_write": last_write,
        }
        if meta["kind"] == "current":
            tie_order = self.tie_order_overrides.get(table, self._tie_order(table))
            if tie_order is not None:
                described["tie_order"] = list(tie_order)
        return described

    def _tie_order(self, table: str) -> list[str]:
        """The tie order the lake serves for current table `table`: its columns other than
        the key columns and the bookkeeping columns `scraped_at`, `dt` and `table`, in the
        table's column order."""
        excluded = {*self._meta[table]["key_columns"], *_LOG_BOOKKEEPING_COLUMNS}
        return [column for column in self._columns(table) if column not in excluded]

    def tables(self, include_logs: bool = False) -> list[dict[str, Any]]:
        return [{"table": name, "kind": meta["kind"]} for name, meta in self._meta.items()]

    def status(self) -> dict[str, Any]:
        return {"generated_at": None, "tables": self.tables()}

    def health(self) -> dict[str, Any]:
        tables = {table: dict(entry) for table, entry in self.write_times.items()}
        return {
            "version": self.version,
            "status": self.health_status,
            "write_times": {"tables": tables},
        }

    def sql(self, query: str, params: list[Any] | None = None) -> pd.DataFrame:
        self.calls.append((query, list(params or [])))
        self.last_version = self.version
        if self.error is not None:
            raise self.error
        if self.fixed_result is not None:
            return self.fixed_result
        with self._sql_lock:
            return self._connection.execute(query, params or []).df()

    def sql_answer(self, query: str, params: list[Any] | None = None) -> LakeAnswer:
        """`sql`'s rows as an Arrow table with `version`, the answer the lake answer cache
        stores: the frame converts back to the frame `sql` returns."""
        frame = self.sql(query, params)
        return LakeAnswer(
            table=pa.Table.from_pandas(frame, preserve_index=False), version=self.version
        )


_REAL_LAKE_ACCESSOR = lakehouse_loader._lake


_NO_NETWORK_MESSAGE = "unit tests must not reach the lakehouse API; use the fake_lake fixture"
# The `last_write` the offline catalogue and `preflight_lake` report for every table.
_CATALOGUE_LAST_WRITE = "2020-01-01T00:00:00Z"


class _OfflineLake:
    """The lake a test without `fake_lake` sees: its catalogue (`describe`)
    answers every table as an empty current table last written at
    `_CATALOGUE_LAST_WRITE`, so a `lake_session` can open, its `health` is
    an ok status with no table entries, so a firm record's health check
    passes, and every other call (any SQL read, `status`, ...) raises."""

    last_version: str | None = None

    def describe(self, table: str) -> dict[str, Any]:
        return {
            "kind": "current",
            "key_columns": [],
            "rebuilt_whole": False,
            "last_write": _CATALOGUE_LAST_WRITE,
        }

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "write_times": {"tables": {}}}

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
def _lake_client_state_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Points the lake client's state directory (its request markers and last timeout, see
    `data.lakehouse_client`) at a fresh directory under the session's temporary root for each
    test, so no test reads or writes the user's own state and no timeout one test records
    refuses another's calls. Subprocesses a test starts inherit it."""
    directory = tmp_path_factory.getbasetemp() / "lake-client-state" / uuid.uuid4().hex
    monkeypatch.setenv(lakehouse_client.STATE_DIR_ENV_VAR, str(directory))


@pytest.fixture(autouse=True)
def _unpinned_lake_reads(request: pytest.FixtureRequest) -> Iterator[None]:
    """Runs a test marked `unpinned_lake_reads` inside
    `lakehouse_loader.unpinned_lake_reads`, so its loader calls with no lake session open read
    the fake lake's current tables. Every other test keeps the refusal of a read with no
    session open."""
    if request.node.get_closest_marker(_UNPINNED_LAKE_READS_MARKER) is None:
        yield
        return
    with lakehouse_loader.unpinned_lake_reads("the test reads the fake lake's current tables"):
        yield


# The test module that specifies the empty-read retry: it checks the real
# `EMPTY_READ_RETRY_SECONDS` and replaces `time.sleep` itself in every test
# that can wait.
_EMPTY_READ_RETRY_SPEC = "test_empty_read_retry.py"


@pytest.fixture(autouse=True)
def _no_empty_read_wait(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Sets the loader's empty-read retry wait (`EMPTY_READ_RETRY_SECONDS`)
    to 0 s, so a test whose lake holds an empty always-populated table
    never really sleeps. Tests in `_EMPTY_READ_RETRY_SPEC` keep the real
    value."""
    if request.path.name == _EMPTY_READ_RETRY_SPEC:
        return
    monkeypatch.setattr(lakehouse_loader, "EMPTY_READ_RETRY_SECONDS", 0.0)


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


@pytest.fixture(autouse=True)
def _no_host_power_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Makes the firm record's power preflight see a host without `pmset`
    (`cli.read_power_state` returns `None`), so no test depends on whether
    the machine running it is on battery. Applies when `agentic_trading.cli`
    is imported, as it is by every test module that invokes a command."""
    cli = sys.modules.get("agentic_trading.cli")
    if cli is not None:
        monkeypatch.setattr(cli, "read_power_state", lambda: None)


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


_CONTRACT_PATH = Path(__file__).parent / "fixtures" / "lakehouse_contract_v2.json"

# The one row `preflight_lake` puts in each always-populated table
# (`lakehouse_loader.ALWAYS_POPULATED_TABLES`), by column: it matches no
# ticker, security or report window and carries no coverage date.
_PLACEHOLDER_ROWS: dict[str, dict[str, Any]] = {
    "cftc_cot_publication": {"report_date": "2000-01-04"},
    "equity_adjustment_factors": {"ticker": "ZZUNREL", "security_id": 999},
    "sp500_membership": {"ticker": "ZZUNREL"},
    "yahoo_split_close_basis": {"ticker": "ZZUNREL"},
}


@pytest.fixture
def preflight_lake(fake_lake: FakeLake) -> FakeLake:
    """A `fake_lake` holding every view of the frozen lake contract
    (`tests/fixtures/lakehouse_contract_v2.json`) with the contract's
    column types, so the CLI and dashboard run paths (the
    data-availability preflight and the price history) bind their SQL
    against the real schema. Every view is empty except the
    always-populated tables, which hold the one `_PLACEHOLDER_ROWS` row
    each (in the table and its log, scraped at `_CATALOGUE_LAST_WRITE`)
    so their table-level reads find rows. Every requirement reports no
    coverage. Each view reports a `last_write` of `_CATALOGUE_LAST_WRITE`,
    so a `lake_session` opened on it has a default lake time."""
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
    assert set(_PLACEHOLDER_ROWS) == lakehouse_loader.ALWAYS_POPULATED_TABLES
    for table, row in _PLACEHOLDER_ROWS.items():
        names = ", ".join([*row, "scraped_at"])
        marks = ", ".join("?" for _ in range(len(row) + 1))
        for relation in (table, f"{table}_log"):
            fake_lake._connection.execute(
                f"INSERT INTO {relation} ({names}) VALUES ({marks})",
                [*row.values(), _CATALOGUE_LAST_WRITE],
            )
    return fake_lake
