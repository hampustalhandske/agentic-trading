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


class FakeLake:
    """In-memory stand-in for `Lake`.

    `add(name, frame, date_columns=...)` registers a table (the named
    columns become `DATE`, as the real lake serves them). `sql` runs the
    query against those tables. Setting `fixed_result` makes every query
    return that frame regardless of the SQL (for testing a loader's own
    guards against a result its query should never produce); setting
    `error` makes every call raise it."""

    def __init__(self) -> None:
        self._connection = duckdb.connect(database=":memory:")
        self.fixed_result: pd.DataFrame | None = None
        self.error: Exception | None = None
        self.calls: list[tuple[str, list[Any]]] = []

    def add(self, name: str, frame: pd.DataFrame, date_columns: tuple[str, ...] = ()) -> None:
        frame = frame.copy()
        for column in date_columns:
            frame[column] = pd.to_datetime(frame[column]).dt.date
        self._connection.register(f"_staged_{name}", frame)
        self._connection.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _staged_{name}")
        self._connection.unregister(f"_staged_{name}")

    def add_empty(self, name: str, columns: dict[str, str]) -> None:
        """Registers a table with no rows; `columns` maps names to DuckDB types."""
        definition = ", ".join(f"{column} {kind}" for column, kind in columns.items())
        self._connection.execute(f"CREATE OR REPLACE TABLE {name} ({definition})")

    def sql(self, query: str, params: list[Any] | None = None) -> pd.DataFrame:
        self.calls.append((query, list(params or [])))
        if self.error is not None:
            raise self.error
        if self.fixed_result is not None:
            return self.fixed_result
        return self._connection.execute(query, params or []).df()


_REAL_LAKE_ACCESSOR = lakehouse_loader._lake


def _no_network_lake() -> Any:
    raise RuntimeError("unit tests must not reach the lakehouse API; use the fake_lake fixture")


@pytest.fixture(autouse=True)
def _block_real_lake(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Replaces the loader's lake accessor with one that raises, so no
    unit test reaches the network. Tests that need lake data request
    `fake_lake`, which overrides this. The accessor cache is cleared on
    teardown so a client built or closed inside one test never leaks."""
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
    against the real schema. Every requirement reports no coverage."""
    contract = json.loads(_CONTRACT_PATH.read_text())
    for view, columns in contract["views"].items():
        fake_lake.add_empty(
            view,
            {
                f'"{name}"': ("VARCHAR" if kind == '"NULL"' else kind)
                for name, kind in columns.items()
            },
        )
    return fake_lake
