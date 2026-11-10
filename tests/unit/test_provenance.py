# test_provenance.py

"""Unit tests for agentic_trading.provenance.

`build_run_provenance` records the code revision (git, quantcore), the full
strategy settings, the CLI arguments, the roster and the lake build a run
used. The lake lookup is stubbed so no test reaches the lakehouse.
"""

import platform
import re
from datetime import UTC, datetime
from importlib import metadata
from typing import Any

import pytest

from agentic_trading import provenance
from agentic_trading.config import settings, temporary_overrides
from agentic_trading.data.loaders import lakehouse_loader
from agentic_trading.data.loaders.lakehouse_loader import LakeProvenance
from agentic_trading.provenance import (
    LibraryVersions,
    RunProvenance,
    RuntimeEnvironment,
    build_run_provenance,
    runtime_environment,
)

_LAKE = LakeProvenance(
    lake_version="abc123-dirty",
    read_at=datetime(2024, 3, 15, 12, 0, tzinfo=UTC),
    status_generated_at=datetime(2024, 3, 15, 11, 0, tzinfo=UTC),
)


@pytest.fixture(autouse=True)
def _stub_lake_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lakehouse_loader, "lake_provenance", lambda: _LAKE)
    monkeypatch.setattr(provenance, "lake_provenance", lambda: _LAKE, raising=False)


def _build(**overrides: Any) -> RunProvenance:
    arguments: dict[str, Any] = {
        "cli_args": {"start_date": "2024-01-02", "end_date": "2024-01-05", "assets": ["AAPL"]},
        "analyst_ids": ["a1", "a2"],
        "manager_ids": ["m1"],
    }
    return build_run_provenance(**{**arguments, **overrides})


def test_run_provenance_has_exactly_the_specified_fields() -> None:
    """The model exposes the specified field names and no others."""
    assert set(RunProvenance.model_fields) == {
        "git_sha",
        "git_dirty",
        "quantcore_sha",
        "settings",
        "cli_args",
        "analyst_ids",
        "manager_ids",
        "lake",
        "lake_contract",
        "environment",
    }


def test_build_passes_inputs_and_lake_version_through_unchanged() -> None:
    """CLI args, roster ids and the lake version (with its -dirty suffix) are recorded verbatim."""
    result = _build()

    assert result.cli_args == {
        "start_date": "2024-01-02",
        "end_date": "2024-01-05",
        "assets": ["AAPL"],
    }
    assert result.analyst_ids == ["a1", "a2"]
    assert result.manager_ids == ["m1"]
    assert result.lake.lake_version == "abc123-dirty"
    assert result.lake == _LAKE
    assert result.lake_contract == "lakehouse-contract-v1"


def test_build_records_empty_inputs() -> None:
    """Empty CLI args and rosters are preserved as empty containers."""
    result = _build(cli_args={}, analyst_ids=[], manager_ids=[])

    assert result.cli_args == {}
    assert result.analyst_ids == []
    assert result.manager_ids == []


def test_settings_snapshot_reflects_overrides() -> None:
    """The recorded settings are the live values, so a temporary override shows up."""
    baseline = _build().settings
    with temporary_overrides({"firm.commission_rate": 0.0123}):
        overridden = _build().settings

    assert overridden["firm"]["commission_rate"] == pytest.approx(0.0123)
    assert baseline == settings.model_dump(mode="json")
    assert baseline["firm"]["commission_rate"] != overridden["firm"]["commission_rate"]


def test_git_fields_are_populated_inside_this_repository() -> None:
    """Run from this git checkout, the revision is a 40-hex sha and dirtiness a bool."""
    result = _build()

    assert result.git_sha is not None
    assert re.fullmatch(r"[0-9a-f]{40}", result.git_sha)
    assert isinstance(result.git_dirty, bool)
    assert result.quantcore_sha is None or isinstance(result.quantcore_sha, str)


def test_provenance_round_trips_through_json() -> None:
    """Serialising to JSON and validating it back reproduces the same model."""
    result = _build()

    assert RunProvenance.model_validate_json(result.model_dump_json()) == result


def _installed(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def test_runtime_environment_records_installed_versions_and_platform() -> None:
    """Each library's version is importlib.metadata's, and the platform is `platform`'s."""
    environment = runtime_environment()

    assert set(LibraryVersions.model_fields) == {
        "numpy",
        "scipy",
        "numba",
        "arch",
        "statsmodels",
        "pandas",
        "pyarrow",
        "duckdb",
        "quantcore",
    }
    for name in LibraryVersions.model_fields:
        assert getattr(environment.libraries, name) == _installed(name), name
    assert environment.libraries.numpy is not None
    assert environment.libraries.quantcore is not None
    assert environment.quantcore_sha == provenance.quantcore_sha()
    assert environment.python_version == platform.python_version()
    assert environment.platform_system == platform.system()
    assert environment.platform_machine == platform.machine()
    assert environment.platform_release == platform.release()
    assert environment.blas_name is None or environment.blas_name


def test_missing_library_is_recorded_as_null(monkeypatch: pytest.MonkeyPatch) -> None:
    """A library that is not installed is null; the others keep their versions."""
    installed_version = metadata.version

    def without_numba(name: str) -> str:
        if name == "numba":
            raise metadata.PackageNotFoundError(name)
        return installed_version(name)

    monkeypatch.setattr(metadata, "version", without_numba)
    environment = runtime_environment()

    assert environment.libraries.numba is None
    assert environment.libraries.numpy == installed_version("numpy")
    assert RuntimeEnvironment.model_validate_json(environment.model_dump_json()) == environment


def test_run_provenance_records_the_runtime_environment() -> None:
    """A run's provenance holds the environment it ran in."""
    assert _build().environment == runtime_environment()
