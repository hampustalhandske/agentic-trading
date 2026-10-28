# test_provenance.py

"""Unit tests for agentic_trading.provenance.

`build_run_provenance` records the code revision (git, quantcore), the full
strategy settings, the CLI arguments, the roster and the lake build a run
used. The lake lookup is stubbed so no test reaches the lakehouse.
"""

import re
from datetime import UTC, datetime
from typing import Any

import pytest

from agentic_trading import provenance
from agentic_trading.config import settings, temporary_overrides
from agentic_trading.data.loaders import lakehouse_loader
from agentic_trading.data.loaders.lakehouse_loader import LakeProvenance
from agentic_trading.provenance import RunProvenance, build_run_provenance

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
