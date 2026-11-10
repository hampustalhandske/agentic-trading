# test_submission.py

"""`firm_record.submission`: sources.json entries map onto edgelab's `DecisionSource` shape
with unique names, a dated source stating `can_publish_after_close` and an undated one its
`limitation`; a prior-exposure block keeps edgelab's `PriorExposure` checks; the replay folder
names the lake as its data source, pinned to the record's lake time and lake version, in both
strategy/strategy.yaml and submission.json, and its data block keeps edgelab's
`LakeQueryBundle` checks."""

import json
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from agentic_trading.firm_record.submission import (
    SUBMISSION_FILE,
    SubmissionData,
    SubmissionPriorExposure,
    SubmissionSource,
    decision_sources,
    submission_for,
    write_submission_folder,
)
from agentic_trading.stage_a.spec import FirmSpec, FirmSpecRecord, firm_spec_hash
from unit.firm_record.test_firm_store import _DAYS, _SOURCES

# A lake version as the lake serves it: '<served code hash>.<data hash>'.
_LAKE_VERSION = "0123abc.4567def"


def test_sources_map_dated_and_undated_entries() -> None:
    """A dated entry and an undated one map field for field and dump back to the same keys."""
    entries = [
        {"name": "fred", "can_publish_after_close": True},
        {"name": "sic", "undated": True, "limitation": "current SIC code screens candidates"},
    ]
    sources = decision_sources(entries)

    assert [source.model_dump(exclude_defaults=True) for source in sources] == entries


def test_sources_refuse_a_repeated_name() -> None:
    """Two entries under one name are refused."""
    entry = {"name": "fred", "can_publish_after_close": True}
    with pytest.raises(ValueError, match="repeat"):
        decision_sources([entry, entry])


@pytest.mark.parametrize(
    "entry",
    [
        {"name": "fred"},
        {"name": "fred", "can_publish_after_close": True, "limitation": "x"},
        {"name": "sic", "undated": True},
        {"name": "sic", "undated": True, "limitation": "x", "can_publish_after_close": False},
        {"name": "fred", "can_publish_after_close": True, "extra": 1},
    ],
    ids=["dated-no-flag", "dated-limitation", "undated-no-limitation", "undated-flag", "extra"],
)
def test_sources_refuse_entries_edgelab_refuses(entry: dict[str, object]) -> None:
    """An entry that is neither a dated nor an undated source, or carries an unknown key, is
    refused."""
    with pytest.raises(ValidationError):
        SubmissionSource.model_validate(entry)


def test_prior_exposure_round_trips_and_keeps_edgelab_checks() -> None:
    """A valid block dumps back unchanged; one without windows, with a reversed window or a
    negative count is refused."""
    block = {
        "note": "Firm Sharpe seen",
        "windows": [{"start": "2010-01-04", "end": "2017-12-29"}],
        "min_count": 3,
    }

    assert SubmissionPriorExposure.model_validate(block).model_dump(mode="json") == block
    for broken in (
        block | {"windows": []},
        block | {"windows": [{"start": "2017-12-29", "end": "2010-01-04"}]},
        block | {"min_count": -1},
    ):
        with pytest.raises(ValidationError):
            SubmissionPriorExposure.model_validate(broken)


def _lake_record(lake_time: datetime) -> FirmSpecRecord:
    """A firm record pinned to `lake_time` and `_LAKE_VERSION`."""
    spec = FirmSpec(
        commander={"class": "agentic_trading.agents.commander.Commander"},
        roster=[{"agent_id": "risk_manager", "class": "x.RiskManager", "parameters": {}}],
        settings={"firm": {"trading_days_per_year": 252}},
        module_hashes={"agentic_trading.agents.commander": "ab" * 32},
        universe=["SPY", "AAPL"],
        window_start=_DAYS[0],
        window_end=_DAYS[-1],
        decision_rule="rule",
        quantcore_sha="e" * 40,
        lake_time=lake_time,
    )
    return FirmSpecRecord(
        spec=spec,
        spec_hash=firm_spec_hash(spec),
        lake_time=lake_time,
        lake_version=_LAKE_VERSION,
        quantcore_version="0.3.1",
    )


def _written(tmp_path: Path, record: FirmSpecRecord) -> tuple[dict[str, Any], dict[str, Any]]:
    """The strategy.yaml and submission.json an uncounted `record` writes into `tmp_path`."""
    submission, meta = submission_for(
        record,
        sources=_SOURCES,
        git_sha="a" * 40,
        src_tree_clean=False,
        counted=False,
        registered_trials=None,
    )
    write_submission_folder(tmp_path, record, submission, meta)
    strategy = yaml.safe_load((tmp_path / "strategy" / "strategy.yaml").read_text())
    return strategy, json.loads((tmp_path / SUBMISSION_FILE).read_text())


@pytest.mark.parametrize(
    ("lake_time", "text"),
    [
        (datetime(2026, 11, 10, 17, 20, 50, tzinfo=UTC), "2026-11-10T17:20:50Z"),
        (
            datetime(2026, 11, 10, 18, 20, 50, tzinfo=timezone(timedelta(hours=1))),
            "2026-11-10T17:20:50Z",
        ),
    ],
    ids=["utc", "offset"],
)
def test_the_folder_reads_the_lake_at_the_records_lake_time_and_version(
    tmp_path: Path, lake_time: datetime, text: str
) -> None:
    """strategy.yaml's data block names the lakehouse provider over the spec's window, pinned to
    the record's lake time as ISO-8601 UTC with a Z suffix and to its lake version, and
    survivorship-free; submission.json's data is a lake bundle at the same lake time string with
    no queries. The two strings are identical and parse back to the record's lake time."""
    record = _lake_record(lake_time)
    strategy, submission = _written(tmp_path, record)

    assert strategy["data"] == {
        "provider": "lakehouse",
        "range": {"start": _DAYS[0], "end": _DAYS[-1]},
        "frequency": "1d",
        "params": {"lake_time": text, "lake_version": _LAKE_VERSION, "survivorship_free": True},
    }
    assert submission["data"] == {"kind": "lake", "lake_time": text, "queries": []}
    strategy_time = strategy["data"]["params"]["lake_time"]
    submission_time = submission["data"]["lake_time"]
    assert isinstance(strategy_time, str) and strategy_time == submission_time
    assert strategy_time.endswith("Z")
    assert datetime.fromisoformat(strategy_time) == record.lake_time
    assert strategy["data"]["params"]["lake_version"] == record.lake_version


def test_the_folder_has_no_data_bundle(tmp_path: Path) -> None:
    """The folder holds no Parquet bundle folder: no data/ beside strategy/."""
    _written(tmp_path, _lake_record(datetime(2026, 11, 10, 17, 20, 50, tzinfo=UTC)))

    assert not (tmp_path / "data").exists()


def test_lake_data_round_trips_and_keeps_edgelab_checks() -> None:
    """A lake bundle with a query dumps back unchanged; one with a lake time off UTC or without
    an offset, another kind, an empty table or query text, or an unknown field is refused."""
    block = {
        "kind": "lake",
        "lake_time": "2026-11-10T17:20:50Z",
        "queries": [{"table": "instrument_prices", "sql": "SELECT 1"}],
    }

    assert SubmissionData.model_validate(block).model_dump(mode="json") == block
    for broken in (
        block | {"lake_time": "2026-11-10T18:20:50+01:00"},
        block | {"lake_time": "2026-11-10T17:20:50"},
        block | {"kind": "parquet"},
        block | {"queries": [{"table": "", "sql": "SELECT 1"}]},
        block | {"queries": [{"table": "instrument_prices", "sql": ""}]},
        block | {"queries": [{"table": "instrument_prices", "sql": "SELECT 1", "x": 1}]},
        block | {"path": "data"},
    ):
        with pytest.raises(ValidationError):
            SubmissionData.model_validate(broken)
