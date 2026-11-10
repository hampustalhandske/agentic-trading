# test_edgelab_grading_golden.py

"""edgelab's pinned JSON schemas and golden cases, vendored at `tests/fixtures/
edgelab_grading_golden/`, and the firm record's outputs checked against them.

The slot holds edgelab's `docs/schema/` at commit afdca2a, byte for byte, under the same
relative path: `SHA256SUMS` and the seven schemas it lists (`bundle.v1`, `decision_file.v1`,
`grading_spec.v1`, `result.v4`, `strategy_scaffold.v1`, `submission.v1`, `submission.v2`), and
`golden/<schema stem>/accept/*.json` and `.../refuse/*.json`. Beside it, `SOURCE` names the
commit and `GOLDEN_SHA256SUMS` lists the sha256 of each golden case, which edgelab's
`SHA256SUMS` does not cover. Both sum files are pinned by sha256 here, so every vendored byte is
pinned.

Every accept case validates against its schema and every refuse case fails it, under the
`jsonschema` Draft 2020-12 validator with format checking, so the validator is shown to refuse.
A firm record's `submission.json` validates against `submission.v1`, the version it writes; with
`schema_version` set to 2 it validates against `submission.v2`. Its `strategy/strategy.yaml`
validates against `strategy_scaffold.v1`, and its grading block is edgelab's golden
`policy_2026.11.0` case. Its decision rows validate against `decision_file.v1`. Each output with
an unknown field planted in it is refused by its schema. The outputs are built from fake data
with the helpers of `test_firm_store.py`; nothing reads the lake."""

import copy
import hashlib
import json
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import jsonschema
import pytest
import yaml

from agentic_trading.firm_record.submission import (
    SUBMISSION_FILE,
    submission_for,
    write_submission_folder,
)
from agentic_trading.stage_a.spec import FirmSpecRecord
from unit.firm_record.test_firm_store import _DAYS, _SOURCES, _day_rows, _json_rows, _open, _record

_SLOT = Path(__file__).resolve().parents[2] / "fixtures" / "edgelab_grading_golden"
_SCHEMAS = _SLOT / "docs" / "schema"
_GOLDEN = _SCHEMAS / "golden"
_SOURCE = _SLOT / "SOURCE"
_SCHEMA_SUMS = _SCHEMAS / "SHA256SUMS"
_GOLDEN_SUMS = _SLOT / "GOLDEN_SHA256SUMS"
_SHA = "afdca2a"
_FULL_SHA = "afdca2a7dec35060719d691de9f9352c25a50b7d"
_SUBJECT = (
    "Policy 2026.11.1 cost table, submission v2 and upgrade, pinned schemas, replay label, "
    "Not assessable (M2 part 1)"
)
# The sha256 of edgelab's SHA256SUMS at afdca2a, and of GOLDEN_SHA256SUMS.
_SCHEMA_SUMS_SHA256 = "083a33283271f7ce29c6e4cd48937b8e924dda304da28e0fa98f00a9f8ee3aff"
_GOLDEN_SUMS_SHA256 = "604f0c9b725e2cd7059dd2b160229bdfdba2c5141249e2508da153b4469c67cd"
_SCHEMA_FILES = {
    "bundle.v1.json",
    "decision_file.v1.json",
    "grading_spec.v1.json",
    "result.v4.json",
    "strategy_scaffold.v1.json",
    "submission.v1.json",
    "submission.v2.json",
}
# The golden grading block a firm record's strategy.yaml carries.
_FIRM_GRADING_CASE = _GOLDEN / "grading_spec.v1" / "accept" / "policy_2026.11.0.json"
_PLANTED = "zzz_unknown_field"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sums(path: Path) -> dict[str, str]:
    """A `sha256sum`-format file as {path relative to `docs/schema/`: hex digest}."""
    found: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            digest, name = line.split(None, 1)
            found[name.strip().lstrip("*")] = digest
    return found


def _validator(stem: str) -> jsonschema.Draft202012Validator:
    """A format-checking validator for the vendored schema whose file stem is `stem`."""
    return jsonschema.Draft202012Validator(
        json.loads((_SCHEMAS / f"{stem}.json").read_text()),
        format_checker=jsonschema.Draft202012Validator.FORMAT_CHECKER,
    )


def _messages(stem: str, value: Any) -> list[str]:
    """Each error the `stem` schema reports for `value`, with the path it is at."""
    return [
        f"{'/'.join(map(str, error.absolute_path))}: {error.message}"
        for error in _validator(stem).iter_errors(value)
    ]


def _golden_cases() -> list[tuple[str, str, Path]]:
    """Every vendored golden case as (schema stem, `accept` or `refuse`, file)."""
    return [
        (schema.name, verdict, case)
        for schema in sorted(path for path in _GOLDEN.iterdir() if path.is_dir())
        for verdict in ("accept", "refuse")
        for case in sorted((schema / verdict).glob("*.json"))
    ]


def _as_json(value: Any) -> Any:
    """`value` as parsed YAML gives it, with each date as its ISO string, as JSON holds it."""
    return json.loads(
        json.dumps(value, default=lambda item: item.isoformat() if isinstance(item, date) else item)
    )


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _write_folder(tmp_path: Path, record: FirmSpecRecord, *, counted: bool) -> Path:
    """The submission folder `write_submission_folder` writes for `record` into `tmp_path`:
    uncounted with one trial, or counted with a registered count and prior exposure."""
    registered = (
        {
            "declared_n": 7,
            "prior_exposure": {
                "note": "Firm Sharpe seen",
                "windows": [{"start": "2010-01-04", "end": "2015-12-31"}],
                "min_count": 3,
            },
        }
        if counted
        else None
    )
    submission, meta = submission_for(
        record,
        sources=_SOURCES,
        git_sha="a" * 40,
        src_tree_clean=counted,
        counted=counted,
        registered_trials=registered,
    )
    write_submission_folder(tmp_path, record, submission, meta)
    return tmp_path


def _submission(tmp_path: Path) -> dict[str, Any]:
    folder = _write_folder(tmp_path, _record(), counted=False)
    return json.loads((folder / SUBMISSION_FILE).read_text())


def _strategy(tmp_path: Path) -> dict[str, Any]:
    folder = _write_folder(tmp_path, _record(), counted=False)
    return _as_json(yaml.safe_load((folder / "strategy" / "strategy.yaml").read_text()))


def _decision_rows(tmp_path: Path) -> list[dict[str, Any]]:
    store, trial = _open(tmp_path)
    for day in _DAYS:
        store.write_day(day, _day_rows(day))
    return _json_rows(trial)


# Vendored bytes


def test_source_names_the_edgelab_commit() -> None:
    """SOURCE names edgelab commit afdca2a by short and full sha and by its subject."""
    fields = dict(line.split(": ", 1) for line in _SOURCE.read_text().splitlines() if line.strip())
    assert fields["repository"] == "edgelab"
    assert fields["sha"] == _SHA
    assert fields["full_sha"] == _FULL_SHA and _FULL_SHA.startswith(_SHA)
    assert fields["subject"] == _SUBJECT


def test_sum_files_are_the_pinned_copies() -> None:
    """edgelab's SHA256SUMS and GOLDEN_SHA256SUMS hash to their pinned sha256."""
    assert _sha256(_SCHEMA_SUMS) == _SCHEMA_SUMS_SHA256
    assert _sha256(_GOLDEN_SUMS) == _GOLDEN_SUMS_SHA256


def test_every_vendored_file_matches_its_sum() -> None:
    """SHA256SUMS lists the seven schemas and GOLDEN_SHA256SUMS the golden cases; the slot
    holds exactly those files, SOURCE and the two sum files, each at its listed sha256."""
    schema_sums = _sums(_SCHEMA_SUMS)
    golden_sums = _sums(_GOLDEN_SUMS)
    listed = schema_sums | golden_sums
    on_disk = {path.relative_to(_SLOT).as_posix() for path in _SLOT.rglob("*") if path.is_file()}

    assert set(schema_sums) == _SCHEMA_FILES
    assert golden_sums and all(name.startswith("golden/") for name in golden_sums)
    assert on_disk == {"SOURCE", "GOLDEN_SHA256SUMS", "docs/schema/SHA256SUMS"} | {
        f"docs/schema/{name}" for name in listed
    }
    assert {name: _sha256(_SCHEMAS / name) for name in listed} == listed


# Golden cases


def test_every_schema_with_golden_cases_has_both_verdicts() -> None:
    """Each schema folder under golden/ is a vendored schema with at least one accept and one
    refuse case."""
    cases = _golden_cases()
    stems = {stem for stem, _, _ in cases}
    assert stems == {
        "bundle.v1",
        "grading_spec.v1",
        "strategy_scaffold.v1",
        "submission.v1",
        "submission.v2",
    }
    for stem in stems:
        assert (_SCHEMAS / f"{stem}.json").is_file()
        assert {verdict for found, verdict, _ in cases if found == stem} == {"accept", "refuse"}


@pytest.mark.parametrize(
    ("stem", "verdict", "case"),
    _golden_cases(),
    ids=lambda value: value.name if isinstance(value, Path) else str(value),
)
def test_a_golden_case_gets_its_verdict(stem: str, verdict: str, case: Path) -> None:
    """An accept case validates against its schema and a refuse case fails it."""
    messages = _messages(stem, json.loads(case.read_text()))
    if verdict == "accept":
        assert messages == []
    else:
        assert messages


# The firm record's outputs


@pytest.mark.parametrize("counted", [False, True], ids=["uncounted", "counted"])
def test_submission_json_validates_against_submission_v1(tmp_path: Path, counted: bool) -> None:
    """A firm record's submission.json, uncounted or counted, is a version 1 file that
    `submission.v1` accepts."""
    folder = _write_folder(tmp_path, _record(), counted=counted)
    submission = json.loads((folder / SUBMISSION_FILE).read_text())

    assert submission["schema_version"] == 1
    assert _messages("submission.v1", submission) == []


def test_submission_json_differs_from_v2_only_in_its_version(tmp_path: Path) -> None:
    """`submission.v2` refuses a firm record's submission.json only for its `schema_version`,
    and accepts it with `schema_version` set to 2."""
    submission = _submission(tmp_path)

    assert _messages("submission.v2", submission) == ["schema_version: 2 was expected"]
    assert _messages("submission.v2", submission | {"schema_version": 2}) == []


def test_strategy_yaml_validates_against_strategy_scaffold_v1(tmp_path: Path) -> None:
    """A firm record's strategy/strategy.yaml, its dates as ISO strings, is a config
    `strategy_scaffold.v1` accepts, and its grading block one `grading_spec.v1` accepts."""
    strategy = _strategy(tmp_path)

    assert _messages("strategy_scaffold.v1", strategy) == []
    assert _messages("grading_spec.v1", strategy["grading"]) == []


def test_strategy_yaml_carries_the_golden_2026_11_0_grading_block(tmp_path: Path) -> None:
    """The grading block of a firm record's strategy.yaml is edgelab's golden
    `policy_2026.11.0` accept case."""
    golden = json.loads(_FIRM_GRADING_CASE.read_text())
    assert _canonical(_strategy(tmp_path)["grading"]) == _canonical(golden)


def test_decision_rows_validate_against_decision_file_v1(tmp_path: Path) -> None:
    """Every row of a firm record's decisions.parquet, as JSON values, is a row
    `decision_file.v1` accepts."""
    rows = _decision_rows(tmp_path)

    assert len(rows) == 4 * len(_DAYS)
    assert [_messages("decision_file.v1", row) for row in rows] == [[]] * len(rows)


def _plant_top(value: dict[str, Any]) -> None:
    value[_PLANTED] = 1


@pytest.mark.parametrize(
    ("stem", "build", "plant"),
    [
        ("submission.v1", _submission, _plant_top),
        ("submission.v1", _submission, lambda value: value["client"].update({_PLANTED: 1})),
        ("strategy_scaffold.v1", _strategy, _plant_top),
        (
            "strategy_scaffold.v1",
            _strategy,
            lambda value: value["grading"]["gates"].update({_PLANTED: 1}),
        ),
        ("decision_file.v1", lambda tmp_path: _decision_rows(tmp_path)[0], _plant_top),
    ],
    ids=["submission", "submission-client", "strategy", "strategy-gates", "decision-row"],
)
def test_an_output_with_a_planted_unknown_field_is_refused(
    tmp_path: Path,
    stem: str,
    build: Callable[[Path], dict[str, Any]],
    plant: Callable[[dict[str, Any]], None],
) -> None:
    """An output its schema accepts is refused once an unknown field is planted in it."""
    output = build(tmp_path)
    planted = copy.deepcopy(output)
    plant(planted)

    assert _messages(stem, output) == []
    with pytest.raises(jsonschema.ValidationError, match=_PLANTED):
        _validator(stem).validate(planted)
