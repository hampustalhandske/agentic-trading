# test_edgelab_grading_golden.py

"""edgelab's pinned JSON schemas and golden cases, vendored at `tests/fixtures/
edgelab_grading_golden/`, and the firm record's outputs checked against them.

The slot holds edgelab's `docs/schema/` at commit 04d7c5c, byte for byte, under the same
relative path: `SHA256SUMS` and the thirteen schemas it lists (`bundle.v1`, `decision_file.v1`,
the four `frame.*.v1`, `grading_spec.v1`, `result.v4`, `result.v5`, `strategy_scaffold.v1`,
`strategy_scaffold.v2`, `submission.v1`, `submission.v2`), and
`golden/<schema stem>/accept/*.json` and `.../refuse/*.json`. Beside it, `SOURCE` names the
commit and `GOLDEN_SHA256SUMS` lists the sha256 of each golden case, which edgelab's
`SHA256SUMS` does not cover. Both sum files are pinned by sha256 here, so every vendored byte is
pinned. When an edgelab checkout is reachable, each schema and `SHA256SUMS` must equal the file
at the same path in it (marked `edgelab_live`), so a schema change on either side fails until
the slot is refreshed from edgelab.

Every accept case validates against its schema and every refuse case fails it, under the
`jsonschema` Draft 2020-12 validator with format checking, so the validator is shown to refuse.
A firm record's `submission.json` validates against `submission.v2`, the version it writes, and
names the policy `edgelab-2026.11.1`; `submission.v1` refuses it. Its `strategy/strategy.yaml`
validates against `strategy_scaffold.v2` and has no `grading:` block. Its decision rows
validate against `decision_file.v1`. Each output with an unknown field planted in it is refused
by its schema. The outputs are built from fake data with the helpers of `test_firm_store.py`;
nothing reads the lake."""

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

from agentic_trading.firm_record.edgelab_strategy import REPLAY_STRATEGY_YAML
from agentic_trading.firm_record.submission import (
    SUBMISSION_FILE,
    submission_for,
    write_submission_folder,
)
from agentic_trading.stage_a.spec import FirmSpecRecord
from edgelab_live import edgelab_path
from unit.firm_record.test_firm_store import _DAYS, _SOURCES, _day_rows, _json_rows, _open, _record

_SLOT = Path(__file__).resolve().parents[2] / "fixtures" / "edgelab_grading_golden"
_SCHEMAS = _SLOT / "docs" / "schema"
_GOLDEN = _SCHEMAS / "golden"
_SOURCE = _SLOT / "SOURCE"
_SCHEMA_SUMS = _SCHEMAS / "SHA256SUMS"
_GOLDEN_SUMS = _SLOT / "GOLDEN_SHA256SUMS"
_SHA = "04d7c5c"
_FULL_SHA = "04d7c5c8869876f7a0b650c98eaea546d067f428"
_SUBJECT = (
    "The top pod breaks a tie by pod id; the row-pick allowlist and the lake contract check "
    "cover the firm-book modules"
)
# The sha256 of edgelab's SHA256SUMS at 04d7c5c, and of GOLDEN_SHA256SUMS.
_SCHEMA_SUMS_SHA256 = "6e83c15ff567e95bcf256f4a36e372a93c4587c8973615e9ebe9a94ccfa6a6a5"
_GOLDEN_SUMS_SHA256 = "da103dcd8ad4539e8e340cb7817dec2597aaebcf5d54748e9065ad3845a897c3"
_SCHEMA_FILES = {
    "bundle.v1.json",
    "decision_file.v1.json",
    "frame.bars.v1.json",
    "frame.book_snapshots.v1.json",
    "frame.features.v1.json",
    "frame.resolutions.v1.json",
    "grading_spec.v1.json",
    "result.v4.json",
    "result.v5.json",
    "strategy_scaffold.v1.json",
    "strategy_scaffold.v2.json",
    "submission.v1.json",
    "submission.v2.json",
}
# The published grading policy a firm record's submission.json names.
_POLICY = "edgelab-2026.11.1"
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


def _messages_for_case(stem: str, value: Any) -> list[str]:
    """Each error the `stem` schema reports for a golden case `value`. A `frame.*` schema
    describes one row of an array, so each row of the case is checked."""
    if stem.startswith("frame."):
        return [message for row in value for message in _messages(stem, row)]
    return _messages(stem, value)


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
    """SOURCE names edgelab commit 04d7c5c by short and full sha and by its subject."""
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
    """SHA256SUMS lists the thirteen schemas and GOLDEN_SHA256SUMS the golden cases; the slot
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


@pytest.mark.edgelab_live
@pytest.mark.parametrize("name", sorted(_SCHEMA_FILES | {"SHA256SUMS"}))
def test_vendored_schema_matches_edgelab(name: str) -> None:
    """When an edgelab checkout is reachable, its `docs/schema/<name>` is byte-identical to the
    vendored copy."""
    source = edgelab_path(Path("docs") / "schema" / name)
    assert source.read_bytes() == (_SCHEMAS / name).read_bytes(), (
        f"edgelab's docs/schema/{name} changed; refresh the slot from it"
    )


# Golden cases


def test_every_schema_with_golden_cases_has_both_verdicts() -> None:
    """Each schema folder under golden/ is a vendored schema with at least one accept case and,
    but for `result.v4`, at least one refuse case."""
    cases = _golden_cases()
    stems = {stem for stem, _, _ in cases}
    assert stems == {
        "bundle.v1",
        "frame.bars.v1",
        "frame.book_snapshots.v1",
        "frame.features.v1",
        "frame.resolutions.v1",
        "grading_spec.v1",
        "result.v4",
        "result.v5",
        "strategy_scaffold.v1",
        "strategy_scaffold.v2",
        "submission.v1",
        "submission.v2",
    }
    for stem in stems:
        assert (_SCHEMAS / f"{stem}.json").is_file()
        verdicts = {verdict for found, verdict, _ in cases if found == stem}
        assert verdicts == ({"accept"} if stem == "result.v4" else {"accept", "refuse"})


@pytest.mark.parametrize(
    ("stem", "verdict", "case"),
    _golden_cases(),
    ids=lambda value: value.name if isinstance(value, Path) else str(value),
)
def test_a_golden_case_gets_its_verdict(stem: str, verdict: str, case: Path) -> None:
    """An accept case validates against its schema and a refuse case fails it; a `frame.*` case
    is an array of rows, each checked against the row schema."""
    messages = _messages_for_case(stem, json.loads(case.read_text()))
    if verdict == "accept":
        assert messages == []
    else:
        assert messages


# The firm record's outputs


@pytest.mark.parametrize("counted", [False, True], ids=["uncounted", "counted"])
def test_submission_json_validates_against_submission_v2(tmp_path: Path, counted: bool) -> None:
    """A firm record's submission.json, uncounted or counted, is a version 2 file that
    `submission.v2` accepts and that names the policy `edgelab-2026.11.1`."""
    folder = _write_folder(tmp_path, _record(), counted=counted)
    submission = json.loads((folder / SUBMISSION_FILE).read_text())

    assert submission["schema_version"] == 2
    assert submission["policy"] == _POLICY
    assert _messages("submission.v2", submission) == []


def test_submission_v1_and_a_policy_less_or_unknown_policy_file_are_refused(
    tmp_path: Path,
) -> None:
    """Negative controls: `submission.v1` refuses a firm record's submission.json, `submission.v2`
    refuses it as a version 1 file and with a policy edgelab has not published."""
    submission = _submission(tmp_path)

    assert _messages("submission.v1", submission)
    assert _messages("submission.v2", submission | {"schema_version": 1})
    assert _messages("submission.v2", submission | {"policy": "edgelab-2099.01.0"})


def test_strategy_yaml_validates_against_strategy_scaffold_v2(tmp_path: Path) -> None:
    """A firm record's strategy/strategy.yaml, its dates as ISO strings, is a config
    `strategy_scaffold.v2` accepts."""
    assert _messages("strategy_scaffold.v2", _strategy(tmp_path)) == []


def test_strategy_yaml_has_no_grading_block_and_a_planted_one_is_checked(tmp_path: Path) -> None:
    """The strategy.yaml of a firm record has no `grading:` block, so edgelab grades it under
    the submission's policy. Negative control: a grading block with a field `grading_spec.v1`
    does not have is refused by `strategy_scaffold.v2`."""
    strategy = _strategy(tmp_path)
    assert "grading" not in strategy

    grading = yaml.safe_load(REPLAY_STRATEGY_YAML)["grading"]
    grading["gates"]["max_spa_p"] = 0.05
    assert _messages("strategy_scaffold.v2", strategy | {"grading": grading})


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
        ("submission.v2", _submission, _plant_top),
        ("submission.v2", _submission, lambda value: value["client"].update({_PLANTED: 1})),
        ("strategy_scaffold.v2", _strategy, _plant_top),
        (
            "strategy_scaffold.v2",
            _strategy,
            lambda value: value["hypothesis"].update({_PLANTED: 1}),
        ),
        ("decision_file.v1", lambda tmp_path: _decision_rows(tmp_path)[0], _plant_top),
    ],
    ids=["submission", "submission-client", "strategy", "strategy-hypothesis", "decision-row"],
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
