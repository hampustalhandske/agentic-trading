# test_replay_strategy_fixture.py

"""`REPLAY_STRATEGY_YAML` is edgelab's replay scaffold, and its grading block is one edgelab's
`GradingSpec` accepts.

`tests/fixtures/edgelab_replay_strategy.yaml`, vendored from edgelab commit 04d7c5c and pinned
by sha256, is the text `edgelab strategy init <slug> --asset-class replay` scaffolds from (its
template `strategy.yaml` with `@KIND@\\n` replaced by `kinds/replay.yaml`). The grading block's
schema is edgelab's `grading_spec.v1.json` from the slot `tests/fixtures/edgelab_grading_golden/`
(pinned in `test_edgelab_grading_golden.py`): `GradingSpec.model_json_schema()` with a top-level
`$id`. Without edgelab, `REPLAY_STRATEGY_YAML` must equal the vendored scaffold with its
comment lines removed, and the scaffold's grading block must validate against the vendored
schema. When an edgelab checkout is reachable (`EDGELAB_DIR`, or `edgelab` beside this
repo's parent directory), its templates must compose to the vendored scaffold; when that
checkout also has a `.venv`, its `GradingSpec` must produce the vendored schema without its
`$id` and measure the block as `edgelab-2026.11.0`; these live checks are marked
`edgelab_live` (`tests/edgelab_live.py`). A change on either side fails here until the copies
are refreshed.

The grading block of `REPLAY_STRATEGY_YAML`, and its cost table on its own, are pinned by
canonical sha256 (`_canonical_sha256`), so an edit to the block fails here even when the vendored
scaffold is changed with it. A firm record's strategy.yaml drops the block: its submission's
policy supplies the grading values.
edgelab's golden accept/refuse cases and pinned schemas (EL-75) in the same slot are checked
against a firm record's outputs in `test_edgelab_grading_golden.py`."""

import copy
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import jsonschema
import pytest
import yaml

from agentic_trading.firm_record.edgelab_strategy import REPLAY_STRATEGY_YAML
from agentic_trading.firm_record.submission import strategy_config
from edgelab_live import edgelab_path
from unit.firm_record.test_firm_store import _record

_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
_SCAFFOLD = _FIXTURES / "edgelab_replay_strategy.yaml"
_SCAFFOLD_SHA256 = "1a56d539098a33b0e3e7525696dce5cfe1325f9ceb0a0209a9c868bdb530a7f2"
_GRADING_SCHEMA = _FIXTURES / "edgelab_grading_golden" / "docs" / "schema" / "grading_spec.v1.json"
_TEMPLATES = Path("src") / "edgelab" / "templates" / "strategy"
_MEASUREMENT = "edgelab-2026.11.0"
# The canonical sha256 of the grading block edgelab scaffolds and measures as
# `_MEASUREMENT`, and of its cost table (`grading.costs`, one entry per asset class).
_GRADING_BLOCK_SHA256 = "aa1fea3ee707b83e9937cbd8231dd708ea4c54332f07b077f207ac1f60c1cfb5"
_COST_TABLE_SHA256 = "35550da6cebd0a2832b97f80e752b41c9ad91e93bd4c248e7bfd73f850c0647c"
_COST_ASSET_CLASSES = {"equity", "future", "option", "fx", "crypto", "prediction_market"}


def _without_comment_lines(text: str) -> str:
    """`text` without its lines whose first non-blank character is `#`."""
    return "".join(
        line for line in text.splitlines(keepends=True) if not line.lstrip().startswith("#")
    )


def _canonical_sha256(value: Any) -> str:
    """The sha256 of `value` as JSON with sorted keys and no whitespace."""
    text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def validate_grading_block(grading: dict[str, Any]) -> None:
    """Raise `jsonschema.ValidationError` when `grading` is not a block edgelab's vendored
    `GradingSpec` schema accepts."""
    jsonschema.validate(grading, json.loads(_GRADING_SCHEMA.read_text()))


def test_vendored_scaffold_is_the_pinned_copy() -> None:
    """The vendored scaffold hashes to its pinned sha256."""
    assert hashlib.sha256(_SCAFFOLD.read_bytes()).hexdigest() == _SCAFFOLD_SHA256


def test_replay_yaml_is_the_scaffold_without_comment_lines() -> None:
    """`REPLAY_STRATEGY_YAML` is the vendored scaffold, character for character, once the
    scaffold's comment lines are removed."""
    assert _without_comment_lines(_SCAFFOLD.read_text()) == REPLAY_STRATEGY_YAML


def test_replay_grading_block_validates() -> None:
    """The grading block of `REPLAY_STRATEGY_YAML` validates against the vendored schema."""
    validate_grading_block(yaml.safe_load(REPLAY_STRATEGY_YAML)["grading"])


def test_grading_block_and_cost_table_are_the_pinned_ones() -> None:
    """The grading block of `REPLAY_STRATEGY_YAML` and its cost table, one entry per asset
    class, hash to their pinned canonical sha256."""
    grading = yaml.safe_load(REPLAY_STRATEGY_YAML)["grading"]
    assert set(grading["costs"]) == _COST_ASSET_CLASSES
    assert _canonical_sha256(grading["costs"]) == _COST_TABLE_SHA256
    assert _canonical_sha256(grading) == _GRADING_BLOCK_SHA256


def test_a_firm_record_writes_no_grading_block() -> None:
    """The strategy.yaml a firm record writes has no `grading:` block, though the scaffold has
    one: the submission's policy supplies the grading values."""
    assert "grading" in yaml.safe_load(REPLAY_STRATEGY_YAML)
    assert "grading" not in strategy_config(_record())


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("gates", "max_spa_p", 0.05),
        ("gates", "min_plateau_share", 0.6),
        ("robustness", "structural_break", None),
    ],
)
def test_a_field_edgelab_does_not_have_is_refused(section: str, field: str, value: Any) -> None:
    """The vendored schema refuses a grading block with a field `GradingSpec` does not have."""
    grading = copy.deepcopy(yaml.safe_load(REPLAY_STRATEGY_YAML)["grading"])
    grading[section][field] = value
    with pytest.raises(jsonschema.ValidationError):
        validate_grading_block(grading)


@pytest.mark.edgelab_live
def test_edgelab_templates_compose_to_the_vendored_scaffold() -> None:
    """When an edgelab checkout is reachable, its replay templates compose to the vendored
    scaffold byte for byte."""
    templates = edgelab_path(_TEMPLATES / "kinds" / "replay.yaml").parents[1]
    composed = (
        (templates / "strategy.yaml")
        .read_text()
        .replace("@KIND@\n", (templates / "kinds" / "replay.yaml").read_text())
    )
    assert composed == _SCAFFOLD.read_text(), (
        f"edgelab's replay scaffold changed; refresh {_SCAFFOLD.name} and REPLAY_STRATEGY_YAML"
    )


@pytest.mark.edgelab_live
def test_edgelab_grading_spec_matches_and_measures_the_block() -> None:
    """When an edgelab checkout with a `.venv` is reachable, its `GradingSpec` schema equals the
    vendored one without its top-level `$id`, and it measures the grading block of
    `REPLAY_STRATEGY_YAML` as `edgelab-2026.11.0`."""
    python = edgelab_path(Path(".venv") / "bin" / "python")
    edgelab = python.parents[2]
    program = (
        "import json, sys\n"
        "from edgelab.grading.policy import GradingSpec\n"
        "from edgelab.grading.versions import measurement_version\n"
        "grading = GradingSpec.model_validate(json.load(sys.stdin))\n"
        "print(json.dumps({'schema': GradingSpec.model_json_schema(),"
        " 'measurement': measurement_version(grading)}))\n"
    )
    grading = yaml.safe_load(REPLAY_STRATEGY_YAML)["grading"]
    completed = subprocess.run(
        [str(python), "-I", "-c", program],
        input=json.dumps(grading),
        capture_output=True,
        text=True,
        cwd=edgelab,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    found = json.loads(completed.stdout)
    vendored = json.loads(_GRADING_SCHEMA.read_text())
    vendored.pop("$id")
    assert found["schema"] == vendored, (
        f"edgelab's GradingSpec changed; refresh {_GRADING_SCHEMA.name}"
    )
    assert found["measurement"] == _MEASUREMENT
