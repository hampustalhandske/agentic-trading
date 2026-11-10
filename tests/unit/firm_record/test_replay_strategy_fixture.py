# test_replay_strategy_fixture.py

"""`REPLAY_STRATEGY_YAML` is edgelab's replay scaffold, and its grading block is one edgelab's
`GradingSpec` accepts.

Two copies from edgelab commit 62a7151 are vendored and pinned by sha256:
`tests/fixtures/edgelab_replay_strategy.yaml`, the text `edgelab strategy init <slug>
--asset-class replay` scaffolds from (its template `strategy.yaml` with `@KIND@\\n` replaced by
`kinds/replay.yaml`), and `tests/fixtures/edgelab_grading_spec.json`, edgelab's
`GradingSpec.model_json_schema()` written as `json.dumps(schema, indent=2, sort_keys=True)` plus
a newline. Without edgelab, `REPLAY_STRATEGY_YAML` must equal the vendored scaffold with its
comment lines removed, and the grading block a firm record writes must validate against the
vendored schema. When an edgelab checkout is reachable (`EDGELAB_DIR`, or `edgelab` beside this
repo's parent directory), its templates must compose to the vendored scaffold; when that
checkout also has a `.venv`, its `GradingSpec` must produce the vendored schema and measure the
block as `edgelab-2026.11.0`. A change on either side fails here until the copies are refreshed."""

import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import jsonschema
import pytest
import yaml

from agentic_trading.firm_record.edgelab_strategy import REPLAY_STRATEGY_YAML

_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
_SCAFFOLD = _FIXTURES / "edgelab_replay_strategy.yaml"
_SCAFFOLD_SHA256 = "1a56d539098a33b0e3e7525696dce5cfe1325f9ceb0a0209a9c868bdb530a7f2"
_GRADING_SCHEMA = _FIXTURES / "edgelab_grading_spec.json"
_GRADING_SCHEMA_SHA256 = "01151817bbe77012a3637ea12373c7fabc6a1f66c4189ba7c3ca5618dee6cbfb"
_TEMPLATES = Path("src") / "edgelab" / "templates" / "strategy"
_MEASUREMENT = "edgelab-2026.11.0"


def _edgelab_dir() -> Path | None:
    """The edgelab checkout: `EDGELAB_DIR` when set, else `edgelab` beside this repo's
    parent directory; `None` when the repo sits too near the filesystem root to have one."""
    configured = os.environ.get("EDGELAB_DIR")
    if configured:
        return Path(configured)
    parents = Path(__file__).resolve().parents
    return parents[5] / "edgelab" if len(parents) > 5 else None


def _without_comment_lines(text: str) -> str:
    """`text` without its lines whose first non-blank character is `#`."""
    return "".join(
        line for line in text.splitlines(keepends=True) if not line.lstrip().startswith("#")
    )


def validate_grading_block(grading: dict[str, Any]) -> None:
    """Raise `jsonschema.ValidationError` when `grading` is not a block edgelab's vendored
    `GradingSpec` schema accepts."""
    jsonschema.validate(grading, json.loads(_GRADING_SCHEMA.read_text()))


def test_vendored_files_are_the_pinned_copies() -> None:
    """The vendored scaffold and grading schema hash to their pinned sha256."""
    assert hashlib.sha256(_SCAFFOLD.read_bytes()).hexdigest() == _SCAFFOLD_SHA256
    assert hashlib.sha256(_GRADING_SCHEMA.read_bytes()).hexdigest() == _GRADING_SCHEMA_SHA256


def test_replay_yaml_is_the_scaffold_without_comment_lines() -> None:
    """`REPLAY_STRATEGY_YAML` is the vendored scaffold, character for character, once the
    scaffold's comment lines are removed."""
    assert _without_comment_lines(_SCAFFOLD.read_text()) == REPLAY_STRATEGY_YAML


def test_replay_grading_block_validates() -> None:
    """The grading block of `REPLAY_STRATEGY_YAML` validates against the vendored schema."""
    validate_grading_block(yaml.safe_load(REPLAY_STRATEGY_YAML)["grading"])


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


def test_edgelab_templates_compose_to_the_vendored_scaffold() -> None:
    """When an edgelab checkout is reachable, its replay templates compose to the vendored
    scaffold byte for byte."""
    edgelab = _edgelab_dir()
    templates = edgelab / _TEMPLATES if edgelab is not None else None
    if templates is None or not (templates / "kinds" / "replay.yaml").exists():
        pytest.skip(f"no edgelab checkout with {_TEMPLATES}; set EDGELAB_DIR to check drift")
    composed = (
        (templates / "strategy.yaml")
        .read_text()
        .replace("@KIND@\n", (templates / "kinds" / "replay.yaml").read_text())
    )
    assert composed == _SCAFFOLD.read_text(), (
        f"edgelab's replay scaffold changed; refresh {_SCAFFOLD.name} and REPLAY_STRATEGY_YAML"
    )


def test_edgelab_grading_spec_matches_and_measures_the_block() -> None:
    """When an edgelab checkout with a `.venv` is reachable, its `GradingSpec` schema equals the
    vendored one and it measures the grading block of `REPLAY_STRATEGY_YAML` as
    `edgelab-2026.11.0`."""
    edgelab = _edgelab_dir()
    python = edgelab / ".venv" / "bin" / "python" if edgelab is not None else None
    if python is None or not python.exists():
        pytest.skip("no edgelab checkout with a .venv; set EDGELAB_DIR to check drift")
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
    assert found["schema"] == json.loads(_GRADING_SCHEMA.read_text()), (
        f"edgelab's GradingSpec changed; refresh {_GRADING_SCHEMA.name}"
    )
    assert found["measurement"] == _MEASUREMENT
