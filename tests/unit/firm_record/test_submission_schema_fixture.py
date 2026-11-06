# test_submission_schema_fixture.py

"""`tests/fixtures/edgelab_submission_v1.json` is a byte copy of edgelab's
`docs/schema/submission.v1.json` at edgelab commit b582efd, the JSON schema of edgelab's
`Submission` model that a firm record's submission.json must satisfy (checked in
`test_runner.py`). edgelab writes that file as `json.dumps(schema, indent=2, sort_keys=True) +
"\\n"` through `edgelab contract schema --out <dir>`. The copy is pinned by sha256, and when an
edgelab checkout is reachable the copy must equal edgelab's current file, so a schema change on
either side fails here until the copy is refreshed from edgelab."""

import hashlib
import os
from pathlib import Path

import pytest

_FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "edgelab_submission_v1.json"
_FIXTURE_SHA256 = "1283fa42a5dffdabc4e41adfec2ec36c7e094e5ebcaec964c4df7e82d6c2075e"
_EDGELAB_SCHEMA = Path("docs") / "schema" / "submission.v1.json"


def _edgelab_dir() -> Path | None:
    """The edgelab checkout: `EDGELAB_DIR` when set, else `edgelab` beside this repo's
    parent directory; `None` when the repo sits too near the filesystem root to have one."""
    configured = os.environ.get("EDGELAB_DIR")
    if configured:
        return Path(configured)
    parents = Path(__file__).resolve().parents
    return parents[5] / "edgelab" if len(parents) > 5 else None


def test_vendored_schema_is_the_pinned_copy() -> None:
    """The vendored schema's bytes hash to the pinned sha256."""
    assert hashlib.sha256(_FIXTURE.read_bytes()).hexdigest() == _FIXTURE_SHA256


def test_vendored_schema_matches_edgelab() -> None:
    """When an edgelab checkout is reachable, its schema is byte-identical to the copy."""
    edgelab = _edgelab_dir()
    source = edgelab / _EDGELAB_SCHEMA if edgelab is not None else None
    if source is None or not source.exists():
        pytest.skip(f"no edgelab checkout with {_EDGELAB_SCHEMA}; set EDGELAB_DIR to check drift")
    assert source.read_bytes() == _FIXTURE.read_bytes(), (
        f"edgelab's {_EDGELAB_SCHEMA} changed; refresh {_FIXTURE.name} from it"
    )
