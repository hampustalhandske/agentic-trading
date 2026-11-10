# test_decision_schema_fixture.py

"""`tests/fixtures/edgelab_decision_file_v1.json` is a byte copy of edgelab's
`docs/schema/decision_file.v1.json` at edgelab commit b5af311, the interface schema a stage (a)
decision file must satisfy (checked in `test_store.py`). The copy is pinned by sha256, and when
an edgelab checkout is reachable the copy must equal edgelab's current file, so a schema change
on either side fails here until the copy is refreshed from edgelab."""

import hashlib
from pathlib import Path

import pytest

from edgelab_live import edgelab_path

_FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "edgelab_decision_file_v1.json"
_FIXTURE_SHA256 = "4c87c72531fd68314926e41e9408c414e9224f9c769dd8a1768ea97bd5d6759e"
_EDGELAB_SCHEMA = Path("docs") / "schema" / "decision_file.v1.json"


def test_vendored_schema_is_the_pinned_copy() -> None:
    """The vendored schema's bytes hash to the pinned sha256."""
    assert hashlib.sha256(_FIXTURE.read_bytes()).hexdigest() == _FIXTURE_SHA256


@pytest.mark.edgelab_live
def test_vendored_schema_matches_edgelab() -> None:
    """When an edgelab checkout is reachable, its schema is byte-identical to the copy."""
    source = edgelab_path(_EDGELAB_SCHEMA)
    assert source.read_bytes() == _FIXTURE.read_bytes(), (
        f"edgelab's {_EDGELAB_SCHEMA} changed; refresh {_FIXTURE.name} from it"
    )
