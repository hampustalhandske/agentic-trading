# test_edgelab_fixture_pins.py

"""Every file vendored from edgelab (`tests/fixtures/edgelab_*`) is pinned by sha256 here, and
these checks need no edgelab checkout, so they run in every suite. The one vendored directory,
`edgelab_grading_golden/` (edgelab's pinned schemas and golden cases), is pinned through its sum
files by `tests/unit/firm_record/test_edgelab_grading_golden.py`. The live comparisons with an
edgelab checkout (marked `edgelab_live`) and what each file is for are in the tests next to the
code that uses it."""

import hashlib
from pathlib import Path

import pytest

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
_PINS = {
    "edgelab_decision_file_v1.json": (
        "4c87c72531fd68314926e41e9408c414e9224f9c769dd8a1768ea97bd5d6759e"
    ),
    "edgelab_grading_spec.json": "01151817bbe77012a3637ea12373c7fabc6a1f66c4189ba7c3ca5618dee6cbfb",
    "edgelab_replay_strategy.yaml": (
        "1a56d539098a33b0e3e7525696dce5cfe1325f9ceb0a0209a9c868bdb530a7f2"
    ),
    "edgelab_submission_v1.json": (
        "1283fa42a5dffdabc4e41adfec2ec36c7e094e5ebcaec964c4df7e82d6c2075e"
    ),
}
_PINNED_ELSEWHERE = {"edgelab_grading_golden"}


def test_every_vendored_edgelab_file_is_pinned() -> None:
    """The edgelab files under tests/fixtures are exactly the pinned ones, and the edgelab
    directories exactly the one pinned elsewhere."""
    vendored = list(_FIXTURES.glob("edgelab_*"))
    assert {path.name for path in vendored if path.is_file()} == set(_PINS)
    assert {path.name for path in vendored if path.is_dir()} == _PINNED_ELSEWHERE


@pytest.mark.parametrize(("name", "sha256"), sorted(_PINS.items()))
def test_vendored_edgelab_file_is_the_pinned_copy(name: str, sha256: str) -> None:
    """The vendored file's bytes hash to its pinned sha256."""
    assert hashlib.sha256((_FIXTURES / name).read_bytes()).hexdigest() == sha256
