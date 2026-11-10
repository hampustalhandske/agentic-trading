# test_runner_resume_environment.py

"""A firm record resumes only under the numerical environment its trial was started in: the
environment its agentic_meta.json records must match `runtime_environment()` in `libraries`,
`quantcore_sha`, `python_version` and `platform_machine`, or the resume is refused before
anything is written. `platform_release` and the BLAS fields may differ. A trial whose
agentic_meta.json records no environment is refused too; a fresh trial records the
environment of the pass that starts it."""

import json
from pathlib import Path
from typing import Any

import pytest

from agentic_trading.provenance import runtime_environment
from unit.firm_record.test_runner import _SESSIONS, _rows, _run

_META = "agentic_meta.json"


def _paused(store_dir: Path) -> Path:
    """Record 3 days into `store_dir` and return the trial directory."""
    paused = _run(store_dir, max_days=3)
    assert paused.days_written == 3 and not paused.complete
    return store_dir / paused.spec_hash


def _edit_environment(trial: Path, edit: Any) -> dict[str, Any]:
    """Apply `edit` to the parsed agentic_meta.json of `trial`, write it back, return it."""
    meta: dict[str, Any] = json.loads((trial / _META).read_text())
    edit(meta)
    (trial / _META).write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def _snapshot(trial: Path) -> dict[str, bytes]:
    """Every file of `trial`, by relative path, with its bytes."""
    return {
        str(path.relative_to(trial)): path.read_bytes()
        for path in sorted(trial.rglob("*"))
        if path.is_file()
    }


def test_fresh_trial_records_the_current_environment(tmp_path: Path) -> None:
    """A fresh trial's agentic_meta.json holds the environment of the pass that wrote it."""
    trial = _paused(tmp_path)
    meta = json.loads((trial / _META).read_text())
    assert meta["environment"] == runtime_environment().model_dump(mode="json")


def test_resume_under_the_same_environment_completes(tmp_path: Path) -> None:
    """Control: a resume under the recorded environment writes the remaining days."""
    trial = _paused(tmp_path / "paused")
    resumed = _run(tmp_path / "paused")
    unbroken = _run(tmp_path / "unbroken")

    assert resumed.days_written == len(_SESSIONS) - 3 and resumed.complete
    assert _rows(tmp_path / "paused", resumed) == _rows(tmp_path / "unbroken", unbroken)
    assert json.loads((trial / _META).read_text())["environment"] == (
        runtime_environment().model_dump(mode="json")
    )


def test_resume_under_another_numpy_version_is_refused(tmp_path: Path) -> None:
    """A trial recorded under another numpy version refuses to resume, naming the field, and
    nothing in the trial changes."""
    trial = _paused(tmp_path)
    _edit_environment(trial, lambda meta: meta["environment"]["libraries"].update(numpy="0.0.1"))
    before = _snapshot(trial)

    with pytest.raises(ValueError, match=r"libraries\.numpy") as refused:
        _run(tmp_path)

    assert "0.0.1" in str(refused.value)
    assert runtime_environment().libraries.numpy in str(refused.value)
    assert _snapshot(trial) == before


@pytest.mark.parametrize(
    ("field", "value"),
    [("quantcore_sha", "0" * 40), ("python_version", "2.7.18"), ("platform_machine", "sparc")],
)
def test_resume_under_another_interpreter_or_machine_is_refused(
    tmp_path: Path, field: str, value: str
) -> None:
    """A different quantcore commit, Python version or machine architecture refuses."""
    trial = _paused(tmp_path)
    _edit_environment(trial, lambda meta: meta["environment"].update({field: value}))

    with pytest.raises(ValueError, match=field):
        _run(tmp_path)


def test_trial_without_a_recorded_environment_is_refused(tmp_path: Path) -> None:
    """A trial whose agentic_meta.json has no environment key is refused, not adopted."""
    trial = _paused(tmp_path)
    _edit_environment(trial, lambda meta: meta.pop("environment"))
    before = _snapshot(trial)

    with pytest.raises(ValueError, match="records no environment"):
        _run(tmp_path)

    assert _snapshot(trial) == before


def test_platform_release_and_blas_differences_resume(tmp_path: Path) -> None:
    """A different OS release or BLAS report does not block a resume."""
    trial = _paused(tmp_path)
    _edit_environment(
        trial,
        lambda meta: meta["environment"].update(
            platform_release="0.0.0-other", blas_name="openblas", blas_version="0.0.1"
        ),
    )

    resumed = _run(tmp_path)

    assert resumed.complete
