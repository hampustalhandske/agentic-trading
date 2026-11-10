# test_firm_record_determinism.py

"""The firm record and the stage (a) batch are pure functions of their inputs: two passes in
two processes, one with string-hash seed 0 and one with a seed drawn per test session, write
byte-identical trial folders; a firm
pass SIGKILLed after some days are checkpointed and then resumed in a fresh process writes the
folder of an unbroken pass; and one Commander decides the same bits with one worker as with
four.

Each pass runs the firm of `_firm.py` through `pipeline_entry.py` in a subprocess against the
tests' `FakeLake`. Every file under the store directory is compared byte for byte, except the
JSON fields of `_PROVENANCE_FIELDS`. Each comparison has a negative control that plants a
defect in the firm (`_firm.PLANT_ENV`) and requires the comparison to catch it."""

import json
import os
import random
import signal
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import pytest

from agentic_trading.data.loaders import lakehouse_loader
from agentic_trading.scoring.scoreboard import PerformanceScoreboard
from agentic_trading.scoring.track_record import TrackRecord
from unit.determinism import _firm

pytestmark = pytest.mark.determinism

_ENTRY = Path(__file__).with_name("pipeline_entry.py")
_REPO_ROOT = Path(__file__).resolve().parents[3]
# Seconds one firm-record subprocess may take.
_PASS_TIMEOUT = 180
# Days checkpointed before a pass is killed.
_KILLED_AFTER_DAYS = 4
# The string-hash seed of the second pass of each comparison: `AGENTIC_DETERMINISM_HASH_SEED`
# when set, which reproduces a failed session, else drawn once per test session. Every
# comparison names it in its failure message. The first pass runs with seed 0.
_HASH_SEED_ENV = "AGENTIC_DETERMINISM_HASH_SEED"
_OTHER_HASH_SEED = os.environ.get(_HASH_SEED_ENV) or str(random.SystemRandom().randrange(1, 2**32))
# Every pass runs single-threaded BLAS and OpenMP: multi-threaded OpenBLAS
# reduces in a thread-dependent order, which can change GARCH-family fits in
# the last bits on Linux.
_BLAS_THREAD_ENV = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
_SEED_NOTE = f"second pass PYTHONHASHSEED={_OTHER_HASH_SEED} (rerun with {_HASH_SEED_ENV})"

# JSON fields that may differ between two passes of the same pipeline, by file (relative to
# the store directory) and key path, with the reason each is provenance rather than output.
# Two passes from one commit and one environment write no such field, so it is empty.
_PROVENANCE_FIELDS: dict[str, tuple[tuple[str, ...], ...]] = {}


def _pass(
    store_dir: Path,
    pipeline: str = "firm",
    *,
    hash_seed: str = "0",
    plant: str | None = None,
    kill_at_day: int | None = None,
    kill_point: str = "after",
) -> subprocess.CompletedProcess[str]:
    """One pass of `pipeline` (`firm` or `stage_a`) under `store_dir` in a fresh interpreter
    with `PYTHONHASHSEED` `hash_seed`, `_BLAS_THREAD_ENV` and the planted defect `plant`, killed per
    `kill_at_day`/`kill_point` (see `pipeline_entry.py`)."""
    env = {key: value for key, value in os.environ.items() if key != _firm.PLANT_ENV}
    env["PYTHONHASHSEED"] = hash_seed
    env.update(_BLAS_THREAD_ENV)
    if plant is not None:
        env[_firm.PLANT_ENV] = plant
    command = [sys.executable, str(_ENTRY), pipeline, str(store_dir)]
    if kill_at_day is not None:
        command += ["--kill-at-day", str(kill_at_day), "--kill-point", kill_point]
    return subprocess.run(
        command,
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=_PASS_TIMEOUT,
        check=False,
    )


def _completed_pass(store_dir: Path, **kwargs: Any) -> Path:
    """Runs a firm `_pass` to completion and returns `store_dir`."""
    completed = _pass(store_dir, **kwargs)
    assert completed.returncode == 0, completed.stderr[-4000:]
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["complete"], result
    return store_dir


def _without(document: Any, path: tuple[str, ...]) -> None:
    """Removes the field at key `path` from `document` when it is there."""
    *parents, leaf = path
    for key in parents:
        if not isinstance(document, dict) or key not in document:
            return
        document = document[key]
    if isinstance(document, dict):
        document.pop(leaf, None)


def _comparable(trial: Path, relative: str) -> bytes:
    """The bytes of `trial / relative`, with the JSON fields `_PROVENANCE_FIELDS` lists for it
    removed."""
    content = (trial / relative).read_bytes()
    fields = _PROVENANCE_FIELDS.get(relative)
    if not fields:
        return content
    document = json.loads(content)
    for path in fields:
        _without(document, path)
    return json.dumps(document, sort_keys=True).encode()


def store_differences(first: Path, second: Path) -> list[str]:
    """Each file that only one of two store directories holds, and each file whose comparable
    bytes differ, as paths relative to the store directory. A firm pass's engine checkpoint,
    which sits beside its trial folder and holds the run's state rather than its output, is
    left out."""
    files = {
        store: {
            path.relative_to(store).as_posix()
            for path in store.rglob("*")
            if path.is_file() and not path.name.endswith(".checkpoint.sqlite")
        }
        for store in (first, second)
    }
    differences = sorted(files[first] ^ files[second])
    for relative in sorted(files[first] & files[second]):
        if _comparable(first, relative) != _comparable(second, relative):
            differences.append(relative)
    return differences


@pytest.fixture(scope="module")
def unbroken(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The store directory of an unbroken firm pass."""
    return _completed_pass(tmp_path_factory.mktemp("unbroken"))


def test_the_record_holds_sized_positions_of_every_pod(unbroken: Path) -> None:
    """The compared record is not empty: every pod has active rows with a non-zero target."""
    (decisions,) = unbroken.glob("*/decisions.parquet")
    rows = pq.read_table(decisions).to_pylist()
    sized = {row["pod_id"] for row in rows if row["status"] == "active" and row["target_weight"]}

    assert sized == {"risk_manager", "alt_data_manager", "fred_pod_manager"}


def test_two_processes_write_identical_trial_folders(unbroken: Path, tmp_path: Path) -> None:
    """A second pass in another process, with another string-hash seed, writes the same
    files with the same bytes, provenance fields aside."""
    again = _completed_pass(tmp_path, hash_seed=_OTHER_HASH_SEED)

    assert store_differences(unbroken, again) == [], _SEED_NOTE


def test_an_unseeded_draw_breaks_the_two_process_comparison(tmp_path: Path) -> None:
    """Negative control: with an unseeded random draw planted in an analyst, two passes write
    different decisions."""
    first = _completed_pass(tmp_path / "first", plant="random")
    second = _completed_pass(tmp_path / "second", plant="random", hash_seed="2")

    assert any(path.endswith("decisions.parquet") for path in store_differences(first, second))


def test_a_hash_order_dependence_breaks_the_two_process_comparison(tmp_path: Path) -> None:
    """Negative control: with an analyst whose signal depends on a set's iteration order, passes
    under string-hash seeds 0 and 1 write different decisions."""
    first = _completed_pass(tmp_path / "first", plant="hash_order")
    second = _completed_pass(tmp_path / "second", plant="hash_order", hash_seed="1")

    assert any(path.endswith("decisions.parquet") for path in store_differences(first, second))


@pytest.mark.parametrize("kill_point", ["before", "after"])
def test_a_killed_pass_resumes_to_the_unbroken_trial_folder(
    unbroken: Path, tmp_path: Path, kill_point: str
) -> None:
    """A pass SIGKILLed on the write of the day after `_KILLED_AFTER_DAYS` checkpointed days,
    before or after that day's rows reach the store, then resumed in a fresh process, ends
    with the unbroken pass's trial folder."""
    killed = _pass(tmp_path, kill_at_day=_KILLED_AFTER_DAYS, kill_point=kill_point)
    assert killed.returncode == -signal.SIGKILL, killed.stderr[-4000:]

    resumed = _completed_pass(tmp_path, hash_seed=_OTHER_HASH_SEED)

    assert store_differences(unbroken, resumed) == [], _SEED_NOTE


def test_state_outside_the_checkpoint_breaks_the_resume_comparison(tmp_path: Path) -> None:
    """Negative control: with an analyst whose signal depends on a call counter the checkpoint
    does not hold, a killed and resumed pass writes other decisions than an unbroken one."""
    unbroken_planted = _completed_pass(tmp_path / "unbroken", plant="unsaved_state")
    killed = _pass(tmp_path / "killed", kill_at_day=_KILLED_AFTER_DAYS, plant="unsaved_state")
    assert killed.returncode == -signal.SIGKILL, killed.stderr[-4000:]
    resumed = _completed_pass(tmp_path / "killed", plant="unsaved_state")

    differences = store_differences(unbroken_planted, resumed)
    assert any(path.endswith("decisions.parquet") for path in differences)


def _stage_a_batch(store_dir: Path, **kwargs: Any) -> Path:
    """Runs a stage (a) `_pass` to completion and returns `store_dir`."""
    completed = _pass(store_dir, "stage_a", **kwargs)
    assert completed.returncode == 0, completed.stderr[-4000:]
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert not result["stopped_for_resources"], result
    assert len(result["analysts"]) == 4, result
    return store_dir


def test_two_processes_write_identical_stage_a_trials(tmp_path: Path) -> None:
    """Two stage (a) batches of the firm's analysts in two processes, with different
    string-hash seeds, write the same files with the same bytes."""
    first = _stage_a_batch(tmp_path / "first")
    second = _stage_a_batch(tmp_path / "second", hash_seed=_OTHER_HASH_SEED)

    assert store_differences(first, second) == [], _SEED_NOTE


def test_an_unseeded_draw_breaks_the_stage_a_comparison(tmp_path: Path) -> None:
    """Negative control: with an unseeded random draw planted in an analyst, two stage (a)
    batches write different signals."""
    first = _stage_a_batch(tmp_path / "first", plant="random")
    second = _stage_a_batch(tmp_path / "second", plant="random", hash_seed="2")

    assert store_differences(first, second) != []


def _decide(parallel_workers: int) -> str:
    """The JSON of the decisions and pod targets of one firm Commander with
    `parallel_workers` workers over the window's first three sessions, inside a lake session."""
    frame = _firm.market_frame()
    sessions = frame.index[frame.index >= pd.Timestamp(_firm.WINDOW_START)][:3]
    commander = _firm.build_commander(parallel_workers=parallel_workers)
    scoreboard = PerformanceScoreboard(TrackRecord())
    days: list[Any] = []
    try:
        with lakehouse_loader.lake_session(_firm.LAKE_TIME):
            for day in sessions:
                decisions = commander.decide(frame.loc[:day], day, scoreboard)
                days.append(
                    {
                        "decisions": [decision.model_dump(mode="json") for decision in decisions],
                        "pod_targets": [
                            target.model_dump(mode="json") for target in commander.pod_targets()
                        ],
                    }
                )
    finally:
        commander.close()
    return json.dumps(days, sort_keys=True)


@pytest.fixture
def firm_lake(fake_lake: Any) -> Iterator[Any]:
    _firm.add_fred_vintages(fake_lake)
    yield fake_lake


def test_one_worker_and_four_decide_the_same_bits(
    firm_lake: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Commander's decisions and pod targets, every float in them, are the same with one
    worker as with four."""
    monkeypatch.delenv(_firm.PLANT_ENV, raising=False)

    assert _decide(4) == _decide(1)


def test_a_thread_dependent_signal_breaks_the_worker_comparison(
    firm_lake: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Negative control: with an analyst whose signal depends on the thread that asks it, one
    worker and four decide different bits."""
    monkeypatch.setenv(_firm.PLANT_ENV, "thread")

    assert _decide(4) != _decide(1)
