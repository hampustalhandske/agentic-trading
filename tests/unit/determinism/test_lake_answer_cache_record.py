# test_lake_answer_cache_record.py

"""The lake answer cache never changes a firm record and spares the lake every answer it holds:
a pass that fills the cache and a pass that reads every answer from it, with the fake lake
refusing every SQL read, write byte-identical trial folders, the same as a pass with no cache;
and a pass SIGKILLed after some days and resumed with the same cache sends the lake none of the
reads the killed pass already made, and ends with the unbroken pass's trial folder.

Each pass runs the firm of `_firm.py` through `pipeline_entry.py` in a subprocess against the
tests' `FakeLake`, as `test_pipeline_determinism.py` does."""

import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from unit.determinism.test_pipeline_determinism import (
    _BLAS_THREAD_ENV,
    _ENTRY,
    _KILLED_AFTER_DAYS,
    _PASS_TIMEOUT,
    _REPO_ROOT,
    store_differences,
)

pytestmark = pytest.mark.determinism

_COUNTS_PREFIX = "lake-answer-cache "


def _pass(store_dir: Path, *options: str) -> subprocess.CompletedProcess[str]:
    """One firm pass under `store_dir` in a fresh interpreter with string-hash seed 0 and
    `options` given to `pipeline_entry.py`."""
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "0"
    env.update(_BLAS_THREAD_ENV)
    return subprocess.run(
        [sys.executable, str(_ENTRY), "firm", str(store_dir), *options],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=_PASS_TIMEOUT,
        check=False,
    )


def _completed(completed: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    """The cache counts of a pass that ran to the end of the record."""
    assert completed.returncode == 0, completed.stderr[-4000:]
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["complete"], result
    (line,) = [line for line in completed.stderr.splitlines() if line.startswith(_COUNTS_PREFIX)]
    counts: dict[str, Any] = json.loads(line.removeprefix(_COUNTS_PREFIX))
    return counts


def _calls(log: Path) -> list[tuple[str, str]]:
    """The SQL reads a pass sent the fake lake, as (query, params JSON) pairs."""
    if not log.exists():
        return []
    return [
        (query, json.dumps(params))
        for query, params in map(json.loads, log.read_text().splitlines())
    ]


def test_a_pass_from_the_cache_alone_writes_the_same_trial_folder(tmp_path: Path) -> None:
    """A pass filling the cache, then one reading only from it while the lake refuses every
    SQL read, write the same bytes as each other and as a pass with no cache."""
    cache = tmp_path / "lake_cache"
    uncached = tmp_path / "uncached"
    assert _pass(uncached).returncode == 0
    filling_log = tmp_path / "filling.jsonl"
    filling = _completed(
        _pass(tmp_path / "filling", "--lake-cache", str(cache), "--calls-log", str(filling_log))
    )
    cached_log = tmp_path / "cached.jsonl"
    cached = _completed(
        _pass(
            tmp_path / "cached",
            "--lake-cache",
            str(cache),
            "--refuse-sql",
            "--calls-log",
            str(cached_log),
        )
    )

    assert filling["stored"] > 0
    assert len(_calls(filling_log)) == filling["stored"] + filling["unstored"]
    assert cached["hits"] > 0
    assert cached["stored"] == cached["unstored"] == cached["discarded"] == 0
    assert _calls(cached_log) == []
    assert store_differences(tmp_path / "filling", tmp_path / "cached") == []
    assert store_differences(uncached, tmp_path / "cached") == []


def test_a_resumed_pass_reads_every_earlier_answer_from_the_cache(tmp_path: Path) -> None:
    """A pass SIGKILLed after `_KILLED_AFTER_DAYS` checkpointed days, resumed in a fresh process
    with the same cache, sends the lake none of the reads the killed pass made, and ends with
    the unbroken pass's trial folder."""
    cache = tmp_path / "lake_cache"
    killed_log = tmp_path / "killed.jsonl"
    killed = _pass(
        tmp_path / "store",
        "--lake-cache",
        str(cache),
        "--calls-log",
        str(killed_log),
        "--kill-at-day",
        str(_KILLED_AFTER_DAYS),
    )
    assert killed.returncode == -signal.SIGKILL, killed.stderr[-4000:]
    resumed_log = tmp_path / "resumed.jsonl"
    resumed = _completed(
        _pass(tmp_path / "store", "--lake-cache", str(cache), "--calls-log", str(resumed_log))
    )
    assert _pass(tmp_path / "unbroken").returncode == 0

    asked_before = set(_calls(killed_log))
    asked_after = _calls(resumed_log)
    assert asked_before
    assert resumed["hits"] > 0
    assert [call for call in asked_after if call in asked_before] == []
    assert store_differences(tmp_path / "unbroken", tmp_path / "store") == []
