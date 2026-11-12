# test_lake_read_error_record.py

"""A failed lake read stops a firm record: with the FRED analyst's lake read failing from the
window's `_firm.LAKE_FAILS_AFTER_DAYS + 1`-th session on (`_firm.LAKE_FAULT_ENV`), the pass
exits non-zero naming the `LakehouseError` instead of recording error signals and completing,
with the days before it written and checkpointed: resumed without the fault, the record writes
only the remaining days and ends with the unbroken pass's trial folder.

Each pass runs the firm of `_firm.py` through `pipeline_entry.py` in a subprocess, as
`test_pipeline_determinism.py` does."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from agentic_trading.data.trading_calendar import sessions_between
from unit.determinism import _firm
from unit.determinism.test_pipeline_determinism import (
    _BLAS_THREAD_ENV,
    _ENTRY,
    _PASS_TIMEOUT,
    _REPO_ROOT,
    _completed_pass,
    store_differences,
)

pytestmark = pytest.mark.determinism


def _pass(store_dir: Path, *, lake_fault: bool) -> subprocess.CompletedProcess[str]:
    """One firm pass under `store_dir` in a fresh interpreter, the lake failing per
    `_firm.LAKE_FAULT_ENV` when `lake_fault`."""
    env = {key: value for key, value in os.environ.items() if key != _firm.LAKE_FAULT_ENV}
    env["PYTHONHASHSEED"] = "0"
    env.update(_BLAS_THREAD_ENV)
    if lake_fault:
        env[_firm.LAKE_FAULT_ENV] = "1"
    return subprocess.run(
        [sys.executable, str(_ENTRY), "firm", str(store_dir)],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=_PASS_TIMEOUT,
        check=False,
    )


def test_a_lake_read_error_stops_the_record_at_the_day_before(tmp_path: Path) -> None:
    failed = _pass(tmp_path / "store", lake_fault=True)
    assert failed.returncode != 0, failed.stdout[-2000:]
    assert "LakehouseError: HTTP 400: the query failed on the lake" in failed.stderr

    resumed = _pass(tmp_path / "store", lake_fault=False)
    assert resumed.returncode == 0, resumed.stderr[-4000:]
    result = json.loads(resumed.stdout.strip().splitlines()[-1])
    sessions = sessions_between(_firm.WINDOW_START, _firm.WINDOW_END)
    assert result["complete"]
    assert result["days_written"] == len(sessions) - _firm.LAKE_FAILS_AFTER_DAYS

    unbroken = _completed_pass(tmp_path / "unbroken")
    assert store_differences(unbroken, tmp_path / "store") == []
