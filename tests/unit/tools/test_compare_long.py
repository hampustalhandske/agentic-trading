# test_compare_long.py

"""`tools/ab/compare_long.py` compares two firm-record run directories and
`tools/ab/compare_and_record.sh` records that comparison while exiting with its status.

The differences the comparison allows: code identity in the trial JSON files, the new run's
`firm.history_horizon_days` in spec.json's settings snapshot and `history_horizon_days` in its
BacktestResult (each only where the old run has no such field), a `decisions_by_day` holding
only the old run's last one or two days, and, with `split_moves_apart`, `unfactored_split_moves`.
Anything else, including a checkpoint table only one side has, makes it DIFFERENT, and the
recording script then exits non-zero."""

import importlib.util
import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from unit.tools.support import TOOLS


def _load_compare_long() -> ModuleType:
    spec = importlib.util.spec_from_file_location("compare_long", TOOLS / "ab" / "compare_long.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compare_long = _load_compare_long()
_RECORD = TOOLS / "ab" / "compare_and_record.sh"
OLD, NEW = Path("store/aaa"), Path("store/bbb")


def spec(firm: dict, **extra: object) -> dict:
    return {"spec": {"settings": {"firm": firm, "portfolio": {"ic_lookback_days": 250}}}, **extra}


def test_an_added_horizon_in_the_spec_settings_is_allowed() -> None:
    old = spec({"low_availability_threshold": 0.5}, spec_hash="aaa")
    new = spec({"low_availability_threshold": 0.5, "history_horizon_days": 300}, spec_hash="bbb")
    assert compare_long.json_same("trial/spec.json", old, new, OLD, NEW)


def test_a_horizon_in_both_specs_must_match() -> None:
    old = spec({"history_horizon_days": 250})
    new = spec({"history_horizon_days": 300})
    assert not compare_long.json_same("trial/spec.json", old, new, OLD, NEW)


def test_any_other_added_setting_fails() -> None:
    old = spec({"low_availability_threshold": 0.5})
    new = spec({"low_availability_threshold": 0.5, "history_horizon_days": 300, "other": 1})
    assert not compare_long.json_same("trial/spec.json", old, new, OLD, NEW)


def test_the_horizon_outside_spec_json_fails() -> None:
    old = spec({"low_availability_threshold": 0.5})
    new = spec({"low_availability_threshold": 0.5, "history_horizon_days": 300})
    assert not compare_long.json_same("trial/agentic_meta.json", old, new, OLD, NEW)


def test_the_horizon_elsewhere_in_the_spec_fails() -> None:
    old = {"spec": {"settings": {"firm": {}}}, "extra": {}}
    new = {"spec": {"settings": {"firm": {}}}, "extra": {"history_horizon_days": 300}}
    assert not compare_long.json_same("trial/spec.json", old, new, OLD, NEW)


def test_code_identity_is_ignored_in_trial_json() -> None:
    old = {"git_sha": "a" * 40, "trials": {"test_keys": ["aaa", "zzz"]}, "value": 1}
    new = {"git_sha": "b" * 40, "trials": {"test_keys": ["bbb", "zzz"]}, "value": 1}
    assert compare_long.json_same("trial/submission.json", old, new, OLD, NEW)
    assert not compare_long.json_same("trial/submission.json", old, {**new, "value": 2}, OLD, NEW)


def test_an_added_horizon_in_the_result_is_dropped() -> None:
    old = {"final_hit_rate": 0.5}
    assert (
        compare_long.result_without_added_horizon(old, {**old, "history_horizon_days": 300}) == old
    )


def test_a_horizon_in_both_results_is_kept() -> None:
    old = {"final_hit_rate": 0.5, "history_horizon_days": None}
    new = {"final_hit_rate": 0.5, "history_horizon_days": 300}
    assert compare_long.result_without_added_horizon(old, new) != old


OLD_DAYS = {"2017-12-27": [1], "2017-12-28": [2], "2017-12-29": [3]}


def test_the_last_two_held_days_match() -> None:
    assert compare_long.held_days_match(OLD_DAYS, {"2017-12-28": [2], "2017-12-29": [3]})


def test_no_held_day_fails() -> None:
    assert not compare_long.held_days_match(OLD_DAYS, {})


def test_held_days_without_the_last_day_fail() -> None:
    assert not compare_long.held_days_match(OLD_DAYS, {"2017-12-28": [2]})


def test_a_held_day_that_differs_fails() -> None:
    assert not compare_long.held_days_match(OLD_DAYS, {"2017-12-28": [9], "2017-12-29": [3]})


def test_more_than_two_held_days_fail() -> None:
    assert not compare_long.held_days_match(OLD_DAYS, dict(OLD_DAYS))


DAY = {"2017-12-29": [3]}
MOVE = {"security_id": 1, "as_of_date": "2017-11-01", "ratio": 0.66}


def _write_run(run: Path, result: dict, trial_bytes: bytes = b"rows", table: str = "cash") -> Path:
    """Writes a run directory: `result` as backtest_result.json, one trial file with
    `trial_bytes`, and a checkpoint holding one row in table `table`."""
    trial = run / "store" / "trial"
    trial.mkdir(parents=True)
    (trial / "decisions.parquet").write_bytes(trial_bytes)
    (run / "backtest_result.json").write_text(json.dumps(result))
    with sqlite3.connect(run / "store" / "trial.checkpoint.sqlite") as connection:
        connection.execute(f"CREATE TABLE {table} (day TEXT, value REAL)")
        connection.execute(f"INSERT INTO {table} VALUES ('2017-12-29', 1.0)")
    connection.close()
    return run


def _write_results(tmp_path: Path, old: dict, new: dict) -> tuple[Path, Path]:
    return _write_run(tmp_path / "old", old), _write_run(tmp_path / "new", new)


def test_split_moves_apart_leaves_only_that_field_out(tmp_path: Path) -> None:
    old, new = _write_results(
        tmp_path,
        {"decisions_by_day": DAY, "unfactored_split_moves": [MOVE], "final_hit_rate": 0.5},
        {"decisions_by_day": DAY, "unfactored_split_moves": None, "final_hit_rate": 0.5},
    )
    assert compare_long.compare_result(old, new, split_moves_apart=True)
    assert not compare_long.compare_result(old, new)


def test_split_moves_apart_still_fails_any_other_field(tmp_path: Path) -> None:
    old, new = _write_results(
        tmp_path,
        {"decisions_by_day": DAY, "unfactored_split_moves": [MOVE], "final_hit_rate": 0.5},
        {"decisions_by_day": DAY, "unfactored_split_moves": [], "final_hit_rate": 0.6},
    )
    assert not compare_long.compare_result(old, new, split_moves_apart=True)


def test_a_checkpoint_table_only_one_side_has_fails(tmp_path: Path) -> None:
    result = {"decisions_by_day": DAY}
    old = _write_run(tmp_path / "old", result)
    new = _write_run(tmp_path / "new", result)
    with sqlite3.connect(new / "store" / "trial.checkpoint.sqlite") as connection:
        connection.execute("CREATE TABLE extra (x INTEGER)")
    connection.close()
    assert not compare_long.compare_checkpoints(old, new)
    assert not compare_long.compare_checkpoints(new, old)


def _record(old: Path, new: Path, report: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(_RECORD), str(old), str(new), str(report)],
        env={"PATH": "/usr/bin:/bin", "PYTHON": sys.executable},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


@pytest.mark.parametrize(
    ("new_result", "new_trial"),
    [
        ({"decisions_by_day": DAY, "final_hit_rate": 0.6}, b"rows"),
        ({"decisions_by_day": DAY}, b"x"),
    ],
)
def test_recording_a_difference_exits_non_zero(
    tmp_path: Path, new_result: dict, new_trial: bytes
) -> None:
    """A planted difference in the result or a trial file is recorded as OVERALL DIFFERENT and
    the recording script exits non-zero."""
    old = _write_run(tmp_path / "old", {"decisions_by_day": DAY})
    new = _write_run(tmp_path / "new", new_result, trial_bytes=new_trial)
    report = tmp_path / "compare.txt"
    done = _record(old, new, report)
    assert done.returncode != 0
    assert "OVERALL DIFFERENT" in report.read_text()


def test_recording_an_identical_pair_exits_zero(tmp_path: Path) -> None:
    """Two identical runs are recorded as OVERALL IDENTICAL and the script exits 0."""
    old = _write_run(tmp_path / "old", {"decisions_by_day": DAY})
    new = _write_run(tmp_path / "new", {"decisions_by_day": DAY})
    report = tmp_path / "compare.txt"
    done = _record(old, new, report)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "OVERALL IDENTICAL" in report.read_text()
    assert report.read_text() == done.stdout
