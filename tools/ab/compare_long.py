"""Compares two firm-record run directories of an A/B run: an old one and a new one that may keep
a history horizon in memory and may have been killed and resumed. A run directory holds
`backtest_result.json` (when the run returned a BacktestResult), `store/` with one trial
directory and one `*.checkpoint.sqlite`, and optionally `sessions.jsonl` and `starts.jsonl`
(RSS per session and per process start).

- The trial files (decisions.parquet and the rest) are compared byte for byte by relative path.
  A JSON file that differs is compared again without its code identity (`CODE_KEYS` at any
  depth, and in submission.json each `trials.test_keys` entry equal to the run's own spec hash).
  The one other difference allowed is the new run's `firm.history_horizon_days` in spec.json's
  settings snapshot (`spec.settings.firm`) when the old run's snapshot has no such field.
- backtest_result.json is compared without `decisions_by_day`, which the new run holds for its
  last days only: it must hold 1 or 2 days, the old run's last among them, each equal to the
  old run's. The new result's `history_horizon_days` is allowed only when the old result has no
  such field. Two runs that both returned no BacktestResult (paused passes) are reported and
  compared as nothing.
- With `--split-moves-apart`, `unfactored_split_moves` is taken out of both results before that
  comparison and reported on its own line as SAME or KNOWN (not failed): a resumed process
  returns only the split-sized moves its own reads flagged.
- The checkpoints must hold the same tables, compared table by table with rows in rowid order. A
  `carried_state` row is compared by agent id, and its pickled state by bytes, reported
  separately. The cursor's fingerprints of code are reported, not failed.

It then prints RSS per 50 sessions from each side's sessions.jsonl, when the run logged one.
Prints OVERALL IDENTICAL or OVERALL DIFFERENT and exits 0 when identical, else 1.

Usage: python tools/ab/compare_long.py <old run dir> <new run dir> [--split-moves-apart]
(tools/ab/compare_and_record.sh also writes the report to a file and keeps the exit status.)
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

CODE_KEYS = {"module_hashes", "spec_hash", "git_sha", "src_tree_clean"}
CURSOR_CODE_COLUMNS = {"settings_fingerprint", "roster_fingerprint"}
HORIZON_FIELD = "history_horizon_days"
SPLIT_MOVES_FIELD = "unfactored_split_moves"


def trial_dir(run: Path) -> Path:
    """The run's one trial directory under store/."""
    dirs = [path for path in (run / "store").iterdir() if path.is_dir()]
    if len(dirs) != 1:
        raise SystemExit(f"{run}: expected one trial dir, found {len(dirs)}")
    return dirs[0]


def without_code(value: object) -> object:
    """`value` with every key in CODE_KEYS removed at any depth."""
    if isinstance(value, dict):
        return {key: without_code(item) for key, item in value.items() if key not in CODE_KEYS}
    if isinstance(value, list):
        return [without_code(item) for item in value]
    return value


def code_neutral(label: str, value: object, spec_hash: str) -> object:
    """`value`, the parsed JSON of the file `label`, without its code identity: the CODE_KEYS
    at any depth and, for trial/submission.json, each `trials.test_keys` entry equal to
    `spec_hash`, the run's own spec hash, replaced by a placeholder."""
    if label == "trial/submission.json" and isinstance(value, dict):
        trials = value.get("trials")
        if isinstance(trials, dict) and isinstance(trials.get("test_keys"), list):
            keys = ["<own spec hash>" if key == spec_hash else key for key in trials["test_keys"]]
            value = {**value, "trials": {**trials, "test_keys": keys}}
    return without_code(value)


def without_added_horizon(label: str, new_value: object, old_value: object) -> object:
    """`new_value`, the parsed new trial/spec.json, without
    `spec.settings.firm.history_horizon_days` when the old spec's `spec.settings.firm` has no
    such field; any other file or value is returned unchanged."""
    if label != "trial/spec.json":
        return new_value
    try:
        new_firm = new_value["spec"]["settings"]["firm"]  # type: ignore[index]
        old_firm = old_value["spec"]["settings"]["firm"]  # type: ignore[index]
    except (KeyError, TypeError):
        return new_value
    if HORIZON_FIELD not in new_firm or HORIZON_FIELD in old_firm:
        return new_value
    firm = {key: item for key, item in new_firm.items() if key != HORIZON_FIELD}
    spec = new_value["spec"]  # type: ignore[index]
    return {**new_value, "spec": {**spec, "settings": {**spec["settings"], "firm": firm}}}


def checkpoint(run: Path) -> Path:
    """The run's one checkpoint file under store/."""
    files = list((run / "store").glob("*.checkpoint.sqlite"))
    if len(files) != 1:
        raise SystemExit(f"{run}: expected one checkpoint, found {len(files)}")
    return files[0]


def json_same(
    label: str, old_value: object, new_value: object, old_trial: Path, new_trial: Path
) -> bool:
    """Whether two parsed trial JSON files match apart from code identity and the added
    horizon field."""
    new_value = without_added_horizon(label, new_value, old_value)
    return code_neutral(label, old_value, old_trial.name) == code_neutral(
        label, new_value, new_trial.name
    )


def compare_trial(old: Path, new: Path) -> bool:
    """Prints and returns whether the two runs' trial files match apart from code identity."""
    old_trial, new_trial = trial_dir(old), trial_dir(new)
    names = {p.relative_to(old_trial).as_posix() for p in old_trial.rglob("*") if p.is_file()}
    names |= {p.relative_to(new_trial).as_posix() for p in new_trial.rglob("*") if p.is_file()}
    same = True
    for name in sorted(names):
        label, left, right = f"trial/{name}", old_trial / name, new_trial / name
        if not (left.exists() and right.exists()):
            print(f"MISSING {label}")
            same = False
        elif left.read_bytes() == right.read_bytes():
            print(f"SAME {label}")
        elif name.endswith(".json") and json_same(
            label, json.loads(left.read_text()), json.loads(right.read_text()), old_trial, new_trial
        ):
            print(f"SAME-BUT-CODE-IDENTITY {label}")
        else:
            print(f"DIFF {label}")
            same = False
    return same


def result_without_added_horizon(old_value: dict, new_value: dict) -> dict:
    """`new_value`, a parsed BacktestResult, without `history_horizon_days` when `old_value` has
    no such field; otherwise unchanged."""
    if HORIZON_FIELD in old_value or HORIZON_FIELD not in new_value:
        return new_value
    return {key: item for key, item in new_value.items() if key != HORIZON_FIELD}


def held_days_match(old_days: dict, new_days: dict) -> bool:
    """Whether `new_days`, the decisions_by_day a horizon run returns, holds 1 or 2 days, among
    them the old run's last day, each equal to the old run's."""
    return (
        1 <= len(new_days) <= 2
        and bool(old_days)
        and max(old_days) in new_days
        and all(old_days.get(day) == value for day, value in new_days.items())
    )


def compare_result(old: Path, new: Path, split_moves_apart: bool = False) -> bool:
    """Prints and returns whether the BacktestResults match apart from `decisions_by_day`, and
    whether every day the new result holds matches the old one's. With `split_moves_apart`,
    `unfactored_split_moves` is reported on its own line and left out of the verdict."""
    old_path, new_path = old / "backtest_result.json", new / "backtest_result.json"
    if not old_path.exists() and not new_path.exists():
        print("NONE backtest_result.json: neither run returned a BacktestResult (paused passes)")
        return True
    if not (old_path.exists() and new_path.exists()):
        print("MISSING backtest_result.json on one side")
        return False
    left, right = json.loads(old_path.read_text()), json.loads(new_path.read_text())
    old_days, new_days = left.pop("decisions_by_day"), right.pop("decisions_by_day")
    if split_moves_apart:
        old_moves = left.pop(SPLIT_MOVES_FIELD, None)
        new_moves = right.pop(SPLIT_MOVES_FIELD, None)
        label = "SAME" if old_moves == new_moves else "KNOWN"
        print(
            f"{label} {SPLIT_MOVES_FIELD} (not in the verdict): old {len(old_moves or [])}, "
            f"new {len(new_moves or [])} entries"
        )
    same = left == result_without_added_horizon(left, right)
    print(f"{'SAME' if same else 'DIFF'} backtest_result.json without decisions_by_day")
    held = held_days_match(old_days, new_days)
    print(
        f"{'SAME' if held else 'DIFF'} decisions_by_day: new holds {sorted(new_days)}, "
        f"old holds {len(old_days)} days"
    )
    return same and held


def _tables(connection: sqlite3.Connection) -> set[str]:
    rows = connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
    return {row[0] for row in rows}


def compare_checkpoints(old: Path, new: Path) -> bool:
    """Prints and returns whether the two checkpoints hold the same tables with the same rows."""
    left, right = sqlite3.connect(checkpoint(old)), sqlite3.connect(checkpoint(new))
    try:
        left_tables, right_tables = _tables(left), _tables(right)
        same = left_tables == right_tables
        for table in sorted(left_tables ^ right_tables):
            side = "old" if table in left_tables else "new"
            print(f"DIFF checkpoint {table}: only in the {side} run")
        for table in sorted(left_tables & right_tables):
            same &= _compare_table(left, right, table)
    finally:
        left.close()
        right.close()
    return same


def _compare_table(left: sqlite3.Connection, right: sqlite3.Connection, table: str) -> bool:
    """Prints and returns whether checkpoint table `table` holds the same rows on both sides."""
    if table == "carried_state":
        query = "SELECT agent_id, state FROM carried_state ORDER BY agent_id"
        a, b = left.execute(query).fetchall(), right.execute(query).fetchall()
        ids_same = [row[0] for row in a] == [row[0] for row in b]
        bytes_same = sum(x[1] == y[1] for x, y in zip(a, b, strict=False))
        print(
            f"{'SAME' if ids_same else 'DIFF'} checkpoint carried_state agent ids "
            f"({len(a)}); pickled state byte-equal for {bytes_same}/{len(a)}"
        )
        return ids_same
    columns = [row[1] for row in left.execute(f"PRAGMA table_info({table})")]
    kept = [c for c in columns if table != "cursor" or c not in CURSOR_CODE_COLUMNS]
    query = f"SELECT {', '.join(kept)} FROM {table} ORDER BY rowid"
    a, b = left.execute(query).fetchall(), right.execute(query).fetchall()
    ok = a == b
    print(f"{'SAME' if ok else 'DIFF'} checkpoint {table} ({len(a)} vs {len(b)} rows)")
    if table == "cursor":
        code = ", ".join(sorted(CURSOR_CODE_COLUMNS))
        query = f"SELECT {code} FROM cursor"
        equal = left.execute(query).fetchall() == right.execute(query).fetchall()
        print(f"INFO cursor {code} {'equal' if equal else 'differ (code identity)'}")
    return ok


def process_key(line: dict) -> tuple:
    """The process image a sessions.jsonl line came from: its pid and, when logged, when the
    image started (a restart for memory keeps the pid)."""
    return line["pid"], line.get("started")


def rss_table(run: Path, label: str) -> None:
    """Prints the run's RSS at every 50th session and at its last, then per process image its
    sessions, its RSS at start (starts.jsonl), at its first and last session, and its peak."""
    path = run / "sessions.jsonl"
    if not path.exists():
        print(f"{label}: no sessions.jsonl (run without RSS logging)")
        return
    lines = [json.loads(text) for text in path.read_text().splitlines()]
    print(f"{label}: {len(lines)} sessions logged")
    for index, line in enumerate(lines, start=1):
        if index % 50 == 0 or index == len(lines):
            print(
                f"  session {index:4} {line['day']}  rss {line['rss_mb']:7.1f} MB  "
                f"peak {line['peak_rss_mb']:7.1f} MB  pid {line['pid']}"
            )
    starts_path = run / "starts.jsonl"
    starts = (
        {}
        if not starts_path.exists()
        else {
            entry["started"]: entry["rss_mb"]
            for entry in map(json.loads, starts_path.read_text().splitlines())
        }
    )
    for process in dict.fromkeys(process_key(line) for line in lines):
        own = [line for line in lines if process_key(line) == process]
        pid, started = process
        at_start = f"rss at start {starts[started]} MB, " if started in starts else ""
        print(
            f"  pid {pid} started {started}: sessions {own[0]['day']}..{own[-1]['day']} "
            f"({len(own)}), {at_start}first-session rss {own[0]['rss_mb']} MB, last "
            f"{own[-1]['rss_mb']} MB, peak rss {max(line['rss_mb'] for line in own)} MB"
        )


def main(old: Path, new: Path, split_moves_apart: bool = False) -> bool:
    """Prints the whole comparison of `old` and `new`; returns whether they are identical."""
    identical = compare_trial(old, new)
    identical &= compare_result(old, new, split_moves_apart)
    identical &= compare_checkpoints(old, new)
    rss_table(old, "old")
    rss_table(new, "new")
    print(f"OVERALL {'IDENTICAL' if identical else 'DIFFERENT'}")
    return identical


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("old", type=Path)
    parser.add_argument("new", type=Path)
    parser.add_argument("--split-moves-apart", action="store_true")
    arguments = parser.parse_args()
    sys.exit(0 if main(arguments.old, arguments.new, arguments.split_moves_apart) else 1)
