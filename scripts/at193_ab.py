# at193_ab.py

"""A/B check of the firm record's outputs between two commits.

Runs the determinism tests' firm (`tests/unit/determinism/_firm.py`) through
`tests/unit/determinism/pipeline_entry.py` against their `FakeLake`, once from
the `--base` commit and once from the working tree this script lives in, each in
fresh interpreters with `PYTHONHASHSEED=0` and single-threaded BLAS, and compares
what the two sides write. No lake or network call is made.

The base side is `git archive <base> src tests .gitignore` extracted into a scratch
directory and committed there as a one-commit repository, so the firm record
finds a clean `src/` tree at a known commit; its `src/` is put first on
`PYTHONPATH`. The head side runs from this checkout, which must be clean for
its submission to say so.

Scenarios, each run on both sides:

- `firm`: one unbroken firm record pass;
- `firm_killed_before` / `firm_killed_after`: a pass SIGKILLed on the write of
  the fifth day, before or after its rows reach the store, then resumed to the
  end in a fresh process;
- `stage_a`: the stage (a) batch of the firm's analysts;
- `probe`: the reads of the unbroken pass's engine checkpoint, through the
  side's own `checkpoint_store`: `read_all`, `read_recent` at every cutoff from
  the window's first day to its last and one or two held decision days, and
  `iter_source_returns` over several windows, dumped as JSON;
- `engine`: the BacktestResult JSON of the firm run through `BacktestEngine`
  without a history horizon, unbroken and stopped after six days then resumed.

Every file the head side writes under a scenario's store directory must equal
the base side's byte for byte, the engine checkpoint aside, except the fields
that name the code each side ran (`PROVENANCE_FIELDS`): the commit, and the
source hashes of the firm's modules with the spec hash they give. A trial
directory is named by its spec hash, so a firm store's only trial directory
is matched with the other side's whatever its name. A killed-and-resumed head pass
is compared with the base side's unbroken pass. Prints one line per
scenario, the differing files, the modules whose source hash differs, and
`OVERALL PASS|FAIL`; exits 1 on FAIL.

    uv run python scripts/at193_ab.py --base 583a50f [--keep DIR]
"""

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
# The JSON fields that name the code a side ran, by file name: its commit, and
# a firm spec's source hash per module with the spec hash (the trial's key)
# they give. The only fields that may differ between the base and the head side.
PROVENANCE_FIELDS: dict[str, tuple[tuple[str, ...], ...]] = {
    "submission.json": (("client", "git_sha"), ("trials", "test_keys")),
    "agentic_meta.json": (("git_sha",),),
    "spec.json": (("spec", "module_hashes"), ("spec_hash",)),
}
# A trial directory's name: its spec hash.
_TRIAL_DIR = re.compile(r"^[0-9a-f]{64}$")
_ENV = {"PYTHONHASHSEED": "0", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
_TIMEOUT = 600
_KILL_AT_DAY = 4

# Run inside a side's interpreter: reads the engine checkpoint at argv[1] with
# that side's checkpoint_store and prints the reads as JSON.
_PROBE = r"""
import base64, json, sys
from datetime import date, timedelta
from pathlib import Path
from agentic_trading.backtest import checkpoint_store

def plain(value):
    if isinstance(value, bytes):
        return base64.b64encode(value).decode()
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value

connection = checkpoint_store.connect(Path(sys.argv[1]), resume=True)
full = checkpoint_store.read_all(connection)
days = sorted(full["decisions_by_day"])
first, last = date.fromisoformat(days[0]), date.fromisoformat(days[-1])
out = {"read_all": plain(full), "read_recent": {}, "iter_source_returns": {}}
cutoff = first - timedelta(days=3)
while cutoff <= last + timedelta(days=1):
    for held in (1, 2):
        payload = checkpoint_store.read_recent(connection, cutoff, held)
        out["read_recent"][f"{cutoff.isoformat()}/{held}"] = plain(payload)
    cutoff += timedelta(days=1)
middle = first + (last - first) / 2
for start, end in ((first, last), (first, middle), (middle, last), (last, last),
                   (first - timedelta(days=30), first - timedelta(days=1))):
    out["iter_source_returns"][f"{start}/{end}"] = [
        list(row) for row in checkpoint_store.iter_source_returns(connection, start, end)
    ]
out["unfilled_orders"] = checkpoint_store.read_unfilled_orders(connection)
connection.close()
Path(sys.argv[2]).write_text(json.dumps(out, indent=1, sort_keys=True))
"""

# Run inside a side's interpreter: the firm through BacktestEngine without a
# horizon, unbroken and resumed after six days; writes both results' JSON.
_ENGINE = r"""
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from agentic_trading.backtest.engine import BacktestEngine
from agentic_trading.data.loaders import lakehouse_loader
from agentic_trading.scoring.track_record import TrackRecord
from conftest import FakeLake
from unit.determinism import _firm

lake = FakeLake()
_firm.add_fred_vintages(lake)
lakehouse_loader._lake = lambda: lake
out = Path(sys.argv[2])
inputs = _firm.engine_inputs()

class Stop(Exception):
    pass

def engine(path, resume):
    return BacktestEngine(
        commander=_firm.build_commander(), portfolio=inputs.portfolio(),
        track_record=TrackRecord(), data_loader=inputs.data_loader,
        start_date=_firm.WINDOW_START, end_date=_firm.WINDOW_END, assets=inputs.assets,
        market_data_source=inputs.market_data_source, execution_lag_days=1,
        checkpoint_path=path, resume=resume,
    )

def stop(index, *_):
    if index == 5:
        raise Stop

with lakehouse_loader.lake_session(_firm.LAKE_TIME):
    unbroken = engine(out / "unbroken.sqlite", False).run()
    try:
        engine(out / "resumed.sqlite", False).run(on_day_complete=stop)
    except Stop:
        pass
    resumed = engine(out / "resumed.sqlite", True).run()
(out / "unbroken.json").write_text(unbroken.model_dump_json(indent=1))
(out / "resumed.json").write_text(
    resumed.model_dump_json(indent=1, exclude={"unfactored_split_moves"})
)
(out / "unbroken_comparable.json").write_text(
    unbroken.model_dump_json(indent=1, exclude={"unfactored_split_moves"})
)
"""


def _export_base(base: str, root: Path) -> Path:
    """`base`'s src/, tests/ and .gitignore extracted under `root / "base"`, committed there as a
    one-commit repository; returns that directory."""
    target = root / "base"
    target.mkdir()
    archive = subprocess.run(
        ["git", "archive", base, "src", "tests", ".gitignore"],
        cwd=REPO,
        capture_output=True,
        check=True,
    )
    subprocess.run(["tar", "-x", "-C", str(target)], input=archive.stdout, check=True)
    git = ["git", "-c", "user.name=ab", "-c", "user.email=ab@example.invalid"]
    subprocess.run([*git, "init", "-q"], cwd=target, check=True)
    subprocess.run([*git, "add", "-A"], cwd=target, check=True)
    subprocess.run([*git, "commit", "-q", "-m", f"base {base}"], cwd=target, check=True)
    return target


def _run(side: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    """`args` run by this interpreter with `side`'s src/ first on the path."""
    env = {key: value for key, value in os.environ.items() if not key.startswith("AGENTIC_")}
    env.update(_ENV)
    env["PYTHONPATH"] = str(side / "src")
    completed = subprocess.run(
        [sys.executable, *args],
        cwd=side,
        env=env,
        capture_output=True,
        text=True,
        timeout=_TIMEOUT,
        check=False,
    )
    if check and completed.returncode != 0:
        raise RuntimeError(f"{args} failed in {side}:\n{completed.stderr[-4000:]}")
    return completed


def _scenarios(side: Path, out: Path) -> None:
    """Every scenario of `side`, written under `out / <scenario>`."""
    entry = str(side / "tests" / "unit" / "determinism" / "pipeline_entry.py")
    _run(side, entry, "firm", str(out / "firm"))
    for point in ("before", "after"):
        store = out / f"firm_killed_{point}"
        killed = _run(
            side,
            entry,
            "firm",
            str(store),
            "--kill-at-day",
            str(_KILL_AT_DAY),
            "--kill-point",
            point,
            check=False,
        )
        if killed.returncode != -signal.SIGKILL:
            raise RuntimeError(f"the {point} kill did not happen: {killed.stderr[-2000:]}")
        _run(side, entry, "firm", str(store))
    _run(side, entry, "stage_a", str(out / "stage_a"))
    (checkpoint,) = (out / "firm").glob("*.checkpoint.sqlite")
    probe = out / "probe"
    probe.mkdir()
    script = out / "_probe.py"
    script.write_text(_PROBE)
    _run(side, str(script), str(checkpoint), str(probe / "reads.json"))
    engine = out / "engine"
    engine.mkdir()
    script = out / "_engine.py"
    script.write_text(_ENGINE)
    _run(side, str(script), str(side / "tests"), str(engine))
    for path in engine.glob("*.sqlite"):
        path.unlink()


def _comparable(path: Path) -> bytes:
    """`path`'s bytes, with the `PROVENANCE_FIELDS` of its file name it holds removed."""
    content = path.read_bytes()
    fields = PROVENANCE_FIELDS.get(path.name)
    if not fields:
        return content
    document = json.loads(content)
    for field in fields:
        holder: Any = document
        for key in field[:-1]:
            holder = holder.get(key, {}) if isinstance(holder, dict) else {}
        if isinstance(holder, dict):
            holder.pop(field[-1], None)
    return json.dumps(document, sort_keys=True).encode()


def _files(store: Path) -> dict[str, Path]:
    """Each file under `store` but an engine checkpoint, by its path relative to `store`; the
    name of a store's only trial directory is written `<trial>`."""
    trials = [
        path.name for path in store.iterdir() if path.is_dir() and _TRIAL_DIR.match(path.name)
    ]
    only = trials[0] if len(trials) == 1 else None
    files = {}
    for path in store.rglob("*"):
        if not path.is_file() or path.name.endswith(".checkpoint.sqlite"):
            continue
        parts = path.relative_to(store).parts
        named = [
            "<trial>" if index == 0 and part == only else part for index, part in enumerate(parts)
        ]
        files["/".join(named)] = path
    return files


def differences(expected: Path, actual: Path) -> list[str]:
    """Each file only one of two directories holds, then each whose comparable bytes differ."""
    expected_files, actual_files = _files(expected), _files(actual)
    found = sorted(expected_files.keys() ^ actual_files.keys())
    for relative in sorted(expected_files.keys() & actual_files.keys()):
        if _comparable(expected_files[relative]) != _comparable(actual_files[relative]):
            found.append(relative)
    return found


def changed_modules(expected: Path, actual: Path) -> list[str]:
    """The modules whose source hash differs between the two stores' firm spec.json."""
    hashes = [
        json.loads(next(store.glob("*/spec.json")).read_text())["spec"]["module_hashes"]
        for store in (expected, actual)
    ]
    return sorted(
        name
        for name in hashes[0].keys() | hashes[1].keys()
        if hashes[0].get(name) != hashes[1].get(name)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", required=True, help="the commit the head side is compared to")
    parser.add_argument("--keep", type=Path, help="keep both sides' outputs under this directory")
    args = parser.parse_args()
    root = Path(tempfile.mkdtemp(prefix="at193_ab_"))
    try:
        base = _export_base(args.base, root)
        outputs = {"base": root / "out_base", "head": root / "out_head"}
        for name, side in (("base", base), ("head", REPO)):
            outputs[name].mkdir()
            _scenarios(side, outputs[name])
        checks = {
            "firm": ("firm", "firm"),
            "firm_killed_before": ("firm", "firm_killed_before"),
            "firm_killed_after": ("firm", "firm_killed_after"),
            "stage_a": ("stage_a", "stage_a"),
            "probe": ("probe", "probe"),
            "engine": ("engine", "engine"),
        }
        failed = False
        for scenario, (expected, actual) in checks.items():
            found = differences(outputs["base"] / expected, outputs["head"] / actual)
            files = len(_files(outputs["head"] / actual))
            print(f"{scenario}: {files} file(s), {'PASS' if not found else 'FAIL'}")
            for relative in found:
                print(f"  differs: {relative}")
            failed = failed or bool(found)
        modules = changed_modules(outputs["base"] / "firm", outputs["head"] / "firm")
        print(f"firm modules whose source hash differs: {', '.join(modules) or 'none'}")
        engine = outputs["head"] / "engine"
        resumed_equal = (engine / "resumed.json").read_bytes() == (
            engine / "unbroken_comparable.json"
        ).read_bytes()
        print(f"engine resume equals unbroken (head): {'PASS' if resumed_equal else 'FAIL'}")
        failed = failed or not resumed_equal
        print(f"OVERALL {'FAIL' if failed else 'PASS'}")
        if args.keep is not None:
            shutil.copytree(root, args.keep, dirs_exist_ok=True)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
