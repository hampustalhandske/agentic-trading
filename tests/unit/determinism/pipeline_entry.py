# pipeline_entry.py

"""Runs one pipeline of the determinism tests' firm (`_firm.py`) in a process of its own,
against the tests' `FakeLake`, and prints its result as JSON.

    python pipeline_entry.py firm <store_dir> [--kill-at-day N --kill-point before|after]
        [--lake-cache DIR] [--refuse-sql] [--calls-log FILE]
    python pipeline_entry.py stage_a <store_dir>

`firm` runs the firm through `run_firm_record`. With `--kill-at-day N` the process SIGKILLs
itself on the write of day N + 1, once N days are checkpointed: `before` the day's rows reach
the store, or `after` they do and before the engine checkpoints the day. With `--lake-cache`
the record runs inside `lakehouse_loader.lake_answer_cache` at that directory, and the fake
lake's version is known before the session opens, as the real lake's catalogue responses
carry it; the cache's counts are written to stderr as one `lake-answer-cache` JSON line.
`--refuse-sql` makes every SQL read of the fake lake raise, so every answer must come from the
cache. `--calls-log` appends each SQL read the fake lake answers to FILE as one JSON line
`[query, params]`, flushed at once so a killed pass keeps its lines. `stage_a` runs the
firm's analysts through the stage (a) `run_batch` as ETFs. `_firm.PLANT_ENV` plants a defect
in the firm."""

import argparse
import contextlib
import json
import os
import signal
import sys
from pathlib import Path
from typing import Any

_TESTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_TESTS))

from agentic_trading.data.loaders import lakehouse_loader  # noqa: E402
from agentic_trading.firm_record.runner import run_firm_record  # noqa: E402
from agentic_trading.stage_a.runner import run_batch  # noqa: E402
from agentic_trading.stage_a.store import FirmDecisionStore  # noqa: E402
from conftest import FakeLake  # noqa: E402
from unit.determinism import _firm  # noqa: E402


def _kill_on_write(day_number: int, point: str) -> None:
    """Makes the `day_number`-th `FirmDecisionStore.write_day` of this process SIGKILL it,
    `before` or `after` the write."""
    write_day = FirmDecisionStore.write_day
    calls = 0

    def killing_write_day(self: FirmDecisionStore, *args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == day_number and point == "before":
            os.kill(os.getpid(), signal.SIGKILL)
        result = write_day(self, *args, **kwargs)
        if calls == day_number and point == "after":
            os.kill(os.getpid(), signal.SIGKILL)
        return result

    FirmDecisionStore.write_day = killing_write_day  # type: ignore[method-assign]


def _log_calls(lake: FakeLake, path: Path) -> None:
    """Makes `lake` append each SQL read it answers to `path` as one JSON line."""
    sql = lake.sql

    def logged_sql(query: str, params: list[Any] | None = None) -> Any:
        frame = sql(query, params)
        with path.open("a") as log:
            log.write(json.dumps([query, list(params or [])], default=str) + "\n")
        return frame

    lake.sql = logged_sql  # type: ignore[method-assign]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pipeline", choices=("firm", "stage_a"))
    parser.add_argument("store_dir", type=Path)
    parser.add_argument("--kill-at-day", type=int)
    parser.add_argument("--kill-point", choices=("before", "after"), default="after")
    parser.add_argument("--lake-cache", type=Path)
    parser.add_argument("--refuse-sql", action="store_true")
    parser.add_argument("--calls-log", type=Path)
    args = parser.parse_args()
    lake = FakeLake()
    _firm.add_fred_vintages(lake)
    if args.calls_log is not None:
        _log_calls(lake, args.calls_log)
    if args.refuse_sql:
        lake.error = RuntimeError("the fake lake refuses every SQL read")
    lakehouse_loader._lake = lambda: lake  # type: ignore[assignment]
    provenance = {
        "lake_time": _firm.LAKE_TIME,
        "lake_version": "fake-lake-1",
        "quantcore_version": "0.3.1",
    }
    if args.pipeline == "stage_a":
        batch = run_batch(
            _firm.analysts(),
            _firm.UNIVERSE,
            "etf",
            _firm.WINDOW_START,
            _firm.WINDOW_END,
            args.store_dir,
            _firm.market_frame(),
            resources_ok=lambda _store_dir: True,
            **provenance,
        )
        print(json.dumps(batch.model_dump(mode="json")))
        return
    if args.kill_at_day is not None:
        _kill_on_write(args.kill_at_day + 1, args.kill_point)
    answers: Any = contextlib.nullcontext()
    if args.lake_cache is not None:
        lake.last_version = lake.version
        answers = lakehouse_loader.lake_answer_cache(args.lake_cache)
    with answers:
        result = run_firm_record(
            _firm.build_commander(),
            _firm.UNIVERSE,
            _firm.WINDOW_START,
            _firm.WINDOW_END,
            args.store_dir,
            _firm.engine_inputs(),
            **provenance,
        )
        counts = lakehouse_loader.lake_answer_cache_counts()
    if counts is not None:
        print("lake-answer-cache " + counts.model_dump_json(), file=sys.stderr)
    print(json.dumps(result.model_dump(mode="json")))


if __name__ == "__main__":
    main()
