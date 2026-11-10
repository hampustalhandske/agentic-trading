# pipeline_entry.py

"""Runs one pipeline of the determinism tests' firm (`_firm.py`) in a process of its own,
against the tests' `FakeLake`, and prints its result as JSON.

    python pipeline_entry.py firm <store_dir> [--kill-at-day N --kill-point before|after]
    python pipeline_entry.py stage_a <store_dir>

`firm` runs the firm through `run_firm_record`. With `--kill-at-day N` the process SIGKILLs
itself on the write of day N + 1, once N days are checkpointed: `before` the day's rows reach
the store, or `after` they do and before the engine checkpoints the day. `stage_a` runs the
firm's analysts through the stage (a) `run_batch` as ETFs. `_firm.PLANT_ENV` plants a defect
in the firm."""

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pipeline", choices=("firm", "stage_a"))
    parser.add_argument("store_dir", type=Path)
    parser.add_argument("--kill-at-day", type=int)
    parser.add_argument("--kill-point", choices=("before", "after"), default="after")
    args = parser.parse_args()
    lake = FakeLake()
    _firm.add_fred_vintages(lake)
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
    result = run_firm_record(
        _firm.build_commander(),
        _firm.UNIVERSE,
        _firm.WINDOW_START,
        _firm.WINDOW_END,
        args.store_dir,
        _firm.engine_inputs(),
        **provenance,
    )
    print(json.dumps(result.model_dump(mode="json")))


if __name__ == "__main__":
    main()
