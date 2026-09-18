# build_performance_time_series.py

"""Builds a per-analyst and per-manager cumulative-attributed-return time
series from a run's resumable checkpoint (`track_record` is the full,
chronological `PerformanceRecord` log both backtest phases wrote into —
see `backtest/engine.py::_source_performance`, which computes the run's
final `source_performance` numbers from the exact same log via
`np.sum(returns)`, a plain running sum, not compounding).

The checkpoint file itself (hundreds of MB — one entry per asset per
source per simulated day) is too large to `json.load` comfortably, so this
streams it with `ijson` instead of loading it whole, accumulating only
what's needed: one (source_id -> {date -> daily_sum_across_assets}) dict,
which is small regardless of the input file's size.

Usage:
    uv run python scripts/build_performance_time_series.py \\
        results/2018_2026_v2/test_checkpoint.json \\
        --start 2018-04-02 \\
        -o results/2018_2026_v2/report/performance_time_series.json
"""

import argparse
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

import ijson

from agentic_trading.logging_config import configure_logging, get_logger

logger = get_logger(__name__)


def build_time_series(checkpoint_path: Path, start: date) -> dict:
    """Streams `checkpoint_path`'s `track_record` array, keeping only
    entries on/after `start` (so a checkpoint whose track record spans
    calibration+test can be restricted to just the test period, matching
    how `_source_performance` computes each phase's own numbers
    independently). Returns `{source_type: {source_id: [[date, cumulative_return], ...]}}`."""
    daily_sum: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    source_type_of: dict[str, str] = {}

    entry_count = 0
    with checkpoint_path.open("rb") as f:
        for entry in ijson.items(f, "track_record.item"):
            entry_count += 1
            if entry_count % 2_000_000 == 0:
                logger.info("Streamed %d track_record entries...", entry_count)
            as_of_date = entry["as_of_date"]
            if as_of_date < start.isoformat():
                continue
            source_id = entry["source_id"]
            source_type_of[source_id] = entry["source_type"]
            # ijson yields Decimal for JSON numbers by default; this project's
            # own numbers are all float, so normalize here rather than letting
            # Decimal leak into arithmetic against floats downstream.
            daily_sum[source_id][as_of_date] += float(entry["attributable_return"])

    logger.info("Streamed %d track_record entries total.", entry_count)

    result: dict[str, dict[str, list]] = {"analyst": {}, "manager": {}}
    for source_id, by_date in daily_sum.items():
        sorted_dates = sorted(by_date)
        cumulative = 0.0
        series = []
        for d in sorted_dates:
            cumulative += by_date[d]
            series.append([d, cumulative])
        result[source_type_of[source_id]][source_id] = series
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Path to a resumable checkpoint JSON file")
    parser.add_argument(
        "--start",
        type=date.fromisoformat,
        required=True,
        help="Only keep entries on/after this date",
    )
    parser.add_argument("-o", "--out", type=Path, required=True, help="Output JSON path")
    args = parser.parse_args()
    configure_logging()

    result = build_time_series(args.checkpoint, args.start)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(result, f)

    n_analysts = len(result["analyst"])
    n_managers = len(result["manager"])
    print(
        f"Wrote {args.out} ({args.out.stat().st_size:,} bytes): "
        f"{n_analysts} analysts, {n_managers} managers"
    )


if __name__ == "__main__":
    main()
