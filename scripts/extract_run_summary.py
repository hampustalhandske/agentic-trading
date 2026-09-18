# extract_run_summary.py

"""CLI wrapper around `agentic_trading.backtest.run_summary.extract_run_summary`:
strips a completed `agentic-trading-experiment` run's `experiment_result.json`
(hundreds of MB — dominated by `decisions_by_day`/`trade_log`, one entry per
asset per simulated day) down to the metrics, `signal_health`, and
`source_performance`/`portfolio_value_history` needed to browse the run —
small enough (order 100s of KB) to load instantly in a Streamlit session or
commit alongside a report, unlike the source file. The GUI's "Load a saved
run" sidebar panel (`gui/dashboard.py`) reads exactly this output format.

Usage: `uv run python scripts/extract_run_summary.py results/<run>/experiment_result.json`
(writes `summary.json` next to it, or pass `-o` for a different path).
"""

import argparse
import json
from pathlib import Path

from agentic_trading.backtest.run_summary import extract_run_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_result", type=Path, help="Path to experiment_result.json")
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="Output path (default: summary.json next to the input file)",
    )
    args = parser.parse_args()

    out_path = args.out or args.experiment_result.with_name("summary.json")
    with args.experiment_result.open() as f:
        experiment_result = json.load(f)

    summary = extract_run_summary(experiment_result)

    with out_path.open("w") as f:
        json.dump(summary, f)

    print(f"Wrote {out_path} ({out_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
