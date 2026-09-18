# run_summary.py

"""Shared shape for a completed run's compact summary: what
`scripts/extract_run_summary.py` writes and what `gui/dashboard.py`'s
"Load a saved run" panel reads. Kept as one module so the CLI extraction
tool and the GUI loader can't silently drift on what fields the summary
file is expected to carry.

A full `experiment_result.json` (or the `test`/`calibration` block of a
resumable checkpoint) is dominated by `decisions_by_day`/`trade_log` — one
entry per asset per simulated day, hundreds of MB for a multi-year,
multi-asset run. The summary strips both, keeping only what a human
actually reads afterwards: final metrics, the equity curve,
`signal_health`, `source_performance` (the attribution-score reading of
analyst/manager skill), and `pod_return_history` (each pod's own real,
weighted daily return — small, one number per pod per day, not one row
per asset per day like the fields this strips)."""

from typing import Any

PERIOD_METRIC_KEYS = (
    "start_date",
    "end_date",
    "portfolio_value_history",
    "final_sharpe_ratio",
    "final_max_drawdown",
    "final_sortino_ratio",
    "final_calmar_ratio",
    "final_information_ratio",
    "final_hit_rate",
    "final_profit_factor",
)


def extract_run_summary(experiment_result: dict[str, Any]) -> dict[str, Any]:
    """The full `experiment_result.json` payload in, the compact summary
    out. `experiment_result` must have `experiment`, `market_data_source`,
    `calibration`, and `test` top-level keys
    (as produced by `agentic-trading-experiment`); `baselines` is optional
    since not every run includes it."""
    summary: dict[str, Any] = {
        "experiment": experiment_result["experiment"],
        "market_data_source": experiment_result["market_data_source"],
    }
    for period in ("calibration", "test"):
        block = experiment_result[period]
        summary[period] = {key: block[key] for key in PERIOD_METRIC_KEYS}
        summary[period]["signal_health"] = block["signal_health"]
        summary[period]["source_performance"] = block["source_performance"]
        summary[period]["pod_return_history"] = block.get("pod_return_history", {})

    summary["baselines"] = {
        name: {key: block[key] for key in PERIOD_METRIC_KEYS}
        for name, block in experiment_result.get("baselines", {}).items()
    }
    return summary
