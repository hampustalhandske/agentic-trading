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

from collections import Counter
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


def _unfilled_order_counts(block: dict[str, Any]) -> dict[str, Any]:
    """`unfilled_orders` (their count) and `unfilled_orders_by_reason` for a
    period or baseline block; both are 0 / `{}` when the block has none."""
    orders = block.get("unfilled_orders") or []
    return {
        "unfilled_orders": len(orders),
        "unfilled_orders_by_reason": dict(Counter(order["reason"] for order in orders)),
    }


def _capital_usage_means(block: dict[str, Any]) -> dict[str, float | None]:
    """`mean_idle_pod_share` (over the days that report one) and
    `mean_cash_share` for a period or baseline block; each is `None` when
    the block has no values to average."""
    usage = block.get("capital_usage") or []
    idle = [entry["idle_pod_share"] for entry in usage if entry["idle_pod_share"] is not None]
    cash = [entry["cash_share"] for entry in usage]
    return {
        "mean_idle_pod_share": sum(idle) / len(idle) if idle else None,
        "mean_cash_share": sum(cash) / len(cash) if cash else None,
    }


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
        summary[period]["pod_performance"] = block.get("pod_performance", {})
        summary[period].update(_unfilled_order_counts(block))
        summary[period].update(_capital_usage_means(block))

    summary["baselines"] = {
        name: {
            **{key: block[key] for key in PERIOD_METRIC_KEYS},
            **_unfilled_order_counts(block),
            **_capital_usage_means(block),
        }
        for name, block in experiment_result.get("baselines", {}).items()
    }
    return summary
