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
per asset per day like the fields this strips). `pod_performance` and
`pod_costs` (each pod's fill costs, option premium and turnover) pass
through as they are, and `execution_costs` gives the run's fill-cost
totals by kind. Per-pod sizing diagnostics are reduced to each pod's
binding-stage shares and mean final-to-target vol ratio, and unfactored
split-sized moves to a count per security (`None` for a resumed run,
which carries no move list); the per-day and per-move entries are not
kept. `lake` carries the lake time the run read at."""

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


# Summary label -> `BacktestResult` field, for the `execution_costs` block.
# `total_transaction_cost` is commission only; `all_in` is commission plus
# slippage, impact and option spread.
EXECUTION_COST_KEYS = {
    "commission": "total_transaction_cost",
    "slippage": "total_slippage_cost",
    "impact": "total_impact_cost",
    "spread": "total_spread_cost",
    "all_in": "total_execution_cost",
}


def _execution_costs(block: dict[str, Any]) -> dict[str, float | None]:
    """A period block's fill-cost totals keyed by the labels of
    `EXECUTION_COST_KEYS`; a total the block does not carry is `None`."""
    return {label: block.get(field) for label, field in EXECUTION_COST_KEYS.items()}


def _unfilled_order_counts(block: dict[str, Any]) -> dict[str, Any]:
    """`unfilled_orders` (their count) and `unfilled_orders_by_reason` for a
    period or baseline block; both are 0 / `{}` when the block has none."""
    orders = block.get("unfilled_orders") or []
    return {
        "unfilled_orders": len(orders),
        "unfilled_orders_by_reason": dict(Counter(order["reason"] for order in orders)),
    }


def _capital_usage_means(block: dict[str, Any]) -> dict[str, Any]:
    """`mean_idle_pod_share`, `mean_unavailable_pod_share` and
    `mean_no_position_pod_share` (each over the days that report one) and
    `mean_cash_share` for a period or baseline block; each is `None` when
    the block has no values to average. `pod_state_days` maps each pod to
    the number of days it spent in each state (`{}` when none reported)."""
    usage = block.get("capital_usage") or []

    def mean_of(key: str) -> float | None:
        values = [entry[key] for entry in usage if entry.get(key) is not None]
        return sum(values) / len(values) if values else None

    state_days: dict[str, Counter[str]] = {}
    for entry in usage:
        for pod_id, state in (entry.get("pod_states") or {}).items():
            state_days.setdefault(pod_id, Counter())[state] += 1
    return {
        "mean_idle_pod_share": mean_of("idle_pod_share"),
        "mean_unavailable_pod_share": mean_of("unavailable_pod_share"),
        "mean_no_position_pod_share": mean_of("no_position_pod_share"),
        "mean_cash_share": mean_of("cash_share"),
        "pod_state_days": {pod_id: dict(counts) for pod_id, counts in state_days.items()},
    }


def _sizing_summary(block: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Per-pod sizing statistics for a period block, from its
    `sizing_diagnostics` entries with `path == "full"`, keyed by
    `manager_id`. `sizing_binding_shares` maps each pod to the share of
    its full-path days on which each stage was `binding_stage` (`"none"`
    when no stage bound). `sizing_vol_ratio` maps each pod to the mean of
    `final_vol / target_vol` over those days. A pod with no full-path days
    is left out; a block with no diagnostics gives `{}` for both."""
    days_by_pod: dict[str, list[dict[str, Any]]] = {}
    for entry in block.get("sizing_diagnostics") or []:
        if entry["path"] == "full":
            days_by_pod.setdefault(entry["manager_id"], []).append(entry)
    binding_shares: dict[str, dict[str, float]] = {}
    vol_ratio: dict[str, float] = {}
    for pod_id, days in days_by_pod.items():
        counts = Counter(entry["binding_stage"] or "none" for entry in days)
        binding_shares[pod_id] = {stage: count / len(days) for stage, count in counts.items()}
        vol_ratio[pod_id] = sum(entry["final_vol"] / entry["target_vol"] for entry in days) / len(
            days
        )
    return {"sizing_binding_shares": binding_shares, "sizing_vol_ratio": vol_ratio}


def _unfactored_split_move_counts(block: dict[str, Any]) -> dict[str, int] | None:
    """The number of `unfactored_split_moves` entries per security in a
    period block, keyed by the security id as a string; `{}` when the block
    has none, and `None` when its `unfactored_split_moves` is null (a run
    resumed from a checkpoint, whose list would be incomplete)."""
    if "unfactored_split_moves" in block and block["unfactored_split_moves"] is None:
        return None
    moves = block.get("unfactored_split_moves") or []
    return dict(Counter(str(move["security_id"]) for move in moves))


def _lake_reading(experiment_result: dict[str, Any]) -> dict[str, Any]:
    """`lake_time` from the payload's `provenance.lake`, `None` for a
    payload that does not carry it."""
    lake = (experiment_result.get("provenance") or {}).get("lake") or {}
    return {"lake_time": lake.get("lake_time")}


def extract_run_summary(experiment_result: dict[str, Any]) -> dict[str, Any]:
    """The full `experiment_result.json` payload in, the compact summary
    out. `experiment_result` must have `experiment`, `market_data_source`,
    `calibration`, and `test` top-level keys
    (as produced by `agentic-trading-experiment`); `baselines` and
    `provenance` are optional since not every payload carries them."""
    summary: dict[str, Any] = {
        "experiment": experiment_result["experiment"],
        "market_data_source": experiment_result["market_data_source"],
        "lake": _lake_reading(experiment_result),
    }
    for period in ("calibration", "test"):
        block = experiment_result[period]
        summary[period] = {key: block[key] for key in PERIOD_METRIC_KEYS}
        summary[period]["signal_health"] = block["signal_health"]
        summary[period]["source_performance"] = block["source_performance"]
        summary[period]["pod_return_history"] = block.get("pod_return_history", {})
        summary[period]["pod_performance"] = block.get("pod_performance", {})
        summary[period]["pod_costs"] = block.get("pod_costs", {})
        summary[period]["execution_costs"] = _execution_costs(block)
        summary[period]["unfactored_split_moves_by_security"] = _unfactored_split_move_counts(block)
        summary[period].update(_unfilled_order_counts(block))
        summary[period].update(_capital_usage_means(block))
        summary[period].update(_sizing_summary(block))

    summary["baselines"] = {
        name: {
            **{key: block[key] for key in PERIOD_METRIC_KEYS},
            **_unfilled_order_counts(block),
            **_capital_usage_means(block),
        }
        for name, block in experiment_result.get("baselines", {}).items()
    }
    return summary
