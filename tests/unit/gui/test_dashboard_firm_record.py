# test_dashboard_firm_record.py

"""The dashboard's saved-run panel on a firm record: its loader accepts the record's summary
JSON (one `record` period) and the record's engine checkpoint, whose summary it builds once,
read-only; the Saved Run view shows the record against SPY with a Pods section, also before any
live simulation has run."""

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import agentic_trading.gui.dashboard as dashboard_module
from agentic_trading.firm_record import review
from unit.firm_record.test_review import record_benchmarks, write_record


def _record_summary() -> dict[str, Any]:
    history = [["2010-01-04", 1_000_000.0], ["2010-01-05", 1_010_000.0], ["2010-01-06", 990_000.0]]
    return {
        "experiment": {
            "spec_hash": "abc",
            "assets": ["SPY", "AAPL"],
            "excluded_analysts": ["pairs_cointegration_analyst"],
        },
        "market_data_source": "lakehouse data",
        "lake": {"lake_time": "2026-11-10T19:55:57+00:00", "lake_version": "aaa.bbb"},
        "record": {
            "start_date": "2010-01-04",
            "end_date": "2010-01-06",
            "portfolio_value_history": history,
            "final_sharpe_ratio": 0.5,
            "final_max_drawdown": 0.02,
            "final_sortino_ratio": 0.6,
            "final_calmar_ratio": 0.4,
            "final_information_ratio": -0.1,
            "final_hit_rate": 0.5,
            "final_profit_factor": 1.1,
            "signal_health": {
                "vix_tail_hedge_analyst": {"active": 3, "unavailable": 0, "error": 0}
            },
            "source_performance": {
                "vix_tail_hedge_analyst": {
                    "cumulative_return": 0.01,
                    "sharpe_ratio": 0.2,
                    "max_drawdown": -0.01,
                    "observations": 3.0,
                },
                "tail_hedge_manager": {
                    "cumulative_return": 0.01,
                    "sharpe_ratio": 0.2,
                    "max_drawdown": -0.01,
                    "observations": 3.0,
                },
            },
            "pod_return_history": {
                "risk_manager": [["2010-01-05", 0.01], ["2010-01-06", -0.02]],
                "tail_hedge_manager": [["2010-01-06", 0.003]],
            },
            "pod_performance": {
                "risk_manager": {
                    "observations": 2,
                    "cumulative_return": -0.0102,
                    "annualized_vol": 0.3,
                    "sharpe_ratio": None,
                    "max_drawdown": 0.02,
                }
            },
            "pod_costs": {
                "risk_manager": {"execution_cost": 12.0, "turnover": 3.0},
                "commander": {"execution_cost": 1.0, "turnover": 0.0},
            },
            "execution_costs": {"all_in": 13.0},
        },
        "baselines": {
            "SPY total return": {
                "start_date": "2010-01-04",
                "end_date": "2010-01-06",
                "portfolio_value_history": [
                    ["2010-01-04", 1_000_000.0],
                    ["2010-01-05", 1_005_000.0],
                    ["2010-01-06", 1_001_000.0],
                ],
                "final_sharpe_ratio": 0.3,
                "final_max_drawdown": 0.004,
                "final_sortino_ratio": 0.3,
                "final_calmar_ratio": 0.3,
                "final_information_ratio": None,
                "final_hit_rate": 0.5,
                "final_profit_factor": 1.2,
            }
        },
    }


def test_a_firm_record_summary_loads_as_it_is(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(_record_summary()))

    assert dashboard_module._load_saved_run_summary(path) == _record_summary()


def test_a_checkpoint_loads_as_its_firm_record_summary_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The loader builds the summary from the checkpoint and the trial folder beside it, reads
    nothing new on a second load, and leaves the record's store as it was."""
    written = write_record(tmp_path / "store")
    reads: list[review.RecordBenchmarks] = []

    def _benchmarks(*_args: object, **_kwargs: object) -> review.RecordBenchmarks:
        reads.append(record_benchmarks())
        return reads[-1]

    monkeypatch.setattr(review, "load_record_benchmarks", _benchmarks)
    before = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in written.store.rglob("*")
        if path.is_file()
    }

    loaded = dashboard_module._load_saved_run_summary(written.checkpoint)
    again = dashboard_module._load_saved_run_summary(written.checkpoint)

    expected = review.firm_record_summary(
        written.checkpoint, written.trial_dir, benchmarks=record_benchmarks()
    )
    assert loaded == expected
    assert again == expected
    assert len(reads) == 1
    after = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in written.store.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_a_payload_without_any_period_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "other.json"
    path.write_text(json.dumps({"experiment": {}}))

    with pytest.raises(ValueError, match="record"):
        dashboard_module._load_saved_run_summary(path)


def test_pod_cumulative_returns_compound_each_pods_own_returns() -> None:
    frame = dashboard_module._pod_cumulative_return_frame(
        _record_summary()["record"]["pod_return_history"]
    )

    assert list(frame.columns) == ["risk_manager", "tail_hedge_manager"]
    assert frame.loc[pd.Timestamp("2010-01-05"), "risk_manager"] == pytest.approx(1.0)
    assert frame.loc[pd.Timestamp("2010-01-06"), "risk_manager"] == pytest.approx(
        (1.01 * 0.98 - 1.0) * 100.0
    )
    assert pd.isna(frame.loc[pd.Timestamp("2010-01-05"), "tail_hedge_manager"])
    assert dashboard_module._pod_cumulative_return_frame({}).empty


def test_the_pod_table_joins_performance_and_costs_per_pod() -> None:
    record = _record_summary()["record"]

    table = dashboard_module._pod_table_frame(record["pod_performance"], record["pod_costs"])

    assert list(table["pod_id"]) == ["risk_manager", "commander"]
    assert list(table["pod"]) == ["Risk", "commander"]
    assert table.loc[0, "execution_cost"] == 12.0
    assert table.loc[0, "observations"] == 2
    assert pd.isna(table.loc[1, "observations"])
    assert dashboard_module._pod_table_frame({}, {}).empty


def test_the_saved_run_view_shows_a_firm_record_without_a_live_run() -> None:
    at = AppTest.from_file(dashboard_module.__file__)
    at.session_state["saved_run_summary"] = _record_summary()
    at.session_state["saved_run_source_path"] = "runs/store/abc.checkpoint.sqlite"

    at.run(timeout=30)

    assert not at.exception
    markdown = " ".join(md.value for md in at.markdown)
    assert "runs/store/abc.checkpoint.sqlite" in markdown
    assert "Pods" in markdown
    assert "Equity curve" in markdown
    periods = [radio for radio in at.radio if radio.key == "saved_run_period"]
    assert periods and list(periods[0].options) == ["record"]
    assert {metric.label for metric in at.metric} >= {"Sharpe", "Information ratio"}
    assert len(at.dataframe) >= 2
