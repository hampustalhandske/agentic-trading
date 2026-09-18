"""Unit tests for agentic_trading.backtest.run_summary.extract_run_summary."""

from agentic_trading.backtest.run_summary import extract_run_summary


def _fake_period_block(**overrides: object) -> dict:
    block = {
        "start_date": "2018-04-02",
        "end_date": "2018-04-03",
        "portfolio_value_history": [["2018-04-02", 10_000.0], ["2018-04-03", 10_050.0]],
        "final_sharpe_ratio": 0.5,
        "final_max_drawdown": -0.01,
        "final_sortino_ratio": 0.6,
        "final_calmar_ratio": 0.4,
        "final_information_ratio": None,
        "final_hit_rate": 0.55,
        "final_profit_factor": 1.1,
        "decisions_by_day": {"2018-04-02": [{"asset": "AAPL"}]},
        "trade_log": [{"asset": "AAPL"}],
        "signal_health": {"risk_minimizing_analyst": {"active": 2, "unavailable": 0, "error": 0}},
        "source_performance": {"risk_minimizing_analyst": {"cumulative_return": 0.01}},
        "pod_return_history": {"risk_manager": [["2018-04-02", 0.001], ["2018-04-03", -0.0005]]},
        "market_data_source": "lakehouse data",
    }
    block.update(overrides)
    return block


def _fake_experiment_result(**extra: object) -> dict:
    result = {
        "experiment": {"assets": ["AAPL"], "test_start": "2018-04-02"},
        "market_data_source": "lakehouse data",
        "calibration": _fake_period_block(),
        "test": _fake_period_block(),
    }
    result.update(extra)
    return result


def test_extract_run_summary_keeps_experiment_metadata() -> None:
    summary = extract_run_summary(_fake_experiment_result())

    assert summary["experiment"] == {"assets": ["AAPL"], "test_start": "2018-04-02"}
    assert summary["market_data_source"] == "lakehouse data"


def test_extract_run_summary_drops_decisions_by_day_and_trade_log() -> None:
    summary = extract_run_summary(_fake_experiment_result())

    for period in ("calibration", "test"):
        assert "decisions_by_day" not in summary[period]
        assert "trade_log" not in summary[period]


def test_extract_run_summary_keeps_metrics_signal_health_and_source_performance() -> None:
    summary = extract_run_summary(_fake_experiment_result())

    test_summary = summary["test"]
    assert test_summary["final_sharpe_ratio"] == 0.5
    assert test_summary["portfolio_value_history"] == [
        ["2018-04-02", 10_000.0],
        ["2018-04-03", 10_050.0],
    ]
    assert test_summary["signal_health"] == {
        "risk_minimizing_analyst": {"active": 2, "unavailable": 0, "error": 0}
    }
    assert test_summary["source_performance"] == {
        "risk_minimizing_analyst": {"cumulative_return": 0.01}
    }
    assert test_summary["pod_return_history"] == {
        "risk_manager": [["2018-04-02", 0.001], ["2018-04-03", -0.0005]]
    }


def test_extract_run_summary_carries_baselines_when_present() -> None:
    experiment_result = _fake_experiment_result(
        baselines={"equal_weight": _fake_period_block(final_sharpe_ratio=0.3)}
    )

    summary = extract_run_summary(experiment_result)

    assert summary["baselines"]["equal_weight"]["final_sharpe_ratio"] == 0.3
    assert "decisions_by_day" not in summary["baselines"]["equal_weight"]


def test_extract_run_summary_defaults_baselines_to_empty_when_absent() -> None:
    summary = extract_run_summary(_fake_experiment_result())

    assert summary["baselines"] == {}


def test_extract_run_summary_defaults_pod_return_history_to_empty_when_absent() -> None:
    """Older result files predating `pod_return_history` still summarize."""
    block = _fake_period_block()
    del block["pod_return_history"]
    experiment_result = _fake_experiment_result(calibration=block, test=block)

    summary = extract_run_summary(experiment_result)

    assert summary["test"]["pod_return_history"] == {}
