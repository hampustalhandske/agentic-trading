"""Unit tests for agentic_trading.cli.

Mock target convention chosen for this test file (mirrors tests/unit/data/test_loader.py's
documented convention): cli.py's `run` command is expected to construct and invoke
`agentic_trading.cli.BacktestEngine` (imported by that exact name into cli.py's own
namespace, e.g. `from agentic_trading.backtest.engine import BacktestEngine`) to execute the
backtest. Patching `BacktestEngine` guarantees no real backtest/LLM call occurs, but `run`
also calls `agentic_trading.cli.load_price_history_with_fallback` *before* constructing the
engine (to log which data source the run used) — that must be patched too, or these tests
would make a real lakehouse read despite BacktestEngine being mocked. `run` also runs the
real-data preflight (`agentic_trading.cli.check_data_availability`) before either of those —
tests that don't care about preflight behavior pass `--yes` to skip its prompt.
"""

import inspect
from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from typer.testing import CliRunner

from agentic_trading.agents.base import DataRequirement
from agentic_trading.agents.schemas import BacktestResult
from agentic_trading.backtest.preflight import (
    AnalystPreflightResult,
    PreflightReport,
    RequirementCoverage,
)
from agentic_trading.cli import app, experiment
from agentic_trading.config import settings, temporary_overrides

runner = CliRunner()


def _fake_backtest_result() -> BacktestResult:
    return BacktestResult(
        start_date=date(2024, 1, 2),
        end_date=date(2024, 1, 5),
        portfolio_value_history=[
            (date(2024, 1, 2), 10_000.0),
            (date(2024, 1, 5), 10_150.0),
        ],
        final_sharpe_ratio=1.2,
        final_max_drawdown=-0.03,
        final_sortino_ratio=1.1,
        final_calmar_ratio=0.9,
        final_hit_rate=0.55,
        final_profit_factor=1.3,
    )


@patch("agentic_trading.cli.load_price_history_with_fallback")
@patch("agentic_trading.cli.BacktestEngine")
def test_run_with_valid_arguments_exits_zero_and_prints_summary(
    mock_engine_cls: MagicMock, mock_load: MagicMock
) -> None:
    """Valid ticker + date-range arguments drive the (mocked) backtest pipeline to completion."""
    mock_load.return_value = (pd.DataFrame(), "lakehouse data")
    mock_engine_cls.return_value.run.return_value = _fake_backtest_result()

    result = runner.invoke(
        app,
        ["AAPL", "MSFT", "--start-date", "2024-01-02", "--end-date", "2024-01-05", "--yes"],
    )

    assert result.exit_code == 0
    mock_engine_cls.return_value.run.assert_called_once()
    assert result.stdout.strip() != ""


@patch("agentic_trading.cli.load_price_history_with_fallback")
@patch("agentic_trading.cli.BacktestEngine")
def test_run_never_executes_a_real_backtest(
    mock_engine_cls: MagicMock, mock_load: MagicMock
) -> None:
    """No LLM calls, no network calls: only the mocked BacktestEngine.run() may be invoked."""
    mock_load.return_value = (pd.DataFrame(), "lakehouse data")
    mock_engine_cls.return_value.run.return_value = _fake_backtest_result()

    runner.invoke(
        app,
        ["AAPL", "--start-date", "2024-01-02", "--end-date", "2024-01-05", "--yes"],
    )

    mock_engine_cls.assert_called_once()
    mock_load.assert_called_once()


def test_run_missing_assets_argument_exits_nonzero() -> None:
    """No tickers supplied at all is also a usage error, not a silent empty-basket run."""
    result = runner.invoke(app, ["--start-date", "2024-01-02", "--end-date", "2024-01-05"])

    assert result.exit_code != 0


def _failing_preflight_report() -> PreflightReport:
    requirement = DataRequirement(
        source="fred", identifier="DGS3MO", asset_scoped=False, description="FRED DGS3MO"
    )
    check = RequirementCoverage(
        requirement=requirement,
        asset=None,
        available_from=None,
        available_to=None,
        covers_run=False,
    )
    return PreflightReport(
        checked_at=date(2024, 1, 2),
        results=[
            AnalystPreflightResult(
                analyst_id="cir_rate_regime_analyst", passed=False, checks=[check]
            ),
            AnalystPreflightResult(analyst_id="risk_minimizing_analyst", passed=True, checks=[]),
        ],
    )


@patch("agentic_trading.cli.check_data_availability")
@patch("agentic_trading.cli.load_price_history_with_fallback")
@patch("agentic_trading.cli.BacktestEngine")
def test_run_yes_flag_auto_drops_failing_analysts_without_prompting(
    mock_engine_cls: MagicMock, mock_load: MagicMock, mock_preflight: MagicMock
) -> None:
    mock_preflight.return_value = _failing_preflight_report()
    mock_load.return_value = (pd.DataFrame(), "lakehouse data")
    mock_engine_cls.return_value.run.return_value = _fake_backtest_result()

    result = runner.invoke(
        app,
        ["AAPL", "--start-date", "2024-01-02", "--end-date", "2024-01-05", "--yes"],
    )

    assert result.exit_code == 0
    mock_engine_cls.return_value.run.assert_called_once()


@patch("agentic_trading.cli.check_data_availability")
@patch("agentic_trading.cli.load_price_history_with_fallback")
@patch("agentic_trading.cli.BacktestEngine")
def test_run_prompts_and_continues_on_confirm(
    mock_engine_cls: MagicMock, mock_load: MagicMock, mock_preflight: MagicMock
) -> None:
    mock_preflight.return_value = _failing_preflight_report()
    mock_load.return_value = (pd.DataFrame(), "lakehouse data")
    mock_engine_cls.return_value.run.return_value = _fake_backtest_result()

    result = runner.invoke(
        app,
        ["AAPL", "--start-date", "2024-01-02", "--end-date", "2024-01-05"],
        input="y\n",
    )

    assert result.exit_code == 0
    assert "cir_rate_regime_analyst" in result.output
    mock_engine_cls.return_value.run.assert_called_once()


@patch("agentic_trading.cli.check_data_availability")
@patch("agentic_trading.cli.load_price_history_with_fallback")
@patch("agentic_trading.cli.BacktestEngine")
def test_run_aborts_on_decline(
    mock_engine_cls: MagicMock, mock_load: MagicMock, mock_preflight: MagicMock
) -> None:
    mock_preflight.return_value = _failing_preflight_report()
    mock_load.return_value = (pd.DataFrame(), "lakehouse data")

    result = runner.invoke(
        app,
        ["AAPL", "--start-date", "2024-01-02", "--end-date", "2024-01-05"],
        input="n\n",
    )

    assert result.exit_code != 0
    mock_engine_cls.return_value.run.assert_not_called()


@patch("agentic_trading.cli.load_price_history_with_fallback")
@patch("agentic_trading.cli.BacktestEngine")
def test_run_builds_engine_with_execution_defaults_from_settings(
    mock_engine_cls: MagicMock, mock_load: MagicMock
) -> None:
    """`run` shares the firm-wide execution assumptions: commission and
    slippage on the Portfolio and the execution lag on the engine all come
    from `settings.firm` rather than being hardcoded per entry point."""
    mock_load.return_value = (pd.DataFrame(), "lakehouse data")
    mock_engine_cls.return_value.run.return_value = _fake_backtest_result()

    with temporary_overrides(
        {
            "firm.commission_rate": 0.003,
            "firm.slippage_bps": 9.0,
            "firm.execution_lag_days": 0,
        }
    ):
        result = runner.invoke(
            app,
            ["AAPL", "--start-date", "2024-01-02", "--end-date", "2024-01-05", "--yes"],
        )

    assert result.exit_code == 0
    kwargs = mock_engine_cls.call_args.kwargs
    assert kwargs["execution_lag_days"] == 0
    assert kwargs["portfolio"].commission_rate == pytest.approx(0.003)
    assert kwargs["portfolio"].slippage_bps == pytest.approx(9.0)


def test_experiment_options_default_to_firm_settings() -> None:
    """The experiment command's overridable execution options carry the
    same settings-derived defaults as every other entry point."""
    signature = inspect.signature(experiment)
    option_defaults = {
        name: signature.parameters[name].default.default
        for name in ("commission_rate", "slippage_bps", "execution_lag_days")
    }

    assert option_defaults == {
        "commission_rate": settings.firm.commission_rate,
        "slippage_bps": settings.firm.slippage_bps,
        "execution_lag_days": settings.firm.execution_lag_days,
    }
