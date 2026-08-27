"""Unit tests for agentic_trading.cli.

Mock target convention chosen for this test file (mirrors tests/unit/data/test_loader.py's
documented convention): cli.py's `run` command is expected to construct and invoke
`agentic_trading.cli.BacktestEngine` (imported by that exact name into cli.py's own
namespace, e.g. `from agentic_trading.backtest.engine import BacktestEngine`) to execute the
backtest. Patching `BacktestEngine` guarantees no real backtest/LLM call occurs, but `run`
also calls `agentic_trading.cli.load_price_history_with_fallback` *before* constructing the
engine (to log which data source — live or synthetic — the run used) — that must be patched
too, or these tests would make a real yfinance network call despite BacktestEngine being
mocked. The implementer should wire `run` through both names so these tests pass without
being rewritten around the implementation.

NOTE: `typer` is imported by cli.py but is not currently declared in pyproject.toml's
dependencies nor installed in this project's .venv (confirmed via `.venv/lib/python*/
site-packages` — no `typer` package present). This file will fail to collect until `typer` is
added as a dependency and installed; that is a pre-existing environment gap, not an issue
with these tests.
"""

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
from typer.testing import CliRunner

from agentic_trading.agents.schemas import BacktestResult
from agentic_trading.cli import app

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
    mock_load.return_value = (pd.DataFrame(), "live yfinance data")
    mock_engine_cls.return_value.run.return_value = _fake_backtest_result()

    result = runner.invoke(
        app,
        ["AAPL", "MSFT", "--start-date", "2024-01-02", "--end-date", "2024-01-05"],
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
    mock_load.return_value = (pd.DataFrame(), "live yfinance data")
    mock_engine_cls.return_value.run.return_value = _fake_backtest_result()

    runner.invoke(
        app,
        ["AAPL", "--start-date", "2024-01-02", "--end-date", "2024-01-05"],
    )

    mock_engine_cls.assert_called_once()
    mock_load.assert_called_once()


def test_run_missing_assets_argument_exits_nonzero() -> None:
    """No tickers supplied at all is also a usage error, not a silent empty-basket run."""
    result = runner.invoke(app, ["--start-date", "2024-01-02", "--end-date", "2024-01-05"])

    assert result.exit_code != 0
