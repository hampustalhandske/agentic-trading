# test_cli_experiment_overrides.py

"""`experiment` resolves commission, slippage and execution lag from
`settings.firm` after `--set` overrides; explicit flags win."""

import json
from collections.abc import Iterator
from datetime import date
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from typer.testing import CliRunner

from agentic_trading.agents.schemas import BacktestResult
from agentic_trading.backtest.preflight import PreflightReport
from agentic_trading.cli import experiment_app
from agentic_trading.config import settings, temporary_overrides

runner = CliRunner()

_BASE_ARGS = [
    "AAPL",
    "--train-start",
    "2024-01-02",
    "--train-end",
    "2024-01-05",
    "--test-start",
    "2024-01-08",
    "--test-end",
    "2024-01-12",
    "--yes",
]


@pytest.fixture(autouse=True)
def _restore_firm_settings() -> Iterator[None]:
    """Restores the firm execution settings that `--set` mutates globally."""
    keys = ("commission_rate", "slippage_bps", "execution_lag_days")
    with temporary_overrides({f"firm.{k}": getattr(settings.firm, k) for k in keys}):
        yield


def _run(extra: list[str]) -> tuple[Any, list[Any]]:
    """Invokes `experiment` with stubbed data and engine; returns the result and engine calls."""
    index = pd.date_range("2024-01-02", periods=4)
    frame = pd.DataFrame({("^GSPC", "close"): [100.0, 101.0, 102.0, 103.0]}, index=index)
    frame.columns = pd.MultiIndex.from_tuples(frame.columns)
    backtest_result = BacktestResult(
        start_date=date(2024, 1, 2),
        end_date=date(2024, 1, 5),
        portfolio_value_history=[(date(2024, 1, 2), 10_000.0), (date(2024, 1, 5), 10_150.0)],
        final_sharpe_ratio=1.2,
        final_max_drawdown=-0.03,
        final_sortino_ratio=1.1,
        final_calmar_ratio=0.9,
        final_hit_rate=0.55,
        final_profit_factor=1.3,
    )
    with (
        patch("agentic_trading.cli.check_data_availability") as preflight,
        patch("agentic_trading.cli.load_risk_free_rate_series") as risk_free,
        patch("agentic_trading.cli.load_price_history_with_fallback") as load,
        patch("agentic_trading.cli.BacktestEngine") as engine_cls,
    ):
        load.return_value = (frame, "lakehouse data")
        risk_free.return_value = pd.Series(0.0, index=index)
        preflight.return_value = PreflightReport(checked_at=date(2024, 1, 2), results=[])
        engine_cls.return_value.run.return_value = backtest_result
        result = runner.invoke(experiment_app, _BASE_ARGS + extra)
    calls: list[MagicMock] = engine_cls.call_args_list
    return result, calls


@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        (["--set", "firm.commission_rate=0.003"], (0.003, 5.0, 1)),
        (["--set", "firm.slippage_bps=7"], (0.0005, 7.0, 1)),
        (["--set", "firm.execution_lag_days=0"], (0.0005, 5.0, 0)),
        (["--commission-rate", "0.001", "--set", "firm.commission_rate=0.003"], (0.001, 5.0, 1)),
        (["--slippage-bps", "2", "--set", "firm.slippage_bps=7"], (0.0005, 2.0, 1)),
        (["--execution-lag-days", "1", "--set", "firm.execution_lag_days=0"], (0.0005, 5.0, 1)),
        ([], (0.0005, 5.0, 1)),
    ],
)
def test_experiment_resolves_execution_options(
    extra: list[str], expected: tuple[float, float, int]
) -> None:
    """Execution options come from the flag, else `--set`, else the settings default."""
    result, calls = _run(extra)

    assert result.exit_code == 0, result.output
    assert calls
    for call in calls:
        portfolio = call.kwargs["portfolio"]
        assert portfolio.commission_rate == pytest.approx(expected[0])
        assert portfolio.slippage_bps == pytest.approx(expected[1])
        assert call.kwargs["execution_lag_days"] == expected[2]


def test_experiment_prints_overridden_commission_in_provenance() -> None:
    """The printed provenance settings reflect the `--set` commission override."""
    result, _ = _run(["--set", "firm.commission_rate=0.003"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout[result.stdout.index("{") :])
    assert payload["provenance"]["settings"]["firm"]["commission_rate"] == pytest.approx(0.003)


def test_experiment_rejects_max_leverage_override() -> None:
    """`--set firm.max_leverage` fails with a message naming the unsupported key."""
    result, calls = _run(["--set", "firm.max_leverage=3"])

    assert result.exit_code != 0
    assert "firm.max_leverage" in result.output
    assert not calls
