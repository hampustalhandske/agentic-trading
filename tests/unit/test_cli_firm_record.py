# test_cli_firm_record.py

"""AT-125: `agentic-trading-firm-record` builds the registered firm (agentic-firm-v1: the
default roster without `FIRM_V1_EXCLUDED_ANALYST_IDS`, over `FIRM_V1_UNIVERSE`), loads its
prices aligned on `stage_a.session_calendar` with the stage (a) warm-up inside one lake
session cut at `stage_a.data_end`, runs `run_firm_record` and prints its result, which holds no
performance figure."""

import inspect
import tomllib
from collections.abc import Callable
from datetime import date, timedelta
from importlib import import_module
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from click.testing import Result
from typer.testing import CliRunner

from agentic_trading.agents.roster import (
    FIRM_V1_EQUITIES,
    FIRM_V1_EQUITY_SECURITY_IDS,
    FIRM_V1_EXCLUDED_ANALYST_IDS,
    FIRM_V1_UNIVERSE,
)
from agentic_trading.backtest.preflight import PriceCoverageError, check_price_coverage
from agentic_trading.backtest.resume_fingerprint import roster_agents
from agentic_trading.cli import firm_record_app
from agentic_trading.config import settings
from agentic_trading.data.loader import load_price_history_with_fallback
from agentic_trading.firm_record.runner import FirmRecordResult, run_firm_record
from agentic_trading.trials.ledger import TrialLedger

runner = CliRunner()
_PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_MARKET = pd.DataFrame(
    {("SPY", "close"): [100.0, 101.0]}, index=pd.DatetimeIndex(["2010-01-04", "2010-01-05"])
)
_RESULT = FirmRecordResult(
    spec_hash="0" * 64,
    days_written=2,
    last_completed=date(2010, 1, 5),
    complete=True,
    counted=False,
)


def _invoke(
    args: list[str], coverage: MagicMock | None = None
) -> tuple[Result, MagicMock, MagicMock, MagicMock]:
    """Runs the command with the price load, the rate load and the record mocked, and the
    price coverage preflight replaced by `coverage` (a passing mock by default)."""
    with (
        patch(
            "agentic_trading.cli.load_price_history_with_fallback",
            return_value=(_MARKET, "lakehouse data"),
        ) as load,
        patch("agentic_trading.cli.load_risk_free_rate_series", return_value=None) as rates,
        patch("agentic_trading.cli.run_firm_record", return_value=_RESULT) as record,
        patch("agentic_trading.cli.check_price_coverage", coverage or MagicMock()),
    ):
        result = runner.invoke(firm_record_app, args)
    return result, load, rates, record


def _bound(function: Callable[..., Any], mock: MagicMock) -> dict[str, Any]:
    mock.assert_called_once()
    bound = inspect.signature(function).bind(*mock.call_args.args, **mock.call_args.kwargs)
    bound.apply_defaults()
    return bound.arguments


def test_entry_point_is_declared() -> None:
    """pyproject declares agentic-trading-firm-record on cli."""
    scripts = tomllib.loads(_PYPROJECT.read_text())["project"]["scripts"]
    module, _, name = scripts["agentic-trading-firm-record"].partition(":")

    assert (module, name) == ("agentic_trading.cli", "firm_record_main")
    assert callable(getattr(import_module(module), name))


def test_firm_record_runs_the_registered_firm(tmp_path: Path, preflight_lake: Any) -> None:
    """The registered roster and universe, session-aligned prices with the stage (a) warm-up,
    --max-days and the ledger reach run_firm_record; its result is printed."""
    args = [
        "--start",
        "2010-01-04",
        "--end",
        "2010-01-05",
        "--store-dir",
        str(tmp_path),
        "--max-days",
        "20",
        "--lake-time",
        "2026-11-06T00:00:00+00:00",
    ]
    result, load, rates, record = _invoke(args)
    loaded = _bound(load_price_history_with_fallback, load)
    called = _bound(run_firm_record, record)

    assert result.exit_code == 0, result.output
    assert list(loaded["tickers"]) == list(FIRM_V1_UNIVERSE)
    assert (loaded["start"], loaded["end"]) == (date(2010, 1, 4), date(2010, 1, 5))
    assert loaded["warmup_days"] == settings.stage_a.warmup_days
    assert (loaded["align"], loaded["calendar"]) == ("sessions", settings.stage_a.session_calendar)
    assert loaded["security_ids"] is FIRM_V1_EQUITY_SECURITY_IDS
    rates.assert_called_once_with(date(2010, 1, 4), date(2010, 1, 5))
    agent_ids = {agent.agent_id for agent in roster_agents(called["commander"])}
    assert agent_ids.isdisjoint(FIRM_V1_EXCLUDED_ANALYST_IDS)
    assert "tail_hedge_manager" in agent_ids and "ou_mean_reversion_analyst" in agent_ids
    assert list(called["universe"]) == list(FIRM_V1_UNIVERSE)
    assert called["market_data"] is _MARKET
    assert called["max_days"] == 20
    assert called["store_dir"] == tmp_path
    assert isinstance(called["ledger"], TrialLedger)
    assert called["repo_root"] == _REPO_ROOT
    assert called["lake_time"].isoformat() == "2026-11-06T00:00:00+00:00"
    assert FirmRecordResult.model_validate_json(result.stdout[result.stdout.index("{") :]) == (
        _RESULT
    )


@pytest.mark.parametrize(
    ("start", "end", "store_dir"),
    [
        ("2010-01-05", "2010-01-04", None),
        ("2024-12-02", (settings.stage_a.data_end + timedelta(days=1)).isoformat(), None),
        ("2010-01-04", "2010-01-05", _REPO_ROOT / "inside"),
    ],
)
def test_usage_errors(tmp_path: Path, start: str, end: str, store_dir: Path | None) -> None:
    """A reversed window, a window past stage_a.data_end or a store inside the repository is a
    usage error, before any read."""
    args = ["--start", start, "--end", end, "--store-dir", str(store_dir or tmp_path)]
    result, load, _, record = _invoke(args)

    assert result.exit_code == 2, result.output
    load.assert_not_called()
    record.assert_not_called()


def test_price_coverage_preflight_runs_before_the_record(tmp_path: Path) -> None:
    """The firm equities' price coverage is checked on the loaded frame from the first warm-up
    day to --end; a failure stops the command before run_firm_record writes anything."""
    args = ["--start", "2010-01-04", "--end", "2010-01-05", "--store-dir", str(tmp_path)]
    passing = MagicMock()
    result, _, _, record = _invoke(args, passing)
    checked = _bound(check_price_coverage, passing)

    assert result.exit_code == 0, result.output
    assert checked["market_data"] is _MARKET
    assert list(checked["members"]) == list(FIRM_V1_EQUITIES)
    warmup_start = date(2010, 1, 4) - timedelta(days=settings.stage_a.warmup_days)
    assert (checked["from_date"], checked["to_date"]) == (warmup_start, date(2010, 1, 5))
    assert checked["calendar"] == settings.stage_a.session_calendar

    failing = MagicMock(side_effect=PriceCoverageError("MON first bar none, last bar none"))
    result, _, _, record = _invoke(args, failing)

    assert isinstance(result.exception, PriceCoverageError)
    record.assert_not_called()
