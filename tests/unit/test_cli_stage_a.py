# test_cli_stage_a.py

"""AT-110 precompute spec item 5: the `agentic-trading-stage-a` command loads the universe's
prices from the window start minus the standard warm-up, runs `run_batch` over the chosen
analysts, prints the `BatchResult` as JSON and exits 3 when the batch stopped for resources."""

import inspect
import tomllib
from collections.abc import Callable, Iterator
from datetime import UTC, date, datetime, timedelta
from importlib import import_module
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from click.testing import Result
from typer.testing import CliRunner

from agentic_trading import cli
from agentic_trading.agents.base import AnalystAgent
from agentic_trading.cli import stage_a_app
from agentic_trading.config import settings, temporary_overrides
from agentic_trading.data.loader import (
    DEFAULT_BACKTEST_WARMUP_DAYS,
    load_price_history_with_fallback,
)
from agentic_trading.stage_a.runner import BatchResult, run_batch
from agentic_trading.trials.ledger import TrialLedger

runner = CliRunner()

_PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"
# The reverse of their roster order, so the order run_batch receives is the caller's.
_ANALYSTS = ("cir_rate_regime_analyst", "egarch_asymmetric_vol_analyst")
_UNIVERSE = ("SHY", "IEF")
_T = datetime(2026, 10, 29, tzinfo=UTC)
_INDEX = pd.bdate_range("2010-01-04", "2010-01-08", name="date")
_CLOSE = [100.0, 101.0, 102.0, 103.0, 104.0]
_MARKET = pd.DataFrame(
    {(ticker, field): _CLOSE for ticker in _UNIVERSE for field in ("open", "close")},
    index=_INDEX,
)


def _args(store_dir: Path | None, *extra: str, analysts: tuple[str, ...] = _ANALYSTS) -> list[str]:
    """Command-line tokens for `analysts` over `_UNIVERSE`, with `--store-dir` when given."""
    tokens = [token for agent_id in analysts for token in ("--analyst", agent_id)]
    tokens += [token for ticker in _UNIVERSE for token in ("--universe", ticker)]
    if store_dir is not None:
        tokens += ["--store-dir", str(store_dir)]
    return [*tokens, *extra]


def _result(stopped: bool) -> BatchResult:
    entry = {"spec_hash": "0" * 64, "days_written": 5, "last_completed": "2010-01-08"}
    return BatchResult.model_validate(
        {"analysts": dict.fromkeys(_ANALYSTS, entry), "stopped_for_resources": stopped}
    )


def _invoke(args: list[str], stopped: bool = False) -> tuple[Result, MagicMock, MagicMock]:
    """Invoke the command with the price loader and `run_batch` patched; returns the result and
    both mocks."""
    with (
        patch(
            "agentic_trading.cli.load_price_history_with_fallback",
            return_value=(_MARKET, "lakehouse data"),
        ) as load,
        patch("agentic_trading.cli.run_batch", return_value=_result(stopped)) as batch,
    ):
        result = runner.invoke(stage_a_app, args)
    return result, load, batch


def _bound(function: Callable[..., Any], mock: MagicMock) -> dict[str, Any]:
    """The arguments of `mock`'s one call, by parameter name of `function`, defaults applied."""
    mock.assert_called_once()
    bound = inspect.signature(function).bind(*mock.call_args.args, **mock.call_args.kwargs)
    bound.apply_defaults()
    return bound.arguments


def _printed(result: Result) -> BatchResult:
    return BatchResult.model_validate_json(result.stdout[result.stdout.index("{") :])


def test_entry_point_is_declared() -> None:
    """pyproject declares agentic-trading-stage-a, beside the other commands, on cli."""
    scripts = tomllib.loads(_PYPROJECT.read_text())["project"]["scripts"]
    module, _, name = scripts["agentic-trading-stage-a"].partition(":")

    assert (module, name) == ("agentic_trading.cli", "stage_a_main")
    assert callable(getattr(import_module(module), name))


@pytest.mark.parametrize(("stopped", "exit_code"), [(False, 0), (True, 3)])
def test_stage_a_runs_the_batch(
    tmp_path: Path, preflight_lake: Any, stopped: bool, exit_code: int
) -> None:
    """Prices load with the warm-up, every option reaches run_batch, the result prints as JSON,
    and the exit code is 3 when the batch stopped for resources, else 0."""
    extra = [
        "--instrument-kind",
        "etf",
        "--window-start",
        "2010-01-04",
        "--window-end",
        "2010-01-08",
        "--lake-time",
        "2026-10-29T00:00:00+00:00",
    ]
    result, load, batch = _invoke(_args(tmp_path, *extra), stopped)
    loaded = _bound(load_price_history_with_fallback, load)
    called = _bound(run_batch, batch)
    warmed_start = date(2010, 1, 4) - timedelta(days=DEFAULT_BACKTEST_WARMUP_DAYS)

    assert result.exit_code == exit_code, result.output
    assert list(loaded["tickers"]) == list(_UNIVERSE)
    assert loaded["start"] - timedelta(days=loaded["warmup_days"]) == warmed_start
    assert loaded["end"] == date(2010, 1, 8)
    assert all(isinstance(analyst, AnalystAgent) for analyst in called["analysts"])
    assert [analyst.agent_id for analyst in called["analysts"]] == list(_ANALYSTS)
    assert list(called["universe"]) == list(_UNIVERSE)
    assert called["instrument_kind"] == "etf"
    assert (called["window_start"], called["window_end"]) == (date(2010, 1, 4), date(2010, 1, 8))
    assert Path(called["store_dir"]) == tmp_path
    assert called["market_data"] is _MARKET
    assert called["lake_time"] == _T
    assert _printed(result) == _result(stopped)


def test_stage_a_defaults(tmp_path: Path, preflight_lake: Any) -> None:
    """Without the optional flags the kind is equity and the window is settings.stage_a's
    window_start to data_end, read at use time."""
    result, load, batch = _invoke(_args(tmp_path))
    called = _bound(run_batch, batch)
    with temporary_overrides(
        {"stage_a.window_start": date(2012, 1, 3), "stage_a.data_end": date(2020, 12, 31)}
    ):
        overridden, overridden_load, overridden_batch = _invoke(_args(tmp_path))
    moved = _bound(run_batch, overridden_batch)
    window = (called["window_start"], called["window_end"])

    assert result.exit_code == 0, result.output
    assert called["instrument_kind"] == "equity"
    assert window == (date(2010, 1, 4), date(2024, 12, 31))
    assert _bound(load_price_history_with_fallback, load)["end"] == date(2024, 12, 31)
    assert overridden.exit_code == 0, overridden.output
    assert (moved["window_start"], moved["window_end"]) == (date(2012, 1, 3), date(2020, 12, 31))
    assert _bound(load_price_history_with_fallback, overridden_load)["end"] == date(2020, 12, 31)


def test_stage_a_usage_errors(tmp_path: Path, preflight_lake: Any) -> None:
    """An unknown analyst id, a missing --store-dir or a naive --lake-time is a usage error that
    loads and runs nothing."""
    cases = {
        "unknown analyst": _args(tmp_path, analysts=(_ANALYSTS[0], "no_such_analyst")),
        "no store dir": _args(None),
        "naive lake time": _args(tmp_path, "--lake-time", "2026-10-29T00:00:00"),
    }
    for label, args in cases.items():
        result, load, batch = _invoke(args)

        assert result.exit_code == 2, (label, result.output)
        load.assert_not_called()
        batch.assert_not_called()


@pytest.fixture
def _restore_ledger_path() -> Iterator[None]:
    """Restores `trials.ledger_path`, which `--set` mutates globally."""
    with temporary_overrides({"trials.ledger_path": settings.trials.ledger_path}):
        yield


def test_stage_a_records_trials_in_the_ledger(
    tmp_path: Path, preflight_lake: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_batch gets a TrialLedger at the repository root joined with trials.ledger_path, read
    at use time, and the repository root as repo_root."""
    repo_root = tmp_path / "repo"
    monkeypatch.setattr(cli, "_REPO_ROOT", repo_root)
    with temporary_overrides({"trials.ledger_path": "trials/ledger.duckdb"}):
        result, _, batch = _invoke(_args(tmp_path / "store"))
    called = _bound(run_batch, batch)

    assert result.exit_code == 0, result.output
    assert isinstance(called["ledger"], TrialLedger)
    assert called["ledger"].path == repo_root / "trials" / "ledger.duckdb"
    assert called["repo_root"] == repo_root
    assert not repo_root.exists()


@pytest.mark.usefixtures("_restore_ledger_path")
def test_stage_a_ledger_path_from_set(
    tmp_path: Path, preflight_lake: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An absolute trials.ledger_path given through --set is the ledger's path as is."""
    repo_root = tmp_path / "repo"
    ledger_path = tmp_path / "elsewhere" / "ledger.duckdb"
    monkeypatch.setattr(cli, "_REPO_ROOT", repo_root)
    result, _, batch = _invoke(
        _args(tmp_path / "store", "--set", f"trials.ledger_path={ledger_path}")
    )
    called = _bound(run_batch, batch)

    assert result.exit_code == 0, result.output
    assert called["ledger"].path == ledger_path
    assert called["repo_root"] == repo_root
