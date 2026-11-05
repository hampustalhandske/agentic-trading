# test_cli_stage_a_inputs.py

"""The `agentic-trading-stage-a` command loads `settings.stage_a.warmup_days` of history before
the window, read at call time, and refuses an analyst in `stage_a.runner.EXCLUDED_ANALYSTS` as a
usage error before any lake read."""

import inspect
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from click.testing import Result
from typer.testing import CliRunner

from agentic_trading.cli import stage_a_app
from agentic_trading.config import settings, temporary_overrides
from agentic_trading.data.loader import load_price_history_with_fallback
from agentic_trading.stage_a.runner import EXCLUDED_ANALYSTS, BatchResult

runner = CliRunner()

_UNIVERSE = ("SHY", "IEF")
_INDEX = pd.bdate_range("2010-01-04", "2010-01-08", name="date")
_MARKET = pd.DataFrame(
    {(ticker, field): 100.0 for ticker in _UNIVERSE for field in ("open", "close")},
    index=_INDEX,
)
_RESULT = BatchResult.model_validate(
    {
        "analysts": {
            "cir_rate_regime_analyst": {
                "spec_hash": "0" * 64,
                "days_written": 5,
                "last_completed": "2010-01-08",
            }
        },
        "stopped_for_resources": False,
    }
)


def _args(store_dir: Path, *analysts: str) -> list[str]:
    tokens = [token for agent_id in analysts for token in ("--analyst", agent_id)]
    tokens += [token for ticker in _UNIVERSE for token in ("--universe", ticker)]
    return [*tokens, "--store-dir", str(store_dir), "--window-end", "2010-01-08"]


def _invoke(args: list[str]) -> tuple[Result, MagicMock, MagicMock]:
    with (
        patch(
            "agentic_trading.cli.load_price_history_with_fallback",
            return_value=(_MARKET, "lakehouse data"),
        ) as load,
        patch("agentic_trading.cli.run_batch", return_value=_RESULT) as batch,
    ):
        result = runner.invoke(stage_a_app, args)
    return result, load, batch


def _bound(function: Callable[..., Any], mock: MagicMock) -> dict[str, Any]:
    mock.assert_called_once()
    bound = inspect.signature(function).bind(*mock.call_args.args, **mock.call_args.kwargs)
    bound.apply_defaults()
    return bound.arguments


def test_stage_a_loads_the_warmup_from_settings(tmp_path: Path, preflight_lake: Any) -> None:
    """The price history starts stage_a.warmup_days before the window, read at call time."""
    args = _args(tmp_path, "cir_rate_regime_analyst")
    default, default_load, _ = _invoke(args)
    with temporary_overrides({"stage_a.warmup_days": 400}):
        overridden, overridden_load, _ = _invoke(args)

    assert default.exit_code == 0, default.output
    assert overridden.exit_code == 0, overridden.output
    loaded = _bound(load_price_history_with_fallback, default_load)
    moved = _bound(load_price_history_with_fallback, overridden_load)
    assert loaded["warmup_days"] == settings.stage_a.warmup_days == 1100
    assert moved["warmup_days"] == 400
    assert loaded["start"] == moved["start"] == date(2010, 1, 4)


@pytest.mark.parametrize("excluded", sorted(EXCLUDED_ANALYSTS))
def test_stage_a_refuses_an_excluded_analyst(
    tmp_path: Path, preflight_lake: Any, excluded: str
) -> None:
    """An excluded analyst is a usage error (exit 2) naming it; nothing is read or run."""
    result, load, batch = _invoke(_args(tmp_path, "cir_rate_regime_analyst", excluded))

    assert result.exit_code == 2, result.output
    assert excluded in result.output
    load.assert_not_called()
    batch.assert_not_called()
    assert preflight_lake.calls == []
