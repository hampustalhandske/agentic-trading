# test_cli_lake_time.py

"""AT-106: `run` and `experiment` accept `--lake-time`, do all their lake reads inside one
session, and record its T in the provenance JSON; `experiment --resume` reuses a checkpoint's T."""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from agentic_trading import provenance
from agentic_trading.agents.schemas import BacktestResult, PortfolioDecision
from agentic_trading.backtest.engine import BacktestEngine
from agentic_trading.backtest.portfolio import Portfolio
from agentic_trading.backtest.preflight import PreflightReport
from agentic_trading.cli import app, experiment_app
from agentic_trading.data.loaders import lakehouse_loader as ll
from agentic_trading.scoring.track_record import TrackRecord

runner = CliRunner()

_T = datetime(2024, 6, 1, 12, tzinfo=UTC)
_DEFAULT_T = datetime(2024, 5, 1, 12, 0, 1, tzinfo=UTC)
_RUN_ARGS = ["AAPL", "--start-date", "2024-01-02", "--end-date", "2024-01-05", "--yes"]
_EXPERIMENT_ARGS = [
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
_INDEX = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])


@pytest.fixture
def cli_lake(fake_lake: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    """A fake lake with one FRED write and both adjustment tables rebuilt, and the real
    `lake_provenance` wired into `build_run_provenance`."""
    fred = pd.DataFrame(
        {
            "series_id": ["DGS3MO"],
            "date": ["2024-04-30"],
            "realtime_start": ["2024-05-01"],
            "value": [5.0],
        }
    )
    fake_lake.add(
        "fred_vintages", fred, ("date", "realtime_start"), scraped_at="2024-05-01T12:00:00Z"
    )
    factors = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "effective_date": ["2024-01-02"],
            "cum_factor": [1.0],
            "security_id": [1],
        }
    )
    fake_lake.add(
        "equity_adjustment_factors",
        factors,
        ("effective_date",),
        scraped_at="2024-04-01T00:00:00Z",
        rebuilt_whole=True,
    )
    splits = pd.DataFrame(
        {"ticker": ["AAA"], "ex_date": ["2024-01-02"], "ratio": [0.5], "in_close_basis": [True]}
    )
    fake_lake.add(
        "yahoo_split_close_basis",
        splits,
        ("ex_date",),
        scraped_at="2024-04-01T00:00:00Z",
        rebuilt_whole=True,
    )
    monkeypatch.setattr(provenance, "lake_provenance", lambda: ll.lake_provenance())
    return fake_lake


def _result() -> BacktestResult:
    return BacktestResult(
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


def _payload(stdout: str) -> dict[str, Any]:
    return json.loads(stdout[stdout.index("{") :])  # type: ignore[no-any-return]


@contextmanager
def _stubbed_cli(seen: dict[str, list[Any]]) -> Iterator[MagicMock]:
    """Stubs the CLI's data and engine collaborators; each records `current_lake_time()` under
    its name in `seen` when called. Yields the engine class mock."""
    frame = pd.DataFrame({("^GSPC", "close"): [100.0, 101.0, 102.0, 103.0]}, index=_INDEX)
    frame.columns = pd.MultiIndex.from_tuples(frame.columns)

    def record(name: str, value: Any) -> Any:
        seen.setdefault(name, []).append(ll.current_lake_time())
        return value

    with (
        patch(
            "agentic_trading.cli.check_data_availability",
            side_effect=lambda *a, **k: record(
                "preflight", PreflightReport(checked_at=date(2024, 1, 2), results=[])
            ),
        ),
        patch(
            "agentic_trading.cli.load_price_history_with_fallback",
            side_effect=lambda *a, **k: record("prices", (frame, "lakehouse data")),
        ),
        patch(
            "agentic_trading.cli.load_risk_free_rate_series",
            side_effect=lambda *a, **k: record("risk_free", pd.Series(0.0, index=_INDEX)),
        ),
        patch("agentic_trading.cli.BacktestEngine") as engine_cls,
    ):
        engine_cls.return_value.run.side_effect = lambda *a, **k: record("engine", _result())
        yield engine_cls


@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        (["--lake-time", "2024-06-01T12:00:00+00:00"], _T),
        (["--lake-time", "2024-06-01T14:00:00+02:00"], _T),
        ([], _DEFAULT_T),
    ],
    ids=["utc", "offset", "default"],
)
def test_run_reads_inside_one_session_and_records_its_time(
    cli_lake: Any, extra: list[str], expected: datetime
) -> None:
    """Every stage of `run` sees the same open session T and the provenance JSON carries it."""
    seen: dict[str, list[Any]] = {}
    with _stubbed_cli(seen):
        result = runner.invoke(app, [*_RUN_ARGS, *extra])

    assert result.exit_code == 0, result.output
    assert set(seen) == {"preflight", "prices", "risk_free", "engine"}
    assert all(stamp == expected for stamps in seen.values() for stamp in stamps)
    lake = _payload(result.stdout)["provenance"]["lake"]
    assert datetime.fromisoformat(lake["lake_time"]) == expected
    assert set(lake["rebuilt_whole_builds"]) >= {
        "equity_adjustment_factors",
        "yahoo_split_close_basis",
    }
    assert ll.current_lake_time() is None


def test_run_rejects_a_naive_lake_time(cli_lake: Any) -> None:
    """A `--lake-time` without a UTC offset is a usage error and runs nothing."""
    seen: dict[str, list[Any]] = {}
    with _stubbed_cli(seen) as engine_cls:
        result = runner.invoke(app, [*_RUN_ARGS, "--lake-time", "2024-06-01T12:00:00"])

    assert result.exit_code != 0
    engine_cls.return_value.run.assert_not_called()


def test_experiment_uses_one_lake_time_for_calibration_and_test(cli_lake: Any) -> None:
    """Calibration, test and baseline runs all see the explicit T, and the provenance records it."""
    seen: dict[str, list[Any]] = {}
    with _stubbed_cli(seen):
        result = runner.invoke(
            experiment_app, [*_EXPERIMENT_ARGS, "--lake-time", "2024-06-01T12:00:00+00:00"]
        )

    assert result.exit_code == 0, result.output
    assert len(seen["engine"]) >= 2
    assert {stamp for stamps in seen.values() for stamp in stamps} == {_T}
    lake = _payload(result.stdout)["provenance"]["lake"]
    assert datetime.fromisoformat(lake["lake_time"]) == _T


def _rebuild_splits(lake: Any) -> BacktestResult:
    splits = pd.DataFrame(
        {"ticker": ["AAA"], "ex_date": ["2024-01-02"], "ratio": [0.25], "in_close_basis": [True]}
    )
    lake.add(
        "yahoo_split_close_basis",
        splits,
        ("ex_date",),
        scraped_at="2024-06-01T13:00:00Z",
        rebuilt_whole=True,
    )
    return _result()


@pytest.mark.parametrize("command", ["run", "experiment"])
def test_a_rebuild_during_the_run_fails_it_before_any_result_is_printed(
    cli_lake: Any, command: str
) -> None:
    """A rebuilt-whole table rebuilt while the engine runs fails the command, naming the table,
    and no result JSON reaches stdout."""
    seen: dict[str, list[Any]] = {}
    with _stubbed_cli(seen) as engine_cls:
        engine_cls.return_value.run.side_effect = lambda *a, **k: _rebuild_splits(cli_lake)
        if command == "run":
            result = runner.invoke(app, [*_RUN_ARGS, "--lake-time", "2024-06-01T12:00:00Z"])
        else:
            result = runner.invoke(
                experiment_app, [*_EXPERIMENT_ARGS, "--lake-time", "2024-06-01T12:00:00Z"]
            )

    assert result.exit_code != 0
    assert isinstance(result.exception, ll.LakeTimeError)
    assert "yahoo_split_close_basis" in str(result.exception)
    assert '"result"' not in result.stdout
    assert ll.current_lake_time() is None


class _Commander:
    def decide(self, market_data, as_of_date, scoreboard, instrument_returns=None):
        return [
            PortfolioDecision(
                asset="AAPL",
                as_of_date=pd.Timestamp(as_of_date).date(),
                target_weight=0.5,
                contributing_signals=[],
            )
        ]

    def idle_pod_share(self) -> float:
        return 0.3


def _write_checkpoint(path: Path, lake_time: datetime) -> None:
    close = pd.Series([100.0, 101.0, 102.0, 103.0], index=_INDEX)
    frame = pd.DataFrame(
        {
            ("AAPL", "open"): close,
            ("AAPL", "close"): close,
            ("AAPL", "log_return"): np.log(close / close.shift(1)),
        }
    )
    engine = BacktestEngine(
        commander=_Commander(),  # type: ignore[arg-type]
        portfolio=Portfolio(cash=100_000.0, positions={}, commission_rate=0.0, slippage_bps=0.0),
        track_record=TrackRecord(),
        data_loader=lambda _assets, _start, _end: frame,
        start_date=_INDEX[0].date(),
        end_date=_INDEX[-1].date(),
        assets=["AAPL"],
        execution_lag_days=1,
        checkpoint_path=path,
    )
    with ll.lake_session(lake_time):
        engine.run()


def test_experiment_resume_without_lake_time_reuses_the_checkpoint_time(
    cli_lake: Any, tmp_path: Path
) -> None:
    """With no `--lake-time`, `experiment --resume` reads at its checkpoint's T."""
    checkpoint_time = datetime(2024, 5, 20, tzinfo=UTC)
    checkpoint = tmp_path / "calibration.db"
    _write_checkpoint(checkpoint, checkpoint_time)
    seen: dict[str, list[Any]] = {}

    with _stubbed_cli(seen):
        result = runner.invoke(
            experiment_app,
            [*_EXPERIMENT_ARGS, "--calibration-checkpoint", str(checkpoint), "--resume"],
        )

    assert result.exit_code == 0, result.output
    assert {stamp for stamps in seen.values() for stamp in stamps} == {checkpoint_time}
    lake = _payload(result.stdout)["provenance"]["lake"]
    assert datetime.fromisoformat(lake["lake_time"]) == checkpoint_time
