# test_cli_firm_record.py

"""AT-125: `agentic-trading-firm-record` builds the registered firm (agentic-firm-v1: the
default roster without `FIRM_V1_EXCLUDED_ANALYST_IDS`, over `FIRM_V1_UNIVERSE`), loads its
prices aligned on `stage_a.session_calendar` with the stage (a) warm-up inside one lake
session cut at `stage_a.data_end`, runs `run_firm_record` and prints its result, which holds no
performance figure."""

import importlib.util
import inspect
import shutil
import sys
import tomllib
from collections.abc import Callable, Iterator
from datetime import date, datetime, timedelta
from importlib import import_module
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from click.testing import Result
from typer.testing import CliRunner

from agentic_trading import cli
from agentic_trading.agents.commander import Commander
from agentic_trading.agents.roster import (
    FIRM_V1_EQUITY_SECURITY_IDS,
    FIRM_V1_EXCLUDED_ANALYST_IDS,
    FIRM_V1_UNIVERSE,
    build_default_analysts,
    build_default_managers,
)
from agentic_trading.agents.schemas import ShortWarmup
from agentic_trading.backtest.preflight import (
    PriceCoverage,
    PriceCoverageError,
    check_price_coverage,
)
from agentic_trading.backtest.resume_fingerprint import roster_agents
from agentic_trading.cli import firm_record_app
from agentic_trading.config import settings, temporary_overrides
from agentic_trading.data.loader import load_price_history_with_fallback
from agentic_trading.data.loaders import lakehouse_loader as ll
from agentic_trading.firm_record.runner import FirmRecordResult, run_firm_record
from agentic_trading.firm_record.spec import build_firm_spec
from agentic_trading.scoring.weighting import Hedge, ShrinkageWeight
from agentic_trading.stage_a.spec import FirmSpecRecord, firm_spec_hash
from agentic_trading.trials.ledger import TrialLedger

runner = CliRunner()
_PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_MARKET = pd.DataFrame(
    {("SPY", "close"): [100.0, 101.0]}, index=pd.DatetimeIndex(["2010-01-04", "2010-01-05"])
)
# A BIS policy-rate frame for every currency of the universe's spot FX pairs.
_POLICY_RATES = pd.DataFrame(
    {currency: [0.01] for currency in ("AUD", "EUR", "GBP", "JPY", "USD")},
    index=pd.DatetimeIndex(["2009-12-31"], name="date"),
)
_RESULT = FirmRecordResult(
    spec_hash="0" * 64,
    days_written=2,
    last_completed=date(2010, 1, 5),
    complete=True,
    counted=False,
)


def _invoke(
    args: list[str], coverage: MagicMock | None = None, record_result: FirmRecordResult = _RESULT
) -> tuple[Result, MagicMock, MagicMock, MagicMock]:
    """Runs the command with the price load, the risk-free rate load and the record mocked
    (the record returning `record_result`, the policy rates by `_policy_rates`), and the price
    coverage preflight replaced by `coverage` (a passing mock by default)."""
    with (
        patch(
            "agentic_trading.backtest.engine_inputs.load_price_history_with_fallback",
            return_value=(_MARKET, "lakehouse data"),
        ) as load,
        patch(
            "agentic_trading.backtest.engine_inputs.load_risk_free_rate_series", return_value=None
        ) as rates,
        patch("agentic_trading.cli.run_firm_record", return_value=record_result) as record,
        patch("agentic_trading.cli.check_price_coverage", coverage or MagicMock()),
    ):
        result = runner.invoke(firm_record_app, args)
    return result, load, rates, record


@pytest.fixture(autouse=True)
def _policy_rates() -> Iterator[MagicMock]:
    """Stubs the BIS policy-rate load (no lake in unit tests) with `_POLICY_RATES`."""
    with patch(
        "agentic_trading.backtest.engine_inputs.load_fx_policy_rates", return_value=_POLICY_RATES
    ) as policy:
        yield policy


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
    assert called["inputs"].market_data is _MARKET
    assert list(called["inputs"].assets) == list(FIRM_V1_UNIVERSE)
    assert called["max_days"] == 20
    assert called["store_dir"] == tmp_path
    assert isinstance(called["ledger"], TrialLedger)
    assert called["repo_root"] == _REPO_ROOT
    assert called["lake_time"].isoformat() == "2026-11-06T00:00:00+00:00"
    assert FirmRecordResult.model_validate_json(result.stdout[result.stdout.index("{") :]) == (
        _RESULT
    )


_ARGS = [
    "--start",
    "2010-01-04",
    "--end",
    "2010-01-05",
    "--lake-time",
    "2026-11-06T00:00:00Z",
]


def test_restart_threshold_defaults_to_4096_mb(tmp_path: Path, preflight_lake: Any) -> None:
    """Without the flag the record restarts above 4096 MB; 0 never restarts; neither result
    restarts the process when the pass did not stop for memory."""
    with patch("agentic_trading.cli._restart_record") as restart:
        default, _, _, record = _invoke([*_ARGS, "--store-dir", str(tmp_path)])
        assert default.exit_code == 0, default.output
        assert _bound(run_firm_record, record)["restart_above_rss_mb"] == 4096
        off, _, _, record = _invoke(
            [*_ARGS, "--store-dir", str(tmp_path), "--restart-above-rss-mb", "0"]
        )
        assert off.exit_code == 0, off.output
        assert _bound(run_firm_record, record)["restart_above_rss_mb"] is None
    restart.assert_not_called()


@pytest.mark.parametrize(("max_days", "extra"), [(None, []), ("20", ["--max-days", "17"])])
def test_pass_paused_for_memory_restarts_pinned(
    tmp_path: Path, max_days: str | None, extra: list[str], preflight_lake: Any
) -> None:
    """A pass that stopped for memory prints its result, then restarts the command pinned to
    the lake time it read and, with --max-days, to the days it has left."""
    paused = _RESULT.model_copy(
        update={"days_written": 3, "complete": False, "paused_for_memory": True}
    )
    args = [*_ARGS, "--store-dir", str(tmp_path), "--restart-above-rss-mb", "1500"]
    args += [] if max_days is None else ["--max-days", max_days]
    with (
        patch("agentic_trading.cli._require_unchanged_source"),
        patch("agentic_trading.cli._restart_record") as restart,
    ):
        result, _, _, record = _invoke(args, record_result=paused)

    assert result.exit_code == 0, result.output
    assert _bound(run_firm_record, record)["restart_above_rss_mb"] == 1500
    assert FirmRecordResult.model_validate_json(result.stdout[result.stdout.index("{") :]) == (
        paused
    )
    restart.assert_called_once_with(["--lake-time", "2026-11-06T00:00:00+00:00", *extra])


def test_restart_pins_the_lake_time_the_session_read(tmp_path: Path, preflight_lake: Any) -> None:
    """Without --lake-time the restart is pinned to the lake time the session chose, which
    parses back to the same instant."""
    paused = _RESULT.model_copy(
        update={"days_written": 3, "complete": False, "paused_for_memory": True}
    )
    args = ["--start", "2010-01-04", "--end", "2010-01-05", "--store-dir", str(tmp_path)]
    with (
        patch("agentic_trading.cli._require_unchanged_source"),
        patch("agentic_trading.cli._restart_record") as restart,
    ):
        result, _, _, record = _invoke(args, record_result=paused)

    assert result.exit_code == 0, result.output
    read = _bound(run_firm_record, record)["lake_time"]
    flag, pinned = restart.call_args.args[0]
    assert flag == "--lake-time"
    assert datetime.fromisoformat(pinned) == read


def test_later_options_override_earlier_ones(tmp_path: Path, preflight_lake: Any) -> None:
    """A repeated --max-days or --lake-time takes its last value, which a restart relies on."""
    args = [*_ARGS, "--store-dir", str(tmp_path), "--max-days", "20", "--max-days", "17"]
    args += ["--lake-time", "2026-11-05T00:00:00+00:00"]
    result, _, _, record = _invoke(args)
    called = _bound(run_firm_record, record)

    assert result.exit_code == 0, result.output
    assert called["max_days"] == 17
    assert called["lake_time"].isoformat() == "2026-11-05T00:00:00+00:00"


def test_restart_record_execs_the_same_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """The restart replaces the process with the interpreter running the original arguments
    followed by the extra ones."""
    monkeypatch.setattr(sys, "orig_argv", ["python3", "/venv/bin/record", "--start", "x"])
    with patch("agentic_trading.cli.os.execv") as execv:
        cli._restart_record(["--max-days", "5"])

    execv.assert_called_once_with(
        sys.executable,
        [sys.executable, "/venv/bin/record", "--start", "x", "--max-days", "5"],
    )


def test_restart_record_refuses_with_a_lake_request_in_flight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The restart raises, without exec, while a lake request is in flight, and execs when none
    is."""
    monkeypatch.setattr(sys, "orig_argv", ["python3", "/venv/bin/record"])
    monkeypatch.setattr(cli, "lake_requests_in_flight", lambda: 1)
    with patch("agentic_trading.cli.os.execv") as execv:
        with pytest.raises(RuntimeError, match="1 lake request"):
            cli._restart_record([])
    execv.assert_not_called()
    monkeypatch.setattr(cli, "lake_requests_in_flight", lambda: 0)
    with patch("agentic_trading.cli.os.execv") as execv:
        cli._restart_record([])
    execv.assert_called_once()


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
    """The universe's price coverage is checked on the loaded frame over the warm-up and the
    window, every firm equity on every session bar the known lake gaps and crypto exempt at
    the start; a short warm-up reaches run_firm_record, and a failure stops the command before
    run_firm_record writes anything."""
    args = ["--start", "2010-01-04", "--end", "2010-01-05", "--store-dir", str(tmp_path)]
    short = ShortWarmup(instrument="V", first_bar=date(2008, 3, 19), warmup_sessions=450)
    entries = [
        PriceCoverage(
            ticker=ticker,
            first_bar=date(2007, 1, 3),
            last_bar=date(2010, 1, 5),
            longest_gap_sessions=0,
            short_warmup=short if ticker == "V" else None,
            covers_run=True,
        )
        for ticker in FIRM_V1_UNIVERSE
    ]
    passing = MagicMock(return_value=entries)
    result, _, _, record = _invoke(args, passing)
    checked = _bound(check_price_coverage, passing)

    assert result.exit_code == 0, result.output
    assert checked["market_data"] is _MARKET
    assert list(checked["members"]) == list(FIRM_V1_UNIVERSE)
    warmup_start = date(2010, 1, 4) - timedelta(days=settings.stage_a.warmup_days)
    assert (checked["warmup_start"], checked["window_start"], checked["window_end"]) == (
        warmup_start,
        date(2010, 1, 4),
        date(2010, 1, 5),
    )
    assert checked["calendar"] == settings.stage_a.session_calendar
    assert checked["complete"] is FIRM_V1_EQUITY_SECURITY_IDS
    assert checked["known_gaps"] == settings.firm.known_price_gaps == []
    assert list(checked["start_exempt"]) == ["BTC-USD", "ETH-USD"]
    assert _bound(run_firm_record, record)["short_warmup"] == [short]

    failing = MagicMock(side_effect=PriceCoverageError("MON missing 2 session(s)"))
    result, _, _, record = _invoke(args, failing)

    assert isinstance(result.exception, PriceCoverageError)
    record.assert_not_called()


def test_firm_record_passes_the_policy_rates(
    tmp_path: Path, preflight_lake: Any, _policy_rates: MagicMock
) -> None:
    """The BIS policy rates of the universe's spot FX pairs are read for --start to --end inside
    the lake session, under the declared window store, which covers the loader's look-back, and
    reach run_firm_record in its engine inputs."""
    windows: list[ll.LoaderWindow | None] = []

    def read(*_args: Any) -> pd.DataFrame:
        windows.append(ll.loader_window())
        return _POLICY_RATES

    _policy_rates.side_effect = read
    args = ["--start", "2010-01-04", "--end", "2010-01-05", "--store-dir", str(tmp_path)]
    result, _, _, record = _invoke(args)

    assert result.exit_code == 0, result.output
    _policy_rates.assert_called_once_with(
        list(FIRM_V1_UNIVERSE), date(2010, 1, 4), date(2010, 1, 5)
    )
    [window] = windows
    assert window is not None
    assert window.start <= date(2010, 1, 4) - timedelta(days=ll.POLICY_RATE_LOOKBACK_DAYS)
    assert window.end == date(2010, 1, 5)
    assert _bound(run_firm_record, record)["inputs"].policy_rates is _POLICY_RATES


_PROBE_MODULE = "agentic_trading._restart_source_probe"


def _paused_trial(store_dir: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Any]:
    """Loads a probe `agentic_trading` module from a file under `store_dir`, writes the
    spec.json of the registered firm's trial (built as the command builds it, the probe among
    its loaded modules) under `store_dir`, and returns the probe's source file and a result of
    that trial paused for memory."""
    probe = store_dir / "probe_source" / "probe.py"
    probe.parent.mkdir()
    probe.write_text("VALUE = 1\n")
    found = importlib.util.spec_from_file_location(_PROBE_MODULE, probe)
    assert found is not None
    monkeypatch.setitem(sys.modules, _PROBE_MODULE, importlib.util.module_from_spec(found))
    analysts = build_default_analysts(excluded_analyst_ids=FIRM_V1_EXCLUDED_ANALYST_IDS)
    commander = Commander(
        managers=build_default_managers(analysts), weighting_strategy=ShrinkageWeight(Hedge())
    )
    spec = build_firm_spec(
        commander,
        list(FIRM_V1_UNIVERSE),
        date(2010, 1, 4),
        date(2010, 1, 5),
        datetime.fromisoformat("2026-11-06T00:00:00+00:00"),
    )
    record = FirmSpecRecord(
        spec=spec,
        spec_hash=firm_spec_hash(spec),
        lake_time=spec.lake_time,
        lake_version="v",
        quantcore_version="q",
    )
    assert _PROBE_MODULE in spec.module_hashes
    trial_dir = store_dir / record.spec_hash
    trial_dir.mkdir()
    (trial_dir / "spec.json").write_text(record.model_dump_json())
    paused = _RESULT.model_copy(
        update={
            "spec_hash": record.spec_hash,
            "days_written": 3,
            "complete": False,
            "paused_for_memory": True,
        }
    )
    return probe, paused


def test_restart_proceeds_while_the_source_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, preflight_lake: Any
) -> None:
    """A pass paused for memory restarts when the source on disk still gives the running
    trial's spec hash."""
    _, paused = _paused_trial(tmp_path, monkeypatch)
    with patch("agentic_trading.cli._restart_record") as restart:
        result, _, _, _ = _invoke([*_ARGS, "--store-dir", str(tmp_path)], record_result=paused)

    assert result.exit_code == 0, result.output
    restart.assert_called_once()


def test_restart_refuses_when_a_module_changed_on_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, preflight_lake: Any
) -> None:
    """A module whose file changed on disk since the pass began, though its loaded code did
    not, would give the restarted process another spec hash, so the restart is refused."""
    probe, paused = _paused_trial(tmp_path, monkeypatch)
    probe.write_text("VALUE = 2\n")
    with patch("agentic_trading.cli._restart_record") as restart:
        result, _, _, _ = _invoke([*_ARGS, "--store-dir", str(tmp_path)], record_result=paused)

    assert result.exit_code != 0
    assert isinstance(result.exception, RuntimeError)
    assert _PROBE_MODULE in str(result.exception)
    assert paused.spec_hash in str(result.exception)
    restart.assert_not_called()


def test_restart_refuses_without_the_running_trials_spec(
    tmp_path: Path, preflight_lake: Any
) -> None:
    """A pass paused for memory whose trial has no readable spec.json under --store-dir is not
    restarted."""
    paused = _RESULT.model_copy(
        update={"days_written": 3, "complete": False, "paused_for_memory": True}
    )
    with patch("agentic_trading.cli._restart_record") as restart:
        result, _, _, _ = _invoke([*_ARGS, "--store-dir", str(tmp_path)], record_result=paused)

    assert isinstance(result.exception, RuntimeError)
    assert "spec.json" in str(result.exception)
    restart.assert_not_called()


@pytest.mark.parametrize(
    ("value", "exit_code"),
    [((settings.stage_a.data_end + timedelta(days=1)).isoformat(), 2), ("2010-06-30", 0)],
)
def test_set_cannot_move_data_end_past_the_holdout(
    tmp_path: Path, value: str, exit_code: int, preflight_lake: Any
) -> None:
    """--set stage_a.data_end later than the locked holdout cut is a usage error before any
    read; an earlier cut is accepted and the record runs."""
    args = [*_ARGS, "--store-dir", str(tmp_path), "--set", f"stage_a.data_end={value}"]
    with temporary_overrides({"stage_a.data_end": settings.stage_a.data_end}):
        result, load, _, record = _invoke(args)

    assert result.exit_code == exit_code, result.output
    assert load.called is (exit_code == 0)
    assert record.called is (exit_code == 0)


_ON_AC = (
    "Now drawing from 'AC Power'\n"
    " -InternalBattery-0 (id=20447331)\t62%; charging; 2:14 remaining present: true\n"
)
_ON_BATTERY = (
    "Now drawing from 'Battery Power'\n"
    " -InternalBattery-0 (id=20447331)\t80%; discharging; 5:10 remaining present: true\n"
)


def _power_settings(low_power_mode: int) -> str:
    """`pmset -g` output with Low Power Mode at `low_power_mode`."""
    return (
        "System-wide power settings:\nCurrently in use:\n standby              1\n"
        f" lowpowermode         {low_power_mode}\n womp                 1\n"
    )


# The real power reader, kept before `tests/conftest.py` stubs it for each test.
_READ_POWER_STATE = getattr(cli, "read_power_state", None)


@pytest.fixture
def pmset(monkeypatch: pytest.MonkeyPatch) -> Callable[[str | None, str | None], None]:
    """Restores the real power reader over a stubbed `pmset`: `pmset -g batt` prints the first
    argument and `pmset -g` the second; `None` for both is a host without pmset."""

    def stub(battery_report: str | None, power_settings: str | None) -> None:
        outputs = {("-g", "batt"): battery_report, ("-g",): power_settings}
        monkeypatch.setattr(cli, "read_power_state", _READ_POWER_STATE, raising=False)
        monkeypatch.setattr(cli, "_run_pmset", lambda *args: outputs[args], raising=False)

    return stub


@pytest.mark.parametrize(
    ("battery_report", "power_settings", "extra", "refused"),
    [
        (_ON_BATTERY, _power_settings(0), [], True),
        (_ON_AC, _power_settings(1), [], True),
        ("no power source here\n", _power_settings(0), [], True),
        (_ON_AC, "System-wide power settings:\n standby 1\n", [], True),
        (_ON_AC, _power_settings(0), [], False),
        (None, None, [], False),
        (_ON_BATTERY, _power_settings(1), ["--allow-battery"], False),
    ],
    ids=[
        "battery",
        "low-power",
        "unparseable-source",
        "no-low-power-line",
        "ac",
        "no-pmset",
        "allow-battery",
    ],
)
def test_power_preflight(
    tmp_path: Path,
    pmset: Callable[[str | None, str | None], None],
    battery_report: str | None,
    power_settings: str | None,
    extra: list[str],
    refused: bool,
    preflight_lake: Any,
) -> None:
    """The record refuses to start on battery, in Low Power Mode or on pmset output it cannot
    read, naming --allow-battery and reading nothing; on AC power, on a host without pmset or
    with --allow-battery it runs."""
    pmset(battery_report, power_settings)
    result, load, _, record = _invoke([*_ARGS, "--store-dir", str(tmp_path), *extra])

    if refused:
        assert result.exit_code == 1, result.output
        assert "--allow-battery" in result.output
        load.assert_not_called()
        record.assert_not_called()
    else:
        assert result.exit_code == 0, result.output
        record.assert_called_once()


def test_power_preflight_refuses_when_pmset_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, preflight_lake: Any
) -> None:
    """An installed pmset that exits non-zero refuses the record."""
    monkeypatch.setattr(cli, "read_power_state", _READ_POWER_STATE, raising=False)
    monkeypatch.setattr(cli.shutil, "which", lambda name: f"/usr/bin/{name}")
    failed = MagicMock(return_value=MagicMock(returncode=1, stdout="", stderr="pmset: boom"))
    monkeypatch.setattr(cli.subprocess, "run", failed)
    result, _, _, record = _invoke([*_ARGS, "--store-dir", str(tmp_path)])

    assert result.exit_code == 1, result.output
    assert "pmset: boom" in result.output
    record.assert_not_called()


def test_run_pmset_without_pmset_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Where pmset is not installed (Linux, CI) there is no output to read, and nothing runs."""
    monkeypatch.setattr(cli.shutil, "which", lambda _name: None)
    ran = MagicMock()
    monkeypatch.setattr(cli.subprocess, "run", ran)

    assert cli._run_pmset("-g", "batt") is None
    ran.assert_not_called()


_PACKAGE_DIR = Path(cli.__file__).resolve().parent


@pytest.mark.parametrize(
    ("change", "named"),
    [
        (None, None),
        (lambda package: (package / "planted.py").write_text("VALUE = 1\n"), "planted.py"),
        (
            lambda package: (package / "cli.py").write_text(
                (package / "cli.py").read_text() + "\nimport agentic_trading.planted\n"
            ),
            "cli.py",
        ),
        (lambda package: (package / "gui" / "dashboard.py").unlink(), "dashboard.py"),
    ],
    ids=["unchanged", "added", "cli-edited", "removed"],
)
def test_restart_refuses_when_a_package_file_changed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: Callable[[Path], Any] | None,
    named: str | None,
    preflight_lake: Any,
) -> None:
    """A .py file under the package added, edited (cli.py among them, which the spec does not
    hash) or removed between the command's start and a restart refuses the restart, naming the
    file; an unchanged package restarts. The package is a copy, changed during the pass."""
    package = tmp_path / "package" / "agentic_trading"
    shutil.copytree(_PACKAGE_DIR, package, ignore=shutil.ignore_patterns("__pycache__"))
    monkeypatch.setattr(cli, "_PACKAGE_DIR", package, raising=False)

    def coverage(*_args: Any, **_kwargs: Any) -> list[PriceCoverage]:
        if change is not None:
            change(package)
        return []

    paused = _RESULT.model_copy(
        update={"days_written": 3, "complete": False, "paused_for_memory": True}
    )
    store = tmp_path / "store"
    with (
        patch("agentic_trading.cli._require_unchanged_source"),
        patch("agentic_trading.cli._restart_record") as restart,
    ):
        result, _, _, _ = _invoke(
            [*_ARGS, "--store-dir", str(store)], MagicMock(side_effect=coverage), paused
        )

    if named is None:
        assert result.exit_code == 0, result.output
        restart.assert_called_once()
    else:
        assert isinstance(result.exception, RuntimeError), result.output
        assert named in str(result.exception)
        restart.assert_not_called()
