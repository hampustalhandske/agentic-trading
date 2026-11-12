# test_cli_firm_record_lake_version.py

"""`agentic-trading-firm-record` pins its lake version the way it pins its lake time: the
record writes the pinned `<code>.<builds>` pair (from --lake-version, the trial's spec.json on
a resume, or the first version the lake answers with), every lake response must carry the
pair's code part, and a restart after a pass that stopped for memory passes the pair along.
A rebuild between the pause and the restart leaves the restarted record on the same store,
ending with the rows of an unbroken record.

The command runs the small lake-reading firm of `firm_record/test_runner_lake_writes.py` in
place of the registered one: `run_firm_record` is wrapped to swap in that firm, its universe
and its inputs, and to read resident memory as `_RSS_MB`; everything else (the lake session,
its pinned version, the store, the restart arguments) is the command's own."""

import json
from collections.abc import Callable, Iterator
from datetime import date, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pyarrow.parquet as pq
import pytest
import yaml
from click.testing import Result
from typer.testing import CliRunner

from agentic_trading import cli
from agentic_trading.agents.commander import Commander
from agentic_trading.agents.roster import (
    FIRM_V1_EXCLUDED_ANALYST_IDS,
    FIRM_V1_UNIVERSE,
    build_default_analysts,
    build_default_managers,
)
from agentic_trading.cli import firm_record_app
from agentic_trading.data.loaders import lakehouse_loader as ll
from agentic_trading.firm_record.runner import FirmRecordResult, run_firm_record
from agentic_trading.firm_record.spec import build_firm_spec
from agentic_trading.scoring.weighting import Hedge, ShrinkageWeight
from agentic_trading.stage_a.spec import FirmSpecRecord, firm_spec_hash
from unit.firm_record.test_runner import _UNIVERSE, _frame, _inputs
from unit.firm_record.test_runner_lake_writes import _commander, _seed, _write_after_t

runner = CliRunner()
_LAKE_TIME = "2026-11-06T00:00:00+00:00"
_ARGS = ["--start", "2016-01-04", "--end", "2016-01-22", "--lake-time", _LAKE_TIME]
# Resident memory the wrapped record reads: above the --restart-above-rss-mb the tests pass.
_RSS_MB = 101.0
_SERVED = "fake-code-1.fake-builds-1"


def _small_record(*args: Any, **kwargs: Any) -> FirmRecordResult:
    """`run_firm_record` as the command calls it, on the lake-reading firm over the runner
    tests' universe and inputs, stopping for memory after three days when the command passes
    a threshold. The ledger is left out, so the record is uncounted."""
    _, _, window_start, window_end, store_dir, _ = args
    for name in ("ledger", "repo_root", "short_warmup"):
        kwargs.pop(name)
    return run_firm_record(
        _commander(),
        _UNIVERSE,
        window_start,
        window_end,
        store_dir,
        _inputs(_frame()),
        restart_min_days=3,
        rss_mb=lambda: _RSS_MB,
        **kwargs,
    )


def _invoke(args: list[str], restart: MagicMock) -> Result:
    """Runs the command with `_small_record`, its engine inputs and price coverage stubbed,
    the restart replaced by `restart` and the restart's source check passed."""
    with (
        patch("agentic_trading.cli.load_engine_inputs", return_value=_inputs(_frame())),
        patch("agentic_trading.cli.check_price_coverage", MagicMock()),
        patch("agentic_trading.cli.run_firm_record", side_effect=_small_record),
        patch("agentic_trading.cli._require_unchanged_source"),
        patch("agentic_trading.cli._restart_record", restart),
    ):
        return runner.invoke(firm_record_app, args)


def _record_until_complete(
    args: list[str], between: Callable[[int], None] = lambda _: None
) -> list[list[str]]:
    """Runs the command on `args`, then, each time it restarts (`_restart_record`), runs it
    again on `args` followed by the restart's arguments, as the exec does, until a pass does
    not restart; `between(n)` runs before restart `n` (from 1). Returns each restart's
    arguments."""
    restarts: list[list[str]] = []
    command = list(args)
    while True:
        restart = MagicMock()
        result = _invoke(command, restart)
        assert result.exit_code == 0, result.output
        if not restart.called:
            return restarts
        extra = list(restart.call_args.args[0])
        restarts.append(extra)
        between(len(restarts))
        command = [*args, *extra]
        assert len(restarts) < 10


def _trial(store_dir: Path) -> Path:
    """The store's one trial folder; the lake answer cache beside it is not a trial."""
    (trial,) = [
        path
        for path in store_dir.iterdir()
        if path.is_dir() and path.name != cli.FIRM_RECORD_LAKE_CACHE_DIR
    ]
    return trial


def _decisions(store_dir: Path) -> list[dict[str, Any]]:
    return pq.read_table(_trial(store_dir) / "decisions.parquet").to_pylist()


def _written_versions(store_dir: Path) -> dict[str, Any]:
    """The lake version each output of the store's one trial records: spec.json (the store's
    identity) and strategy.yaml's data params. Both outputs and submission.json also carry the
    pinned lake time, checked here."""
    trial = _trial(store_dir)
    spec = json.loads((trial / "spec.json").read_text())
    params = yaml.safe_load((trial / "strategy" / "strategy.yaml").read_text())["data"]["params"]
    submission = json.loads((trial / "submission.json").read_text())
    assert datetime.fromisoformat(spec["lake_time"]) == datetime.fromisoformat(_LAKE_TIME)
    assert params["lake_time"] == submission["data"]["lake_time"] == "2026-11-06T00:00:00Z"
    return {"spec.json": spec["lake_version"], "strategy.yaml": params["lake_version"]}


@pytest.fixture
def lake(fake_lake: Any) -> Iterator[Any]:
    _seed(fake_lake)
    yield fake_lake


def test_a_restart_across_a_rebuild_resumes_the_same_store_to_the_unbroken_rows(
    tmp_path: Path, lake: Any
) -> None:
    """A record without --lake-version writes the first version the lake answers with; a
    rebuild between its pause and its restart moves the lake's builds part, yet the restart,
    pinned to the written pair, opens the same store and ends with an unbroken record's
    rows."""
    unbroken = tmp_path / "unbroken"
    assert _record_until_complete([*_ARGS, "--store-dir", str(unbroken)]) == []
    paused = tmp_path / "paused"

    def rebuild_before_the_first_restart(restart: int) -> None:
        if restart == 1:
            _write_after_t(lake)

    restarts = _record_until_complete(
        [*_ARGS, "--store-dir", str(paused), "--restart-above-rss-mb", "100"],
        rebuild_before_the_first_restart,
    )

    assert lake.version == "fake-code-1.fake-builds-2"
    assert len(restarts) >= 2
    assert all(
        extra[:4] == ["--lake-time", _LAKE_TIME, "--lake-version", _SERVED] for extra in restarts
    )
    assert _written_versions(paused) == dict.fromkeys(_written_versions(paused), _SERVED)
    assert _decisions(paused) == _decisions(unbroken)


def test_a_builds_only_change_leaves_every_output_on_the_pinned_pair(
    tmp_path: Path, lake: Any
) -> None:
    """With --lake-version, every output carries that pair, not the version the lake answers
    with, even when the lake's builds part changes during the pass."""
    pinned = "fake-code-1.registered-builds"
    store_dir = tmp_path / "store"
    real_seed_write = _write_after_t

    def rebuild_mid_pass(*args: Any, **kwargs: Any) -> FirmRecordResult:
        real_seed_write(lake)
        return _small_record(*args, **kwargs)

    with (
        patch("agentic_trading.cli.load_engine_inputs", return_value=_inputs(_frame())),
        patch("agentic_trading.cli.check_price_coverage", MagicMock()),
        patch("agentic_trading.cli.run_firm_record", side_effect=rebuild_mid_pass),
    ):
        result = runner.invoke(
            firm_record_app,
            [*_ARGS, "--store-dir", str(store_dir), "--lake-version", pinned],
        )

    assert result.exit_code == 0, result.output
    assert lake.version == "fake-code-1.fake-builds-2"
    assert _written_versions(store_dir) == dict.fromkeys(_written_versions(store_dir), pinned)


def test_a_served_code_part_other_than_the_pinned_one_refuses_the_record(
    tmp_path: Path, lake: Any
) -> None:
    """A lake answering with another code part than --lake-version's raises LakeTimeError,
    naming both, and writes no day."""
    store_dir = tmp_path / "store"
    result = _invoke(
        [*_ARGS, "--store-dir", str(store_dir), "--lake-version", "other-code.fake-builds-1"],
        MagicMock(),
    )

    assert isinstance(result.exception, ll.LakeTimeError), result.output
    assert "served code" in str(result.exception)
    assert "other-code.fake-builds-1" in str(result.exception)
    assert not store_dir.exists() or not any(store_dir.rglob("decisions.parquet"))


@pytest.mark.parametrize("value", ["5941cb993106", "a.b.c", "a .b", ".b"])
def test_a_lake_version_not_of_the_pair_form_is_a_usage_error(
    tmp_path: Path, lake: Any, value: str
) -> None:
    restart = MagicMock()
    result = _invoke([*_ARGS, "--store-dir", str(tmp_path), "--lake-version", value], restart)

    assert result.exit_code == 2
    assert "--lake-version" in result.output


def _firm_commander() -> Commander:
    analysts = build_default_analysts(excluded_analyst_ids=FIRM_V1_EXCLUDED_ANALYST_IDS)
    return Commander(
        managers=build_default_managers(analysts), weighting_strategy=ShrinkageWeight(Hedge())
    )


def test_a_counted_record_at_a_lake_time_needs_its_lake_version(
    tmp_path: Path, lake: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the firm's registry file present, --lake-time without --lake-version is a usage
    error before the lake is read, unless the store already holds the trial: then the
    spec.json's pair is the pin."""
    monkeypatch.setattr(cli, "firm_registered_trials", lambda repo_root: {})
    args = ["--start", "2010-01-04", "--end", "2010-01-05", "--lake-time", _LAKE_TIME]
    args += ["--store-dir", str(tmp_path)]
    with patch("agentic_trading.cli.run_firm_record") as record:
        refused = runner.invoke(firm_record_app, args)
    assert refused.exit_code == 2
    assert "--lake-version" in refused.output
    record.assert_not_called()
    assert lake.calls == []

    spec = build_firm_spec(
        _firm_commander(),
        list(FIRM_V1_UNIVERSE),
        date(2010, 1, 4),
        date(2010, 1, 5),
        datetime.fromisoformat(_LAKE_TIME),
    )
    stored = FirmSpecRecord(
        spec=spec,
        spec_hash=firm_spec_hash(spec),
        lake_time=spec.lake_time,
        lake_version="fake-code-1.recorded-builds",
        quantcore_version="q",
    )
    (tmp_path / stored.spec_hash).mkdir()
    (tmp_path / stored.spec_hash / "spec.json").write_text(stored.model_dump_json())
    with (
        patch("agentic_trading.cli.load_engine_inputs", return_value=_inputs(_frame())),
        patch("agentic_trading.cli.check_price_coverage", MagicMock()),
        patch(
            "agentic_trading.cli.run_firm_record",
            return_value=FirmRecordResult(
                spec_hash=stored.spec_hash,
                days_written=0,
                last_completed=None,
                complete=True,
                counted=True,
            ),
        ) as record,
    ):
        resumed = runner.invoke(firm_record_app, args)

    assert resumed.exit_code == 0, resumed.output
    assert record.call_args.kwargs["lake_version"] == "fake-code-1.recorded-builds"
