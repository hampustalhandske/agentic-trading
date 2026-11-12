# test_cli_firm_review.py

"""`agentic-trading-firm-review` writes a finished firm record's run summary
(`firm_record.review.firm_record_summary`) as JSON outside the record's store, and prints no
performance figure."""

import json
import subprocess
import sys
import tomllib
from importlib import import_module
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from agentic_trading.cli import firm_review_app
from agentic_trading.firm_record import review

runner = CliRunner()
_PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"
_SUMMARY = {
    "experiment": {"spec_hash": "abc"},
    "record": {
        "portfolio_value_history": [["2010-01-04", 1_000_000.0], ["2010-01-05", 1_001_234.5]],
        "final_sharpe_ratio": 0.123456,
    },
}


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    store = tmp_path / "store"
    store.mkdir()
    path = store / "abc.checkpoint.sqlite"
    path.write_bytes(b"")
    return path


@pytest.fixture
def summary_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, ...]]:
    calls: list[tuple[Any, ...]] = []

    def _summary(
        checkpoint_path: Path, trial_dir: Path | None = None, *, lake_cache_dir: Path | None = None
    ) -> dict[str, Any]:
        calls.append((checkpoint_path, trial_dir, lake_cache_dir))
        return _SUMMARY

    monkeypatch.setattr(review, "firm_record_summary", _summary)
    return calls


def test_entry_point_is_declared() -> None:
    scripts = tomllib.loads(_PYPROJECT.read_text())["project"]["scripts"]
    module, _, name = scripts["agentic-trading-firm-review"].partition(":")

    assert (module, name) == ("agentic_trading.cli", "firm_review_main")
    assert callable(getattr(import_module(module), name))


def test_the_summary_is_written_to_out(
    checkpoint: Path, summary_calls: list[tuple[Any, ...]], tmp_path: Path
) -> None:
    out = tmp_path / "review" / "summary.json"
    cache = tmp_path / "cache"
    trial = tmp_path / "trial"

    result = runner.invoke(
        firm_review_app,
        [
            str(checkpoint),
            "--out",
            str(out),
            "--trial-dir",
            str(trial),
            "--lake-cache-dir",
            str(cache),
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(out.read_text()) == _SUMMARY
    assert summary_calls == [(checkpoint, trial, cache)]
    assert "2 days" in result.output
    assert "0.12" not in result.output and "1001234" not in result.output.replace(",", "")


def test_the_trial_folder_and_cache_default_to_none(
    checkpoint: Path, summary_calls: list[tuple[Any, ...]], tmp_path: Path
) -> None:
    result = runner.invoke(firm_review_app, [str(checkpoint), "--out", str(tmp_path / "s.json")])

    assert result.exit_code == 0, result.output
    assert summary_calls == [(checkpoint, None, None)]


@pytest.mark.parametrize("flag", ["--out", "--lake-cache-dir"])
def test_nothing_is_written_inside_the_records_store(
    checkpoint: Path, summary_calls: list[tuple[Any, ...]], tmp_path: Path, flag: str
) -> None:
    inside = checkpoint.parent / "inside"
    args = [str(checkpoint), "--out", str(tmp_path / "s.json"), flag, str(inside)]

    result = runner.invoke(firm_review_app, args)

    assert result.exit_code == 2
    assert summary_calls == []
    assert sorted(path.name for path in checkpoint.parent.iterdir()) == [checkpoint.name]


def test_a_missing_checkpoint_is_a_usage_error(
    summary_calls: list[tuple[Any, ...]], tmp_path: Path
) -> None:
    result = runner.invoke(
        firm_review_app, [str(tmp_path / "absent.sqlite"), "--out", str(tmp_path / "s.json")]
    )

    assert result.exit_code == 2
    assert summary_calls == []


def test_the_cli_module_does_not_load_the_review() -> None:
    """A firm record hashes every loaded `agentic_trading` module, so importing the command line
    leaves the review unloaded until the command runs."""
    code = (
        "import sys, agentic_trading.cli; "
        "print('agentic_trading.firm_record.review' in sys.modules)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert completed.stdout.strip() == "False"
