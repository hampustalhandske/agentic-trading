# test_cli_firm_record_lake_cache.py

"""`agentic-trading-firm-record` reads the lake through the lake answer cache at
--lake-cache-dir, by default `<store-dir>/lake_cache`, so a restart (the same command line) or a
resume reuses it, and writes the cache's counts to stderr as one `lake-answer-cache` JSON
line."""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from agentic_trading import cli
from agentic_trading.data.lake_answer_cache import LakeAnswerCache, LakeAnswerCacheCounts
from agentic_trading.data.loaders import lakehouse_loader as ll
from unit.test_cli_firm_record import _ARGS, _invoke
from unit.test_cli_firm_record import _policy_rates as _policy_rates


@pytest.fixture
def opened(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """The directory of each lake answer cache the command opens."""
    directories: list[Path] = []
    real = ll.lake_answer_cache

    @contextmanager
    def recording(directory: Path) -> Iterator[LakeAnswerCache]:
        directories.append(directory)
        with real(directory) as answers:
            yield answers

    monkeypatch.setattr(cli, "lake_answer_cache", recording)
    return directories


def _counts(stderr: str) -> LakeAnswerCacheCounts:
    (line,) = [
        line for line in stderr.splitlines() if line.startswith(cli.LAKE_ANSWER_CACHE_PREFIX)
    ]
    return LakeAnswerCacheCounts.model_validate(
        json.loads(line.removeprefix(cli.LAKE_ANSWER_CACHE_PREFIX))
    )


def test_the_cache_defaults_to_the_store_directory(
    tmp_path: Path, preflight_lake: Any, opened: list[Path]
) -> None:
    """Without --lake-cache-dir the record's cache is `<store-dir>/lake_cache`; its counts are
    written to stderr."""
    result, _, _, _ = _invoke([*_ARGS, "--store-dir", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert opened == [tmp_path / cli.FIRM_RECORD_LAKE_CACHE_DIR]
    assert _counts(result.stderr) == LakeAnswerCacheCounts()
    assert ll.lake_answer_cache_counts() is None


def test_the_cache_directory_can_be_named(
    tmp_path: Path, preflight_lake: Any, opened: list[Path]
) -> None:
    named = tmp_path / "elsewhere"
    result, _, _, _ = _invoke(
        [*_ARGS, "--store-dir", str(tmp_path / "store"), "--lake-cache-dir", str(named)]
    )
    assert result.exit_code == 0, result.output
    assert opened == [named]


def test_the_records_reads_inside_the_session_go_through_the_cache(
    tmp_path: Path, preflight_lake: Any
) -> None:
    """A read made inside the record's session (here by the coverage preflight) is stored in
    the cache; a second command with the same store reads it from there and sends the lake
    nothing for it."""

    def reading_coverage(*args: Any, **kwargs: Any) -> list[Any]:
        ll._query("SELECT count(*) AS n FROM sp500_price_coverage_log", [])
        return []

    first, _, _, _ = _invoke(
        [*_ARGS, "--store-dir", str(tmp_path)], coverage=MagicMock(side_effect=reading_coverage)
    )
    assert first.exit_code == 0, first.output
    calls_after_first = len(preflight_lake.calls)
    second, _, _, _ = _invoke(
        [*_ARGS, "--store-dir", str(tmp_path)], coverage=MagicMock(side_effect=reading_coverage)
    )
    assert second.exit_code == 0, second.output

    assert _counts(first.stderr).stored == 1
    assert _counts(second.stderr).hits == 1
    assert len(preflight_lake.calls) == calls_after_first
