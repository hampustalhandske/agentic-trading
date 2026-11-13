# test_runner_lake_health.py

"""A firm record checks the lake's health (`require_lake_health`) before it starts: a degraded
answer refuses the pass, naming the bad table, before anything is written; a healthy one lets
it run."""

from pathlib import Path
from typing import Any

import pytest

from agentic_trading.data.loaders.lakehouse_loader import LakeHealthError
from unit.firm_record.test_runner import _run


def test_a_degraded_lake_refuses_the_record_before_anything_is_written(
    tmp_path: Path, fake_lake: Any
) -> None:
    fake_lake.write_times = {"instrument_prices": {"ok": False}}
    store_dir = tmp_path / "store"

    with pytest.raises(LakeHealthError, match="instrument_prices"):
        _run(store_dir)

    assert not store_dir.exists() or not any(store_dir.iterdir())
    assert fake_lake.calls == []


def test_a_healthy_lake_lets_the_record_run(tmp_path: Path, fake_lake: Any) -> None:
    fake_lake.write_times = {"instrument_prices": {"ok": True}, "reddit_comments": {"ok": False}}

    result = _run(tmp_path / "store")

    assert result.days_written > 0
