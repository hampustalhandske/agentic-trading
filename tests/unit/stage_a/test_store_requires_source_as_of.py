# test_store_requires_source_as_of.py

"""AT-118 item 3: `SignalStore.write_day` refuses an active signal without a `source_as_of` from
an analyst that declares a data source, raising `LookAheadError` before anything is written. A
price-only analyst's signals, and every unavailable or error signal, still pass without one."""

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest

from agentic_trading.agents.analysts import CIRRateRegimeAnalyst, GBMMomentumAnalyst
from agentic_trading.agents.analysts.factor_pod.earnings_surprise_analyst import (
    EarningsSurpriseAnalyst,
)
from agentic_trading.agents.analysts.macro_rates_pod.cot_positioning_analyst import (
    COTPositioningAnalyst,
)
from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.stage_a.decisions import LookAheadError
from agentic_trading.stage_a.spec import SpecRecord, build_spec, spec_hash
from agentic_trading.stage_a.store import SignalStore

_DAYS = [date(2010, 1, 4), date(2010, 1, 5)]
_RATE_IDS = {"SHY": "SHY", "IEF": "IEF", "TLT": "TLT"}
_EQUITY_IDS = {"AAPL": "14593", "MSFT": "10107"}
_WRITTEN = ("signals.parquet", "decisions.parquet", "progress.json")


def _record(analyst: AnalystAgent, universe: list[str]) -> SpecRecord:
    spec = build_spec(analyst, universe, date(2010, 1, 4), date(2024, 12, 31))
    return SpecRecord(
        spec=spec,
        spec_hash=spec_hash(spec),
        lake_time=datetime(2026, 10, 29, tzinfo=UTC),
        lake_version="f89aef2f428a",
        quantcore_version="0.4.0",
    )


def _open(
    store_dir: Path, analyst: AnalystAgent | None = None, ids: dict[str, str] = _RATE_IDS
) -> tuple[SignalStore, Path]:
    """An opened store for `analyst` (a CIR analyst by default) over `ids`, and its trial dir."""
    record = _record(analyst or CIRRateRegimeAnalyst(), list(ids))
    store = SignalStore(store_dir, record)
    store.open()
    return store, store_dir / record.spec_hash


def _signal(
    asset: str, day: date, direction: str = "long", confidence: float = 0.5, **fields: Any
) -> AgentSignal:
    return AgentSignal(
        agent_id="test",
        asset=asset,
        as_of_date=day,
        direction=direction,
        confidence=confidence,
        rationale="",
        **fields,
    )


def _sourced(day: date, assets: list[str]) -> list[AgentSignal]:
    """Each of `assets` long 0.5 on data published the day before `day`."""
    return [_signal(asset, day, source_as_of=day - timedelta(days=1)) for asset in assets]


def _rows(path: Path) -> list[dict[str, Any]]:
    """The Parquet file's rows sorted by date, then instrument_id."""
    table = pq.read_table(path)
    day = "date" if "date" in table.schema.names else "decision_date"
    return sorted(table.to_pylist(), key=lambda row: (row[day], row["instrument_id"]))


def _stored(trial: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The rows of the trial's signals.parquet and decisions.parquet."""
    return _rows(trial / "signals.parquet"), _rows(trial / "decisions.parquet")


def _progress(trial: Path) -> Any:
    return json.loads((trial / "progress.json").read_text())


def test_write_day_refuses_an_unsourced_active_signal(tmp_path: Path) -> None:
    """An unsourced active signal, flat included, raises and leaves the store as it was."""
    first, second = _DAYS
    analysts: list[tuple[AnalystAgent, dict[str, str]]] = [
        (CIRRateRegimeAnalyst(), _RATE_IDS),
        (COTPositioningAnalyst(agent_id="cot_positioning_analyst_rates"), _RATE_IDS),
        (EarningsSurpriseAnalyst(), _EQUITY_IDS),
    ]
    for n, (analyst, ids) in enumerate(analysts):
        bare, *rest = ids
        for direction, confidence in (("long", 0.5), ("flat", 0.0)):
            store, trial = _open(tmp_path / f"{n}-{direction}", analyst, ids)
            unsourced = _signal(bare, first, direction, confidence)
            with pytest.raises(LookAheadError):
                store.write_day(first, [unsourced, *_sourced(first, rest)], ids, "equity")
            assert store.last_completed() is None
            assert [name for name in _WRITTEN if (trial / name).exists()] == []

            store.write_day(first, _sourced(first, list(ids)), ids, "equity")
            written = _stored(trial)
            unsourced = _signal(bare, second, direction, confidence)
            with pytest.raises(LookAheadError):
                store.write_day(second, [unsourced, *_sourced(second, rest)], ids, "equity")
            assert _stored(trial) == written
            assert store.last_completed() == first
            assert _progress(trial) == {"last_completed": first.isoformat()}


def test_write_day_passes_none_for_price_only_and_inactive_signals(tmp_path: Path) -> None:
    """Price-only active rows and sourced inactive rows are written without a source."""
    day = _DAYS[0]
    price_only: list[AnalystAgent] = [
        GBMMomentumAnalyst(),
        CIRRateRegimeAnalyst(data_requirements=()),
    ]
    for n, analyst in enumerate(price_only):
        store, trial = _open(tmp_path / f"price-{n}", analyst)
        store.write_day(day, [_signal(asset, day) for asset in _RATE_IDS], _RATE_IDS, "equity")
        decisions = _rows(trial / "decisions.parquet")
        rows = [(row["status"], row["source"], row["source_as_of"]) for row in decisions]

        assert rows == [("active", None, None)] * len(_RATE_IDS)
        assert store.last_completed() == day

    store, trial = _open(tmp_path / "sourced")
    known = day - timedelta(days=1)
    signals = [
        _signal("SHY", day, "long", 0.6, source_as_of=known),
        _signal("IEF", day, "flat", 0.0, status="unavailable"),
        _signal("TLT", day, "short", 0.4, status="error"),
    ]
    store.write_day(day, signals, _RATE_IDS, "equity")
    decisions = _rows(trial / "decisions.parquet")
    columns = ("instrument_id", "status", "source", "source_as_of")

    assert [tuple(row[column] for column in columns) for row in decisions] == [
        ("IEF", "unavailable", None, None),
        ("SHY", "active", "fred", known),
        ("TLT", "error", None, None),
    ]
    assert store.last_completed() == day
