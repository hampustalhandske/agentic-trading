# test_store.py

"""AT-110 precompute spec item 4: `SignalStore` keeps one directory per spec hash holding
spec.json, sources.json, signals.parquet, decisions.parquet (decision_file.v1 with `status`,
`source` and `source_as_of`), progress.json for resume and counts.json (coverage and turnover
only). It refuses a look-ahead row or an out-of-order day before writing it."""

import ast
import importlib.util
import json
import math
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from agentic_trading.agents.analysts import CIRRateRegimeAnalyst, GBMMomentumAnalyst
from agentic_trading.agents.analysts.factor_pod.earnings_surprise_analyst import (
    EarningsSurpriseAnalyst,
)
from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.stage_a.decisions import LookAheadError
from agentic_trading.stage_a.spec import SpecRecord, build_spec, spec_hash
from agentic_trading.stage_a.store import SignalStore, StoreMismatchError, StoreOrderError

_DAYS = [date(2010, 1, 4), date(2010, 1, 5), date(2010, 1, 6), date(2010, 1, 7)]
_RATE_IDS = {"SHY": "SHY", "IEF": "IEF", "TLT": "TLT"}
_EQUITY_IDS = {"AAPL": "14593", "MSFT": "10107"}
_REQUIRED = ["decision_date", "instrument_kind", "instrument_id", "target_weight"]
_INSTRUMENT_KINDS = {"equity", "etf", "future", "fx", "crypto", "option"}
_STATUSES = {"active", "unavailable", "error"}
_SIGNAL_COLUMNS = [
    "date",
    "instrument_id",
    "status",
    "direction",
    "confidence",
    "source_as_of",
    "sample_count",
]
_DECISION_COLUMNS = [*_REQUIRED, "status", "source", "source_as_of"]
_SCHEMA = Path(__file__).resolve().parents[5] / "edgelab/docs/schema/decision_file.v1.json"
_PACKAGE = "agentic_trading.stage_a"


def _record(analyst: AnalystAgent, universe: list[str], **overrides: Any) -> SpecRecord:
    spec = build_spec(analyst, universe, date(2010, 1, 4), date(2024, 12, 31))
    fields: dict[str, Any] = {
        "lake_time": datetime(2026, 10, 29, tzinfo=UTC),
        "lake_version": "f89aef2f428a",
        "quantcore_version": "0.4.0",
        "quantcore_sha": None,
    }
    return SpecRecord(spec=spec, spec_hash=spec_hash(spec), **(fields | overrides))


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


def _rates_day(day: date, shy_as_of: date | None = None) -> list[AgentSignal]:
    """SHY long 0.6 on FRED of `shy_as_of` (default two days before), IEF short 0.2, TLT
    unavailable."""
    shy_known = shy_as_of or day - timedelta(days=2)
    return [
        _signal("SHY", day, "long", 0.6, source_as_of=shy_known, sample_count=250),
        _signal("IEF", day, "short", 0.2, source_as_of=day - timedelta(days=1), sample_count=249),
        _signal("TLT", day, "flat", 0.0, status="unavailable"),
    ]


def _price_day(day: date, shy_as_of: date | None = None) -> list[AgentSignal]:
    """Every rates instrument long 0.5, only SHY carrying a source_as_of, `shy_as_of`."""
    return [
        _signal(asset, day, source_as_of=shy_as_of if asset == "SHY" else None)
        for asset in _RATE_IDS
    ]


def _equity_day(day: date, aapl_as_of: date) -> list[AgentSignal]:
    """AAPL long 0.4 on earnings published `aapl_as_of`, MSFT in error."""
    return [
        _signal("AAPL", day, "long", 0.4, source_as_of=aapl_as_of),
        _signal("MSFT", day, "flat", 0.0, status="error"),
    ]


def _rows(path: Path) -> list[dict[str, Any]]:
    """The Parquet file's rows sorted by date, then instrument_id."""
    table = pq.read_table(path)
    day = "date" if "date" in table.schema.names else "decision_date"
    return sorted(table.to_pylist(), key=lambda row: (row[day], row["instrument_id"]))


def _progress(trial: Path) -> Any:
    return json.loads((trial / "progress.json").read_text())


def _is_text(kind: pa.DataType) -> bool:
    checks = (pa.types.is_string, pa.types.is_large_string, pa.types.is_string_view)
    return any(check(kind) for check in checks)


def test_open_writes_spec_and_sources(tmp_path: Path) -> None:
    """open() writes spec.json and the row sources' after-close flags, and reopens unchanged."""
    analysts: list[tuple[AnalystAgent, dict[str, str], list[dict[str, Any]]]] = [
        (CIRRateRegimeAnalyst(), _RATE_IDS, [{"name": "fred", "after_close": True}]),
        (EarningsSurpriseAnalyst(), _EQUITY_IDS, [{"name": "earnings", "after_close": True}]),
        (GBMMomentumAnalyst(), _RATE_IDS, []),
    ]
    for analyst, ids, sources in analysts:
        store_dir = tmp_path / analyst.agent_id
        record = _record(analyst, list(ids))
        store, trial = _open(store_dir, analyst, ids)
        spec_text = (trial / "spec.json").read_text()

        assert trial.is_dir()
        assert SpecRecord.model_validate_json(spec_text) == record
        assert json.loads((trial / "sources.json").read_text()) == sources
        assert store.last_completed() is None

        SignalStore(store_dir, record).open()
        assert (trial / "spec.json").read_text() == spec_text


def test_open_refuses_a_changed_record(tmp_path: Path) -> None:
    """A same-hash record with another lake time, lake version or quantcore field is refused."""
    _, trial = _open(tmp_path)
    original = (trial / "spec.json").read_text()
    changes: list[dict[str, Any]] = [
        {"lake_time": datetime(2026, 11, 5, 12, tzinfo=UTC)},
        {"lake_version": "c69d5dd"},
        {"quantcore_version": "0.5.0"},
        {"quantcore_sha": "0123456789abcdef"},
    ]
    for change in changes:
        record = _record(CIRRateRegimeAnalyst(), list(_RATE_IDS), **change)
        assert record.spec_hash == trial.name
        with pytest.raises(StoreMismatchError):
            SignalStore(tmp_path, record).open()
    assert (trial / "spec.json").read_text() == original


def test_write_day_rows(tmp_path: Path) -> None:
    """write_day appends one signal and one decision row per instrument and records progress."""
    store, trial = _open(tmp_path)
    day, previous, two_before = _DAYS[0], _DAYS[0] - timedelta(days=1), _DAYS[0] - timedelta(days=2)
    assert store.last_completed() is None
    store.write_day(day, _rates_day(day), _RATE_IDS, "equity")
    signals, decisions = _rows(trial / "signals.parquet"), _rows(trial / "decisions.parquet")
    decision_fields = [column for column in _DECISION_COLUMNS if column != "target_weight"]

    assert all(set(row) == set(_SIGNAL_COLUMNS) for row in signals)
    assert all(set(row) == set(_DECISION_COLUMNS) for row in decisions)
    assert [tuple(row[column] for column in _SIGNAL_COLUMNS) for row in signals] == [
        (day, "IEF", "active", "short", 0.2, previous, 249),
        (day, "SHY", "active", "long", 0.6, two_before, 250),
        (day, "TLT", "unavailable", "flat", 0.0, None, None),
    ]
    assert [tuple(row[column] for column in decision_fields) for row in decisions] == [
        (day, "equity", "IEF", "active", "fred", previous),
        (day, "equity", "SHY", "active", "fred", two_before),
        (day, "equity", "TLT", "unavailable", None, None),
    ]
    assert [row["target_weight"] for row in decisions] == pytest.approx([-0.25, 0.75, 0.0])
    assert _progress(trial) == {"last_completed": "2010-01-04"}
    assert store.last_completed() == day

    store.write_day(_DAYS[1], _rates_day(_DAYS[1]), _RATE_IDS, "equity")
    assert len(_rows(trial / "signals.parquet")) == len(_rows(trial / "decisions.parquet")) == 6
    assert _progress(trial) == {"last_completed": "2010-01-05"}
    assert store.last_completed() == _DAYS[1]


def test_write_day_refuses_look_ahead(tmp_path: Path) -> None:
    """A row whose source_as_of the day cannot know raises LookAheadError and writes nothing."""
    first, second, third = _DAYS[:3]
    known = date(2009, 12, 31)
    earnings = EarningsSurpriseAnalyst()
    cases = [
        (earnings, _EQUITY_IDS, _equity_day(first, known), _equity_day(second, second)),
        (earnings, _EQUITY_IDS, _equity_day(first, known), _equity_day(second, third)),
        (CIRRateRegimeAnalyst(), _RATE_IDS, _rates_day(first), _rates_day(second, third)),
        (GBMMomentumAnalyst(), _RATE_IDS, _price_day(first), _price_day(second, first)),
    ]
    trials = []
    for n, (analyst, ids, good_day, bad_day) in enumerate(cases):
        store, trial = _open(tmp_path / str(n), analyst, ids)
        store.write_day(first, good_day, ids, "equity")
        written = (_rows(trial / "signals.parquet"), _rows(trial / "decisions.parquet"))
        trials.append(trial)

        with pytest.raises(LookAheadError):
            store.write_day(second, bad_day, ids, "equity")
        assert (_rows(trial / "signals.parquet"), _rows(trial / "decisions.parquet")) == written
        assert store.last_completed() == first
        assert _progress(trial) == {"last_completed": first.isoformat()}

    decisions = _rows(trials[0] / "decisions.parquet")
    assert [(row["instrument_id"], row["source"], row["source_as_of"]) for row in decisions] == [
        ("10107", None, None),
        ("14593", "earnings", known),
    ]


def test_write_day_refuses_out_of_order(tmp_path: Path) -> None:
    """A decision date at or before the last one raises StoreOrderError and writes nothing."""
    store, trial = _open(tmp_path)
    store.write_day(_DAYS[1], _rates_day(_DAYS[1]), _RATE_IDS, "equity")

    for day in (_DAYS[1], _DAYS[0]):
        with pytest.raises(StoreOrderError):
            store.write_day(day, _rates_day(day), _RATE_IDS, "equity")
    assert len(_rows(trial / "signals.parquet")) == len(_RATE_IDS)
    assert len(_rows(trial / "decisions.parquet")) == len(_RATE_IDS)
    assert store.last_completed() == _DAYS[1]

    store.write_day(_DAYS[2], _rates_day(_DAYS[2]), _RATE_IDS, "equity")
    assert store.last_completed() == _DAYS[2]


def test_counts(tmp_path: Path) -> None:
    """Per-instrument status day counts and annual turnover, the first day measured from 0."""
    store, trial = _open(tmp_path)
    fred = [day - timedelta(days=1) for day in _DAYS]
    days = [
        [
            _signal("SHY", _DAYS[0], "long", 0.5, source_as_of=fred[0]),
            _signal("IEF", _DAYS[0], "short", 0.5, source_as_of=fred[0]),
        ],
        [
            _signal("SHY", _DAYS[1], "long", 1.0, source_as_of=fred[1]),
            _signal("IEF", _DAYS[1], "flat", 0.0, status="unavailable"),
        ],
        [
            _signal("SHY", _DAYS[2], "long", 0.3, source_as_of=fred[2]),
            _signal("IEF", _DAYS[2], "long", 0.1, source_as_of=fred[2]),
        ],
        [
            _signal("SHY", _DAYS[3], "flat", 0.0, status="error"),
            _signal("IEF", _DAYS[3], "short", 0.2, source_as_of=fred[3]),
        ],
    ]
    for day, signals in zip(_DAYS, days, strict=True):
        store.write_day(day, signals, _RATE_IDS, "equity")
    daily_turnover = [1.0, 1.0, 0.5, 2.0]
    counts = store.counts()
    store.write_counts()

    assert counts["instruments"] == {
        "SHY": {"active": 3, "unavailable": 0, "error": 1},
        "IEF": {"active": 3, "unavailable": 1, "error": 0},
    }
    assert counts["annual_turnover"] == pytest.approx(sum(daily_turnover) / (4 / 252))
    assert json.loads((trial / "counts.json").read_text()) == counts


def test_store_imports_nothing_from_data() -> None:
    """The store module imports nothing from agentic_trading.data, so it reads no price."""
    found = importlib.util.find_spec(f"{_PACKAGE}.store")
    assert found is not None and found.origin is not None
    imported: list[str] = []
    for node in ast.walk(ast.parse(Path(found.origin).read_bytes())):
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                base = importlib.util.resolve_name("." * node.level + base, _PACKAGE)
            imported += [base, *(f"{base}.{alias.name}" for alias in node.names)]
    data = "agentic_trading.data"

    assert imported
    assert [name for name in imported if name == data or name.startswith(f"{data}.")] == []


def test_decisions_conform_to_decision_file_v1(tmp_path: Path) -> None:
    """decisions.parquet has decision_file.v1's required columns and types, plus a valid status."""
    store, trial = _open(tmp_path)
    days = {
        _DAYS[0]: _rates_day(_DAYS[0]),
        _DAYS[1]: _rates_day(_DAYS[1]),
        _DAYS[2]: [_signal(asset, _DAYS[2], "long", 0.9, status="error") for asset in _RATE_IDS],
    }
    for day, signals in days.items():
        store.write_day(day, signals, _RATE_IDS, "equity")
    table = pq.read_table(trial / "decisions.parquet")
    rows = table.to_pylist()

    assert set(_REQUIRED) <= set(table.schema.names)
    assert pa.types.is_date(table.schema.field("decision_date").type)
    assert pa.types.is_floating(table.schema.field("target_weight").type)
    for name in ("instrument_kind", "instrument_id", "status"):
        assert _is_text(table.schema.field(name).type), name
    assert len(rows) == len(days) * len(_RATE_IDS)
    for row in rows:
        weight = row["target_weight"]
        assert math.isfinite(weight) and abs(weight) <= 6.0
        assert row["status"] in _STATUSES
        assert row["status"] == "active" or weight == 0.0
        assert row["instrument_kind"] in _INSTRUMENT_KINDS and row["instrument_id"]

    if not _SCHEMA.exists():
        pytest.skip(f"edgelab decision-file schema not reachable at {_SCHEMA}")
    schema = json.loads(_SCHEMA.read_text())
    properties = schema["properties"]
    min_id_length = properties["instrument_id"]["minLength"]
    low, high = properties["target_weight"]["minimum"], properties["target_weight"]["maximum"]
    assert schema["required"] == _REQUIRED
    assert properties["decision_date"]["format"] == "date"
    assert {row["instrument_kind"] for row in rows} <= set(properties["instrument_kind"]["enum"])
    assert all(len(row["instrument_id"]) >= min_id_length for row in rows)
    assert all(low <= row["target_weight"] <= high for row in rows)


def test_write_and_resume(tmp_path: Path) -> None:
    """A new store on the same dir resumes after day 3 and appends day 4."""
    store, trial = _open(tmp_path)
    for day in _DAYS[:3]:
        store.write_day(day, _rates_day(day), _RATE_IDS, "equity")
    resumed, _ = _open(tmp_path)

    assert resumed.last_completed() == _DAYS[2]
    with pytest.raises(StoreOrderError):
        resumed.write_day(_DAYS[2], _rates_day(_DAYS[2]), _RATE_IDS, "equity")
    resumed.write_day(_DAYS[3], _rates_day(_DAYS[3]), _RATE_IDS, "equity")

    decisions = _rows(trial / "decisions.parquet")
    assert sorted({row["decision_date"] for row in decisions}) == _DAYS
    assert len(decisions) == len(_rows(trial / "signals.parquet")) == 4 * len(_RATE_IDS)
    assert resumed.last_completed() == _DAYS[3]
    assert _progress(trial) == {"last_completed": "2010-01-07"}
    assert all(sum(days.values()) == 4 for days in resumed.counts()["instruments"].values())


def test_no_machine_path_in_json(tmp_path: Path) -> None:
    """spec.json, sources.json and counts.json name no user, temp or home directory."""
    store, trial = _open(tmp_path)
    store.write_day(_DAYS[0], _rates_day(_DAYS[0]), _RATE_IDS, "equity")
    store.write_counts()
    home = str(Path.home())
    forbidden = ["/Users", "/private", str(tmp_path), *([home] if home != "/" else [])]

    for name in ("spec.json", "sources.json", "counts.json"):
        text = (trial / name).read_text()
        assert [part for part in forbidden if part in text] == [], name
