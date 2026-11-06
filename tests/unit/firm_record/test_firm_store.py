# test_firm_store.py

"""AT-125: `FirmDecisionStore` writes a firm record's decision rows as a `decision_file.v1`
file through the stage (a) store's writer: one row per (decision_date, instrument_id, pod_id),
every row valid under the vendored edgelab schema, every row's source and source_as_of passing
the stage (a) look-ahead audit, a repeated (date, instrument) refused unless each of its rows
carries its own non-null pod_id, and the net weight of a (date, instrument) within [-6, 6]."""

import json
import math
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from agentic_trading.stage_a.decisions import LookAheadError
from agentic_trading.stage_a.spec import FirmSpec, FirmSpecRecord, firm_spec_hash
from agentic_trading.stage_a.store import (
    DecisionRow,
    DuplicateRowError,
    FirmDecisionStore,
    StoreMismatchError,
    StoreOrderError,
)

_SCHEMA = Path(__file__).resolve().parents[2] / "fixtures" / "edgelab_decision_file_v1.json"
_T = datetime(2026, 11, 6, tzinfo=UTC)
_DAYS = [date(2016, 1, 4), date(2016, 1, 5), date(2016, 1, 6)]
_SOURCES: list[dict[str, str | bool]] = [
    {"name": "fred", "can_publish_after_close": True},
    {"name": "options_chain", "can_publish_after_close": False},
    {"name": "sec_sic", "undated": True, "limitation": "current SIC code"},
]


def _record(**changes: Any) -> FirmSpecRecord:
    fields: dict[str, Any] = {
        "commander": {"class": "agentic_trading.agents.commander.Commander"},
        "roster": [{"agent_id": "risk_manager", "class": "x.RiskManager", "parameters": {}}],
        "settings": {"firm": {"trading_days_per_year": 252}},
        "module_hashes": {"agentic_trading.agents.commander": "ab" * 32},
        "universe": ["SPY", "AAPL"],
        "window_start": _DAYS[0],
        "window_end": _DAYS[-1],
        "decision_rule": "rule",
        "quantcore_sha": "e" * 40,
        "lake_time": _T,
    }
    spec = FirmSpec(**(fields | changes))
    return FirmSpecRecord(
        spec=spec,
        spec_hash=firm_spec_hash(spec),
        lake_time=_T,
        lake_version="lake",
        quantcore_version="0.3.1",
    )


def _open(tmp_path: Path, record: FirmSpecRecord | None = None) -> tuple[FirmDecisionStore, Path]:
    record = record or _record()
    store = FirmDecisionStore(tmp_path, record, sources=_SOURCES)
    store.open()
    return store, tmp_path / record.spec_hash


def _row(day: date, instrument: str, pod: str | None, weight: float, **fields: Any) -> DecisionRow:
    values: dict[str, Any] = {
        "decision_date": day,
        "instrument_kind": "equity",
        "instrument_id": instrument,
        "target_weight": weight,
        "status": "active",
        "pod_id": pod,
    }
    return DecisionRow(**(values | fields))


def _day_rows(day: date) -> list[DecisionRow]:
    return [
        _row(day, "SPY", "risk_manager", 0.3),
        _row(
            day, "SPY", "macro_rates_manager", -0.1, source="fred", source_as_of=date(2015, 12, 31)
        ),
        _row(day, "AAPL", "risk_manager", 0.2),
        _row(
            day,
            "SPY 2016-02-19 P 190.00",
            "tail_hedge_manager",
            0.004,
            instrument_kind="option",
            delta_equiv_weight=-0.6,
            source="options_chain",
            source_as_of=day,
        ),
    ]


def _matches(schema: dict[str, Any], value: Any) -> bool:
    """Whether `value` satisfies one property's JSON schema (the keywords the vendored schema
    uses: anyOf, type, enum, format date, minLength, minimum, maximum)."""
    if "anyOf" in schema:
        return any(_matches(option, value) for option in schema["anyOf"])
    kind = schema.get("type")
    if kind == "null" and value is not None:
        return False
    if kind == "string":
        if not isinstance(value, str) or len(value) < schema.get("minLength", 0):
            return False
        if schema.get("format") == "date":
            date.fromisoformat(value)
    if kind == "number" and (not isinstance(value, int | float) or isinstance(value, bool)):
        return False
    if "enum" in schema and value not in schema["enum"]:
        return False
    if "minimum" in schema and value < schema["minimum"]:
        return False
    return not ("maximum" in schema and value > schema["maximum"])


def _schema_errors(row: dict[str, Any]) -> list[str]:
    """What the vendored decision_file.v1 row schema refuses in `row` (JSON values)."""
    schema = json.loads(_SCHEMA.read_text())
    errors = [f"missing {name}" for name in schema["required"] if name not in row]
    for name, value in row.items():
        if name not in schema["properties"]:
            errors.append(f"unknown {name}")
        elif not _matches(schema["properties"][name], value):
            errors.append(f"{name}={value!r}")
    return errors


def _json_rows(trial: Path) -> list[dict[str, Any]]:
    """decisions.parquet's rows as the JSON values a submission carries."""
    rows = pq.read_table(trial / "decisions.parquet").to_pylist()
    return [
        {
            name: value.isoformat() if isinstance(value, date) else value
            for name, value in row.items()
        }
        for row in rows
    ]


def test_rows_conform_to_decision_file_v1(tmp_path: Path) -> None:
    """Every stored row, pod_id and delta_equiv_weight included, passes the vendored schema."""
    store, trial = _open(tmp_path)
    for day in _DAYS:
        store.write_day(day, _day_rows(day))
    rows = _json_rows(trial)
    table = pq.read_table(trial / "decisions.parquet")

    assert len(rows) == 4 * len(_DAYS)
    assert [_schema_errors(row) for row in rows] == [[]] * len(rows)
    assert pa.types.is_date(table.schema.field("decision_date").type)
    assert pa.types.is_date(table.schema.field("source_as_of").type)
    assert pa.types.is_floating(table.schema.field("delta_equiv_weight").type)
    option = next(row for row in rows if row["instrument_kind"] == "option")
    assert option["delta_equiv_weight"] == -0.6
    assert {row["delta_equiv_weight"] for row in rows if row["instrument_kind"] != "option"} == {
        None
    }


def test_vendored_schema_validates_rows_one_by_one(tmp_path: Path) -> None:
    """The vendored schema describes one row, so two pod rows of one (date, instrument) each
    pass it: the duplicate rule is the writer's own check."""
    store, trial = _open(tmp_path)
    store.write_day(_DAYS[0], _day_rows(_DAYS[0]))
    spy = [row for row in _json_rows(trial) if row["instrument_id"] == "SPY"]

    assert len(spy) == 2
    assert {row["pod_id"] for row in spy} == {"risk_manager", "macro_rates_manager"}
    assert all(_schema_errors(row) == [] for row in spy)


def test_open_writes_spec_and_sources(tmp_path: Path) -> None:
    """spec.json holds the record and sources.json the sources the store was given."""
    record = _record()
    _, trial = _open(tmp_path, record)

    assert FirmSpecRecord.model_validate_json((trial / "spec.json").read_text()) == record
    assert json.loads((trial / "sources.json").read_text()) == _SOURCES
    with pytest.raises(StoreMismatchError):
        FirmDecisionStore(tmp_path, record, sources=_SOURCES[:1]).open()


@pytest.mark.parametrize(
    ("rows", "named"),
    [
        (
            [_row(_DAYS[0], "SPY", None, 0.1), _row(_DAYS[0], "SPY", "risk_manager", 0.1)],
            "None",
        ),
        (
            [
                _row(_DAYS[0], "SPY", "risk_manager", 0.1),
                _row(_DAYS[0], "SPY", "risk_manager", 0.2),
            ],
            "risk_manager",
        ),
    ],
)
def test_repeat_needs_distinct_non_null_pods(
    tmp_path: Path, rows: list[DecisionRow], named: str
) -> None:
    """A repeated (date, instrument) with a null or shared pod_id is refused, naming the
    instrument, the date and the pod, and nothing is written."""
    store, trial = _open(tmp_path)

    with pytest.raises(DuplicateRowError) as raised:
        store.write_day(_DAYS[0], rows)
    message = str(raised.value)
    assert "SPY" in message and "2016-01-04" in message and named in message
    assert not (trial / "decisions.parquet").exists()
    assert store.last_completed() is None


def test_single_row_may_have_no_pod(tmp_path: Path) -> None:
    """An instrument with one row may keep pod_id null."""
    store, trial = _open(tmp_path)
    store.write_day(_DAYS[0], [_row(_DAYS[0], "SPY", None, 0.1)])

    assert _json_rows(trial)[0]["pod_id"] is None


def test_net_weight_must_stay_within_six(tmp_path: Path) -> None:
    """Pod rows of one instrument netting beyond 6 are refused."""
    store, _ = _open(tmp_path)
    rows = [_row(_DAYS[0], "SPY", "a_manager", 4.0), _row(_DAYS[0], "SPY", "b_manager", 2.5)]

    with pytest.raises(ValueError, match=r"SPY.*2016-01-04"):
        store.write_day(_DAYS[0], rows)


def test_row_model_refuses_what_the_schema_refuses() -> None:
    """A non-finite or out-of-range weight, an inactive row with a weight, an unknown kind, an
    empty instrument id and a non-finite delta are refused when the row is built."""
    bad: list[dict[str, Any]] = [
        {"target_weight": math.nan},
        {"target_weight": 6.5},
        {"status": "unavailable", "target_weight": 0.1},
        {"instrument_kind": "etf"},
        {"instrument_id": ""},
        {"delta_equiv_weight": math.inf},
    ]
    for fields in bad:
        values: dict[str, Any] = {
            "decision_date": _DAYS[0],
            "instrument_kind": "equity",
            "instrument_id": "SPY",
            "target_weight": 0.0,
            "status": "active",
        } | fields
        with pytest.raises(ValidationError):
            DecisionRow(**values)


@pytest.mark.parametrize(
    "fields",
    [
        {"source": "fred", "source_as_of": _DAYS[0]},
        {"source": "options_chain", "source_as_of": _DAYS[1]},
        {"source": "fred", "source_as_of": None},
        {"source": "sec_edgar_facts", "source_as_of": date(2015, 6, 1)},
    ],
)
def test_rows_pass_the_stage_a_audit(tmp_path: Path, fields: dict[str, Any]) -> None:
    """A row the decision close cannot know, a half-filled source, or a source the store's
    sources do not list is refused before anything is written."""
    store, trial = _open(tmp_path)

    with pytest.raises(LookAheadError):
        store.write_day(_DAYS[0], [_row(_DAYS[0], "SPY", "macro_rates_manager", 0.1, **fields)])
    assert not (trial / "decisions.parquet").exists()


def test_rows_must_be_dated_the_day_and_inside_the_window(tmp_path: Path) -> None:
    """A row dated another day, or a day outside the spec's window, is refused."""
    store, _ = _open(tmp_path)

    with pytest.raises(ValueError, match="dated"):
        store.write_day(_DAYS[0], [_row(_DAYS[1], "SPY", "risk_manager", 0.1)])
    with pytest.raises(ValueError, match="window"):
        store.write_day(date(2016, 1, 7), [_row(date(2016, 1, 7), "SPY", "risk_manager", 0.1)])


def test_write_and_resume(tmp_path: Path) -> None:
    """A store reopened on the same directory resumes after its last completed day; writing
    that day again replaces its rows, and an earlier day is refused."""
    store, trial = _open(tmp_path)
    store.write_day(_DAYS[0], _day_rows(_DAYS[0]))
    store.write_day(_DAYS[1], _day_rows(_DAYS[1]))
    resumed, _ = _open(tmp_path)

    assert resumed.last_completed() == _DAYS[1]
    with pytest.raises(StoreOrderError):
        resumed.write_day(_DAYS[0], _day_rows(_DAYS[0]))
    resumed.write_day(_DAYS[1], _day_rows(_DAYS[1]))
    assert len(_json_rows(trial)) == len(_day_rows(_DAYS[0])) + len(_day_rows(_DAYS[1]))
    resumed.rewind(_DAYS[0])
    assert resumed.last_completed() == _DAYS[0]
    assert resumed.day_rows(_DAYS[1]) == []
    assert sorted(
        resumed.day_rows(_DAYS[0]), key=lambda row: (row.instrument_id, row.pod_id or "")
    ) == sorted(_day_rows(_DAYS[0]), key=lambda row: (row.instrument_id, row.pod_id or ""))
    with pytest.raises(StoreOrderError):
        resumed.rewind(_DAYS[2])
    resumed.write_day(_DAYS[1], _day_rows(_DAYS[1]))

    assert sorted({row["decision_date"] for row in _json_rows(trial)}) == [
        "2016-01-04",
        "2016-01-05",
    ]
    assert json.loads((trial / "progress.json").read_text()) == {"last_completed": "2016-01-05"}


def test_rows_are_sorted_by_instrument_then_pod(tmp_path: Path) -> None:
    """A day's rows are stored by instrument id, then pod id."""
    store, trial = _open(tmp_path)
    store.write_day(_DAYS[0], _day_rows(_DAYS[0]))
    keys = [(row["instrument_id"], row["pod_id"]) for row in _json_rows(trial)]

    assert keys == sorted(keys)


def test_resume_on_another_commit_is_refused(tmp_path: Path) -> None:
    """A record that ran on one commit resumes only on it; on another the store says to
    restart, or resume on the same commit."""
    store, _ = _open(tmp_path, _record())
    store.write_day(_DAYS[0], _day_rows(_DAYS[0]))
    moved = _record().model_copy(update={"git_sha": "f" * 40})

    with pytest.raises(StoreMismatchError, match=r"restart the record.*same commit"):
        _open(tmp_path, moved)
