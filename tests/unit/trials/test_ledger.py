# test_ledger.py

"""AT-40 round 1: `agentic_trading.trials.ledger` writes pitch-edge's shared, hash-chained trial
ledger format (PE-21) byte for byte, so edgelab can recompute our declared trial count from an
exported file. The format core is pinned by pitch-edge's conformance vectors
(`tests/unit/test_research_ledger_format.py` there); the DuckDB store validates every row before
chaining it, verifies the chain, and exports it as pitch-edge's `export_chain` does."""

import hashlib
import json
import math
import re
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import duckdb
import pytest
from pydantic import BaseModel

from agentic_trading.trials.ledger import (
    COLUMNS,
    DIRECTIONS,
    GENESIS,
    REVIEW_KINDS,
    ChainReport,
    LedgerError,
    Registration,
    TrialLedger,
    TrialRow,
    canonical,
    chain_hash,
    row_json,
    row_key,
    write_export,
)
from agentic_trading.trials.registry import config_hash_of, file_registration

REGISTRATION = {
    "id": "demo-v1",
    "family": "demo",
    "sha256": "ab" * 32,
    "path": "research/registry/demo-v1.yaml",
    "registered_at": datetime(2026, 10, 28, 1, 2, 3),
    "recorded_at": datetime(2026, 10, 28, 1, 2, 4, 500000),
}
TRIAL = {
    "trial_id": "t1",
    "hypothesis_id": "demo-v1",
    "family": "demo",
    "config_json": '{"theta":0.02}',
    "config_hash": "cd" * 32,
    "git_sha": "0" * 40,
    "data_snapshot": "lake@2026-10-28",
    "run_at": datetime(2026, 10, 28, 2, 0, 0),
    "n": 300,
    "metric_name": "clv_log",
    "metric_value": 0.0123,
    "ci_lo": float("nan"),
    "ci_hi": None,
    "p_value": 0.01,
    "direction": "positive",
    "extra_json": "{}",
}
REVIEW = {
    "hypothesis_id": "demo-v1",
    "kind": "amendment",
    "verdict": "RECORDED",
    "detail_json": '{"amends_sha256": "' + "ab" * 32 + '", "rules": ["a rule"]}',
    "recorded_at": datetime(2026, 10, 28, 1, 30, 0, 250000),
}
REGISTRATION_HASH = "13c98f3cdd15afc3fd9573cd933721ee65a81746ae35992fe81d064f361d3894"
TRIAL_HASH = "1887978452d3a9f6fde6560cf0fafbd63acf36bf81a3b20ae06dd1eb2ad1d623"
REVIEW_HASH = "297a6a2f82ae5ea5c8ced6c545e51085a274b03030101d26add9d144e9048bf4"

PITCH_EDGE_COLUMNS = {
    "registrations": ("id", "family", "sha256", "path", "registered_at", "recorded_at"),
    "trials": (
        "trial_id",
        "hypothesis_id",
        "family",
        "config_json",
        "config_hash",
        "git_sha",
        "data_snapshot",
        "run_at",
        "n",
        "metric_name",
        "metric_value",
        "ci_lo",
        "ci_hi",
        "p_value",
        "direction",
        "extra_json",
    ),
    "reviews": ("hypothesis_id", "kind", "verdict", "detail_json", "recorded_at"),
}
_CHAIN_COLUMNS = ("seq", "table_name", "row_key", "prev_hash", "row_hash", "chained_at")
_TABLES = ("registrations", "trials", "reviews", "ledger_chain")
_TYPES = {
    "seq": "BIGINT",
    "n": "BIGINT",
    "metric_value": "DOUBLE",
    "ci_lo": "DOUBLE",
    "ci_hi": "DOUBLE",
    "p_value": "DOUBLE",
    "registered_at": "TIMESTAMP",
    "recorded_at": "TIMESTAMP",
    "run_at": "TIMESTAMP",
    "chained_at": "TIMESTAMP",
}

_REGISTERED_AT = datetime(2026, 10, 28, 1, 2, 3, tzinfo=UTC)
_CONFIG = {"theta": 0.02}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _naive_utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _query(db: Path, sql: str) -> list[dict[str, Any]]:
    """The rows of `sql` against the ledger file `db`, as column -> value dicts."""
    con = duckdb.connect(str(db))
    try:
        cursor = con.execute(sql)
        names = [d[0] for d in cursor.description]
        return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]
    finally:
        con.close()


def _execute(db: Path, sql: str, params: list[Any] | None = None) -> None:
    """Runs `sql` against the ledger file `db` directly, around `TrialLedger`."""
    con = duckdb.connect(str(db))
    try:
        con.execute(sql, params or [])
    finally:
        con.close()


def _hashes(db: Path) -> list[str]:
    return [e["row_hash"] for e in _query(db, "SELECT row_hash FROM ledger_chain ORDER BY seq")]


def _trial(config: dict[str, Any] = _CONFIG, **overrides: Any) -> TrialRow:
    """A valid `demo-v1` trial of `config`, with `overrides` applied."""
    fields: dict[str, Any] = {
        "hypothesis_id": "demo-v1",
        "family": "demo",
        "config_json": json.dumps(config),
        "config_hash": config_hash_of(config),
        "git_sha": "0" * 40,
        "data_snapshot": "lake@2026-10-28",
        "n": 300,
        "metric_name": "sharpe",
        "metric_value": 0.5,
        "ci_lo": 0.1,
        "ci_hi": 0.9,
        "p_value": 0.01,
        "direction": "positive",
        "extra_json": "{}",
    }
    return TrialRow(**(fields | overrides))


def _insert_raw_trial(
    db: Path, config: dict[str, Any] = _CONFIG, **overrides: Any
) -> dict[str, Any]:
    """Inserts a trial row straight into the file, with no chain entry, and returns it."""
    dumped = _trial(config, **overrides).model_dump()
    row = {c: dumped[c] for c in COLUMNS["trials"]}
    marks = ", ".join("?" * len(row))
    _execute(db, f"INSERT INTO trials ({', '.join(row)}) VALUES ({marks})", list(row.values()))
    return row


def _summary(
    upto: str, n_tests: int, keys: list[str], return_keys: list[str] | None = None
) -> dict[str, Any]:
    """The expected summary line of an export of family `demo`."""
    return {
        "summary": {
            "family": "demo",
            "upto": upto,
            "n_tests": n_tests,
            "n_distinct_hypothesis_config": len(keys),
            "n_return_kind": None if return_keys is None else len(return_keys),
            "return_test_keys": return_keys,
            "test_keys": keys,
        }
    }


def _report(report: ChainReport) -> tuple[bool, int, list[str], dict[str, list[str]]]:
    return report.ok, report.n_chained, report.broken, report.unchained_after_start


@pytest.fixture
def db(tmp_path: Path) -> Path:
    return tmp_path / "state" / "ledger" / "trials.duckdb"


@pytest.fixture
def ledger(db: Path, tmp_path: Path) -> TrialLedger:
    return TrialLedger(db, repo_root=tmp_path)


@pytest.fixture
def registration(registry_file: Path, tmp_path: Path) -> Registration:
    return file_registration(
        registry_file, tmp_path, family="demo", registered_at=_REGISTERED_AT, metric_kind="return"
    )


@pytest.fixture
def registered(ledger: TrialLedger, registration: Registration) -> TrialLedger:
    assert ledger.register(registration) is True
    return ledger


def test_constants() -> None:
    """GENESIS, DIRECTIONS, REVIEW_KINDS and COLUMNS are pitch-edge's, in pitch-edge's order."""
    assert GENESIS == "0" * 64
    assert DIRECTIONS == ("positive", "negative", "two-sided")
    assert REVIEW_KINDS == ("audit", "edgelab", "paper", "amendment")
    assert COLUMNS == PITCH_EDGE_COLUMNS
    assert list(COLUMNS) == ["registrations", "trials", "reviews"]


def test_canonical() -> None:
    """canonical makes datetimes and dates ISO strings and NaN "NaN", and keeps the rest."""
    assert canonical(datetime(2026, 10, 28, 1, 2, 4, 500000)) == "2026-10-28T01:02:04.500000"
    assert canonical(datetime(2026, 10, 28, 1, 2, 3, tzinfo=UTC)) == "2026-10-28T01:02:03+00:00"
    assert canonical(date(2026, 10, 28)) == "2026-10-28"
    assert canonical(float("nan")) == "NaN"
    for value in (math.inf, -math.inf, None, "NaN", "x", 300, 0.0123, 0.0, True):
        assert canonical(value) == value
        assert type(canonical(value)) is type(value)


def test_row_json() -> None:
    """row_json is the sorted, compact JSON of the table's columns, as pitch-edge writes it."""
    assert row_json("registrations", REGISTRATION) == (
        '{"family":"demo","id":"demo-v1","path":"research/registry/demo-v1.yaml",'
        '"recorded_at":"2026-10-28T01:02:04.500000","registered_at":"2026-10-28T01:02:03",'
        f'"sha256":"{"ab" * 32}"}}'
    )
    assert row_json("reviews", REVIEW) == (
        '{"detail_json":"{\\"amends_sha256\\": \\"'
        + "ab" * 32
        + '\\", \\"rules\\": [\\"a rule\\"]}",'
        '"hypothesis_id":"demo-v1","kind":"amendment","recorded_at":"2026-10-28T01:30:00.250000",'
        '"verdict":"RECORDED"}'
    )
    trial = row_json("trials", TRIAL)
    assert '"ci_lo":"NaN"' in trial
    assert '"ci_hi":null' in trial
    assert '"n":300,' in trial
    assert list(json.loads(trial)) == sorted(COLUMNS["trials"])
    assert row_json("trials", TRIAL | {"unlisted": 1}) == trial

    edge = row_json(
        "trials",
        TRIAL
        | {
            "metric_value": math.inf,
            "ci_lo": -math.inf,
            "p_value": None,
            "run_at": date(2026, 10, 28),
        },
    )
    assert '"metric_value":Infinity' in edge
    assert '"ci_lo":-Infinity' in edge
    assert '"p_value":null' in edge
    assert '"run_at":"2026-10-28"' in edge


def test_row_key() -> None:
    """Registrations key on id, trials on trial_id, reviews on the sha256 of their row JSON."""
    assert row_key("registrations", REGISTRATION) == "demo-v1"
    assert row_key("trials", TRIAL) == "t1"
    assert row_key("trials", TRIAL | {"trial_id": 7}) == "7"
    assert row_key("reviews", REVIEW) == _sha(row_json("reviews", REVIEW))


def test_chain_hash_vectors() -> None:
    """chain_hash reproduces pitch-edge's three conformance vectors, chained in order."""
    first = chain_hash(GENESIS, "registrations", REGISTRATION)
    assert first == REGISTRATION_HASH
    assert first == _sha(f"{GENESIS}|registrations|{row_json('registrations', REGISTRATION)}")
    assert chain_hash(first, "trials", TRIAL) == TRIAL_HASH
    assert chain_hash(TRIAL_HASH, "reviews", REVIEW) == REVIEW_HASH


def test_schema(ledger: TrialLedger, registration: Registration, db: Path) -> None:
    """The first write creates the file, its parent directories and pitch-edge's four tables."""
    assert not db.parent.exists()
    ledger.verify_chain()
    assert not db.exists()
    ledger.register(registration)
    assert db.exists()

    found: dict[str, list[tuple[str, str]]] = {}
    for c in _query(
        db,
        "SELECT table_name, column_name, data_type FROM information_schema.columns "
        "ORDER BY table_name, ordinal_position",
    ):
        if c["table_name"] in _TABLES:
            found.setdefault(c["table_name"], []).append((c["column_name"], c["data_type"]))
    columns = PITCH_EDGE_COLUMNS | {"ledger_chain": _CHAIN_COLUMNS}
    expected = {t: [(c, _TYPES.get(c, "VARCHAR")) for c in cols] for t, cols in columns.items()}
    assert found == expected

    keys = _query(
        db,
        "SELECT table_name, constraint_column_names FROM duckdb_constraints() "
        "WHERE constraint_type = 'PRIMARY KEY'",
    )
    assert {k["table_name"]: list(k["constraint_column_names"]) for k in keys} == {
        "registrations": ["id"],
        "trials": ["trial_id"],
        "ledger_chain": ["seq"],
    }


def test_register(ledger: TrialLedger, registry_file: Path, tmp_path: Path, db: Path) -> None:
    """register chains a naive-UTC row with the repo-relative path, once per id and file hash."""
    registered_at = datetime(2026, 10, 28, 3, 2, 3, tzinfo=timezone(timedelta(hours=2)))
    registration = file_registration(
        registry_file, tmp_path, family="demo", registered_at=registered_at, metric_kind="return"
    )
    before = _naive_utc_now()
    assert ledger.register(registration) is True
    after = _naive_utc_now()

    [row] = _query(db, "SELECT * FROM registrations")
    assert {c: row[c] for c in ("id", "family", "sha256", "path", "registered_at")} == {
        "id": "demo-v1",
        "family": "demo",
        "sha256": hashlib.sha256(registry_file.read_bytes()).hexdigest(),
        "path": "research/registry/demo-v1.yaml",
        "registered_at": datetime(2026, 10, 28, 1, 2, 3),
    }
    assert before <= row["recorded_at"] <= after
    [entry] = _query(db, "SELECT * FROM ledger_chain")
    assert (entry["seq"], entry["table_name"], entry["row_key"], entry["prev_hash"]) == (
        0,
        "registrations",
        "demo-v1",
        GENESIS,
    )
    assert entry["row_hash"] == chain_hash(GENESIS, "registrations", row)

    assert ledger.register(registration) is False
    registry_file.write_text(registry_file.read_text() + "# edited\n")
    edited = file_registration(
        registry_file, tmp_path, family="demo", registered_at=registered_at, metric_kind="return"
    )
    with pytest.raises(LedgerError):
        ledger.register(edited)
    assert _query(db, "SELECT * FROM registrations") == [row]
    assert _query(db, "SELECT * FROM ledger_chain") == [entry]


def test_record_trial(registered: TrialLedger, db: Path) -> None:
    """record_trial chains every valid row, boundary values included, under its trial_id."""
    before = _naive_utc_now()
    first, second = _trial(), _trial()
    after = _naive_utc_now()
    assert re.fullmatch(r"[0-9a-f]{32}", first.trial_id)
    assert first.trial_id != second.trial_id
    assert first.run_at.tzinfo is None
    assert before <= first.run_at <= after

    accepted = [
        first,
        _trial(p_value=0.0, metric_value=0.3, ci_lo=0.3, ci_hi=0.3),
        _trial({"theta": 0.05}, p_value=1.0, direction="negative"),
        _trial(p_value=None, metric_value=None, ci_lo=float("nan"), ci_hi=-1.0),
        _trial(ci_lo=None, ci_hi=None, direction="two-sided"),
        _trial(ci_lo=-math.inf, ci_hi=0.0),
    ]
    for row in accepted:
        registered.record_trial(row)

    chain = _query(db, "SELECT * FROM ledger_chain ORDER BY seq")
    stored = {r["trial_id"]: r for r in _query(db, "SELECT * FROM trials")}
    assert [e["table_name"] for e in chain] == ["registrations", *["trials"] * len(accepted)]
    assert [e["row_key"] for e in chain[1:]] == [r.trial_id for r in accepted]
    for previous, entry, row in zip(chain, chain[1:], accepted, strict=False):
        assert entry["prev_hash"] == previous["row_hash"]
        assert entry["row_hash"] == chain_hash(entry["prev_hash"], "trials", stored[row.trial_id])
        assert entry["row_hash"] == chain_hash(entry["prev_hash"], "trials", row.model_dump())
    assert registered.verify_chain().ok


def test_record_trial_refuses(registered: TrialLedger, db: Path) -> None:
    """record_trial raises LedgerError for each invalid row and leaves the ledger unchanged."""
    first = _trial()
    registered.record_trial(first)
    refused = [
        _trial(hypothesis_id="ghost-v1"),
        _trial(family="other"),
        _trial(config_hash=config_hash_of({"theta": 0.03})),
        _trial(config_json=json.dumps({"theta": 0.03})),
        _trial(p_value=-0.01),
        _trial(p_value=1.01),
        _trial(ci_lo=0.9, ci_hi=0.1),
        _trial(direction="up"),
        _trial(trial_id=first.trial_id),
    ]
    for row in refused:
        with pytest.raises(LedgerError):
            registered.record_trial(row)
    assert [r["trial_id"] for r in _query(db, "SELECT trial_id FROM trials")] == [first.trial_id]
    assert len(_hashes(db)) == 2


def test_record_review(registered: TrialLedger, db: Path) -> None:
    """record_review stores detail as default-separator JSON, chains it, and guards amendments."""
    before = _naive_utc_now()
    detail = {"rules": ["a rule"], "amends_sha256": "ab" * 32}
    registered.record_review("demo-v1", "amendment", "RECORDED", detail)
    registered.record_review("demo-v1", "audit", "PASS")
    after = _naive_utc_now()

    [amendment] = _query(db, "SELECT * FROM reviews WHERE kind = 'amendment'")
    [audit] = _query(db, "SELECT * FROM reviews WHERE kind = 'audit'")
    assert amendment["detail_json"] == REVIEW["detail_json"]
    assert (amendment["hypothesis_id"], amendment["verdict"]) == ("demo-v1", "RECORDED")
    assert audit["detail_json"] == "{}"
    assert before <= amendment["recorded_at"] <= audit["recorded_at"] <= after
    chain = _query(db, "SELECT * FROM ledger_chain WHERE table_name = 'reviews' ORDER BY seq")
    assert [e["row_key"] for e in chain] == [row_key("reviews", r) for r in (amendment, audit)]
    assert chain[0]["row_hash"] == chain_hash(chain[0]["prev_hash"], "reviews", amendment)
    assert chain[1]["row_hash"] == chain_hash(chain[1]["prev_hash"], "reviews", audit)

    with pytest.raises(LedgerError):
        registered.record_review("demo-v1", "vibes", "PASS")
    with pytest.raises(LedgerError):
        registered.record_review("ghost-v1", "audit", "PASS")
    registered.record_trial(_trial())
    with pytest.raises(LedgerError):
        registered.record_review("demo-v1", "amendment", "RECORDED", {"rules": ["late"]})
    for kind in ("audit", "edgelab", "paper"):
        registered.record_review("demo-v1", kind, "PASS", {"n": 1})
    assert len(_query(db, "SELECT * FROM reviews")) == 5
    assert len(_hashes(db)) == 1 + 2 + 1 + 3


def test_verify_chain(registered: TrialLedger, db: Path, tmp_path: Path) -> None:
    """verify_chain passes an intact ledger and catches edited, deleted and unchained rows."""
    fresh_path = tmp_path / "fresh" / "ledger.duckdb"
    fresh = TrialLedger(fresh_path, repo_root=tmp_path)
    assert _report(fresh.verify_chain()) == (True, 0, [], {})
    assert not fresh_path.exists()
    with pytest.raises(LedgerError):
        fresh.record_review("demo-v1", "audit", "PASS")
    assert _report(fresh.verify_chain()) == (True, 0, [], {})

    trial = _trial()
    registered.record_trial(trial)
    registered.record_review("demo-v1", "audit", "PASS")
    report = registered.verify_chain()
    assert isinstance(report, ChainReport)
    assert isinstance(report, BaseModel)
    assert _report(report) == (True, 3, [], {})

    _execute(db, "UPDATE trials SET n = n + 1")
    report = registered.verify_chain()
    assert report.ok is False
    assert any(entry.startswith(f"trials:{trial.trial_id}") for entry in report.broken)
    assert report.unchained_after_start == {}
    _execute(db, "UPDATE trials SET n = n - 1")
    assert _report(registered.verify_chain()) == (True, 3, [], {})

    _insert_raw_trial(db, trial_id="raw1")
    assert _report(registered.verify_chain()) == (False, 3, [], {"trials": ["raw1"]})
    _execute(db, "DELETE FROM trials WHERE trial_id = 'raw1'")
    assert registered.verify_chain().ok is True

    _execute(db, "DELETE FROM reviews")
    report = registered.verify_chain()
    assert report.ok is False
    assert any(entry.startswith("reviews:") for entry in report.broken)


def test_export_chain(
    registered: TrialLedger,
    registration: Registration,
    registry_file: Path,
    db: Path,
    tmp_path: Path,
) -> None:
    """export_chain emits the chain to upto, the family's registry files and its trial summary."""
    other_file = registry_file.with_name("other-v1.yaml")
    other_file.write_text("id: other-v1\n")
    other = file_registration(
        other_file, tmp_path, family="other", registered_at=_REGISTERED_AT, metric_kind="return"
    )
    assert registered.register(other) is True
    registered.record_trial(_trial(trial_id="t1"))
    registered.record_trial(_trial(trial_id="t2"))
    registered.record_trial(_trial({"theta": 0.05}, trial_id="t3"))
    registered.record_trial(_trial(hypothesis_id="other-v1", family="other", trial_id="o1"))
    registered.record_review("demo-v1", "audit", "PASS")
    hashes = _hashes(db)
    keys = sorted(f"demo-v1|{config_hash_of(c)}" for c in (_CONFIG, {"theta": 0.05}))

    lines = registered.export_chain("demo", hashes[-1])
    chained = lines[: len(hashes)]
    assert [line["row_hash"] for line in chained] == hashes
    assert [line["table"] for line in chained] == [
        "registrations",
        "registrations",
        "trials",
        "trials",
        "trials",
        "trials",
        "reviews",
    ]
    prev = GENESIS
    for line in chained:
        assert set(line) == {"table", "row_canonical", "prev_hash", "row_hash"}
        assert line["prev_hash"] == prev
        assert _sha(f"{prev}|{line['table']}|{line['row_canonical']}") == line["row_hash"]
        prev = line["row_hash"]
    yaml_line = {
        "table": "registration_yaml",
        "hypothesis_id": "demo-v1",
        "sha256": registration.source_sha256,
        "yaml": registry_file.read_bytes().decode("utf-8"),
    }
    assert lines[len(hashes) :] == [yaml_line, _summary(hashes[-1], 3, keys)]

    with_kinds = registered.export_chain("demo", hashes[-1], {"demo-v1": "return"})
    assert with_kinds[-1] == _summary(hashes[-1], 3, keys, keys)
    no_return = {"demo-v1": None, "other-v1": "return"}
    assert registered.export_chain("demo", hashes[-1], no_return)[-1] == (
        _summary(hashes[-1], 3, keys, [])
    )

    partial = registered.export_chain("demo", hashes[3])
    assert [line.get("row_hash") for line in partial[:4]] == hashes[:4]
    first_key = [f"demo-v1|{config_hash_of(_CONFIG)}"]
    assert partial[4:] == [yaml_line, _summary(hashes[3], 2, first_key)]


def test_export_chain_unchained_and_mismatch(
    registered: TrialLedger, registration: Registration, registry_file: Path, db: Path
) -> None:
    """export_chain lists unchained family rows and flags an edited or missing registry file."""
    registered.record_trial(_trial(trial_id="t1"))
    upto = _hashes(db)[-1]
    raw = _insert_raw_trial(db, {"theta": 0.07}, trial_id="raw1")
    _insert_raw_trial(db, trial_id="raw2", hypothesis_id="other-v1", family="other")
    keys = sorted(f"demo-v1|{config_hash_of(c)}" for c in (_CONFIG, {"theta": 0.07}))

    lines = registered.export_chain("demo", upto)
    assert len(lines) == 5
    assert lines[2] == {
        "table": "trials",
        "row_canonical": row_json("trials", raw),
        "chained": False,
    }
    assert lines[3]["yaml"] == registry_file.read_bytes().decode("utf-8")
    assert lines[4] == _summary(upto, 2, keys)

    mismatch = {
        "table": "registration_yaml",
        "hypothesis_id": "demo-v1",
        "sha256": registration.source_sha256,
        "yaml": None,
        "mismatch": True,
    }
    registry_file.write_text(registry_file.read_text() + "# edited\n")
    assert registered.export_chain("demo", upto)[3] == mismatch
    registry_file.unlink()
    assert registered.export_chain("demo", upto)[3] == mismatch


def test_export_chain_refuses(registered: TrialLedger, db: Path, tmp_path: Path) -> None:
    """export_chain raises LedgerError for no ledger, an unknown upto, or a break before upto."""
    with pytest.raises(LedgerError):
        TrialLedger(tmp_path / "none.duckdb", repo_root=tmp_path).export_chain("demo", GENESIS)
    registered.record_trial(_trial())
    registered.record_review("demo-v1", "audit", "PASS")
    hashes = _hashes(db)
    with pytest.raises(LedgerError):
        registered.export_chain("demo", "f" * 64)
    with pytest.raises(LedgerError):
        registered.export_chain("demo", GENESIS)

    _execute(db, "UPDATE trials SET n = n + 1")
    for upto in hashes[1:]:
        with pytest.raises(LedgerError):
            registered.export_chain("demo", upto)
    lines = registered.export_chain("demo", hashes[0])
    assert len(lines) == 3
    assert lines[0]["row_hash"] == hashes[0]
    assert lines[-1]["summary"]["n_tests"] == 0


def test_write_export(registered: TrialLedger, db: Path, tmp_path: Path) -> None:
    """write_export writes one sorted-key JSON line per entry, from which the chain recomputes."""
    out = tmp_path / "small.jsonl"
    write_export([{"b": 1, "a": None}, {"summary": {"z": [1, 2], "y": "x"}}], out)
    assert out.read_text() == '{"a": null, "b": 1}\n{"summary": {"y": "x", "z": [1, 2]}}\n'

    registered.record_trial(_trial(ci_lo=float("nan")))
    lines = registered.export_chain("demo", _hashes(db)[-1])
    out = tmp_path / "ledger" / "demo.jsonl"
    out.parent.mkdir()
    write_export(lines, out)
    assert out.read_text() == "".join(json.dumps(line, sort_keys=True) + "\n" for line in lines)

    parsed = [json.loads(text) for text in out.read_text().splitlines()]
    prev = GENESIS
    for line in (p for p in parsed if "row_hash" in p):
        assert _sha(f"{prev}|{line['table']}|{line['row_canonical']}") == line["row_hash"]
        prev = line["row_hash"]
    assert prev == _hashes(db)[-1]
    assert parsed[-1]["summary"]["n_tests"] == 1
