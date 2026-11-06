# test_rows.py

"""AT-40/AT-95 round 2a: `agentic_trading.trials.rows` turns stage (a) batches and imported past
windows into the trial-ledger rows edgelab counts. One firm family (`settings.trials`); each stage
(a) analyst is its own hypothesis; a trial is keyed by (hypothesis_id, config_hash), the config
being the frozen spec with its window; a rerun is another row; each imported (window, issue) is
its own key and is never merged with another."""

import hashlib
import json
import re
from datetime import UTC, date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import duckdb
import pytest

from agentic_trading import provenance
from agentic_trading.agents import roster
from agentic_trading.config import settings, temporary_overrides
from agentic_trading.data.trading_calendar import sessions_between
from agentic_trading.stage_a.spec import DECISION_RULE, SpecRecord, StageASpec, spec_hash
from agentic_trading.trials.ledger import LedgerError, Registration, TrialLedger, TrialRow
from agentic_trading.trials.registry import config_hash_of, file_registration
from agentic_trading.trials.rows import (
    TrialProvenance,
    current_trial_provenance,
    ensure_stage_a_registration,
    import_prior_exposure,
    imported_trial_rows,
    stage_a_hypothesis_id,
    stage_a_registration_text,
    stage_a_trial_row,
)

# The registry's id pattern (`<name>-v<version>`), as `trials.registry` states it.
_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9-]*-v[0-9]+")
_FAMILY = "agentic-firm"
_AGENT = "cir_rate_regime_analyst"
_CLASS = "agentic_trading.agents.analysts.rates.cir_rate_regime.CIRRateRegimeAnalyst"
_IMPORTED_ID = "prior-exposure-firm-v1"
_FIRST_AT = datetime(2026, 11, 5, 9, 30, tzinfo=UTC)
_LATER_AT = datetime(2026, 11, 6, 9, 30, tzinfo=UTC)
_SPEC = StageASpec(
    agent_id=_AGENT,
    analyst_class=_CLASS,
    analyst_params={"agent_id": _AGENT, "lookback_days": 252},
    settings={"stage_a": {"data_end": "2024-12-31", "session_calendar": "XNYS"}},
    module_hashes={"agentic_trading.agents.analysts.rates.cir_rate_regime": "ab" * 32},
    universe=["TLT", "SHY", "IEF"],
    window_start=date(2010, 1, 4),
    window_end=date(2024, 12, 31),
    decision_rule=DECISION_RULE,
    quantcore_sha="e" * 40,
)
_RECORD = SpecRecord(
    spec=_SPEC,
    spec_hash=spec_hash(_SPEC),
    lake_time=datetime(2026, 10, 29, 12, 0, tzinfo=UTC),
    lake_version="f89aef2f428a",
    quantcore_version="0.3.0",
)
_PROVENANCE = TrialProvenance(agentic_sha="a" * 40, agentic_dirty=True, quantcore_sha="b" * 40)


def _naive_utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _query(db: Path, sql: str) -> list[dict[str, Any]]:
    """The rows of `sql` against the ledger file `db`, as column -> value dicts."""
    con = duckdb.connect(str(db), read_only=True)
    try:
        cursor = con.execute(sql)
        names = [d[0] for d in cursor.description]
        return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]
    finally:
        con.close()


def _ledger(root: Path) -> TrialLedger:
    """A ledger at `settings.trials.ledger_path` under the repository root `root`."""
    return TrialLedger(root / settings.trials.ledger_path, repo_root=root)


def _text(registered_at: datetime) -> str:
    return stage_a_registration_text(
        agent_id=_AGENT, analyst_class=_CLASS, family=_FAMILY, registered_at=registered_at
    )


def _ensure(ledger: TrialLedger, root: Path, now: datetime, agent_id: str = _AGENT) -> Registration:
    return ensure_stage_a_registration(
        ledger=ledger,
        repo_root=root,
        agent_id=agent_id,
        analyst_class=_CLASS,
        family=_FAMILY,
        now=now,
    )


def _canonical(spec: StageASpec) -> str:
    """The string `spec_hash` hashes, spelled out."""
    payload = spec.model_dump(mode="json")
    payload["universe"] = sorted(payload["universe"])
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _register_imported(ledger: TrialLedger, root: Path) -> None:
    """Registers `_IMPORTED_ID` in `_FAMILY` from a registry file under `root`."""
    path = root / settings.trials.registry_dir / f"{_IMPORTED_ID}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"id: {_IMPORTED_ID}\n")
    registration = file_registration(
        path, root, family=_FAMILY, registered_at=_FIRST_AT, metric_kind="return"
    )
    assert ledger.register(registration) is True


def _last_hash(ledger: TrialLedger) -> str:
    [entry] = _query(ledger.path, "SELECT row_hash FROM ledger_chain ORDER BY seq DESC LIMIT 1")
    return str(entry["row_hash"])


def test_trials_settings() -> None:
    """settings.trials holds the ledger path, registry dir and firm family; overrides apply."""
    defaults = {
        "ledger_path": "research/ledger.duckdb",
        "registry_dir": "research/registry",
        "firm_family": "agentic-firm",
    }
    assert settings.trials.model_dump() == defaults
    assert type(settings.trials)().model_dump() == defaults
    overrides = {
        "trials.ledger_path": "state/trials.duckdb",
        "trials.registry_dir": "state/registry",
        "trials.firm_family": "other-firm",
    }
    with temporary_overrides(overrides):
        assert settings.trials.ledger_path == "state/trials.duckdb"
        assert settings.trials.registry_dir == "state/registry"
        assert settings.trials.firm_family == "other-firm"
    assert settings.trials.model_dump() == defaults


def test_stage_a_hypothesis_id() -> None:
    """A stage (a) hypothesis id is stage-a-<agent id, hyphenated>-v1 and a valid registry id."""
    known = {
        "cir_rate_regime_analyst": "stage-a-cir-rate-regime-analyst-v1",
        "gbm_momentum_analyst": "stage-a-gbm-momentum-analyst-v1",
        "ou_mean_reversion_analyst": "stage-a-ou-mean-reversion-analyst-v1",
    }
    for agent_id, hypothesis_id in known.items():
        assert stage_a_hypothesis_id(agent_id) == hypothesis_id
    roster_ids: set[str] = set().union(
        *(ids for name, ids in vars(roster).items() if name.endswith("_ANALYST_IDS"))
    )
    assert _AGENT in roster_ids
    for agent_id in roster_ids:
        hypothesis_id = stage_a_hypothesis_id(agent_id)
        assert hypothesis_id == "stage-a-" + agent_id.replace("_", "-") + "-v1"
        assert _ID_PATTERN.fullmatch(hypothesis_id), hypothesis_id


def test_stage_a_registration_text() -> None:
    """The registration text is exactly its eight lines, in order, registered_at in UTC with Z."""
    expected = (
        "id: stage-a-cir-rate-regime-analyst-v1\n"
        "family: agentic-firm\n"
        "registered_at: 2026-11-05T09:30:00Z\n"
        "kind: stage_a_signal_batch\n"
        "agent_id: cir_rate_regime_analyst\n"
        f"analyst_class: {_CLASS}\n"
        "metric_kind: return\n"
        "primary_metric: edgelab stage (a) grade\n"
    )
    assert _text(_FIRST_AT) == expected
    assert _text(_FIRST_AT) == _text(_FIRST_AT)
    assert _text(_FIRST_AT.astimezone(timezone(timedelta(hours=2)))) == expected


def test_ensure_stage_a_registration(tmp_path: Path) -> None:
    """The first call writes and registers the file; later calls reuse it; an edit is refused."""
    ledger = _ledger(tmp_path)
    hypothesis_id = stage_a_hypothesis_id(_AGENT)
    relative = f"{settings.trials.registry_dir}/{hypothesis_id}.yaml"
    path = tmp_path / relative
    assert not path.parent.exists()

    first = _ensure(ledger, tmp_path, _FIRST_AT)
    raw = path.read_bytes()
    assert raw == _text(_FIRST_AT).encode()
    assert first == file_registration(
        path, tmp_path, family=_FAMILY, registered_at=_FIRST_AT, metric_kind="return"
    )
    registrations = _query(ledger.path, "SELECT * FROM registrations")
    assert [
        {c: r[c] for c in ("id", "family", "sha256", "path", "registered_at")}
        for r in registrations
    ] == [
        {
            "id": hypothesis_id,
            "family": _FAMILY,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "path": relative,
            "registered_at": datetime(2026, 11, 5, 9, 30),
        }
    ]
    chain = _query(ledger.path, "SELECT * FROM ledger_chain")
    assert len(chain) == 1

    second = _ensure(ledger, tmp_path, _LATER_AT)
    assert path.read_bytes() == raw
    assert (second.id, second.family, second.metric_kind) == (hypothesis_id, _FAMILY, "return")
    assert (second.source_path, second.source_sha256) == (first.source_path, first.source_sha256)
    assert _query(ledger.path, "SELECT * FROM registrations") == registrations
    assert _query(ledger.path, "SELECT * FROM ledger_chain") == chain

    path.write_bytes(raw + b"# edited\n")
    with pytest.raises(LedgerError):
        _ensure(ledger, tmp_path, _LATER_AT)
    assert path.read_bytes() == raw + b"# edited\n"
    assert _query(ledger.path, "SELECT * FROM registrations") == registrations
    assert _query(ledger.path, "SELECT * FROM ledger_chain") == chain

    with temporary_overrides({"trials.registry_dir": "state/registry"}):
        moved = _ensure(ledger, tmp_path, _FIRST_AT, agent_id="gbm_momentum_analyst")
    assert moved.source_path == Path("state/registry/stage-a-gbm-momentum-analyst-v1.yaml")
    assert (tmp_path / moved.source_path).is_file()
    assert ledger.verify_chain().ok


def test_current_trial_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    """current_trial_provenance returns a TrialProvenance, with null git fields without git."""
    assert set(TrialProvenance.model_fields) == {"agentic_sha", "agentic_dirty", "quantcore_sha"}
    current = current_trial_provenance()
    assert isinstance(current, TrialProvenance)
    assert current.agentic_dirty is None or isinstance(current.agentic_dirty, bool)

    monkeypatch.setattr(provenance, "_git_output", lambda *args: None)
    unknown = current_trial_provenance()
    assert isinstance(unknown, TrialProvenance)
    assert (unknown.agentic_sha, unknown.agentic_dirty) == (None, None)


def test_stage_a_trial_row(tmp_path: Path) -> None:
    """A stage (a) trial row carries the spec's canonical JSON keyed by its spec_hash."""
    hypothesis_id = stage_a_hypothesis_id(_AGENT)
    run_at = datetime(2026, 11, 5, 9, 30)
    row = stage_a_trial_row(
        _RECORD, hypothesis_id, _FAMILY, n_days=7, provenance=_PROVENANCE, run_at=run_at
    )
    extra = {"agentic_sha": "a" * 40, "agentic_dirty": True, "quantcore_sha": "b" * 40}
    assert isinstance(row, TrialRow)
    assert row.model_dump(exclude={"trial_id"}) == {
        "hypothesis_id": hypothesis_id,
        "family": _FAMILY,
        "config_json": _canonical(_SPEC),
        "config_hash": _RECORD.spec_hash,
        "git_sha": "a" * 40,
        "data_snapshot": "2026-10-29T12:00:00+00:00|f89aef2f428a",
        "n": 7,
        "metric_name": "submitted",
        "metric_value": None,
        "ci_lo": None,
        "ci_hi": None,
        "p_value": None,
        "direction": "two-sided",
        "extra_json": json.dumps(extra | {"imported": False}, sort_keys=True),
        "run_at": run_at,
    }
    config = json.loads(row.config_json)
    assert hashlib.sha256(row.config_json.encode()).hexdigest() == _RECORD.spec_hash
    assert config_hash_of(config) == row.config_hash
    assert config["universe"] == ["IEF", "SHY", "TLT"]
    assert (config["window_start"], config["window_end"]) == ("2010-01-04", "2024-12-31")

    unknown = TrialProvenance(agentic_sha=None, agentic_dirty=None, quantcore_sha=None)
    before = _naive_utc_now()
    rerun = stage_a_trial_row(_RECORD, hypothesis_id, _FAMILY, n_days=0, provenance=unknown)
    after = _naive_utc_now()
    assert before <= rerun.run_at <= after
    assert rerun.git_sha == "unknown"
    assert rerun.n == 0
    assert rerun.extra_json == json.dumps(
        {"agentic_sha": None, "agentic_dirty": None, "quantcore_sha": None, "imported": False},
        sort_keys=True,
    )
    assert rerun.config_hash == row.config_hash
    assert rerun.trial_id != row.trial_id

    ledger = _ledger(tmp_path)
    _ensure(ledger, tmp_path, _FIRST_AT)
    ledger.record_trial(row)
    ledger.record_trial(rerun)
    assert ledger.verify_chain().ok


def test_imported_trial_rows(prior_exposure: dict[str, Any]) -> None:
    """Each (results_seen window, issue) is one deterministic row with its own key; nothing else."""
    rows = imported_trial_rows(prior_exposure, _IMPORTED_ID, _FAMILY)
    expected = []
    for entry in prior_exposure["results_seen"]:
        start, end = date.fromisoformat(entry["start"]), date.fromisoformat(entry["end"])
        for issue in entry["issues"]:
            config = {
                "window_start": entry["start"],
                "window_end": entry["end"],
                "universe": sorted(entry["universe"]),
                "issue": issue,
            }
            config_hash = config_hash_of(config)
            extra = {
                "imported": True,
                "spec_reconstructed": False,
                "seen": entry["seen"],
                "printed": entry["printed"],
                "agentic_sha": None,
                "agentic_dirty": None,
                "quantcore_sha": None,
            }
            trial_key = f"imported|{_IMPORTED_ID}|{config_hash}"
            expected.append(
                TrialRow(
                    hypothesis_id=_IMPORTED_ID,
                    family=_FAMILY,
                    config_json=json.dumps(config, sort_keys=True, separators=(",", ":")),
                    config_hash=config_hash,
                    git_sha="unknown",
                    data_snapshot="unknown",
                    n=len(sessions_between(start, end, "XNYS")),
                    metric_name="firm_sharpe_diagnostic",
                    metric_value=None,
                    ci_lo=None,
                    ci_hi=None,
                    p_value=None,
                    direction="two-sided",
                    extra_json=json.dumps(extra, sort_keys=True),
                    trial_id=hashlib.sha256(trial_key.encode()).hexdigest()[:32],
                    run_at=datetime.combine(date.fromisoformat(entry["seen"]), time()),
                ).model_dump()
            )

    def by_id(row: dict[str, Any]) -> str:
        return str(row["trial_id"])

    assert all(isinstance(row, TrialRow) for row in rows)
    assert sorted((row.model_dump() for row in rows), key=by_id) == sorted(expected, key=by_id)
    n_by_issue = {json.loads(row.config_json)["issue"]: row.n for row in rows}
    assert n_by_issue == {"AT-1": 5, "AT-2 part 1": 5, "AT-3": 4}
    assert len({row.config_hash for row in rows}) == len(rows) == 3
    assert imported_trial_rows(prior_exposure, _IMPORTED_ID, _FAMILY) == rows
    assert imported_trial_rows(prior_exposure | {"results_seen": []}, _IMPORTED_ID, _FAMILY) == []


def test_import_prior_exposure(tmp_path: Path, prior_exposure: dict[str, Any]) -> None:
    """import_prior_exposure records each new imported row once; the export counts every key."""
    ledger = _ledger(tmp_path)
    _register_imported(ledger, tmp_path)
    rows = imported_trial_rows(prior_exposure, _IMPORTED_ID, _FAMILY)

    assert import_prior_exposure(ledger, prior_exposure, _IMPORTED_ID, _FAMILY) == 3
    stored = {r["trial_id"]: r for r in _query(ledger.path, "SELECT * FROM trials")}
    assert stored == {row.trial_id: row.model_dump() for row in rows}
    chain = _query(ledger.path, "SELECT * FROM ledger_chain")
    assert import_prior_exposure(ledger, prior_exposure, _IMPORTED_ID, _FAMILY) == 0
    assert _query(ledger.path, "SELECT * FROM ledger_chain") == chain

    keys = sorted(f"{_IMPORTED_ID}|{row.config_hash}" for row in rows)
    export = ledger.export_chain(_FAMILY, _last_hash(ledger), {_IMPORTED_ID: "return"})
    summary = export[-1]["summary"]
    assert summary["test_keys"] == keys
    assert summary["return_test_keys"] == keys
    assert summary["n_tests"] == summary["n_distinct_hypothesis_config"] == 3

    later = {
        "start": "2014-06-02",
        "end": "2014-06-06",
        "universe": ["GLD"],
        "issues": ["AT-6"],
        "seen": "2026-11-01",
        "printed": "firm Sharpe",
    }
    grown = prior_exposure | {"results_seen": [*prior_exposure["results_seen"], later]}
    assert import_prior_exposure(ledger, grown, _IMPORTED_ID, _FAMILY) == 1
    assert len(_query(ledger.path, "SELECT trial_id FROM trials")) == 4
    assert ledger.verify_chain().ok
