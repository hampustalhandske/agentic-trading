# test_runner_ledger.py

"""AT-40/AT-95 round 2a: `run_batch(..., ledger=, repo_root=)` registers each analyst's stage (a)
hypothesis in `settings.trials.firm_family` before writing its first day and, after counts.json,
records one trial row for a call that wrote a day, keyed by the trial's spec hash with n the
decision days stored. A rerun that writes nothing adds no row; a resumed batch adds an exposure
row under the same key; without a ledger the batch writes nothing outside its store."""

import inspect
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pytest
from pydantic import PrivateAttr

from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.config import settings, temporary_overrides
from agentic_trading.stage_a import runner
from agentic_trading.stage_a.runner import BatchResult, run_batch
from agentic_trading.stage_a.spec import SpecRecord
from agentic_trading.trials import rows
from agentic_trading.trials.ledger import COLUMNS, TrialLedger
from agentic_trading.trials.rows import TrialProvenance, stage_a_hypothesis_id, stage_a_trial_row

_UNIVERSE = ("SHY", "IEF", "TLT")
_START, _END = date(2010, 1, 14), date(2010, 1, 21)
# The NYSE sessions from _START to _END: 16-17 January is a weekend, 18 January Martin Luther
# King Day.
_SESSIONS = [date(2010, 1, 14), date(2010, 1, 15), date(2010, 1, 19), date(2010, 1, 20), _END]
_META: dict[str, Any] = {
    "lake_time": datetime(2026, 10, 29, tzinfo=UTC),
    "lake_version": "f89aef2f428a",
    "quantcore_version": "0.4.0",
}
_PROVENANCE = TrialProvenance(agentic_sha="c" * 40, agentic_dirty=False, quantcore_sha="d" * 40)


def _query(db: Path, sql: str) -> list[dict[str, Any]]:
    """The rows of `sql` against the ledger file `db`, as column -> value dicts."""
    con = duckdb.connect(str(db), read_only=True)
    try:
        cursor = con.execute(sql)
        names = [d[0] for d in cursor.description]
        return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]
    finally:
        con.close()


def _is_registered(db: Path, hypothesis_id: str) -> bool:
    """Whether the ledger file `db` holds a registration of `hypothesis_id`."""
    if not db.exists():
        return False
    return bool(_query(db, f"SELECT id FROM registrations WHERE id = '{hypothesis_id}'"))


class _Probe(AnalystAgent):
    """Goes long every asset at confidence 0.5; when `ledger_file` is set, records per decision
    day whether that ledger holds the analyst's stage (a) registration."""

    agent_id: str = "fake_probe"
    ledger_file: str = ""
    _registered: dict[date, bool] = PrivateAttr(default_factory=dict)

    def generate_signal(
        self, asset: str, market_data: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> AgentSignal:
        if self.ledger_file:
            hypothesis_id = stage_a_hypothesis_id(self.agent_id)
            self._registered[as_of_date.date()] = _is_registered(
                Path(self.ledger_file), hypothesis_id
            )
        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction="long",
            confidence=0.5,
            rationale="",
        )


class _Resources:
    """A fake resource check answering `answers` in order, then True."""

    def __init__(self, *answers: bool) -> None:
        self._answers = list(answers)

    def __call__(self, store_dir: Path) -> bool:
        return self._answers.pop(0) if self._answers else True


def _market_data() -> pd.DataFrame:
    """Prices in the (asset, field) column shape of `load_price_history_from_lakehouse`, on every
    business day from before the window to after it."""
    index = pd.bdate_range("2009-12-01", "2010-02-26", name="date")
    close = pd.Series(np.linspace(100.0, 110.0, len(index)), index=index)
    fields = {
        "open": close,
        "close": close,
        "raw_close": close,
        "volume": pd.Series(1e6, index=index),
        "log_return": np.log(close / close.shift(1)),
    }
    return pd.DataFrame({(a, field): v for a in _UNIVERSE for field, v in fields.items()})


def _run(
    analysts: list[_Probe],
    store_dir: Path,
    ledger: TrialLedger | None = None,
    repo_root: Path | None = None,
    *,
    start: date = _START,
    end: date = _END,
    resources: _Resources | None = None,
) -> BatchResult:
    """`run_batch` over `_UNIVERSE` as etf with `_META`, passing `ledger` and `repo_root` only
    when a ledger is given."""
    extra: dict[str, Any] = {} if ledger is None else {"ledger": ledger, "repo_root": repo_root}
    batch: list[AnalystAgent] = list(analysts)
    return run_batch(
        batch,
        list(_UNIVERSE),
        "etf",
        start,
        end,
        store_dir,
        _market_data(),
        resources_ok=_Resources() if resources is None else resources,
        **_META,
        **extra,
    )


def _trials(ledger: TrialLedger) -> list[dict[str, Any]]:
    """The ledger's trial rows in chain order."""
    return _query(
        ledger.path,
        "SELECT t.* FROM trials t JOIN ledger_chain c "
        "ON c.table_name = 'trials' AND c.row_key = t.trial_id ORDER BY c.seq",
    )


def _content(row: dict[str, Any]) -> dict[str, Any]:
    """A trial row's columns other than its trial_id and run_at."""
    return {c: row[c] for c in COLUMNS["trials"] if c not in ("trial_id", "run_at")}


def _last_hash(ledger: TrialLedger) -> str:
    [entry] = _query(ledger.path, "SELECT row_hash FROM ledger_chain ORDER BY seq DESC LIMIT 1")
    return str(entry["row_hash"])


@pytest.fixture(autouse=True)
def _fixed_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pins current_trial_provenance to `_PROVENANCE` wherever the runner looks it up."""
    monkeypatch.setattr(rows, "current_trial_provenance", lambda: _PROVENANCE)
    monkeypatch.setattr(runner, "current_trial_provenance", lambda: _PROVENANCE, raising=False)


@pytest.fixture
def ledger(tmp_path: Path) -> TrialLedger:
    return TrialLedger(tmp_path / settings.trials.ledger_path, repo_root=tmp_path)


def test_run_batch_records_registrations_and_trials(tmp_path: Path, ledger: TrialLedger) -> None:
    """A batch with a ledger registers each analyst first and records one trial per analyst."""
    store = tmp_path / "store"
    family = settings.trials.firm_family
    registry = tmp_path / settings.trials.registry_dir
    analysts = [
        _Probe(agent_id="fake_first", ledger_file=str(ledger.path)),
        _Probe(agent_id="fake_second", ledger_file=str(ledger.path)),
    ]
    result = _run(analysts, store, ledger, tmp_path)
    ids = [stage_a_hypothesis_id(analyst.agent_id) for analyst in analysts]

    assert _query(ledger.path, "SELECT id, family, path FROM registrations ORDER BY id") == [
        {"id": i, "family": family, "path": f"{settings.trials.registry_dir}/{i}.yaml"} for i in ids
    ]
    trials = _trials(ledger)
    assert [trial["hypothesis_id"] for trial in trials] == ids
    for analyst, hypothesis_id, trial in zip(analysts, ids, trials, strict=True):
        entry = result.analysts[analyst.agent_id]
        record = SpecRecord.model_validate_json((store / entry.spec_hash / "spec.json").read_text())
        expected = stage_a_trial_row(
            record, hypothesis_id, family, n_days=len(_SESSIONS), provenance=_PROVENANCE
        )
        registry_text = (registry / f"{hypothesis_id}.yaml").read_text()

        assert entry.days_written == len(_SESSIONS)
        assert trial["config_hash"] == record.spec_hash == entry.spec_hash
        assert trial["n"] == len(_SESSIONS)
        assert _content(trial) == _content(expected.model_dump())
        assert f"analyst_class: {record.spec.analyst_class}\n" in registry_text
        assert set(analyst._registered) == set(_SESSIONS)
        assert all(analyst._registered[day] for day in _SESSIONS[1:])
    assert ledger.verify_chain().ok


def test_run_batch_without_a_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ledger and repo_root are keyword-only and default None; no ledger writes nothing extra."""
    parameters = inspect.signature(run_batch).parameters
    for name in ("ledger", "repo_root"):
        assert parameters[name].default is None
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    monkeypatch.chdir(tmp_path)

    result = _run([_Probe()], tmp_path / "store")
    assert result.analysts["fake_probe"].days_written == len(_SESSIONS)
    assert [path.name for path in tmp_path.iterdir()] == ["store"]

    orphan = TrialLedger(tmp_path / "orphan.duckdb", repo_root=tmp_path)
    with pytest.raises((TypeError, ValueError)):
        _run([_Probe(agent_id="fake_other")], tmp_path / "other", orphan, None)


def test_run_batch_rerun_writes_no_trial(tmp_path: Path, ledger: TrialLedger) -> None:
    """Rerunning a completed batch, or a window with no session, records no trial row."""
    store = tmp_path / "store"
    _run([_Probe()], store, ledger, tmp_path)
    trials = _trials(ledger)
    chain = _query(ledger.path, "SELECT * FROM ledger_chain")

    rerun = _run([_Probe()], store, ledger, tmp_path)
    assert rerun.analysts["fake_probe"].days_written == 0
    assert len(trials) == 1
    assert _trials(ledger) == trials
    assert _query(ledger.path, "SELECT * FROM ledger_chain") == chain

    empty = _run(
        [_Probe(agent_id="fake_weekend")],
        tmp_path / "weekend",
        ledger,
        tmp_path,
        start=date(2010, 1, 16),
        end=date(2010, 1, 18),
    )
    assert empty.analysts["fake_weekend"].days_written == 0
    assert _trials(ledger) == trials
    assert ledger.verify_chain().ok


def test_run_batch_resume_records_an_exposure_row(tmp_path: Path, ledger: TrialLedger) -> None:
    """A resumed batch records a second row with the same config_hash and a larger n."""
    store = tmp_path / "store"
    with temporary_overrides({"stage_a.check_every_days": 2}):
        stopped = _run([_Probe()], store, ledger, tmp_path, resources=_Resources(False))
    resumed = _run([_Probe()], store, ledger, tmp_path)
    hypothesis_id = stage_a_hypothesis_id("fake_probe")
    spec_hash = stopped.analysts["fake_probe"].spec_hash
    first, second = _trials(ledger)

    assert stopped.stopped_for_resources is True
    assert stopped.analysts["fake_probe"].last_completed == _SESSIONS[1]
    assert resumed.analysts["fake_probe"].days_written == len(_SESSIONS) - 2
    assert (first["n"], second["n"]) == (2, len(_SESSIONS))
    assert first["config_hash"] == second["config_hash"] == spec_hash
    assert first["trial_id"] != second["trial_id"]
    assert _content(first) | {"n": len(_SESSIONS)} == _content(second)

    summary = ledger.export_chain(settings.trials.firm_family, _last_hash(ledger))[-1]["summary"]
    assert summary["n_tests"] == 2
    assert summary["n_distinct_hypothesis_config"] == 1
    assert summary["test_keys"] == [f"{hypothesis_id}|{spec_hash}"]
    assert ledger.verify_chain().ok
