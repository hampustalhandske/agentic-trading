# test_firm_registration.py

"""AT-125: `ensure_firm_registration` registers agentic-firm-v1 from its hand-approved registry
file as it stands, returns `None` when the file is absent, and refuses a file naming another
id or family or carrying a naive or missing `registered_at`, recording nothing."""

from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pytest

from agentic_trading.trials.ledger import LedgerError, TrialLedger
from agentic_trading.trials.rows import FIRM_HYPOTHESIS_ID, ensure_firm_registration

_FAMILY = "agentic-firm"
_REGISTRATION = (
    "id: agentic-firm-v1\n"
    "family: agentic-firm\n"
    "registered_at: 2026-11-07T09:00:00Z   # UTC, set when the master approves\n"
    "kind: firm\n"
)


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "research" / "registry" / f"{FIRM_HYPOTHESIS_ID}.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(text)
    return path


def _ledger(tmp_path: Path) -> TrialLedger:
    return TrialLedger(tmp_path / "research" / "ledger.duckdb", tmp_path)


def _registered_ids(tmp_path: Path) -> list[str]:
    path = tmp_path / "research" / "ledger.duckdb"
    if not path.exists():
        return []
    with duckdb.connect(str(path), read_only=True) as connection:
        tables = {row[0] for row in connection.execute("SHOW TABLES").fetchall()}
        if "registrations" not in tables:
            return []
        return [row[0] for row in connection.execute("SELECT id FROM registrations").fetchall()]


def test_absent_file_is_no_registration(tmp_path: Path) -> None:
    """Without the registry file nothing is registered and the ledger is not created."""
    assert (
        ensure_firm_registration(ledger=_ledger(tmp_path), repo_root=tmp_path, family=_FAMILY)
        is None
    )
    assert not (tmp_path / "research" / "ledger.duckdb").exists()


def test_file_is_registered_as_it_stands(tmp_path: Path) -> None:
    """The file's id, family and UTC registered_at (comment stripped) are registered, and a
    second call with the same file is accepted."""
    _write(tmp_path, _REGISTRATION)
    ledger = _ledger(tmp_path)

    registration = ensure_firm_registration(ledger=ledger, repo_root=tmp_path, family=_FAMILY)
    again = ensure_firm_registration(ledger=ledger, repo_root=tmp_path, family=_FAMILY)

    assert registration is not None and again == registration
    assert registration.id == FIRM_HYPOTHESIS_ID
    assert registration.family == _FAMILY
    assert registration.registered_at == datetime(2026, 11, 7, 9, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    "text",
    [
        _REGISTRATION.replace("id: agentic-firm-v1", "id: agentic-firm-v2"),
        _REGISTRATION.replace("family: agentic-firm", "family: other-firm"),
        _REGISTRATION.replace("2026-11-07T09:00:00Z", "2026-11-07T09:00:00"),
        _REGISTRATION.replace("2026-11-07T09:00:00Z", "soon"),
        "id: agentic-firm-v1\nfamily: agentic-firm\n",
    ],
    ids=["other-id", "other-family", "naive-time", "bad-time", "no-time"],
)
def test_bad_file_is_refused(tmp_path: Path, text: str) -> None:
    """A file naming another id or family, or without an aware ISO registered_at, is refused
    and nothing is registered."""
    _write(tmp_path, text)

    with pytest.raises(LedgerError, match=FIRM_HYPOTHESIS_ID):
        ensure_firm_registration(ledger=_ledger(tmp_path), repo_root=tmp_path, family=_FAMILY)
    assert _registered_ids(tmp_path) == []
