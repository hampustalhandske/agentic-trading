# models.py

"""The structures the trial ledger reads and writes, shared by `trials.ledger` and
`trials.registry`.

- `LedgerError`: raised for every write or export the ledger refuses;
- `Registration`: one hypothesis's registry file, as `TrialLedger.register` records it;
- `TrialRow`: one row of the `trials` table, as `TrialLedger.record_trial` records it;
- `ChainReport`: the result of `TrialLedger.verify_chain`.

The ledger stores every timestamp as a naive UTC datetime (`naive_utc_now`)."""

import uuid
from datetime import UTC, datetime
from pathlib import Path

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator


class LedgerError(Exception):
    """A registration, trial, review or export the ledger refuses; the message says why."""


def naive_utc_now() -> datetime:
    """The current UTC time as a naive datetime."""
    return datetime.now(UTC).replace(tzinfo=None)


def _new_trial_id() -> str:
    return uuid.uuid4().hex


class Registration(BaseModel):
    """One hypothesis's registry file.

    - `id`: the hypothesis id, the file's stem (`<name>-v<version>`);
    - `family`: the family of hypotheses whose trials are counted together;
    - `registered_at`: when the hypothesis was declared, timezone-aware;
    - `metric_kind`: the kind of the hypothesis's primary metric (`"return"` makes its trials
      count in an export's `return_test_keys`), or None when undeclared;
    - `source_path`: the file's path relative to the repository root;
    - `source_sha256`: the hex sha256 of the file's raw bytes.

    Raises `pydantic.ValidationError` when `source_path` is absolute."""

    model_config = ConfigDict(frozen=True)

    id: str
    family: str
    registered_at: AwareDatetime
    metric_kind: str | None
    source_path: Path
    source_sha256: str

    @field_validator("source_path")
    @classmethod
    def _relative(cls, value: Path) -> Path:
        if value.is_absolute():
            raise ValueError(f"source_path must be relative to the repository root, got {value}")
        return value


class TrialRow(BaseModel):
    """One row of the `trials` table; its fields are the table's columns.

    `trial_id` defaults to a fresh uuid4 hex and `run_at` to now; an aware `run_at` is converted
    to naive UTC. `config_json` is the JSON object of the trial's config and `config_hash` its
    `registry.config_hash_of`; `extra_json` is a JSON object of anything else the trial records.
    `p_value` is one-sided in `direction` (two-sided for `two-sided`); `metric_value`, `ci_lo`,
    `ci_hi` and `p_value` may be None or NaN. `TrialLedger.record_trial` checks the row against
    the ledger before writing it; this model checks only the types. `model_dump()` gives the
    row with plain Python values, which hash the same as the row read back from the ledger."""

    model_config = ConfigDict(frozen=True)

    hypothesis_id: str
    family: str
    config_json: str
    config_hash: str
    git_sha: str
    data_snapshot: str
    n: int
    metric_name: str
    metric_value: float | None
    ci_lo: float | None
    ci_hi: float | None
    p_value: float | None
    direction: str
    extra_json: str
    trial_id: str = Field(default_factory=_new_trial_id)
    run_at: datetime = Field(default_factory=naive_utc_now)

    @field_validator("run_at")
    @classmethod
    def _naive_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value
        return value.astimezone(UTC).replace(tzinfo=None)


class ChainReport(BaseModel):
    """The result of `TrialLedger.verify_chain`.

    - `ok`: False when any chain entry no longer matches its row or any row without a chain entry
      is at least as new as the chain's first entry;
    - `n_chained`: the number of chain entries;
    - `broken`: one `"<table>:<key>: <reason>"` (or `"chain:<seq>: <reason>"`) per entry that no
      longer matches;
    - `unchained_after_start`: table -> keys of the rows without a chain entry that are at least
      as new as the chain's first entry;
    - `legacy_unchained`: table -> count of the rows without a chain entry that are older than
      the chain's first entry (or that sit in a ledger with no chain entry); they do not make
      the report fail."""

    ok: bool
    n_chained: int
    broken: list[str] = Field(default_factory=list)
    unchained_after_start: dict[str, list[str]] = Field(default_factory=dict)
    legacy_unchained: dict[str, int] = Field(default_factory=dict)
