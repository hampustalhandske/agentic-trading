# ledger.py

"""The append-only, hash-chained trial ledger, in pitch-edge's shared ledger format.

The ledger is one DuckDB file with four tables:

- `registrations`: one row per hypothesis id: its family, the sha256 of its registry file, the
  file's path relative to the repository root (POSIX), the declared `registered_at`, and when the
  ledger first recorded it (`recorded_at`);
- `trials`: one row per trial (`TrialRow`);
- `reviews`: reviews of a registered hypothesis, of a kind in `REVIEW_KINDS`;
- `ledger_chain`: one entry per row written through `TrialLedger`, in write order, with
  `row_hash = sha256(f"{prev_hash}|{table}|{row_json(table, row)}")` and `GENESIS` as the first
  entry's `prev_hash`.

`TrialLedger.verify_chain` recomputes the chain, so an edited or deleted row, or a row inserted
around `TrialLedger`, is detected. `TrialLedger.export_chain` writes the chain out with the exact
strings that were hashed, so another project can recompute it, and one family's trial count,
from the exported file alone (`write_export`).

Reads open read-only connections and never create the file. Each write opens a read-write
connection, creates the file, its parent directories and the tables as needed, and runs in one
transaction, so a refused write leaves the tables and the chain unchanged. Every timestamp is
stored as a naive UTC datetime. There is no update or delete path."""

import hashlib
import json
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import duckdb

from agentic_trading.trials.models import (
    ChainReport,
    LedgerError,
    Registration,
    TrialRow,
    naive_utc_now,
)
from agentic_trading.trials.registry import config_hash_of

__all__ = [
    "COLUMNS",
    "DIRECTIONS",
    "GENESIS",
    "REVIEW_KINDS",
    "ChainReport",
    "LedgerError",
    "Registration",
    "TrialLedger",
    "TrialRow",
    "canonical",
    "chain_hash",
    "row_json",
    "row_key",
    "write_export",
]

GENESIS = "0" * 64
DIRECTIONS = ("positive", "negative", "two-sided")
REVIEW_KINDS = ("audit", "edgelab", "paper", "amendment")

# Each table's columns, in table order; a row's hash covers exactly these.
COLUMNS: dict[str, tuple[str, ...]] = {
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

# The column dating each table's rows, compared with the chain's first `chained_at`.
_TIME_COLUMN = {"registrations": "recorded_at", "trials": "run_at", "reviews": "recorded_at"}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS registrations (
    id VARCHAR PRIMARY KEY,
    family VARCHAR NOT NULL,
    sha256 VARCHAR NOT NULL,
    path VARCHAR NOT NULL,
    registered_at TIMESTAMP NOT NULL,
    recorded_at TIMESTAMP NOT NULL
);
CREATE TABLE IF NOT EXISTS trials (
    trial_id VARCHAR PRIMARY KEY,
    hypothesis_id VARCHAR NOT NULL,
    family VARCHAR NOT NULL,
    config_json VARCHAR NOT NULL,
    config_hash VARCHAR NOT NULL,
    git_sha VARCHAR NOT NULL,
    data_snapshot VARCHAR NOT NULL,
    run_at TIMESTAMP NOT NULL,
    n BIGINT NOT NULL,
    metric_name VARCHAR NOT NULL,
    metric_value DOUBLE,
    ci_lo DOUBLE,
    ci_hi DOUBLE,
    p_value DOUBLE,
    direction VARCHAR NOT NULL,
    extra_json VARCHAR NOT NULL
);
CREATE TABLE IF NOT EXISTS reviews (
    hypothesis_id VARCHAR NOT NULL,
    kind VARCHAR NOT NULL,
    verdict VARCHAR NOT NULL,
    detail_json VARCHAR NOT NULL,
    recorded_at TIMESTAMP NOT NULL
);
CREATE TABLE IF NOT EXISTS ledger_chain (
    seq BIGINT PRIMARY KEY,
    table_name VARCHAR NOT NULL,
    row_key VARCHAR NOT NULL,
    prev_hash VARCHAR NOT NULL,
    row_hash VARCHAR NOT NULL,
    chained_at TIMESTAMP NOT NULL
);
"""

# One `ledger_chain` entry: seq, table_name, row_key, prev_hash, row_hash, chained_at.
type _ChainEntry = tuple[Any, ...]
# Every stored row, by table, then by row key (reviews can share a key).
type _RowsByKey = dict[str, dict[str, list[dict[str, Any]]]]


def canonical(value: Any) -> Any:
    """`value` as it enters a row's JSON: a datetime or date as its ISO string, a NaN float as
    the string `"NaN"`, anything else unchanged."""
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, float) and math.isnan(value):
        return "NaN"
    return value


def row_json(table: str, row: Mapping[str, Any]) -> str:
    """The JSON a row of `table` is hashed on: its `COLUMNS[table]` values through `canonical`,
    keys sorted, no whitespace; an infinite float is written `Infinity` / `-Infinity`. Keys of
    `row` outside the table's columns are ignored."""
    values = {c: canonical(row[c]) for c in COLUMNS[table]}
    return json.dumps(values, sort_keys=True, separators=(",", ":"))


def row_key(table: str, row: Mapping[str, Any]) -> str:
    """The key a chain entry names its row by: a registration's `id`, a trial's `trial_id`, and
    for a review, which has no natural key, the hex sha256 of its `row_json`."""
    if table == "registrations":
        return str(row["id"])
    if table == "trials":
        return str(row["trial_id"])
    return hashlib.sha256(row_json(table, row).encode()).hexdigest()


def chain_hash(prev_hash: str, table: str, row: Mapping[str, Any]) -> str:
    """The hex sha256 of `f"{prev_hash}|{table}|{row_json(table, row)}"`: the `row_hash` of `row`
    chained after the entry whose `row_hash` is `prev_hash`."""
    return hashlib.sha256(f"{prev_hash}|{table}|{row_json(table, row)}".encode()).hexdigest()


def write_export(lines: Sequence[Mapping[str, Any]], out: str | Path) -> None:
    """Writes `lines` (as `TrialLedger.export_chain` returns them) to `out` as JSON lines: one
    `json.dumps(line, sort_keys=True)` per line, each ending in a newline, UTF-8. The parent
    directory of `out` must exist."""
    text = "".join(json.dumps(line, sort_keys=True) + "\n" for line in lines)
    Path(out).write_bytes(text.encode("utf-8"))


def _has_table(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    found = con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [table]
    ).fetchone()
    return found is not None and bool(found[0])


def _registration_row(con: duckdb.DuckDBPyConnection, hypothesis_id: str) -> dict[str, Any] | None:
    cursor = con.execute("SELECT * FROM registrations WHERE id = ?", [hypothesis_id])
    found = cursor.fetchone()
    if found is None:
        return None
    return dict(zip([d[0] for d in cursor.description], found, strict=True))


def _append(con: duckdb.DuckDBPyConnection, table: str, row: Mapping[str, Any]) -> None:
    """Inserts `row` into `table` and chains it after the newest chain entry."""
    columns = COLUMNS[table]
    con.execute(
        f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join('?' * len(columns))})",
        [row[c] for c in columns],
    )
    last = con.execute(
        "SELECT seq, row_hash FROM ledger_chain ORDER BY seq DESC LIMIT 1"
    ).fetchone()
    seq, prev = (0, GENESIS) if last is None else (int(last[0]) + 1, str(last[1]))
    con.execute(
        "INSERT INTO ledger_chain VALUES (?, ?, ?, ?, ?, ?)",
        [seq, table, row_key(table, row), prev, chain_hash(prev, table, row), naive_utc_now()],
    )


def _json_object(text: str, column: str) -> dict[str, Any]:
    """`text` parsed as JSON; raises `LedgerError` unless it is a JSON object."""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LedgerError(f"{column} is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise LedgerError(f"{column} must be a JSON object, got {text!r}")
    return parsed


def _check_trial(row: TrialRow) -> None:
    """Raises `LedgerError` when `row` is invalid on its own, without the ledger: a direction
    outside `DIRECTIONS`, a `config_json` that is not a JSON object or does not hash to
    `config_hash`, an `extra_json` that is not a JSON object, a `p_value` outside [0, 1] (NaN
    included), or `ci_lo > ci_hi`."""
    if row.direction not in DIRECTIONS:
        raise LedgerError(f"direction must be one of {DIRECTIONS}, got {row.direction!r}")
    if config_hash_of(_json_object(row.config_json, "config_json")) != row.config_hash:
        raise LedgerError("config_hash is not the hash of config_json")
    _json_object(row.extra_json, "extra_json")
    p, lo, hi = row.p_value, row.ci_lo, row.ci_hi
    if p is not None and not (0.0 <= p <= 1.0):
        raise LedgerError(f"p_value must lie in [0, 1], got {p}")
    if lo is not None and hi is not None and lo > hi:
        raise LedgerError(f"ci_lo {lo} > ci_hi {hi}")


class TrialLedger:
    """Append-only access to the trial ledger file at `path`. `repo_root` is the repository root
    that the registrations' stored paths are relative to."""

    def __init__(self, path: str | Path, repo_root: str | Path) -> None:
        self._path = Path(path)
        self._repo_root = Path(repo_root)

    @property
    def path(self) -> Path:
        """The ledger file."""
        return self._path

    @contextmanager
    def _reader(self) -> Iterator[duckdb.DuckDBPyConnection | None]:
        """A read-only connection, or None when the ledger file does not exist."""
        if not self._path.exists():
            yield None
            return
        con = duckdb.connect(str(self._path), read_only=True)
        try:
            yield con
        finally:
            con.close()

    def _write[T](self, operation: Callable[[duckdb.DuckDBPyConnection], T]) -> T:
        """Runs `operation` in one transaction on a read-write connection, after creating the
        file, its parent directories and the tables as needed; rolls everything back when
        `operation` raises."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect(str(self._path))
        try:
            con.execute("BEGIN TRANSACTION")
            try:
                con.execute(_SCHEMA)
                result = operation(con)
                con.execute("COMMIT")
            except BaseException:
                con.execute("ROLLBACK")
                raise
            return result
        finally:
            con.close()

    def register(self, registration: Registration) -> bool:
        """Records `registration` and chains it. Returns True when newly recorded, and False when
        its id is already registered with the same file hash. Raises `LedgerError` when the id is
        registered with a different hash: a registration is immutable, and an edited file needs a
        new id with a bumped version."""
        sha = registration.source_sha256

        def operation(con: duckdb.DuckDBPyConnection) -> bool:
            existing = _registration_row(con, registration.id)
            if existing is not None:
                if existing["sha256"] != sha:
                    raise LedgerError(
                        f"{registration.id} is registered with sha256 {existing['sha256']} but "
                        f"its file now hashes to {sha}; register an edited file under a new id "
                        f"with a bumped version"
                    )
                return False
            row = {
                "id": registration.id,
                "family": registration.family,
                "sha256": sha,
                "path": registration.source_path.as_posix(),
                "registered_at": registration.registered_at.astimezone(UTC).replace(tzinfo=None),
                "recorded_at": naive_utc_now(),
            }
            _append(con, "registrations", row)
            return True

        return self._write(operation)

    def record_trial(self, row: TrialRow) -> None:
        """Records `row` and chains it. Raises `LedgerError` when the row is invalid on its own
        (a direction outside `DIRECTIONS`, a `config_json` that is not a JSON object or does not
        hash to `config_hash`, an `extra_json` that is not a JSON object, a `p_value` outside
        [0, 1], `ci_lo > ci_hi`), when its hypothesis is not registered or is registered under
        another family, or when its `trial_id` is already recorded."""
        _check_trial(row)
        values = row.model_dump()

        def operation(con: duckdb.DuckDBPyConnection) -> None:
            registered = _registration_row(con, row.hypothesis_id)
            if registered is None:
                raise LedgerError(
                    f"{row.hypothesis_id} is not registered; register it before recording trials"
                )
            if row.family != registered["family"]:
                raise LedgerError(
                    f"trial family {row.family!r} differs from the registered family "
                    f"{registered['family']!r}"
                )
            duplicate = con.execute(
                "SELECT count(*) FROM trials WHERE trial_id = ?", [row.trial_id]
            ).fetchone()
            if duplicate is not None and duplicate[0]:
                raise LedgerError(f"trial {row.trial_id} is already recorded")
            _append(con, "trials", values)

        self._write(operation)

    def record_review(
        self, hypothesis_id: str, kind: str, verdict: str, detail: dict[str, Any] | None = None
    ) -> None:
        """Records a review of the registered `hypothesis_id` and chains it. `kind` is one of
        `REVIEW_KINDS`; `detail` is stored as `json.dumps(detail, sort_keys=True)` (`"{}"` when
        None). An `amendment` adds a rule to the registration and is accepted only while the
        hypothesis has no trial, so it is on the chain before any result. Raises `LedgerError`
        for an unknown kind, an unregistered hypothesis, or an amendment after a trial."""
        if kind not in REVIEW_KINDS:
            raise LedgerError(f"review kind must be one of {REVIEW_KINDS}, got {kind!r}")
        row = {
            "hypothesis_id": hypothesis_id,
            "kind": kind,
            "verdict": verdict,
            "detail_json": json.dumps(detail or {}, sort_keys=True),
            "recorded_at": naive_utc_now(),
        }

        def operation(con: duckdb.DuckDBPyConnection) -> None:
            if _registration_row(con, hypothesis_id) is None:
                raise LedgerError(f"{hypothesis_id} is not registered")
            if kind == "amendment":
                trials = con.execute(
                    "SELECT count(*) FROM trials WHERE hypothesis_id = ?", [hypothesis_id]
                ).fetchone()
                if trials is not None and trials[0]:
                    raise LedgerError(
                        f"{hypothesis_id} already has trials; an amendment must precede every "
                        f"result"
                    )
            _append(con, "reviews", row)

        self._write(operation)

    def _snapshot(self) -> tuple[_RowsByKey, list[_ChainEntry]] | None:
        """Every stored row by table and row key, and the chain entries in `seq` order; None when
        the ledger file does not exist."""
        with self._reader() as con:
            if con is None:
                return None
            rows: dict[str, list[dict[str, Any]]] = {}
            for table, columns in COLUMNS.items():
                if not _has_table(con, table):
                    rows[table] = []
                    continue
                cursor = con.execute(f"SELECT {', '.join(columns)} FROM {table}")
                rows[table] = [dict(zip(columns, r, strict=True)) for r in cursor.fetchall()]
            chain = (
                con.execute(
                    "SELECT seq, table_name, row_key, prev_hash, row_hash, chained_at "
                    "FROM ledger_chain ORDER BY seq"
                ).fetchall()
                if _has_table(con, "ledger_chain")
                else []
            )
        by_key: _RowsByKey = {table: {} for table in COLUMNS}
        for table, stored in rows.items():
            for r in stored:
                by_key[table].setdefault(row_key(table, r), []).append(r)
        return by_key, chain

    def verify_chain(self) -> ChainReport:
        """Recomputes the hash chain over every chained row and finds the rows without a chain
        entry (see `ChainReport`). A ledger file that does not exist verifies, with no entry."""
        snapshot = self._snapshot()
        if snapshot is None:
            return ChainReport(ok=True, n_chained=0)
        by_key, chain = snapshot

        broken: list[str] = []
        chained: dict[str, set[str]] = {table: set() for table in COLUMNS}
        prev = GENESIS
        for seq, table, key, prev_hash, row_hash, _ in chain:
            if prev_hash != prev:
                broken.append(f"chain:{seq}: prev_hash does not link to the previous entry")
            candidates = by_key.get(table, {}).get(key, [])
            if not candidates:
                broken.append(f"{table}:{key}: row missing or altered")
            elif not any(chain_hash(prev_hash, table, r) == row_hash for r in candidates):
                broken.append(f"{table}:{key}: row content changed since it was written")
            chained.setdefault(table, set()).add(key)
            prev = row_hash

        start = min((entry[5] for entry in chain), default=None)
        after: dict[str, list[str]] = {}
        legacy: dict[str, int] = {}
        for table, keyed in by_key.items():
            for key, stored in keyed.items():
                if key in chained[table]:
                    continue
                for r in stored:
                    written = r[_TIME_COLUMN[table]]
                    if start is not None and written is not None and written >= start:
                        after.setdefault(table, []).append(key)
                    else:
                        legacy[table] = legacy.get(table, 0) + 1
        return ChainReport(
            ok=not broken and not after,
            n_chained=len(chain),
            broken=broken,
            unchained_after_start=after,
            legacy_unchained=legacy,
        )

    def _registry_text(self, path: str, sha256: str) -> str | None:
        """The registry file at `path` (relative to the repository root) decoded as UTF-8, or
        None when it cannot be read or its raw bytes do not hash to `sha256`."""
        try:
            raw = (self._repo_root / path).read_bytes()
            if hashlib.sha256(raw).hexdigest() != sha256:
                return None
            return raw.decode("utf-8")
        except (OSError, ValueError):
            return None

    def export_chain(
        self, family: str, upto: str, metric_kinds: Mapping[str, str | None] | None = None
    ) -> list[dict[str, Any]]:
        """The ledger as lines another project can verify on its own, for `family`'s trial count.

        Lines, in order:
        - every chain entry from genesis to the entry whose `row_hash` is `upto`, of every table
          and family: `{"table", "row_canonical", "prev_hash", "row_hash"}`, where
          `row_canonical` is the exact string that was hashed, so
          `sha256(f"{prev_hash}|{table}|{row_canonical}") == row_hash`;
        - the family's rows without a chain entry: `{"table", "row_canonical", "chained": false}`;
        - one line per registration of the family, chained ones in chain order, then the
          unchained ones: `{"table": "registration_yaml", "hypothesis_id", "sha256", "yaml"}`,
          where `sha256` is the digest in the registration row and `yaml` the registry file's
          text, so `sha256(yaml.encode("utf-8")).hexdigest() == sha256`. An unchained
          registration adds `"chained": false`. When the file is missing or its bytes no longer
          hash to the recorded digest, `yaml` is None and `"mismatch": true` is added. These
          lines are not part of the chain;
        - `{"summary": {family, upto, n_tests, n_distinct_hypothesis_config, n_return_kind,
          return_test_keys, test_keys}}` over the family's trial rows in the lines above:
          `test_keys` are the sorted distinct `"<hypothesis_id>|<config_hash>"`,
          `return_test_keys` those whose hypothesis has metric kind `"return"` in
          `metric_kinds`, and `n_return_kind` their number (both None when `metric_kinds` is
          None).

        A trial or registration belongs to the family by its `family`, a review by its
        hypothesis's registration. Raises `LedgerError` when the ledger file does not exist,
        `upto` is not a `row_hash` in the chain, or the chain does not verify up to `upto`."""
        snapshot = self._snapshot()
        if snapshot is None:
            raise LedgerError(f"no ledger at {self._path}")
        by_key, chain = snapshot
        lines: list[dict[str, Any]] = []
        family_rows: list[tuple[str, dict[str, Any]]] = []
        registrations: list[tuple[dict[str, Any], bool]] = []
        hypotheses = {
            str(r["id"])
            for stored in by_key["registrations"].values()
            for r in stored
            if r["family"] == family
        }

        def in_family(table: str, row: Mapping[str, Any]) -> bool:
            if table == "reviews":
                return str(row["hypothesis_id"]) in hypotheses
            return bool(row["family"] == family)

        prev, reached = GENESIS, False
        for seq, table, key, prev_hash, row_hash, _ in chain:
            if prev_hash != prev:
                raise LedgerError(f"chain:{seq}: prev_hash does not link to the previous entry")
            row = next(
                (
                    r
                    for r in by_key[table].get(key, [])
                    if chain_hash(prev_hash, table, r) == row_hash
                ),
                None,
            )
            if row is None:
                raise LedgerError(f"{table}:{key}: no stored row matches chain entry {seq}")
            lines.append(
                {
                    "table": table,
                    "row_canonical": row_json(table, row),
                    "prev_hash": prev_hash,
                    "row_hash": row_hash,
                }
            )
            if in_family(table, row):
                family_rows.append((table, row))
                if table == "registrations":
                    registrations.append((row, True))
            prev = row_hash
            if row_hash == upto:
                reached = True
                break
        if not reached:
            raise LedgerError(f"{upto} is not a row_hash in the chain")

        chained_keys = {(entry[1], entry[2]) for entry in chain}
        for table, keyed in by_key.items():
            for key, stored in sorted(keyed.items()):
                if (table, key) in chained_keys:
                    continue
                for r in stored:
                    if in_family(table, r):
                        lines.append(
                            {"table": table, "row_canonical": row_json(table, r), "chained": False}
                        )
                        family_rows.append((table, r))
                        if table == "registrations":
                            registrations.append((r, False))

        for registration, is_chained in registrations:
            recorded = str(registration["sha256"])
            text = self._registry_text(str(registration["path"]), recorded)
            line: dict[str, Any] = {
                "table": "registration_yaml",
                "hypothesis_id": str(registration["id"]),
                "sha256": recorded,
                "yaml": text,
            }
            if not is_chained:
                line["chained"] = False
            if text is None:
                line["mismatch"] = True
            lines.append(line)

        trials = [r for table, r in family_rows if table == "trials"]
        keys = sorted({f"{r['hypothesis_id']}|{r['config_hash']}" for r in trials})
        return_keys = (
            None
            if metric_kinds is None
            else [k for k in keys if metric_kinds.get(k.split("|", 1)[0]) == "return"]
        )
        lines.append(
            {
                "summary": {
                    "family": family,
                    "upto": upto,
                    "n_tests": len(trials),
                    "n_distinct_hypothesis_config": len(keys),
                    "n_return_kind": None if return_keys is None else len(return_keys),
                    "return_test_keys": return_keys,
                    "test_keys": keys,
                }
            }
        )
        return lines
