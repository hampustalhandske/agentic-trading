"""The trial-ledger rows of this project: stage (a) signal batches, firm records and the imported
record of past windows whose results were seen.

Every hypothesis of the firm is counted in one family (`settings.trials.firm_family`), so a
grade is deflated by the firm's whole trial count. A trial is keyed by
`(hypothesis_id, config_hash)`:

- each stage (a) analyst is its own hypothesis (`stage_a_hypothesis_id`), declared by a
  registry file under `settings.trials.registry_dir` (`ensure_stage_a_registration`); its
  config is the frozen `StageASpec`, window included, and its `config_hash` the spec hash. Each
  batch that writes a day records one row (`stage_a_trial_row`), so a rerun or a resumed batch
  is another row under the same key;
- the firm is one hypothesis (`FIRM_HYPOTHESIS_ID`), declared by a hand-approved registry file
  that is registered as it stands (`ensure_firm_registration`); its config is the frozen
  `FirmSpec` and its `config_hash` the firm spec hash, and each record pass that writes a day
  records one row (`firm_trial_row`);
- each (window, issue) of `docs/prior_exposure.json`'s `results_seen` is its own key under one
  imported hypothesis (`imported_trial_rows`, `import_prior_exposure`), never merged with
  another; its other lists hold no result and count as no trial."""

import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

from agentic_trading import provenance as code_provenance
from agentic_trading.config import settings
from agentic_trading.data.trading_calendar import sessions_between
from agentic_trading.stage_a.spec import (
    FirmSpecRecord,
    SpecRecord,
    canonical_firm_spec_json,
    canonical_spec_json,
)
from agentic_trading.trials.ledger import LedgerError, Registration, TrialLedger, TrialRow
from agentic_trading.trials.registry import config_hash_of, file_registration

# The registered firm's hypothesis id: its registry file is
# `<settings.trials.registry_dir>/agentic-firm-v1.yaml`, approved by hand.
FIRM_HYPOTHESIS_ID = "agentic-firm-v1"
# The registry file's `kind` of a stage (a) hypothesis.
_STAGE_A_KIND = "stage_a_signal_batch"
# A stage (a) hypothesis's primary metric and its kind; `"return"` counts its trials in an
# export's `return_test_keys`.
_METRIC_KIND = "return"
_PRIMARY_METRIC = "edgelab stage (a) grade"
# The metric of a stage (a) row: the batch is submitted to edgelab, which grades it.
_SUBMITTED_METRIC = "submitted"
# The metric of an imported row: a firm Sharpe that was seen, recorded as a diagnostic.
_IMPORTED_METRIC = "firm_sharpe_diagnostic"
_DIRECTION = "two-sided"
# `git_sha` or `data_snapshot` when it is not known.
_UNKNOWN = "unknown"
# The forms an `expected_availability` entry of the firm's registry file
# takes (`firm_expected_availability`).
_EXPECTED_AVAILABILITY_FORMS = ("from YYYY-MM-DD", "none in window", "full")


class TrialProvenance(BaseModel):
    """The code a trial ran: this repository's HEAD sha (`agentic_sha`), whether its working
    tree was dirty (`agentic_dirty`), and the commit quantcore was installed from
    (`quantcore_sha`), and the machine it ran on (`environment`: library versions and
    platform). Each is None when unknown."""

    model_config = ConfigDict(frozen=True)

    agentic_sha: str | None
    agentic_dirty: bool | None
    quantcore_sha: str | None
    environment: code_provenance.RuntimeEnvironment | None = None


class ExposureWindow(BaseModel):
    """One `results_seen` entry of the prior-exposure record: a window from `start` to `end`
    (inclusive) over `universe` whose results were seen on `seen`, under each of `issues`;
    `printed` says which results were shown."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    start: date
    end: date
    universe: list[str]
    issues: list[str]
    seen: date
    printed: str


class PriorExposure(BaseModel):
    """The part of the prior-exposure record that counts as trials: its `results_seen`
    windows. Every other key of the record is ignored."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    results_seen: list[ExposureWindow]


def stage_a_hypothesis_id(agent_id: str) -> str:
    """The hypothesis id of the stage (a) analyst `agent_id`:
    `stage-a-<agent_id with underscores hyphenated>-v1`."""
    return f"stage-a-{agent_id.replace('_', '-')}-v1"


def stage_a_registration_text(
    *, agent_id: str, analyst_class: str, family: str, registered_at: datetime
) -> str:
    """The registry file of the stage (a) analyst `agent_id`: eight `key: value` lines, each
    ending in a newline, in the order id, family, registered_at (in UTC, to the second, with a
    `Z`), kind, agent_id, analyst_class, metric_kind and primary_metric. Raises `ValueError`
    when `registered_at` is naive."""
    if registered_at.tzinfo is None:
        raise ValueError("registered_at must be timezone-aware")
    stamp = registered_at.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        f"id: {stage_a_hypothesis_id(agent_id)}",
        f"family: {family}",
        f"registered_at: {stamp}",
        f"kind: {_STAGE_A_KIND}",
        f"agent_id: {agent_id}",
        f"analyst_class: {analyst_class}",
        f"metric_kind: {_METRIC_KIND}",
        f"primary_metric: {_PRIMARY_METRIC}",
    ]
    return "".join(f"{line}\n" for line in lines)


def _registered_at(text: str, path: Path) -> datetime:
    """The `registered_at` declared in the registry file `text` read from `path`. Raises
    `LedgerError` when it has no parseable `registered_at` line."""
    prefix = "registered_at: "
    for line in text.splitlines():
        if line.startswith(prefix):
            try:
                parsed = datetime.fromisoformat(line.removeprefix(prefix))
            except ValueError:
                break
            if parsed.tzinfo is not None:
                return parsed
            break
    raise LedgerError(f"{path} has no timezone-aware registered_at line")


def ensure_stage_a_registration(
    *,
    ledger: TrialLedger,
    repo_root: Path,
    agent_id: str,
    analyst_class: str,
    family: str,
    now: datetime,
) -> Registration:
    """The registration of the stage (a) analyst `agent_id`, recorded in `ledger`.

    The registry file is `<repo_root>/<settings.trials.registry_dir>/<hypothesis id>.yaml`
    (read at call time). When it does not exist, it is written as
    `stage_a_registration_text(...)` with `registered_at=now` (timezone-aware), creating its
    directory, and registered; when the ledger refuses it, the new file is removed again. An
    existing file is kept byte for byte and must read exactly as the text its own
    `registered_at` gives with these arguments; it is then registered unless already recorded
    with the same hash. Raises `LedgerError`, recording nothing, when the existing file differs
    from that text or the ledger refuses the registration (the id registered with another
    hash)."""
    hypothesis_id = stage_a_hypothesis_id(agent_id)
    path = Path(repo_root) / settings.trials.registry_dir / f"{hypothesis_id}.yaml"
    created = not path.exists()
    if created:
        registered_at = now
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            stage_a_registration_text(
                agent_id=agent_id, analyst_class=analyst_class, family=family, registered_at=now
            ).encode("utf-8")
        )
    else:
        raw = path.read_bytes()
        registered_at = _registered_at(raw.decode("utf-8", errors="replace"), path)
        expected = stage_a_registration_text(
            agent_id=agent_id,
            analyst_class=analyst_class,
            family=family,
            registered_at=registered_at,
        )
        if raw != expected.encode("utf-8"):
            raise LedgerError(
                f"{path} is not the registration of {agent_id} ({analyst_class}) in family "
                f"{family!r}; an edited hypothesis needs a new id with a bumped version"
            )
    registration = file_registration(
        path, repo_root, family=family, registered_at=registered_at, metric_kind=_METRIC_KIND
    )
    try:
        ledger.register(registration)
    except LedgerError:
        if created:
            path.unlink()
        raise
    return registration


def current_trial_provenance() -> TrialProvenance:
    """The code running now: this repository's git state (`provenance`'s HEAD sha and dirty
    flag, None outside a git checkout), the installed quantcore commit and the runtime
    environment (`provenance.runtime_environment`)."""
    sha, dirty = code_provenance._git_state()
    return TrialProvenance(
        agentic_sha=sha,
        agentic_dirty=dirty,
        quantcore_sha=code_provenance.quantcore_sha(),
        environment=code_provenance.runtime_environment(),
    )


def _extra_json(provenance: TrialProvenance, **fields: Any) -> str:
    """`provenance`'s fields and `fields` as one JSON object with sorted keys; `environment`
    only when it is known."""
    exclude = {"environment"} if provenance.environment is None else set()
    return json.dumps(provenance.model_dump(mode="json", exclude=exclude) | fields, sort_keys=True)


def stage_a_trial_row(
    record: SpecRecord,
    hypothesis_id: str,
    family: str,
    *,
    n_days: int,
    provenance: TrialProvenance,
    run_at: datetime | None = None,
) -> TrialRow:
    """The trial row of one stage (a) batch of the trial `record`, under `hypothesis_id` in
    `family`.

    `config_json` is `canonical_spec_json(record.spec)`, the exact string `spec_hash` hashes,
    and `config_hash` is `record.spec_hash`. `git_sha` is `provenance.agentic_sha` (`"unknown"`
    when None), `data_snapshot` is `"<lake_time ISO>|<lake_version>"`, `n` is `n_days` (the
    trial's decision days stored), the metric is `"submitted"` with no value, interval or p
    value, and the direction two-sided. `extra_json` holds `provenance`'s fields (its
    `environment` only when known) and `"imported": false`. `run_at` defaults to now;
    `trial_id` is fresh, so a rerun is another row under the same key."""
    fields: dict[str, Any] = {} if run_at is None else {"run_at": run_at}
    return TrialRow(
        hypothesis_id=hypothesis_id,
        family=family,
        config_json=canonical_spec_json(record.spec),
        config_hash=record.spec_hash,
        git_sha=provenance.agentic_sha or _UNKNOWN,
        data_snapshot=f"{record.lake_time.isoformat()}|{record.lake_version}",
        n=n_days,
        metric_name=_SUBMITTED_METRIC,
        metric_value=None,
        ci_lo=None,
        ci_hi=None,
        p_value=None,
        direction=_DIRECTION,
        extra_json=_extra_json(provenance, imported=False),
        **fields,
    )


def _registry_field(text: str, name: str) -> str | None:
    """The value of the first `<name>: <value>` line of the registry file
    `text`, with any `# comment` after it removed; `None` when no line has
    it."""
    prefix = f"{name}:"
    for line in text.splitlines():
        if line.startswith(prefix):
            return line.removeprefix(prefix).split(" #", 1)[0].strip()
    return None


def ensure_firm_registration(
    *, ledger: TrialLedger, repo_root: Path, family: str
) -> Registration | None:
    """The registration of the firm hypothesis `FIRM_HYPOTHESIS_ID`, recorded
    in `ledger`, or `None` when its registry file
    `<repo_root>/<settings.trials.registry_dir>/<FIRM_HYPOTHESIS_ID>.yaml`
    (read at call time) does not exist. The firm's registry file is written
    and approved by hand, never by this function; it is registered as it
    stands, unless already recorded with the same hash.

    Raises `LedgerError`, recording nothing, when the file's `id` line is
    not `FIRM_HYPOTHESIS_ID`, its `family` line is not `family`, its
    `registered_at` is not a timezone-aware ISO time, or the ledger refuses
    the registration (the id registered with another hash)."""
    path = firm_registry_path(repo_root)
    if not path.exists():
        return None
    text = path.read_bytes().decode("utf-8", errors="replace")
    declared_id = _registry_field(text, "id")
    declared_family = _registry_field(text, "family")
    if declared_id != FIRM_HYPOTHESIS_ID or declared_family != family:
        raise LedgerError(
            f"{path} declares id {declared_id!r} in family {declared_family!r}, not "
            f"{FIRM_HYPOTHESIS_ID!r} in {family!r}"
        )
    stamp = _registry_field(text, "registered_at") or ""
    try:
        registered_at = datetime.fromisoformat(stamp)
    except ValueError:
        raise LedgerError(f"{path} has no ISO registered_at, got {stamp!r}") from None
    if registered_at.tzinfo is None:
        raise LedgerError(f"{path} has a registered_at without a UTC offset: {stamp!r}")
    registration = file_registration(
        path,
        repo_root,
        family=family,
        registered_at=registered_at,
        metric_kind=_registry_field(text, "metric_kind") or _METRIC_KIND,
    )
    ledger.register(registration)
    return registration


def firm_registry_path(repo_root: Path) -> Path:
    """The firm's registry file,
    `<repo_root>/<settings.trials.registry_dir>/<FIRM_HYPOTHESIS_ID>.yaml`
    (read at call time)."""
    return Path(repo_root) / settings.trials.registry_dir / f"{FIRM_HYPOTHESIS_ID}.yaml"


def _firm_registry_content(repo_root: Path) -> dict[str, Any] | None:
    """The firm's registry file (`firm_registry_path`) read as YAML, `None`
    when the file does not exist. Raises `LedgerError` when it is not valid
    YAML or not a mapping."""
    path = firm_registry_path(repo_root)
    if not path.exists():
        return None
    try:
        content = yaml.safe_load(path.read_bytes().decode("utf-8", errors="replace"))
    except yaml.YAMLError as exc:
        raise LedgerError(f"{path} is not valid YAML: {exc}") from None
    if not isinstance(content, dict):
        raise LedgerError(f"{path} does not hold a YAML mapping")
    return content


def firm_registered_trials(repo_root: Path) -> dict[str, Any] | None:
    """The `trials` mapping of the firm's registry file (`firm_registry_path`)
    as the file states it, `{}` when the file has no `trials` key, and
    `None` when the file does not exist. The file is read as YAML. Raises
    `LedgerError` when it is not valid YAML, is not a mapping, or its
    `trials` is not a mapping."""
    content = _firm_registry_content(repo_root)
    if content is None:
        return None
    trials = content.get("trials", {})
    if not isinstance(trials, dict):
        raise LedgerError(
            f"{firm_registry_path(repo_root)} has a trials entry that is not a mapping"
        )
    return trials


def firm_expected_availability(repo_root: Path) -> dict[str, str]:
    """The `expected_availability` mapping of the firm's registry file
    (`firm_registry_path`), the sibling of its `trials` key: each analyst id
    to when the registration expects the analyst to have data in the
    record's window, `"from YYYY-MM-DD"` (from that day on), `"none in
    window"` or `"full"`, in the file's order. Empty when the file does not
    exist or has no such key. Raises `LedgerError` when the file is not
    valid YAML or not a mapping, or the entry is not a mapping of analyst
    ids to one of those three forms."""
    content = _firm_registry_content(repo_root)
    if content is None:
        return {}
    path = firm_registry_path(repo_root)
    declared = content.get("expected_availability", {})
    if not isinstance(declared, dict):
        raise LedgerError(f"{path} has an expected_availability entry that is not a mapping")
    expected: dict[str, str] = {}
    for analyst_id, availability in declared.items():
        if not isinstance(analyst_id, str) or not _is_expected_availability(availability):
            raise LedgerError(
                f"{path} declares expected availability {availability!r} for {analyst_id!r}; "
                f"expected {' or '.join(repr(form) for form in _EXPECTED_AVAILABILITY_FORMS)}"
            )
        expected[analyst_id] = availability
    return expected


def _is_expected_availability(value: object) -> bool:
    """Whether `value` is `"full"`, `"none in window"` or `"from "` followed
    by an ISO date."""
    if not isinstance(value, str):
        return False
    if value in ("full", "none in window"):
        return True
    prefix, _, day = value.partition(" ")
    if prefix != "from" or len(day) != len("YYYY-MM-DD"):
        return False
    try:
        date.fromisoformat(day)
    except ValueError:
        return False
    return True


def firm_trial_row(
    record: FirmSpecRecord,
    family: str,
    *,
    n_days: int,
    provenance: TrialProvenance,
    run_at: datetime | None = None,
) -> TrialRow:
    """The trial row of one firm record pass of `record`, under
    `FIRM_HYPOTHESIS_ID` in `family`, shaped as `stage_a_trial_row`'s:
    `config_json` is `canonical_firm_spec_json(record.spec)` and
    `config_hash` `record.spec_hash`; `n` is `n_days`, the sessions stored;
    the metric is `"submitted"` with no value, since edgelab grades the
    record. The firm's Sharpe, recorded elsewhere as
    `"firm_sharpe_diagnostic"`, never enters this row."""
    fields: dict[str, Any] = {} if run_at is None else {"run_at": run_at}
    return TrialRow(
        hypothesis_id=FIRM_HYPOTHESIS_ID,
        family=family,
        config_json=canonical_firm_spec_json(record.spec),
        config_hash=record.spec_hash,
        git_sha=provenance.agentic_sha or _UNKNOWN,
        data_snapshot=f"{record.lake_time.isoformat()}|{record.lake_version}",
        n=n_days,
        metric_name=_SUBMITTED_METRIC,
        metric_value=None,
        ci_lo=None,
        ci_hi=None,
        p_value=None,
        direction=_DIRECTION,
        extra_json=_extra_json(provenance, imported=False),
        **fields,
    )


def imported_trial_rows(
    prior_exposure: Mapping[str, Any], hypothesis_id: str, family: str
) -> list[TrialRow]:
    """One row per (window, issue) of `prior_exposure`'s `results_seen`, in record order,
    under `hypothesis_id` in `family`; `prior_exposure` is the parsed prior-exposure record
    (`PriorExposure`), and its other keys add no row.

    A row's config is the window's ISO `window_start` and `window_end`, its sorted `universe`
    and the `issue`, as sorted-key JSON without whitespace, and `config_hash` its
    `config_hash_of`. `n` is the number of NYSE sessions in the window; the metric is
    `"firm_sharpe_diagnostic"` with no value, the direction two-sided, `git_sha` and
    `data_snapshot` `"unknown"`, and `extra_json` records `"imported": true`,
    `"spec_reconstructed": false`, the window's `seen` and `printed`, and null provenance.
    `run_at` is midnight of `seen`, and `trial_id` the first 32 hex characters of the sha256 of
    `"imported|<hypothesis_id>|<config_hash>"`, so the same record always gives the same rows.
    Raises `pydantic.ValidationError` when `results_seen` is missing or malformed."""
    record = PriorExposure.model_validate(prior_exposure)
    unknown = TrialProvenance(agentic_sha=None, agentic_dirty=None, quantcore_sha=None)
    rows = []
    for window in record.results_seen:
        n = len(sessions_between(window.start, window.end))
        for issue in window.issues:
            config = {
                "window_start": window.start.isoformat(),
                "window_end": window.end.isoformat(),
                "universe": sorted(window.universe),
                "issue": issue,
            }
            config_hash = config_hash_of(config)
            trial_key = f"imported|{hypothesis_id}|{config_hash}"
            rows.append(
                TrialRow(
                    hypothesis_id=hypothesis_id,
                    family=family,
                    config_json=json.dumps(config, sort_keys=True, separators=(",", ":")),
                    config_hash=config_hash,
                    git_sha=_UNKNOWN,
                    data_snapshot=_UNKNOWN,
                    n=n,
                    metric_name=_IMPORTED_METRIC,
                    metric_value=None,
                    ci_lo=None,
                    ci_hi=None,
                    p_value=None,
                    direction=_DIRECTION,
                    extra_json=_extra_json(
                        unknown,
                        imported=True,
                        spec_reconstructed=False,
                        seen=window.seen.isoformat(),
                        printed=window.printed,
                    ),
                    trial_id=hashlib.sha256(trial_key.encode()).hexdigest()[:32],
                    run_at=datetime.combine(window.seen, time()),
                )
            )
    return rows


def import_prior_exposure(
    ledger: TrialLedger, prior_exposure: Mapping[str, Any], hypothesis_id: str, family: str
) -> int:
    """Records in `ledger` each row of `imported_trial_rows(prior_exposure, hypothesis_id,
    family)` whose `trial_id` is not yet recorded, in order, and returns how many it recorded.
    `hypothesis_id` must already be registered in `family`; this does not register it. Raises
    `LedgerError` when the ledger refuses a row, with the rows before it recorded."""
    recorded = 0
    for row in imported_trial_rows(prior_exposure, hypothesis_id, family):
        if ledger.has_trial(row.trial_id):
            continue
        ledger.record_trial(row)
        recorded += 1
    return recorded
