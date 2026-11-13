"""The edgelab replay submission a firm record's trial directory forms.

Beside the store's spec.json, sources.json, decisions.parquet and
progress.json, the trial directory holds:

- submission.json: a `Submission`, the `submission.json` of an edgelab
  `kind: replay` folder, whose models here mirror edgelab's
  (`edgelab.submission`) field for field, with the same checks, so a file
  is refused before it is written rather than by `edgelab submission
  check`;
- strategy/: `strategy.yaml`, edgelab's replay scaffold set to the record
  (`strategy_config`), and `strategy.py`, whose `Replay` subclasses
  edgelab's `TargetWeightsReplay`;
- agentic_meta.json: an `AgenticMeta`, what edgelab's model has no field
  for: whether the record is counted in the trial ledger, whether its
  `src/` tree was clean, how many orders went unfilled, which analysts
  had too few available signals, and the library versions and platform
  of the pass that wrote it;
- unfilled_orders.json: an `UnfilledOrdersRecord`, the orders the engine
  could not fill and their count per reason;
- analyst_availability.json: an `AnalystAvailability`, each analyst's
  signals by status per year and over the record, and its EWMA
  volatility fallbacks when it records a volatility estimator
  (`firm_record.availability`).

The folder names the lake as its data source: strategy.yaml's data block
reads the `lakehouse` provider pinned to the record's lake time and lake
version, and submission.json's `data` is a lake bundle at the same lake
time (`lake_time_text`), so the folder holds no data bundle of its own.

`submission_for` builds the `Submission` of a record,
`write_submission_folder` writes these files but the last two,
`write_unfilled_orders` writes unfilled_orders.json and the count, and
`write_analyst_availability` writes analyst_availability.json and the
analysts under the threshold or without a signal. Nothing here imports
edgelab."""

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Final, Literal, Self

import yaml
from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_serializer,
    field_validator,
    model_validator,
)

from agentic_trading.agents.schemas import ShortWarmup, UnfilledOrder
from agentic_trading.config import settings
from agentic_trading.firm_record.availability import (
    AnalystAvailability,
    LowAvailability,
    low_availability,
)
from agentic_trading.firm_record.edgelab_strategy import REPLAY_STRATEGY_PY, REPLAY_STRATEGY_YAML
from agentic_trading.firm_record.rows import row_instrument_id
from agentic_trading.provenance import RuntimeEnvironment, runtime_environment
from agentic_trading.stage_a.spec import FirmSpecRecord
from agentic_trading.stage_a.store import _write_text
from agentic_trading.trials.rows import FIRM_HYPOTHESIS_ID

SUBMISSION_FILE = "submission.json"
META_FILE = "agentic_meta.json"
UNFILLED_ORDERS_FILE = "unfilled_orders.json"
ANALYST_AVAILABILITY_FILE = "analyst_availability.json"
STRATEGY_DIR = "strategy"
DECISIONS_FILE = "decisions.parquet"
# The published edgelab grading policy submission.json names; strategy.yaml has no
# `grading:` block, so the run is graded under this policy's registered values.
GRADING_POLICY: Final = "edgelab-2026.11.1"
# The client project name edgelab records the submission under.
CLIENT_PROJECT = "agentic-trading"
# The edgelab data provider strategy.yaml names: the lake, read through its HTTP API.
LAKE_PROVIDER = "lakehouse"

_SPEC = ConfigDict(frozen=True, extra="forbid")
_GIT_SHA = r"^[0-9a-fA-F]{7,40}$"
_PROJECT = r"^[a-z0-9-]+$"


class SubmissionClient(BaseModel):
    """The client project, its commit (7 to 40 hex characters) and the
    hypothesis under test; edgelab's `ClientRef`."""

    model_config = _SPEC

    project: str = Field(pattern=_PROJECT)
    git_sha: str = Field(pattern=_GIT_SHA)
    hypothesis_id: str = Field(min_length=1)


class SubmissionSource(BaseModel):
    """One source a decision file's rows may name; edgelab's
    `DecisionSource`. A dated source states `can_publish_after_close` and
    no `limitation`; an undated one (`undated`) states a `limitation` and
    no `can_publish_after_close`, and is never named by a row."""

    model_config = _SPEC

    name: str = Field(min_length=1)
    can_publish_after_close: StrictBool | None = None
    undated: StrictBool = False
    limitation: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _dated_or_undated(self) -> Self:
        if self.undated:
            if self.limitation is None or self.can_publish_after_close is not None:
                raise ValueError(
                    f"undated source {self.name!r} needs a limitation and no "
                    "can_publish_after_close"
                )
        elif self.can_publish_after_close is None or self.limitation is not None:
            raise ValueError(
                f"dated source {self.name!r} needs can_publish_after_close and no limitation"
            )
        return self


class SubmissionDecisions(BaseModel):
    """The decision file's path in the folder and its sources, each name
    listed once; edgelab's `DecisionFileSpec`."""

    model_config = _SPEC

    path: str = Field(min_length=1)
    sources: tuple[SubmissionSource, ...] = ()

    @model_validator(mode="after")
    def _unique_names(self) -> Self:
        names = [source.name for source in self.sources]
        if len(set(names)) != len(names):
            raise ValueError(f"source names must be unique, got {names}")
        return self


class ExposureWindow(BaseModel):
    """A closed date range, `start <= end`; edgelab's `DateWindow`."""

    model_config = _SPEC

    start: date
    end: date

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.start > self.end:
            raise ValueError(f"start {self.start} is after end {self.end}")
        return self


class SubmissionPriorExposure(BaseModel):
    """Earlier looks of the family at the data that its trial count does
    not hold: a `note`, the `windows` they looked at (at least one) and the
    known lower bound of their number (`min_count`); edgelab's
    `PriorExposure`."""

    model_config = _SPEC

    note: str = Field(min_length=1)
    windows: tuple[ExposureWindow, ...] = Field(min_length=1)
    min_count: StrictInt = Field(ge=0)


class SubmissionTrials(BaseModel):
    """The trial declaration; edgelab's `TrialDeclaration`: the family, the
    trials counted in it (`declared_n`, at least 1), the ledger reference,
    the unique test keys (at most `declared_n`), the ledger export path and
    the prior exposure."""

    model_config = _SPEC

    family_id: str = Field(min_length=1)
    declared_n: StrictInt = Field(ge=1)
    ledger_ref: str | None = None
    test_keys: tuple[str, ...] | None = None
    ledger_export_path: str | None = None
    prior_exposure: SubmissionPriorExposure | None = None

    @model_validator(mode="after")
    def _keys_fit(self) -> Self:
        keys = self.test_keys
        if keys is not None:
            if len(set(keys)) != len(keys) or any(not key for key in keys):
                raise ValueError("test keys must be unique and non-empty")
            if len(keys) > self.declared_n:
                raise ValueError(f"{len(keys)} test keys exceed declared_n={self.declared_n}")
        return self


def lake_time_text(lake_time: datetime) -> str:
    """The timezone-aware `lake_time` in UTC as ISO-8601 with a `Z` suffix,
    e.g. `2026-11-10T17:20:50Z`: the one spelling of a record's lake time
    in its strategy.yaml and submission.json."""
    return lake_time.astimezone(UTC).isoformat().replace("+00:00", "Z")


class SubmissionLakeQuery(BaseModel):
    """One query against a lake table, its `table` and `sql` text, both
    non-empty; edgelab's `LakeQuery`."""

    model_config = _SPEC

    table: str = Field(min_length=1)
    sql: str = Field(min_length=1)


class SubmissionData(BaseModel):
    """The data the submission reads from the lake as of `lake_time`, a
    UTC timestamp, and its `queries` (possibly none); edgelab's
    `LakeQueryBundle`. `lake_time` dumps to JSON as `lake_time_text`
    gives it. Raises `pydantic.ValidationError` for a lake time without a
    UTC offset of zero."""

    model_config = _SPEC

    kind: Literal["lake"] = "lake"
    lake_time: AwareDatetime
    queries: tuple[SubmissionLakeQuery, ...] = ()

    @field_validator("lake_time")
    @classmethod
    def _utc(cls, value: datetime) -> datetime:
        offset = value.utcoffset()
        if offset is None or offset.total_seconds() != 0:
            raise ValueError("lake_time must be a UTC timestamp")
        return value

    @field_serializer("lake_time", when_used="json")
    def _dump_lake_time(self, value: datetime) -> str:
        return lake_time_text(value)


class ScoreWindow(BaseModel):
    """The window the evidence is scored on, `start < end`; edgelab's
    `ScoreWindow`."""

    model_config = _SPEC

    start: date
    end: date

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.end <= self.start:
            raise ValueError(f"end {self.end} is not after start {self.start}")
        return self


class Submission(BaseModel):
    """A replay folder's `submission.json`; edgelab's `Submission` at
    schema version 2 with `kind` fixed to `"replay"`. `policy` names the
    published grading policy the run is graded under (`GRADING_POLICY`), so
    the strategy folder carries no `grading:` block."""

    model_config = _SPEC

    schema_version: Literal[2] = 2
    kind: Literal["replay"] = "replay"
    client: SubmissionClient
    strategy_dir: str = STRATEGY_DIR
    data: SubmissionData
    decisions: SubmissionDecisions
    trials: SubmissionTrials
    groups_path: str | None = None
    train_end: date | None = None
    test_window: ScoreWindow
    seed: StrictInt = Field(ge=0)
    profile: Literal["fast", "full"]
    policy: Literal["edgelab-2026.11.1"] = GRADING_POLICY


class AgenticMeta(BaseModel):
    """What a firm record's folder states beyond edgelab's model: whether
    the record is `counted` in the trial ledger, the commit it ran from
    (`git_sha`), whether its `src/` tree had no changes
    (`src_tree_clean`), and whether its `declared_n` comes from the
    registry file (`"registration"`) or is the uncounted default of 1
    (`"uncounted_default"`), how many of the engine's orders went
    unfilled over the days its checkpoint holds (`unfilled_orders`, listed
    in unfilled_orders.json), the instruments whose bars start inside
    the warm-up (`short_warmup`, in universe order), the analysts whose
    share of available signals over those days is below
    `settings.firm.low_availability_threshold` or that emitted no signal
    (`low_availability`, in analyst order; analyst_availability.json holds
    every roster analyst's counts),
    the registry file's `expected_availability` as it states it, empty
    without one, and the runtime `environment` (library versions and
    platform) of the pass that wrote it, which the spec hash does not
    cover."""

    model_config = _SPEC

    counted: bool
    git_sha: str
    src_tree_clean: bool
    declared_n_from: Literal["registration", "uncounted_default"]
    unfilled_orders: int = Field(default=0, ge=0)
    short_warmup: tuple[ShortWarmup, ...] = ()
    low_availability: tuple[LowAvailability, ...] = ()
    expected_availability: dict[str, str] = Field(default_factory=dict)
    environment: RuntimeEnvironment


class UnfilledOrdersRecord(BaseModel):
    """What unfilled_orders.json holds: every order the engine could not
    fill (`orders`, in day order; a `price` that was missing or NaN is
    null) and their count per reason (`by_reason`, reasons sorted)."""

    model_config = _SPEC

    orders: list[UnfilledOrder]
    by_reason: dict[str, int]


def decision_sources(entries: Sequence[Mapping[str, Any]]) -> tuple[SubmissionSource, ...]:
    """The sources.json entries `entries` as `SubmissionSource`s, in order;
    their keys (`name`, `can_publish_after_close`, or `undated` and
    `limitation`) are edgelab's own. Raises `pydantic.ValidationError` for
    an entry that is neither a dated nor an undated source, and
    `ValueError` when two entries share a name."""
    sources = tuple(SubmissionSource.model_validate(dict(entry)) for entry in entries)
    names = [source.name for source in sources]
    if len(set(names)) != len(names):
        raise ValueError(f"the record's sources repeat a name: {names}")
    return sources


def submission_for(
    record: FirmSpecRecord,
    *,
    sources: Sequence[Mapping[str, Any]],
    git_sha: str,
    src_tree_clean: bool,
    counted: bool,
    registered_trials: Mapping[str, Any] | None,
    short_warmup: Sequence[ShortWarmup] = (),
    expected_availability: Mapping[str, str] | None = None,
) -> tuple[Submission, AgenticMeta]:
    """The `Submission` of the firm record `record` whose sources.json
    holds `sources`, run from `git_sha` with a clean `src/` tree or not
    (`src_tree_clean`), and its `AgenticMeta`, which lists `short_warmup`
    and `expected_availability` (the registry file's, `None` without one)
    and the `runtime_environment()` of this process.

    `client` is `CLIENT_PROJECT` at `git_sha` under `FIRM_HYPOTHESIS_ID`;
    `data` is a lake bundle at `record.lake_time` in UTC with no queries;
    the family, ledger reference and hypothesis are `FIRM_HYPOTHESIS_ID`;
    the only test key is `record.spec_hash`; `test_window` is the spec's
    window and `train_end` is null; `seed` is `settings.firm.edgelab_seed`
    (read at call time); `profile` is `"full"` when `counted`, else
    `"fast"`. `declared_n` and `prior_exposure` are the registry file's
    `trials.declared_n` and `trials.prior_exposure` (`registered_trials`,
    `None` without a registry file), the latter copied as it stands. An
    uncounted record without them declares 1 trial and no prior exposure.

    Raises `ValueError` when `counted` and `registered_trials` lacks
    `declared_n` or `prior_exposure`, and `pydantic.ValidationError` when
    any field breaks edgelab's model (a window of one day among them)."""
    trials = dict(registered_trials or {})
    if counted:
        missing = [key for key in ("declared_n", "prior_exposure") if trials.get(key) is None]
        if missing:
            raise ValueError(
                f"a counted firm record needs trials.{' and trials.'.join(missing)} in the "
                f"registry file of {FIRM_HYPOTHESIS_ID}; the count is never inferred"
            )
    from_registration = trials.get("declared_n") is not None
    spec = record.spec
    submission = Submission(
        client=SubmissionClient(
            project=CLIENT_PROJECT, git_sha=git_sha, hypothesis_id=FIRM_HYPOTHESIS_ID
        ),
        data=SubmissionData(lake_time=record.lake_time.astimezone(UTC)),
        decisions=SubmissionDecisions(path=DECISIONS_FILE, sources=decision_sources(sources)),
        trials=SubmissionTrials(
            family_id=FIRM_HYPOTHESIS_ID,
            declared_n=trials["declared_n"] if from_registration else 1,
            ledger_ref=FIRM_HYPOTHESIS_ID,
            test_keys=(record.spec_hash,),
            ledger_export_path=None,
            prior_exposure=trials.get("prior_exposure"),
        ),
        test_window=ScoreWindow(start=spec.window_start, end=spec.window_end),
        seed=settings.firm.edgelab_seed,
        profile="full" if counted else "fast",
    )
    meta = AgenticMeta(
        counted=counted,
        git_sha=git_sha,
        src_tree_clean=src_tree_clean,
        declared_n_from="registration" if from_registration else "uncounted_default",
        short_warmup=tuple(short_warmup),
        expected_availability=dict(expected_availability or {}),
        environment=runtime_environment(),
    )
    return submission, meta


def strategy_config(record: FirmSpecRecord) -> dict[str, Any]:
    """`REPLAY_STRATEGY_YAML` set to the firm record `record`, as
    `edgelab`'s replay fixtures set a scaffold: the strategy is
    `strategy.py:Replay` over every instrument id of the universe, rebalanced
    daily with an empty parameter grid, short sales allowed within the
    firm's gross exposure (`settings.firm.max_gross_exposure`, net alike)
    and the decision file's per-instrument bound
    (`settings.firm.max_leverage`); the data block reads the `lakehouse`
    provider over the spec's window, its params exactly the record's lake
    time (`lake_time_text`), its lake version and `survivorship_free: true`,
    the rest of the block as scaffolded; there is no `grading:` block, the
    submission's policy supplies it; the capital is
    `settings.firm.starting_cash` and the seed `settings.firm.edgelab_seed`;
    there is no replication claim. Settings are read at call time."""
    spec = record.spec
    firm = settings.firm
    config: dict[str, Any] = yaml.safe_load(REPLAY_STRATEGY_YAML)
    config["name"] = f"{FIRM_HYPOTHESIS_ID} firm record replay"
    config["author"] = CLIENT_PROJECT
    config["created"] = record.lake_time.date()
    config["strategy"] = "strategy.py:Replay"
    params = config["params"]
    params["universe"]["symbols"] = list(
        dict.fromkeys(row_instrument_id(ticker) for ticker in spec.universe)
    )
    params["schedule"]["rebalance"] = "daily"
    params["risk_limits"] = {
        "max_gross": float(firm.max_gross_exposure),
        "max_net": float(firm.max_gross_exposure),
        "max_weight_per_instrument": float(firm.max_leverage),
    }
    params["shorting"] = "allowed"
    config["parameters"] = {"grid": {}, "selected": {}}
    config["hypothesis"]["rationale"] = (
        f"The registered hypothesis {FIRM_HYPOTHESIS_ID}: the multi-pod firm's daily targets, "
        "decided after each close and filled at the next open, earn a positive net excess return."
    )
    config["data"]["range"] = {"start": spec.window_start, "end": spec.window_end}
    config["data"]["provider"] = LAKE_PROVIDER
    config["data"]["params"] = {
        "lake_time": lake_time_text(record.lake_time),
        "lake_version": record.lake_version,
        "survivorship_free": True,
    }
    del config["grading"]
    config["capital"] = float(firm.starting_cash)
    config["seed"] = settings.firm.edgelab_seed
    config["replication"] = None
    return config


def write_submission_folder(
    trial_dir: Path, record: FirmSpecRecord, submission: Submission, meta: AgenticMeta
) -> None:
    """Write `submission` as `trial_dir`'s submission.json, `meta` as its
    agentic_meta.json, `strategy/strategy.yaml` (`strategy_config(record)`)
    and `strategy/strategy.py`, each atomically and replacing what is
    there, creating the strategy folder. Each source is written
    with the keys it states, as in sources.json: a dated one as `name` and
    `can_publish_after_close`, an undated one as `name`, `undated` and
    `limitation`."""
    trial_dir = Path(trial_dir)
    strategy_dir = trial_dir / submission.strategy_dir
    strategy_dir.mkdir(parents=True, exist_ok=True)
    _write_text(
        strategy_dir / "strategy.yaml", yaml.safe_dump(strategy_config(record), sort_keys=False)
    )
    _write_text(strategy_dir / "strategy.py", REPLAY_STRATEGY_PY)
    _write_text(trial_dir / META_FILE, meta.model_dump_json(indent=2) + "\n")
    content = submission.model_dump(mode="json")
    content["decisions"]["sources"] = [
        source.model_dump(mode="json", exclude_defaults=True)
        for source in submission.decisions.sources
    ]
    _write_text(trial_dir / SUBMISSION_FILE, json.dumps(content, indent=2) + "\n")


def write_unfilled_orders(
    trial_dir: Path, orders: Sequence[UnfilledOrder], meta: AgenticMeta
) -> AgenticMeta:
    """Write `orders` to `trial_dir`'s unfilled_orders.json as an
    `UnfilledOrdersRecord`, a non-finite `price` as null, then `meta` with
    `unfilled_orders` set to their number as its agentic_meta.json, each
    atomically and replacing what is there; return that `meta`."""
    trial_dir = Path(trial_dir)
    written = [
        order
        if order.price is None or math.isfinite(order.price)
        else order.model_copy(update={"price": None})
        for order in orders
    ]
    counts = Counter(order.reason for order in written)
    record = UnfilledOrdersRecord(
        orders=written, by_reason={reason: counts[reason] for reason in sorted(counts)}
    )
    _write_text(trial_dir / UNFILLED_ORDERS_FILE, record.model_dump_json(indent=2) + "\n")
    updated = meta.model_copy(update={"unfilled_orders": len(written)})
    _write_text(trial_dir / META_FILE, updated.model_dump_json(indent=2) + "\n")
    return updated


def write_analyst_availability(
    trial_dir: Path, availability: AnalystAvailability, meta: AgenticMeta
) -> AgenticMeta:
    """Write `availability` to `trial_dir`'s analyst_availability.json, then
    `meta` with `low_availability` set to its analysts under
    `settings.firm.low_availability_threshold` (read at call time) or
    without any signal (`firm_record.availability.low_availability`), each
    `undeclared` when `meta.expected_availability` does not name it, as its
    agentic_meta.json, each atomically and replacing what is there; return
    that `meta`."""
    trial_dir = Path(trial_dir)
    _write_text(
        trial_dir / ANALYST_AVAILABILITY_FILE, availability.model_dump_json(indent=2) + "\n"
    )
    flagged = low_availability(
        availability, settings.firm.low_availability_threshold, meta.expected_availability
    )
    updated = meta.model_copy(update={"low_availability": tuple(flagged)})
    _write_text(trial_dir / META_FILE, updated.model_dump_json(indent=2) + "\n")
    return updated
