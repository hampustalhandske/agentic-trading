# store.py

"""The on-disk store of one trial's `decision_file.v1` decisions: a stage
(a) trial's signals and decisions (`SignalStore`), or a firm record's pod
rows (`FirmDecisionStore`).

`SignalStore` keeps one directory per spec hash under a store directory:

- spec.json: the trial's `SpecRecord`;
- sources.json: the source the trial's rows carry (`decisions.source_of`
  of the analyst's declared data requirements) with whether it can
  publish after the close, `{"name": ..., "can_publish_after_close": ...}`, then each
  undated input the analyst declares with its look-ahead limitation,
  `{"name": ..., "undated": true, "limitation": ...}`
  (`decisions.undated_declarations`); `[]` for a price-only analyst with
  no undated input;
- signals.parquet: one row per instrument per day, `date`,
  `instrument_id`, `status`, `direction`, `confidence`, `source_as_of`,
  `sample_count`;
- decisions.parquet: a `decision_file.v1` file, `decision_date`,
  `instrument_kind`, `instrument_id`, `target_weight` (from
  `decisions.decision_weights`), plus `status`, `source` and
  `source_as_of`;
- progress.json: `{"last_completed": "<ISO date>"}`, the last day both
  Parquet files hold;
- counts.json: per-instrument status day counts and annual turnover
  (`SignalStore.counts`), written on request.

`FirmDecisionStore` keeps the same layout for a firm record, keyed by its
`FirmSpecRecord`'s hash: spec.json holds that record, sources.json the
sources it is given (the same entry shapes), and decisions.parquet one
`DecisionRow` per (decision_date, instrument_id, pod_id), with `pod_id`
and `delta_equiv_weight` beside the stage (a) columns; it writes no
signals.parquet or counts.json.

Each file is written to a temporary file in the trial directory and
renamed over the old one, and progress.json is written after every
Parquet file, so a write killed part way leaves every file whole and
progress.json naming the last day fully written; rows dated after it are
dropped when the store is next opened. Dates are stored as dates. The
store reads no price or return.

By default each day's write rewrites every Parquet file whole. Inside
`appending()`, a pass of many days, each day's rows go instead to a part
file of their own under the trial directory's `.parts/` directory
(`<file name>/<ISO day>.parquet`, written whole the same way, before
progress.json), and each Parquet file is written once when the block
ends, with the same bytes the day-by-day rewrites give, and `.parts/` is
removed. Inside the block, a Parquet file holds only the days written
before it began. A pass killed inside the block leaves its part files;
the next `open` merges those of days up to progress.json's into the
Parquet files, drops the rest and removes `.parts/`, so the trial
directory then holds what it would had every day rewritten the files."""

import json
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Any, ClassVar, Literal, Self, TypedDict

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from agentic_trading.agents.schemas import AgentSignal, SignalStatus
from agentic_trading.stage_a.decisions import (
    SOURCES,
    LookAheadError,
    check_source_as_of,
    decision_weights,
    source_of,
    undated_declarations,
)
from agentic_trading.stage_a.spec import FirmSpecRecord, SpecRecord

# `decision_file.v1`'s instrument kinds.
InstrumentKind = Literal["equity", "etf", "future", "fx", "crypto", "option"]
# The instrument kinds the vendored `decision_file.v1` row schema accepts; a
# firm record's rows use only these (bond ETFs are `equity`).
DecisionKind = Literal["equity", "future", "fx", "crypto", "option"]
# `decision_file.v1`'s bound on a row's `target_weight`, and on the net
# weight of the rows one (decision_date, instrument_id) repeats over.
MAX_DECISION_WEIGHT = 6.0

# Each Parquet file's columns, in order, to their dtype in the file: the
# pyarrow-backed dtypes write `date32` dates, nullable `int64` and text.
_SIGNAL_COLUMNS = {
    "date": "date32[day][pyarrow]",
    "instrument_id": "large_string[pyarrow]",
    "status": "large_string[pyarrow]",
    "direction": "large_string[pyarrow]",
    "confidence": "double[pyarrow]",
    "source_as_of": "date32[day][pyarrow]",
    "sample_count": "int64[pyarrow]",
}
_DECISION_COLUMNS = {
    "decision_date": "date32[day][pyarrow]",
    "instrument_kind": "large_string[pyarrow]",
    "instrument_id": "large_string[pyarrow]",
    "target_weight": "double[pyarrow]",
    "status": "large_string[pyarrow]",
    "source": "large_string[pyarrow]",
    "source_as_of": "date32[day][pyarrow]",
}
_FIRM_DECISION_COLUMNS = {
    **_DECISION_COLUMNS,
    "pod_id": "large_string[pyarrow]",
    "delta_equiv_weight": "double[pyarrow]",
}
# A file dtype to the numpy-backed dtype the store holds the column in
# between writes; numpy-backed columns stay contiguous as days are
# appended, so each rewrite costs time linear in the rows.
_MEMORY_DTYPES: dict[str, Any] = {
    "date32[day][pyarrow]": "datetime64[s]",
    "large_string[pyarrow]": object,
    "double[pyarrow]": "float64",
    "int64[pyarrow]": "Int64",
}


# The directory of the trial directory holding the part files of a pass
# written inside `_TrialStore.appending`.
PARTS_DIR = ".parts"


class StoreMismatchError(ValueError):
    """The trial directory holds a spec.json or sources.json other than
    the one the store was opened with."""


class StoreOrderError(ValueError):
    """A day written on or before the store's last completed day."""


class DuplicateRowError(ValueError):
    """A repeated (decision_date, instrument_id) whose rows do not each
    carry their own non-null `pod_id`."""


class StatusCounts(TypedDict):
    """Days an instrument had each signal status."""

    active: int
    unavailable: int
    error: int


class StoreCounts(TypedDict):
    """What counts.json holds: `instruments`, each instrument id to its
    `StatusCounts`, and `annual_turnover`, the mean daily turnover
    `sum_i |w_i,t - w_i,t-1|` (the first day measured from a flat book, a
    missing instrument weighing 0) times the trading days per year."""

    instruments: dict[str, StatusCounts]
    annual_turnover: float


class DecisionRow(BaseModel):
    """One row of a firm record's `decision_file.v1` file.

    `decision_date` is the day the weight was decided, after its close;
    `instrument_kind` one of the schema's kinds; `instrument_id` a lake
    ticker, a future's root or an option's unpadded OSI symbol; `target_weight` finite and within
    +-`MAX_DECISION_WEIGHT`, a signed premium weight for an option;
    `status` the row's status, a non-active row weighing exactly 0;
    `source` and `source_as_of` the newest-dated input the row rests on,
    both set or both empty; `pod_id` the pod (manager `agent_id`) the row
    belongs to; `delta_equiv_weight` an option row's finite delta-equivalent
    weight, `None` otherwise or without a mark. Raises
    `pydantic.ValidationError` for a row breaking any of these, except the
    both-or-neither rule, which `FirmDecisionStore.write_day` audits."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    decision_date: date
    instrument_kind: DecisionKind
    instrument_id: str = Field(min_length=1)
    target_weight: float = Field(ge=-MAX_DECISION_WEIGHT, le=MAX_DECISION_WEIGHT)
    status: SignalStatus = "active"
    source: str | None = None
    source_as_of: date | None = None
    pod_id: str | None = None
    delta_equiv_weight: float | None = None

    @model_validator(mode="after")
    def _inactive_rows_weigh_zero(self) -> Self:
        if self.status != "active" and self.target_weight != 0.0:
            raise ValueError(
                f"{self.status} row for {self.instrument_id} has weight {self.target_weight}"
            )
        return self


class _TrialStore:
    """What every trial store shares: the trial directory `store_dir /
    record.spec_hash`, its spec.json and sources.json, its Parquet files
    (`_files`, in write order, each to its columns and day column) and
    progress.json. Call `open` before any other method."""

    _files: ClassVar[dict[str, tuple[dict[str, str], str]]] = {}

    def __init__(
        self,
        store_dir: Path,
        record: SpecRecord | FirmSpecRecord,
        sources: list[dict[str, str | bool]],
    ) -> None:
        self.record = record
        self.trial_dir = Path(store_dir) / record.spec_hash
        self._sources = sources
        self._tables = {name: _frame([], columns) for name, (columns, _) in self._files.items()}
        # Inside `appending`, each file's rows of the days written in the
        # block, by day in write order; `None` outside it.
        self._pending: dict[str, dict[date, pd.DataFrame]] | None = None
        self._last_completed: date | None = None
        self._opened = False

    def open(self) -> None:
        """Create the trial directory, or check an existing one, and load
        its progress and rows. Writes spec.json and sources.json when
        absent. Raises `StoreMismatchError`, writing nothing, when an
        existing spec.json holds a record other than this store's or an
        existing sources.json other sources. The whole record is compared,
        the fields outside the hash included, so a firm record that ran on
        one commit (`FirmSpecRecord.git_sha`) resumes only on that commit:
        on another it must restart in a new store."""
        self.trial_dir.mkdir(parents=True, exist_ok=True)
        spec_path = self.trial_dir / "spec.json"
        if spec_path.exists():
            stored = type(self.record).model_validate_json(spec_path.read_text())
            if stored != self.record:
                raise StoreMismatchError(
                    f"{spec_path.name} of trial {self.record.spec_hash} holds another record "
                    "(another lake version, quantcore version or commit); restart the record "
                    "in a new store, or resume it on the same commit"
                )
        sources_path = self.trial_dir / "sources.json"
        if sources_path.exists() and json.loads(sources_path.read_text()) != self._sources:
            raise StoreMismatchError(
                f"{sources_path.name} of trial {self.record.spec_hash} lists other sources"
            )
        if not spec_path.exists():
            _write_text(spec_path, self.record.model_dump_json(indent=2) + "\n")
        if not sources_path.exists():
            _write_text(sources_path, json.dumps(self._sources, indent=2) + "\n")

        progress_path = self.trial_dir / "progress.json"
        if progress_path.exists():
            last = json.loads(progress_path.read_text())["last_completed"]
            self._last_completed = None if last is None else date.fromisoformat(last)
        self._tables = {
            name: self._load(self.trial_dir / name, columns, day_column)
            for name, (columns, day_column) in self._files.items()
        }
        if (self.trial_dir / PARTS_DIR).exists():
            self._merge_parts_left()
        self._opened = True

    @contextmanager
    def appending(self) -> Iterator[None]:
        """Write the days of the block as part files and each Parquet file
        once at its end, also when the block raises (see the module
        docstring). Raises `RuntimeError` when the store is not open or
        already appending."""
        self._require_open()
        if self._pending is not None:
            raise RuntimeError(f"{type(self).__name__} is already appending")
        self._pending = {name: {} for name in self._files}
        try:
            yield
        finally:
            pending, self._pending = self._pending, None
            self._merge(pending)

    def _part_path(self, name: str, day: date) -> Path:
        """The part file of `name`'s rows of `day`."""
        return self.trial_dir / PARTS_DIR / name / f"{day.isoformat()}.parquet"

    def _merge(self, pending: Mapping[str, Mapping[date, pd.DataFrame]]) -> None:
        """Write each file holding rows of `pending` (file name to its new
        rows by day, in day order) with its stored rows followed by those,
        then remove the part files."""
        for name, (columns, _) in self._files.items():
            frames = list(pending[name].values())
            if not frames:
                continue
            table = _append(self._tables[name], pd.concat(frames, ignore_index=True), columns)
            _write_parquet(self.trial_dir / name, table, columns)
            self._tables[name] = table
        shutil.rmtree(self.trial_dir / PARTS_DIR, ignore_errors=True)

    def _merge_parts_left(self) -> None:
        """Merge the part files a killed pass left (see the module
        docstring): each file's parts of days up to the last completed one
        replace its stored rows of those days and follow the rest."""
        pending: dict[str, dict[date, pd.DataFrame]] = {}
        for name, (columns, day_column) in self._files.items():
            pending[name] = {}
            for path in sorted((self.trial_dir / PARTS_DIR / name).glob("*.parquet")):
                day = date.fromisoformat(path.stem)
                if self._last_completed is None or day > self._last_completed:
                    continue
                pending[name][day] = pd.read_parquet(path).astype(_memory_dtypes(columns))
            table = self._tables[name]
            replaced = table[day_column].isin([pd.Timestamp(day) for day in pending[name]])
            self._tables[name] = table[~replaced].reset_index(drop=True)
        self._merge(pending)

    def _table(self, name: str) -> pd.DataFrame:
        """Every row of `name` written so far, the rows of an open
        `appending` block included."""
        frames = [] if self._pending is None else list(self._pending[name].values())
        if not frames:
            return self._tables[name]
        columns = self._files[name][0]
        return _append(self._tables[name], pd.concat(frames, ignore_index=True), columns)

    def last_completed(self) -> date | None:
        """The last day written in full, `None` before the first."""
        self._require_open()
        return self._last_completed

    def _require_open(self) -> None:
        if not self._opened:
            raise RuntimeError(f"{type(self).__name__}.open() has not been called")

    def _check_order_and_window(self, day: date) -> None:
        """Raise `StoreOrderError` when `day` is on or before the last
        completed day, and `ValueError` when it is outside the spec's
        window."""
        if self._last_completed is not None and day <= self._last_completed:
            raise StoreOrderError(
                f"day {day.isoformat()} is not after the last completed day "
                f"{self._last_completed.isoformat()}"
            )
        spec = self.record.spec
        if not spec.window_start <= day <= spec.window_end:
            raise ValueError(
                f"day {day.isoformat()} is outside the window {spec.window_start.isoformat()} "
                f"to {spec.window_end.isoformat()}"
            )

    def _commit(self, day: date, rows: Mapping[str, list[dict[str, Any]]]) -> None:
        """Append `rows` (file name to that file's new rows) to each file,
        after dropping any row dated `day` already there, write the files in
        order, then record `day` in progress.json. Inside `appending`, the
        rows are written as `day`'s part file of each file instead."""
        if self._pending is not None:
            self._commit_part(day, rows)
            return
        stamp = pd.Timestamp(day)
        tables = {
            name: _append(
                self._tables[name][self._tables[name][day_column] != stamp],
                _frame(rows[name], columns),
                columns,
            )
            for name, (columns, day_column) in self._files.items()
        }
        for name, (columns, _) in self._files.items():
            _write_parquet(self.trial_dir / name, tables[name], columns)
        _write_text(
            self.trial_dir / "progress.json",
            json.dumps({"last_completed": day.isoformat()}) + "\n",
        )
        self._tables, self._last_completed = tables, day

    def _commit_part(self, day: date, rows: Mapping[str, list[dict[str, Any]]]) -> None:
        """`_commit` inside `appending`: write each file's part file of
        `day` in order, record `day` in progress.json, then hold `rows` as
        `day`'s rows of each file, replacing any held or stored before."""
        assert self._pending is not None
        frames = {name: _frame(rows[name], columns) for name, (columns, _) in self._files.items()}
        for name, (columns, _) in self._files.items():
            path = self._part_path(name, day)
            path.parent.mkdir(parents=True, exist_ok=True)
            _write_parquet(path, frames[name], columns)
        _write_text(
            self.trial_dir / "progress.json",
            json.dumps({"last_completed": day.isoformat()}) + "\n",
        )
        stamp = pd.Timestamp(day)
        for name, (_, day_column) in self._files.items():
            table = self._tables[name]
            stored = table[day_column] == stamp
            if stored.any():
                self._tables[name] = table[~stored].reset_index(drop=True)
            self._pending[name][day] = frames[name]
        self._last_completed = day

    def _load(self, path: Path, columns: dict[str, str], day_column: str) -> pd.DataFrame:
        """The rows of `path` dated on or before the last completed day, as
        a frame with `columns`' dtypes; empty when the file is absent or no
        day is completed. Rewrites the file when it holds later rows."""
        if not path.exists():
            return _frame([], columns)
        stored = pd.read_parquet(path).astype(_memory_dtypes(columns))
        if self._last_completed is None:
            kept = stored.iloc[0:0]
        else:
            last = pd.Timestamp(self._last_completed)
            kept = stored[stored[day_column] <= last].reset_index(drop=True)
        if len(kept) != len(stored):
            _write_parquet(path, kept, columns)
        return kept


class SignalStore(_TrialStore):
    """The store of the stage (a) trial `record` defines, in `store_dir /
    record.spec_hash`. Call `open` before any other method."""

    _files: ClassVar[dict[str, tuple[dict[str, str], str]]] = {
        "signals.parquet": (_SIGNAL_COLUMNS, "date"),
        "decisions.parquet": (_DECISION_COLUMNS, "decision_date"),
    }

    def __init__(self, store_dir: Path, record: SpecRecord) -> None:
        declared = [
            requirement["source"]
            for requirement in record.spec.analyst_params.get("data_requirements", [])
        ]
        self._source = source_of(declared)
        sources: list[dict[str, str | bool]] = (
            []
            if self._source is None
            else [{"name": self._source, "can_publish_after_close": SOURCES[self._source]}]
        )
        sources.extend(undated_declarations(declared))
        super().__init__(store_dir, record, sources)
        self.record: SpecRecord = record

    def write_day(
        self,
        day: date,
        signals: Sequence[AgentSignal],
        instrument_ids: Mapping[str, str],
        instrument_kind: InstrumentKind,
    ) -> None:
        """Append `day`'s signals and decisions and record `day` as
        completed. `signals` holds one signal per asset, dated `day`;
        `instrument_ids` maps each asset to the instrument id its rows are
        keyed by; every decision row is of `instrument_kind`. A row's
        `source` is the trial's source when its signal carries a
        `source_as_of`, else empty.

        Every check runs before anything is written. Raises
        `StoreOrderError` when `day` is on or before `last_completed()`;
        `ValueError` when `day` is outside the spec's window, a signal is
        dated another day, two signals share an asset, or an asset is
        outside the spec's universe or `instrument_ids`; and
        `LookAheadError` when the trial has a source and an active signal
        carries no `source_as_of`, or a row's `source` and `source_as_of`
        fail `decisions.check_source_as_of`."""
        self._require_open()
        self._check_day(day, signals, instrument_ids)
        weights = decision_weights(signals)
        sources: list[str | None] = []
        for signal in signals:
            active = signal.status == "active"
            if self._source is not None and active and signal.source_as_of is None:
                raise LookAheadError(
                    f"active signal for {signal.asset!r} on {day.isoformat()} has no "
                    f"source_as_of, but the analyst declares source {self._source!r}"
                )
            source = self._source if signal.source_as_of is not None else None
            check_source_as_of(signal.source_as_of, day, source)
            sources.append(source)

        rows = sorted(
            zip(signals, sources, strict=True), key=lambda row: instrument_ids[row[0].asset]
        )
        self._commit(
            day,
            {
                "signals.parquet": [
                    {
                        "date": day,
                        "instrument_id": instrument_ids[signal.asset],
                        "status": signal.status,
                        "direction": signal.direction,
                        "confidence": signal.confidence,
                        "source_as_of": signal.source_as_of,
                        "sample_count": signal.sample_count,
                    }
                    for signal, _ in rows
                ],
                "decisions.parquet": [
                    {
                        "decision_date": day,
                        "instrument_kind": instrument_kind,
                        "instrument_id": instrument_ids[signal.asset],
                        "target_weight": weights[signal.asset],
                        "status": signal.status,
                        "source": source,
                        "source_as_of": signal.source_as_of,
                    }
                    for signal, source in rows
                ],
            },
        )

    def counts(self) -> StoreCounts:
        """Each instrument's status day counts, in order of first
        appearance, and the annual turnover of the stored decisions (see
        `StoreCounts`); turnover is 0.0 for an empty store."""
        self._require_open()
        decisions = self._table("decisions.parquet")
        by_status = pd.crosstab(decisions["instrument_id"], decisions["status"])
        instruments: dict[str, StatusCounts] = {}
        for instrument in decisions["instrument_id"].unique():
            row = by_status.loc[instrument]
            instruments[str(instrument)] = StatusCounts(
                active=int(row.get("active", 0)),
                unavailable=int(row.get("unavailable", 0)),
                error=int(row.get("error", 0)),
            )

        weights = (
            decisions.pivot(index="decision_date", columns="instrument_id", values="target_weight")
            .sort_index()
            .fillna(0.0)
        )
        turnover = float((weights - weights.shift(fill_value=0.0)).abs().to_numpy().sum())
        days = len(weights)
        trading_days: int = self.record.spec.settings["firm"]["trading_days_per_year"]
        annual = turnover / days * trading_days if days else 0.0
        return StoreCounts(instruments=instruments, annual_turnover=annual)

    def write_counts(self) -> None:
        """Write `counts()` to counts.json."""
        _write_text(self.trial_dir / "counts.json", json.dumps(self.counts(), indent=2) + "\n")

    def _check_day(
        self, day: date, signals: Sequence[AgentSignal], instrument_ids: Mapping[str, str]
    ) -> None:
        """Raise `StoreOrderError` or `ValueError` as `write_day` documents."""
        self._check_order_and_window(day)
        universe = set(self.record.spec.universe)
        seen: set[str] = set()
        for signal in signals:
            if signal.as_of_date != day:
                raise ValueError(
                    f"signal for {signal.asset!r} is dated {signal.as_of_date.isoformat()}, "
                    f"not {day.isoformat()}"
                )
            if signal.asset in seen:
                raise ValueError(f"two signals for asset {signal.asset!r}")
            seen.add(signal.asset)
            if signal.asset not in universe:
                raise ValueError(f"asset {signal.asset!r} is outside the spec's universe")
            if signal.asset not in instrument_ids:
                raise ValueError(f"asset {signal.asset!r} has no instrument id")


class FirmDecisionStore(_TrialStore):
    """The store of the firm record `record` defines, in `store_dir /
    record.spec_hash`, whose sources.json holds `sources` (entries shaped
    as a stage (a) trial's). Call `open` before any other method."""

    _files: ClassVar[dict[str, tuple[dict[str, str], str]]] = {
        "decisions.parquet": (_FIRM_DECISION_COLUMNS, "decision_date")
    }

    def __init__(
        self, store_dir: Path, record: FirmSpecRecord, *, sources: list[dict[str, str | bool]]
    ) -> None:
        super().__init__(store_dir, record, sources)
        self.record: FirmSpecRecord = record
        self._dated_sources = {
            str(entry["name"]) for entry in sources if not entry.get("undated", False)
        }

    def day_rows(self, day: date) -> list[DecisionRow]:
        """The rows stored for `day`, in stored order; empty for a day not
        written."""
        self._require_open()
        table = self._table("decisions.parquet")
        picked = table[table["decision_date"] == pd.Timestamp(day)]
        return [
            DecisionRow(
                decision_date=day,
                instrument_kind=row["instrument_kind"],
                instrument_id=row["instrument_id"],
                target_weight=float(row["target_weight"]),
                status=row["status"],
                source=None if pd.isna(row["source"]) else str(row["source"]),
                source_as_of=(
                    None
                    if pd.isna(row["source_as_of"])
                    else pd.Timestamp(row["source_as_of"]).date()
                ),
                pod_id=None if pd.isna(row["pod_id"]) else str(row["pod_id"]),
                delta_equiv_weight=(
                    None if pd.isna(row["delta_equiv_weight"]) else float(row["delta_equiv_weight"])
                ),
            )
            for row in picked.to_dict("records")
        ]

    def rewind(self, day: date | None) -> None:
        """Drop every row dated after `day` (every row for `None`) and
        record `day` as the last completed day, so those days are written
        again. Raises `StoreOrderError` when `day` is after
        `last_completed()`, and `RuntimeError` inside `appending`."""
        self._require_open()
        if self._pending is not None:
            raise RuntimeError(f"{type(self).__name__} cannot rewind while appending")
        last = self._last_completed
        if day is not None and (last is None or day > last):
            raise StoreOrderError(
                f"cannot rewind to {day.isoformat()}, after the last completed day {last}"
            )
        if day == last:
            return
        tables = {}
        for name, (columns, day_column) in self._files.items():
            table = self._tables[name]
            kept = table.iloc[0:0] if day is None else table[table[day_column] <= pd.Timestamp(day)]
            tables[name] = kept.reset_index(drop=True)
            _write_parquet(self.trial_dir / name, tables[name], columns)
        _write_text(
            self.trial_dir / "progress.json",
            json.dumps({"last_completed": None if day is None else day.isoformat()}) + "\n",
        )
        self._tables, self._last_completed = tables, day

    def write_day(self, day: date, rows: Sequence[DecisionRow]) -> None:
        """Write `day`'s decision rows, sorted by instrument id then pod
        id, and record `day` as completed. Writing the last completed day
        again replaces its rows, so a day's write is idempotent.

        The rows come from `firm_record.rows.firm_day_rows`: a row with a
        nonzero weight is active, since the firm holds that target; a
        zero-weight row is active when any of its signals is, `error` when
        none is and one errored, else `unavailable`; a non-active row always
        weighs 0.

        Every check runs before anything is written. Raises
        `StoreOrderError` when `day` is before `last_completed()`;
        `ValueError` when `day` is outside the spec's window, a row is
        dated another day, or the rows of one instrument net beyond
        +-`MAX_DECISION_WEIGHT`; `DuplicateRowError` when an instrument has
        several rows and one of them has no `pod_id` or two share one, the
        message naming the instrument, the date and the pod; and
        `LookAheadError` when a row's `source` is not among the store's
        dated sources or its `source` and `source_as_of` fail
        `decisions.check_source_as_of`."""
        self._require_open()
        if day != self._last_completed:
            self._check_order_and_window(day)
        for row in rows:
            if row.decision_date != day:
                raise ValueError(
                    f"row for {row.instrument_id!r} is dated {row.decision_date.isoformat()}, "
                    f"not {day.isoformat()}"
                )
            if row.source is not None and row.source not in self._dated_sources:
                raise LookAheadError(
                    f"row for {row.instrument_id!r} on {day.isoformat()} names source "
                    f"{row.source!r}, which the record's sources do not list"
                )
            check_source_as_of(row.source_as_of, day, row.source)
        _check_pod_rows(day, rows)

        ordered = sorted(rows, key=lambda row: (row.instrument_id, row.pod_id or ""))
        self._commit(day, {"decisions.parquet": [row.model_dump() for row in ordered]})


def _check_pod_rows(day: date, rows: Sequence[DecisionRow]) -> None:
    """Raise `DuplicateRowError` when an instrument repeated among `rows`
    has a row without a `pod_id` or two rows sharing one, and `ValueError`
    when an instrument's rows net beyond +-`MAX_DECISION_WEIGHT`."""
    by_instrument: dict[str, list[DecisionRow]] = {}
    for row in rows:
        by_instrument.setdefault(row.instrument_id, []).append(row)
    for instrument, group in by_instrument.items():
        if len(group) > 1:
            pods = Counter(row.pod_id for row in group)
            if None in pods:
                raise DuplicateRowError(
                    f"{instrument} on {day.isoformat()} has {len(group)} rows and one with "
                    "pod_id None; a repeated instrument needs its own pod_id on every row"
                )
            shared = sorted(str(pod) for pod, count in pods.items() if count > 1)
            if shared:
                raise DuplicateRowError(
                    f"{instrument} on {day.isoformat()} has several rows for pod_id "
                    f"{', '.join(shared)}; each pod has at most one row per instrument"
                )
        net = sum(row.target_weight for row in group)
        if abs(net) > MAX_DECISION_WEIGHT:
            raise ValueError(
                f"{instrument} on {day.isoformat()} nets {net}, beyond the decision file's "
                f"+-{MAX_DECISION_WEIGHT}"
            )


def _memory_dtypes(columns: dict[str, str]) -> dict[str, Any]:
    """`columns` with each file dtype replaced by its in-memory dtype."""
    return {name: _MEMORY_DTYPES[dtype] for name, dtype in columns.items()}


def _frame(rows: list[dict[str, Any]], columns: dict[str, str]) -> pd.DataFrame:
    """`rows` as a frame with exactly `columns`, in order, at their
    in-memory dtypes."""
    return pd.DataFrame(rows, columns=list(columns)).astype(_memory_dtypes(columns))


def _append(frame: pd.DataFrame, rows: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    """`frame` followed by `rows`, at `columns`' in-memory dtypes."""
    return pd.concat([frame, rows], ignore_index=True).astype(_memory_dtypes(columns))


def _write_atomically(path: Path, write: Callable[[Path], Any]) -> None:
    """Call `write` on a temporary file beside `path`, then rename it over
    `path`; the temporary file is removed if `write` raises."""
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    try:
        write(Path(temporary))
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _write_text(path: Path, text: str) -> None:
    """Write `text` to `path` atomically."""
    _write_atomically(path, lambda target: target.write_text(text))


def _write_parquet(path: Path, frame: pd.DataFrame, columns: dict[str, str]) -> None:
    """Write `frame` to `path` as Parquet at `columns`' file dtypes,
    without its index, atomically."""
    typed = frame.astype(columns)
    _write_atomically(path, lambda target: typed.to_parquet(target, index=False))
