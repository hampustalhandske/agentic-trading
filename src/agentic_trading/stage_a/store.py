# store.py

"""The on-disk store of one stage (a) trial's signals and decisions.

`SignalStore` keeps one directory per spec hash under a store directory:

- spec.json: the trial's `SpecRecord`;
- sources.json: the source the trial's rows carry (`decisions.source_of`
  of the analyst's declared data requirements) with whether it can
  publish after the close, `[{"name": ..., "after_close": ...}]`, or `[]`
  for a price-only analyst;
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

Each file is written to a temporary file in the trial directory and
renamed over the old one, and progress.json is written after both
Parquet files, so a write killed part way leaves every file whole and
progress.json naming the last day fully written; rows dated after it are
dropped when the store is next opened. Dates are stored as dates. The
store reads no price or return."""

import json
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any, Literal, TypedDict

import pandas as pd

from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.stage_a.decisions import (
    SOURCES,
    LookAheadError,
    check_source_as_of,
    decision_weights,
    source_of,
)
from agentic_trading.stage_a.spec import SpecRecord

# `decision_file.v1`'s instrument kinds.
InstrumentKind = Literal["equity", "etf", "future", "fx", "crypto", "option"]

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
# A file dtype to the numpy-backed dtype the store holds the column in
# between writes; numpy-backed columns stay contiguous as days are
# appended, so each rewrite costs time linear in the rows.
_MEMORY_DTYPES: dict[str, Any] = {
    "date32[day][pyarrow]": "datetime64[s]",
    "large_string[pyarrow]": object,
    "double[pyarrow]": "float64",
    "int64[pyarrow]": "Int64",
}


class StoreMismatchError(ValueError):
    """The trial directory holds a spec.json or sources.json other than
    the one the store was opened with."""


class StoreOrderError(ValueError):
    """A day written on or before the store's last completed day."""


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


class SignalStore:
    """The store of the trial `record` defines, in `store_dir /
    record.spec_hash`. Call `open` before any other method."""

    def __init__(self, store_dir: Path, record: SpecRecord) -> None:
        self.record = record
        self.trial_dir = Path(store_dir) / record.spec_hash
        self._source = source_of(
            requirement["source"]
            for requirement in record.spec.analyst_params.get("data_requirements", [])
        )
        self._signals = _frame([], _SIGNAL_COLUMNS)
        self._decisions = _frame([], _DECISION_COLUMNS)
        self._last_completed: date | None = None
        self._opened = False

    def open(self) -> None:
        """Create the trial directory, or check an existing one, and load
        its progress and rows. Writes spec.json and sources.json when
        absent. Raises `StoreMismatchError`, writing nothing, when an
        existing spec.json holds a record other than this store's or an
        existing sources.json other sources."""
        self.trial_dir.mkdir(parents=True, exist_ok=True)
        spec_path = self.trial_dir / "spec.json"
        if spec_path.exists():
            stored = SpecRecord.model_validate_json(spec_path.read_text())
            if stored != self.record:
                raise StoreMismatchError(
                    f"{spec_path.name} of trial {self.record.spec_hash} holds another record"
                )
        sources = (
            []
            if self._source is None
            else [{"name": self._source, "after_close": SOURCES[self._source]}]
        )
        sources_path = self.trial_dir / "sources.json"
        if sources_path.exists() and json.loads(sources_path.read_text()) != sources:
            raise StoreMismatchError(
                f"{sources_path.name} of trial {self.record.spec_hash} lists other sources"
            )
        if not spec_path.exists():
            _write_text(spec_path, self.record.model_dump_json(indent=2) + "\n")
        if not sources_path.exists():
            _write_text(sources_path, json.dumps(sources, indent=2) + "\n")

        progress_path = self.trial_dir / "progress.json"
        if progress_path.exists():
            last = json.loads(progress_path.read_text())["last_completed"]
            self._last_completed = None if last is None else date.fromisoformat(last)
        self._signals = self._load(self.trial_dir / "signals.parquet", _SIGNAL_COLUMNS, "date")
        self._decisions = self._load(
            self.trial_dir / "decisions.parquet", _DECISION_COLUMNS, "decision_date"
        )
        self._opened = True

    def last_completed(self) -> date | None:
        """The last day written in full, `None` before the first."""
        self._require_open()
        return self._last_completed

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
        signal_rows = _frame(
            [
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
            _SIGNAL_COLUMNS,
        )
        decision_rows = _frame(
            [
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
            _DECISION_COLUMNS,
        )
        signals_table = _append(self._signals, signal_rows, _SIGNAL_COLUMNS)
        decisions_table = _append(self._decisions, decision_rows, _DECISION_COLUMNS)

        _write_parquet(self.trial_dir / "signals.parquet", signals_table, _SIGNAL_COLUMNS)
        _write_parquet(self.trial_dir / "decisions.parquet", decisions_table, _DECISION_COLUMNS)
        _write_text(
            self.trial_dir / "progress.json",
            json.dumps({"last_completed": day.isoformat()}) + "\n",
        )
        self._signals, self._decisions, self._last_completed = (
            signals_table,
            decisions_table,
            day,
        )

    def counts(self) -> StoreCounts:
        """Each instrument's status day counts, in order of first
        appearance, and the annual turnover of the stored decisions (see
        `StoreCounts`); turnover is 0.0 for an empty store."""
        self._require_open()
        decisions = self._decisions
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

    def _require_open(self) -> None:
        if not self._opened:
            raise RuntimeError("SignalStore.open() has not been called")

    def _check_day(
        self, day: date, signals: Sequence[AgentSignal], instrument_ids: Mapping[str, str]
    ) -> None:
        """Raise `StoreOrderError` or `ValueError` as `write_day` documents."""
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
        universe = set(spec.universe)
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
