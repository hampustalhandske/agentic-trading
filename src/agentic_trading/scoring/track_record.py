# track_record.py

from bisect import bisect_left, bisect_right
from datetime import date, timedelta
from typing import Literal

from agentic_trading.agents.schemas import PerformanceRecord


class _SourceHistory:
    """One index key's records in insertion order, with their dates kept
    alongside for binary search.

    `in_date_order` stays True while every record was appended with an
    `as_of_date` no earlier than the one before it. Only then are the
    records inside a date window one contiguous run of the list, so
    `window` can slice by binary search; otherwise it filters linearly.
    Either way it returns the records inside the window in insertion
    order, as a new list."""

    __slots__ = ("dates", "earliest", "in_date_order", "records")

    def __init__(self) -> None:
        self.records: list[PerformanceRecord] = []
        self.dates: list[date] = []
        self.in_date_order = True
        self.earliest: date | None = None

    def append(self, entry: PerformanceRecord) -> None:
        entry_date = entry.as_of_date
        if self.dates and entry_date < self.dates[-1]:
            self.in_date_order = False
        if self.earliest is None or entry_date < self.earliest:
            self.earliest = entry_date
        self.records.append(entry)
        self.dates.append(entry_date)

    def window(self, lower_bound: date, upper_bound: date) -> list[PerformanceRecord]:
        """Records with `lower_bound <= as_of_date <= upper_bound`."""
        if self.in_date_order:
            start = bisect_left(self.dates, lower_bound)
            stop = bisect_right(self.dates, upper_bound)
            return self.records[start:stop]
        return [
            entry
            for entry, entry_date in zip(self.records, self.dates, strict=True)
            if lower_bound <= entry_date <= upper_bound
        ]


def _index_into(
    by_source: dict[str, _SourceHistory],
    by_source_and_type: dict[tuple[str, str], _SourceHistory],
    entry: PerformanceRecord,
) -> None:
    """Appends `entry` to its `source_id` history and to its
    `(source_id, source_type)` history in the given dicts, creating
    either history on first use."""
    history = by_source.get(entry.source_id)
    if history is None:
        history = by_source[entry.source_id] = _SourceHistory()
    history.append(entry)
    key = (entry.source_id, entry.source_type)
    typed_history = by_source_and_type.get(key)
    if typed_history is None:
        typed_history = by_source_and_type[key] = _SourceHistory()
    typed_history.append(entry)


class TrackRecord:
    """Append-only log of realized, attributable performance for every
    analyst and manager in the system.

    Storage is an in-memory list for now — a stub for a future persistent
    store (e.g. a database or file-backed log). Callers should not rely on
    this being in-memory; treat `record`/`history_for` as the durable
    interface regardless of what backs it later.

    `_entries` is the full log in insertion order. Beside it sits an index
    of the same records keyed by `source_id` and by `(source_id,
    source_type)`, so per-source lookups cost the size of that source's
    history rather than the whole log.

    The index follows the log through `record`, `restore` and any
    assignment to `_entries`, each of which updates or rebuilds it. Beyond
    those, the only change it detects is a change in the log's length:
    every lookup rebuilds the index when `len(_entries)` differs from the
    count indexed, which covers appending, extending, popping or clearing
    the list in place. An in-place edit that keeps the length (replacing
    or reordering items) is not detected, nor is a change to a recorded
    record's `source_id`, `source_type` or `as_of_date`; records must not
    be changed once recorded.
    """

    def __init__(self) -> None:
        self._log: list[PerformanceRecord] = []
        self._by_source: dict[str, _SourceHistory] = {}
        self._by_source_and_type: dict[tuple[str, str], _SourceHistory] = {}
        self._indexed_count = 0

    @property
    def _entries(self) -> list[PerformanceRecord]:
        """The full log in insertion order (the list itself, not a copy)."""
        return self._log

    @_entries.setter
    def _entries(self, entries: list[PerformanceRecord]) -> None:
        self._log = entries
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        """Indexes the whole log into new dicts and a new count, and
        assigns them to the instance only once complete, so the index
        attributes never hold a partly built index."""
        by_source: dict[str, _SourceHistory] = {}
        by_source_and_type: dict[tuple[str, str], _SourceHistory] = {}
        indexed_count = 0
        for entry in self._log:
            _index_into(by_source, by_source_and_type, entry)
            indexed_count += 1
        self._by_source = by_source
        self._by_source_and_type = by_source_and_type
        self._indexed_count = indexed_count

    def _sync_index(self) -> None:
        """Rebuilds the index when the log's length differs from what was
        indexed, i.e. when the list was changed in place rather than
        through `record` or `restore`."""
        if self._indexed_count != len(self._log):
            self._rebuild_index()

    def record(self, entry: PerformanceRecord) -> None:
        """Appends one PerformanceRecord to the log. Does not deduplicate
        or validate against existing entries for the same source_id/asset/
        as_of_date — that is the caller's responsibility."""
        self._sync_index()
        self._log.append(entry)
        _index_into(self._by_source, self._by_source_and_type, entry)
        self._indexed_count += 1

    @property
    def entries(self) -> list[PerformanceRecord]:
        """Chronological storage view used by durable backtest checkpoints.
        Copies the full list — O(n). For an append-only checkpoint that
        only needs *new* entries since some earlier count, use
        `entries_since` instead, which is O(k) in the number of new
        entries rather than O(n) in the whole history."""
        return list(self._log)

    def __len__(self) -> int:
        return len(self._log)

    def entries_since(self, count: int) -> list[PerformanceRecord]:
        """Every entry recorded after the first `count` — a plain list
        slice, O(k) in the number of new entries, not O(n) in the whole
        history. Pair with `len(track_record)` taken before a batch of
        `record()` calls to get exactly that batch back afterward."""
        return self._log[count:]

    def restore(self, entries: list[PerformanceRecord]) -> None:
        """Replace in-memory history from a validated checkpoint payload."""
        self._entries = list(entries)

    def history_for(
        self,
        source_id: str,
        as_of_date: date,
        lookback_days: int,
        source_type: Literal["analyst", "manager"] | None = None,
    ) -> list[PerformanceRecord]:
        """Returns every PerformanceRecord for `source_id` with
        `as_of_date - lookback_days <= entry.as_of_date <= as_of_date`,
        across all assets, in the order they were recorded, as a new list.
        `source_type` narrows the result to records of that one type (a
        source tracked as both an analyst and a manager keeps the two
        histories apart); None returns both. Costs the size of the
        source's own history, not of the whole log."""
        self._sync_index()
        if source_type is None:
            history = self._by_source.get(source_id)
        else:
            history = self._by_source_and_type.get((source_id, source_type))
        if history is None:
            return []
        return history.window(as_of_date - timedelta(days=lookback_days), as_of_date)

    def entries_for(self, source_id: str) -> list[PerformanceRecord]:
        """Every PerformanceRecord ever recorded for `source_id`, across all
        assets and dates, sorted chronologically (records on the same date
        keep their recording order). Unlike history_for, this is not
        windowed by an as_of_date/lookback — it's the full history, for
        callers building a complete performance curve (e.g. a GUI plotting
        a source's cumulative return over an entire backtest) rather than
        evaluating a trailing window as of one day."""
        self._sync_index()
        history = self._by_source.get(source_id)
        if history is None:
            return []
        if history.in_date_order:
            return list(history.records)
        return sorted(history.records, key=lambda entry: entry.as_of_date)

    def source_ids_as_of(self, as_of_date: date) -> set[str]:
        """Every source_id with at least one record dated on or before
        `as_of_date`, either source type. Costs the number of distinct
        sources, not the size of the log."""
        self._sync_index()
        return {
            source_id
            for source_id, history in self._by_source.items()
            if history.earliest is not None and history.earliest <= as_of_date
        }
