# track_record.py

from bisect import bisect_left, bisect_right
from collections.abc import Mapping
from datetime import date, timedelta
from typing import Literal

from agentic_trading.agents.schemas import PerformanceRecord


class HistoryHorizonError(LookupError):
    """A read reached history that a horizon-bounded holder has dropped:
    a `TrackRecord` window starting before its cutoff, a whole-history
    read of a `TrackRecord` that dropped records, or a backtest engine's
    decisions of a day it no longer holds. Raised instead of returning
    the part still held."""


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

    def drop_before(self, cutoff: date) -> None:
        """Removes the records dated before `cutoff`, keeping the rest in
        insertion order, and sets `earliest` to the earliest kept date."""
        if self.in_date_order:
            stop = bisect_left(self.dates, cutoff)
            if stop:
                del self.records[:stop]
                del self.dates[:stop]
            self.earliest = self.dates[0] if self.dates else None
            return
        kept = [
            (entry, entry_date)
            for entry, entry_date in zip(self.records, self.dates, strict=True)
            if entry_date >= cutoff
        ]
        self.records = [entry for entry, _ in kept]
        self.dates = [entry_date for _, entry_date in kept]
        self.earliest = min(self.dates) if self.dates else None

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


def _drop_from[Key: (str, tuple[str, str])](
    histories: dict[Key, _SourceHistory], cutoff: date
) -> None:
    """Drops the records dated before `cutoff` from every history in
    `histories`, removing the histories left empty."""
    for key in list(histories):
        history = histories[key]
        history.drop_before(cutoff)
        if not history.records:
            del histories[key]


class TrackRecord:
    """Append-only log of realized, attributable performance for every
    analyst and manager in the system.

    Storage is an in-memory list for now — a stub for a future persistent
    store (e.g. a database or file-backed log). Callers should not rely on
    this being in-memory; treat `record`/`history_for` as the durable
    interface regardless of what backs it later.

    `_entries` is the log in insertion order. Beside it sits an index
    of the same records keyed by `source_id` and by `(source_id,
    source_type)`, so per-source lookups cost the size of that source's
    history rather than the whole log.

    The index follows the log through `record`, `restore`, `drop_before`
    and any assignment to `_entries`, each of which updates or rebuilds
    it. Beyond those, the only change it detects is a change in the log's
    length: every lookup rebuilds the index when `len(_entries)` differs
    from the count indexed, which covers appending, extending, popping or
    clearing the list in place. An in-place edit that keeps the length
    (replacing or reordering items) is not detected, nor is a change to a
    recorded record's `source_id`, `source_type` or `as_of_date`; records
    must not be changed once recorded.

    A record built with `horizon_days` may be bounded in memory: the
    holder calls `drop_before(cutoff)` (the backtest engine does so each
    session with the day minus `horizon_days`), which discards every
    record dated before `cutoff` and remembers the latest cutoff. From then
    on a `history_for` window starting before that cutoff, and `entries`,
    `entries_for` and `_entries` once a record was dropped, raise
    `HistoryHorizonError`; a window starting on or after it returns
    exactly what the unpruned record would. `len` and `entries_since`
    keep counting every record ever written, and `source_ids_as_of`
    keeps knowing the sources of dropped records. Without `horizon_days`
    the record keeps everything and `drop_before` refuses.
    """

    def __init__(self, horizon_days: int | None = None) -> None:
        if horizon_days is not None and horizon_days < 1:
            raise ValueError(f"horizon_days must be at least 1, got {horizon_days}")
        self.horizon_days = horizon_days
        self._log: list[PerformanceRecord] = []
        self._by_source: dict[str, _SourceHistory] = {}
        self._by_source_and_type: dict[tuple[str, str], _SourceHistory] = {}
        self._indexed_count = 0
        # Whether every record was appended with an `as_of_date` no earlier
        # than the one before it, so the records before a cutoff are a
        # prefix of the log.
        self._log_in_date_order = True
        self._reset_horizon()

    def _reset_horizon(self) -> None:
        """Forgets every cutoff and dropped record: the log becomes the
        whole history."""
        # The latest `drop_before` cutoff; None until the first.
        self._cutoff: date | None = None
        # Records dropped so far; `len` adds them to the log's length.
        self._dropped_count = 0
        # The smallest `count` `entries_since` can answer in full: every
        # record written at or after that position is still in the log.
        self._complete_from = 0
        # Each source's earliest date among the dropped records.
        self._dropped_first_dates: dict[str, date] = {}

    @property
    def _entries(self) -> list[PerformanceRecord]:
        """The full log in insertion order (the list itself, not a copy).
        Raises `HistoryHorizonError` once a record was dropped."""
        self._require_whole_history("_entries")
        return self._log

    @_entries.setter
    def _entries(self, entries: list[PerformanceRecord]) -> None:
        self._log = entries
        self._reset_horizon()
        self._rebuild_index()

    def _require_whole_history(self, reader: str) -> None:
        """Raises `HistoryHorizonError` naming `reader` when a record was
        dropped, so the log no longer holds the whole history."""
        if self._dropped_count:
            raise HistoryHorizonError(
                f"{reader} needs the whole track record, but {self._dropped_count} record(s) "
                f"dated before {self._cutoff} were dropped"
            )

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
        self._log_in_date_order = all(
            earlier.as_of_date <= later.as_of_date
            for earlier, later in zip(self._log, self._log[1:], strict=False)
        )

    def _sync_index(self) -> None:
        """Rebuilds the index when the log's length differs from what was
        indexed, i.e. when the list was changed in place rather than
        through `record` or `restore`."""
        if self._indexed_count != len(self._log):
            self._rebuild_index()

    def record(self, entry: PerformanceRecord) -> None:
        """Appends one PerformanceRecord to the log. Does not deduplicate
        or validate against existing entries for the same source_id/asset/
        as_of_date — that is the caller's responsibility. Raises
        `ValueError` for a record dated before the latest `drop_before`
        cutoff, which no read could return."""
        if self._cutoff is not None and entry.as_of_date < self._cutoff:
            raise ValueError(
                f"a record dated {entry.as_of_date} is before the track record's "
                f"cutoff {self._cutoff}"
            )
        self._sync_index()
        if self._log and entry.as_of_date < self._log[-1].as_of_date:
            self._log_in_date_order = False
        self._log.append(entry)
        _index_into(self._by_source, self._by_source_and_type, entry)
        self._indexed_count += 1

    def drop_before(self, cutoff: date) -> None:
        """Discards every record dated before `cutoff` and remembers the
        cutoff (the later of it and any earlier one), after which reads
        reaching before it raise `HistoryHorizonError` (see the class
        docstring). Costs the number of records dropped when the log is in
        date order, and the size of the log otherwise. Raises `ValueError`
        on a record built without `horizon_days`."""
        if self.horizon_days is None:
            raise ValueError("a TrackRecord built without horizon_days keeps every record")
        self._sync_index()
        if self._cutoff is not None and cutoff <= self._cutoff:
            return
        self._cutoff = cutoff
        log = self._log
        last_dropped = -1
        if self._log_in_date_order:
            while last_dropped + 1 < len(log) and log[last_dropped + 1].as_of_date < cutoff:
                last_dropped += 1
            kept = log[last_dropped + 1 :]
        else:
            kept = []
            for position, entry in enumerate(log):
                if entry.as_of_date < cutoff:
                    last_dropped = position
                else:
                    kept.append(entry)
        if last_dropped < 0:
            return
        # Every record after the last dropped one is kept, so `entries_since`
        # answers in full from the position just past it.
        recorded = len(self)
        self._complete_from = max(self._complete_from, recorded - (len(log) - 1 - last_dropped))
        for source_id, history in self._by_source.items():
            if history.earliest is not None and history.earliest < cutoff:
                first = self._dropped_first_dates.get(source_id)
                if first is None or history.earliest < first:
                    self._dropped_first_dates[source_id] = history.earliest
        _drop_from(self._by_source, cutoff)
        _drop_from(self._by_source_and_type, cutoff)
        self._dropped_count += len(log) - len(kept)
        self._log = kept
        self._indexed_count = len(kept)

    @property
    def entries(self) -> list[PerformanceRecord]:
        """Chronological storage view used by durable backtest checkpoints.
        Copies the full list — O(n). For an append-only checkpoint that
        only needs *new* entries since some earlier count, use
        `entries_since` instead, which is O(k) in the number of new
        entries rather than O(n) in the whole history. Raises
        `HistoryHorizonError` once a record was dropped."""
        self._require_whole_history("entries")
        return list(self._log)

    def __len__(self) -> int:
        """The number of records ever written, dropped ones included."""
        return self._dropped_count + len(self._log)

    def entries_since(self, count: int) -> list[PerformanceRecord]:
        """Every entry recorded after the first `count`, O(k) in the number
        of new entries, not O(n) in the whole history. Pair with
        `len(track_record)` taken before a batch of `record()` calls to get
        exactly that batch back afterward. Raises `HistoryHorizonError`
        when a record written after the first `count` was dropped."""
        if count < self._complete_from:
            raise HistoryHorizonError(
                f"records written after the first {count} were dropped before "
                f"{self._cutoff}; the track record holds every record from position "
                f"{self._complete_from}"
            )
        newer = len(self) - count
        if newer <= 0:
            return []
        return self._log[len(self._log) - newer :]

    def restore(
        self,
        entries: list[PerformanceRecord],
        *,
        cutoff: date | None = None,
        recorded_count: int | None = None,
        dropped_first_dates: Mapping[str, date] | None = None,
    ) -> None:
        """Replace in-memory history from a validated checkpoint payload.

        Without `cutoff`, `entries` is the whole history. With `cutoff`,
        on a record built with `horizon_days`, the record stands as if
        `drop_before(cutoff)` had been called on the whole history:
        `entries` are the last records written, in insertion order and
        each dated on or after `cutoff`, out of `recorded_count` ever
        written (`len(entries)` when omitted); the earlier ones count as
        dropped, and `dropped_first_dates` gives each source's earliest
        date among them. Raises `ValueError` when `cutoff` is given to a
        record without `horizon_days`, when `recorded_count` is below
        `len(entries)` or is given without `cutoff`, and when an entry is
        dated before `cutoff`."""
        if cutoff is None:
            if recorded_count is not None or dropped_first_dates is not None:
                raise ValueError("recorded_count and dropped_first_dates need a cutoff")
            self._entries = list(entries)
            return
        if self.horizon_days is None:
            raise ValueError("a TrackRecord built without horizon_days keeps every record")
        recorded = len(entries) if recorded_count is None else recorded_count
        if recorded < len(entries):
            raise ValueError(
                f"recorded_count {recorded} is below the {len(entries)} entries restored"
            )
        early = [entry for entry in entries if entry.as_of_date < cutoff]
        if early:
            raise ValueError(
                f"{len(early)} restored record(s) are dated before the cutoff {cutoff}"
            )
        self._entries = list(entries)
        self._cutoff = cutoff
        self._dropped_count = recorded - len(entries)
        self._complete_from = self._dropped_count
        self._dropped_first_dates = dict(dropped_first_dates or {})

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
        source's own history, not of the whole log. Raises
        `HistoryHorizonError` when the window starts before the latest
        `drop_before` cutoff."""
        lower_bound = as_of_date - timedelta(days=lookback_days)
        if self._cutoff is not None and lower_bound < self._cutoff:
            raise HistoryHorizonError(
                f"a {lookback_days}-day window ending {as_of_date} starts on {lower_bound}, "
                f"before the track record's cutoff {self._cutoff}"
            )
        self._sync_index()
        if source_type is None:
            history = self._by_source.get(source_id)
        else:
            history = self._by_source_and_type.get((source_id, source_type))
        if history is None:
            return []
        return history.window(lower_bound, as_of_date)

    def entries_for(self, source_id: str) -> list[PerformanceRecord]:
        """Every PerformanceRecord ever recorded for `source_id`, across all
        assets and dates, sorted chronologically (records on the same date
        keep their recording order). Unlike history_for, this is not
        windowed by an as_of_date/lookback — it's the full history, for
        callers building a complete performance curve (e.g. a GUI plotting
        a source's cumulative return over an entire backtest) rather than
        evaluating a trailing window as of one day. Raises
        `HistoryHorizonError` once a record was dropped."""
        self._require_whole_history("entries_for")
        self._sync_index()
        history = self._by_source.get(source_id)
        if history is None:
            return []
        if history.in_date_order:
            return list(history.records)
        return sorted(history.records, key=lambda entry: entry.as_of_date)

    def source_ids_as_of(self, as_of_date: date) -> set[str]:
        """Every source_id with at least one record dated on or before
        `as_of_date`, either source type, dropped records included. Costs
        the number of distinct sources, not the size of the log."""
        self._sync_index()
        source_ids = {
            source_id
            for source_id, history in self._by_source.items()
            if history.earliest is not None and history.earliest <= as_of_date
        }
        source_ids.update(
            source_id
            for source_id, first in self._dropped_first_dates.items()
            if first <= as_of_date
        )
        return source_ids
