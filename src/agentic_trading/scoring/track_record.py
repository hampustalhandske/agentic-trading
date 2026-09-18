# track_record.py

from datetime import date, timedelta
from typing import Literal

from agentic_trading.agents.schemas import PerformanceRecord


class TrackRecord:
    """Append-only log of realized, attributable performance for every
    analyst and manager in the system.

    Storage is an in-memory list for now — a stub for a future persistent
    store (e.g. a database or file-backed log). Callers should not rely on
    this being in-memory; treat `record`/`history_for` as the durable
    interface regardless of what backs it later.
    """

    def __init__(self) -> None:
        self._entries: list[PerformanceRecord] = []

    def record(self, entry: PerformanceRecord) -> None:
        """Appends one PerformanceRecord to the log. Does not deduplicate
        or validate against existing entries for the same source_id/asset/
        as_of_date — that is the caller's responsibility."""
        self._entries.append(entry)

    @property
    def entries(self) -> list[PerformanceRecord]:
        """Chronological storage view used by durable backtest checkpoints.
        Copies the full list — O(n). For an append-only checkpoint that
        only needs *new* entries since some earlier count, use
        `entries_since` instead, which is O(k) in the number of new
        entries rather than O(n) in the whole history."""
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def entries_since(self, count: int) -> list[PerformanceRecord]:
        """Every entry recorded after the first `count` — a plain list
        slice, O(k) in the number of new entries, not O(n) in the whole
        history. Pair with `len(track_record)` taken before a batch of
        `record()` calls to get exactly that batch back afterward."""
        return self._entries[count:]

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
        as_of_date in (as_of_date - lookback_days, as_of_date], across all
        assets. `source_type` narrows the result to records of that one
        type (a source tracked as both an analyst and a manager keeps
        the two histories apart); None returns both."""
        lower_bound = as_of_date - timedelta(days=lookback_days)

        return [
            entry
            for entry in self._entries
            if entry.source_id == source_id
            and lower_bound <= entry.as_of_date <= as_of_date
            and (source_type is None or entry.source_type == source_type)
        ]

    def entries_for(self, source_id: str) -> list[PerformanceRecord]:
        """Every PerformanceRecord ever recorded for `source_id`, across all
        assets and dates, sorted chronologically. Unlike history_for, this
        is not windowed by an as_of_date/lookback — it's the full history,
        for callers building a complete performance curve (e.g. a GUI
        plotting a source's cumulative return over an entire backtest)
        rather than evaluating a trailing window as of one day."""
        return sorted(
            (entry for entry in self._entries if entry.source_id == source_id),
            key=lambda entry: entry.as_of_date,
        )
