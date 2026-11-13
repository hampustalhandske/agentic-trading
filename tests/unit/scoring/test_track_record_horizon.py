# test_track_record_horizon.py

"""AT-140: a `TrackRecord` built with `horizon_days` discards the records dated before each
`drop_before` cutoff and remembers the cutoff. A trailing window inside what it kept returns
exactly what an unpruned record returns; a window reaching before the cutoff, and the
whole-history readers once a record was dropped, raise `HistoryHorizonError`; `len` and
`entries_since` keep counting every record ever written."""

import gc
import types
from collections.abc import Iterator
from datetime import date, timedelta

import pytest

from agentic_trading.agents.schemas import PerformanceRecord
from agentic_trading.scoring.track_record import HistoryHorizonError, TrackRecord

_HORIZON_DAYS = 7
_START = date(2024, 1, 1)
# Weekdays only, so many cutoffs fall on a weekend without records.
_DAYS = [
    day for day in (_START + timedelta(days=offset) for offset in range(63)) if day.weekday() < 5
]
# One source tracked both as an analyst and as a manager.
_SOURCES = (
    ("analyst-a", "analyst"),
    ("analyst-b", "analyst"),
    ("pod-m", "manager"),
    ("analyst-a", "manager"),
)
# Every (source_id, source_type) lookup checked, untyped and unknown sources included.
_LOOKUPS = (*_SOURCES, ("analyst-a", None), ("pod-m", None), ("never-seen", None))
# Containers that lead out of the object graph under test into shared interpreter state.
_OPAQUE = (
    types.ModuleType,
    type,
    types.FunctionType,
    types.BuiltinFunctionType,
    types.CodeType,
    types.FrameType,
)


def _day_records(day: date) -> list[PerformanceRecord]:
    """One record per source and asset on `day`, with returns that vary by day."""
    offset = (day - _START).days
    return [
        PerformanceRecord(
            source_id=source_id,
            source_type=source_type,
            asset=asset,
            as_of_date=day,
            attributable_return=0.001 * ((offset + index) % 5 - 2),
            raw_attributable_return=0.002 * ((offset + index) % 3 - 1),
            signal_score=0.1 * ((offset * 7 + index) % 9 - 4),
            realized_return=0.003 * ((offset * 3 + index) % 7 - 3),
        )
        for index, (source_id, source_type) in enumerate(_SOURCES)
        for asset in ("AAPL", "MSFT")
    ]


def _days_recorded() -> Iterator[tuple[date, TrackRecord, TrackRecord]]:
    """Each day, a horizon record pruned to `day - horizon` and an unpruned one, same records."""
    pruned, plain = TrackRecord(horizon_days=_HORIZON_DAYS), TrackRecord()
    for day in _DAYS:
        for record in _day_records(day):
            pruned.record(record)
            plain.record(record)
        pruned.drop_before(day - timedelta(days=_HORIZON_DAYS))
        yield day, pruned, plain


def _reachable(root: object, kind: type) -> int:
    """The number of distinct `kind` instances reachable from `root` outside shared state."""
    seen: set[int] = set()
    stack = [root]
    count = 0
    while stack:
        item = stack.pop()
        if id(item) in seen or isinstance(item, _OPAQUE):
            continue
        seen.add(id(item))
        count += isinstance(item, kind)
        stack.extend(gc.get_referents(item))
    return count


def test_history_for_inside_the_horizon() -> None:
    """A window starting on or after the cutoff returns the unpruned list, in the same order."""
    on_cutoff = 0
    for day, pruned, plain in _days_recorded():
        for source_id, source_type in _LOOKUPS:
            for as_of_date, lookback in (
                (day, 0),
                (day, 3),
                (day, _HORIZON_DAYS),
                (day - timedelta(days=2), _HORIZON_DAYS - 2),
            ):
                expected = plain.history_for(source_id, as_of_date, lookback, source_type)
                assert pruned.history_for(source_id, as_of_date, lookback, source_type) == expected
                start = as_of_date - timedelta(days=lookback)
                on_cutoff += any(entry.as_of_date == start for entry in expected)

    assert on_cutoff


def test_reads_before_the_cutoff() -> None:
    """Windows before the cutoff raise; full-history reads raise once a record has been dropped."""
    fresh = TrackRecord(horizon_days=_HORIZON_DAYS)
    first = _day_records(_DAYS[0])
    for record in first:
        fresh.record(record)
    pod = [record for record in first if record.source_id == "pod-m"]
    assert fresh.history_for("pod-m", _DAYS[0], 365) == pod

    fresh.drop_before(_DAYS[0])

    assert fresh.entries == first
    assert fresh.entries_for("pod-m") == pod
    assert fresh.history_for("pod-m", _DAYS[0], 0) == pod
    with pytest.raises(HistoryHorizonError):
        fresh.history_for("pod-m", _DAYS[0], 1)

    *_, (last, pruned, _plain) = _days_recorded()
    for source_id, source_type in _LOOKUPS:
        with pytest.raises(HistoryHorizonError):
            pruned.history_for(source_id, last, _HORIZON_DAYS + 1, source_type)
        with pytest.raises(HistoryHorizonError):
            pruned.history_for(source_id, last - timedelta(days=1), _HORIZON_DAYS, source_type)
        with pytest.raises(HistoryHorizonError):
            pruned.entries_for(source_id)
    with pytest.raises(HistoryHorizonError):
        _ = pruned.entries


def test_counts_cover_every_record_written() -> None:
    """`len` and `entries_since` count every record ever written, before and after each cutoff."""
    pruned, plain = TrackRecord(horizon_days=_HORIZON_DAYS), TrackRecord()
    for day in _DAYS:
        count_before = len(pruned)
        assert count_before == len(plain)
        for record in _day_records(day):
            pruned.record(record)
            plain.record(record)
        batch = plain.entries_since(count_before)

        assert pruned.entries_since(count_before) == batch
        pruned.drop_before(day - timedelta(days=_HORIZON_DAYS))
        assert pruned.entries_since(count_before) == batch
        assert len(pruned) == len(plain)
        assert pruned.entries_since(len(pruned)) == []


def test_retained_records_are_bounded() -> None:
    """A pruned record never holds more records than are dated on or after its cutoff."""
    for day, pruned, plain in _days_recorded():
        cutoff = day - timedelta(days=_HORIZON_DAYS)
        inside = sum(entry.as_of_date >= cutoff for entry in plain.entries)

        assert _reachable(pruned, PerformanceRecord) <= inside

    assert inside * 4 < len(plain)
