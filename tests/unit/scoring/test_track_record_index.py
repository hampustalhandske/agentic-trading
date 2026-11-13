"""Parity and scaling tests for TrackRecord's per-source index.

Every lookup is compared against a linear scan over the full insertion-order
log, written here independently of the implementation, on seeded random
histories: several sources and assets, both source types, one source
tracked as both, same-day records, out-of-order appends, and a `restore`
midway followed by further records.
"""

import random
import time
from datetime import date, timedelta
from typing import Literal

import pytest

from agentic_trading.agents.schemas import PerformanceRecord
from agentic_trading.scoring.track_record import TrackRecord

SourceType = Literal["analyst", "manager"]

START = date(2015, 1, 1)
ASSETS = ["AAPL", "MSFT", "SPY", "TLT"]
# (source_id, source_types it is recorded under); "dual" is tracked as both.
SOURCES: list[tuple[str, tuple[SourceType, ...]]] = [
    ("analyst-a", ("analyst",)),
    ("analyst-b", ("analyst",)),
    ("analyst-c", ("analyst",)),
    ("manager-x", ("manager",)),
    ("manager-y", ("manager",)),
    ("dual", ("analyst", "manager")),
]
QUERY_SOURCE_IDS = [source_id for source_id, _ in SOURCES] + ["never-recorded"]
QUERY_SOURCE_TYPES: list[SourceType | None] = [None, "analyst", "manager"]


def _reference_history(
    entries: list[PerformanceRecord],
    source_id: str,
    as_of_date: date,
    lookback_days: int,
    source_type: SourceType | None,
) -> list[PerformanceRecord]:
    lower_bound = as_of_date - timedelta(days=lookback_days)
    return [
        entry
        for entry in entries
        if entry.source_id == source_id
        and lower_bound <= entry.as_of_date <= as_of_date
        and (source_type is None or entry.source_type == source_type)
    ]


def _reference_entries_for(
    entries: list[PerformanceRecord], source_id: str
) -> list[PerformanceRecord]:
    return sorted(
        (entry for entry in entries if entry.source_id == source_id),
        key=lambda entry: entry.as_of_date,
    )


def _reference_source_ids(entries: list[PerformanceRecord], as_of_date: date) -> set[str]:
    return {entry.source_id for entry in entries if entry.as_of_date <= as_of_date}


def _make_record(
    rng: random.Random, source_id: str, source_type: SourceType, as_of_date: date
) -> PerformanceRecord:
    value = rng.uniform(-0.02, 0.02)
    return PerformanceRecord(
        source_id=source_id,
        source_type=source_type,
        asset=rng.choice(ASSETS),
        as_of_date=as_of_date,
        attributable_return=value,
        raw_attributable_return=value,
    )


def _random_batch(
    rng: random.Random, first_day: int, days: int, out_of_order_rate: float
) -> list[PerformanceRecord]:
    """Records over `days` consecutive days starting `first_day` days after
    START: several per source per day (same-day records), and with
    probability `out_of_order_rate` a record dated before the current day."""
    batch: list[PerformanceRecord] = []
    for day in range(first_day, first_day + days):
        current = START + timedelta(days=day)
        for _ in range(rng.randint(3, 12)):
            source_id, source_types = rng.choice(SOURCES)
            source_type = rng.choice(source_types)
            as_of_date = current
            if rng.random() < out_of_order_rate:
                as_of_date = current - timedelta(days=rng.randint(1, 40))
            batch.append(_make_record(rng, source_id, source_type, as_of_date))
    return batch


def _assert_parity(track_record: TrackRecord, rng: random.Random, queries: int) -> None:
    entries = track_record.entries
    assert len(track_record) == len(entries)
    last_day = max((entry.as_of_date for entry in entries), default=START)
    span = (last_day - START).days + 60
    for _ in range(queries):
        source_id = rng.choice(QUERY_SOURCE_IDS)
        source_type = rng.choice(QUERY_SOURCE_TYPES)
        as_of_date = START + timedelta(days=rng.randint(-50, span))
        lookback_days = rng.choice([0, 1, 2, 7, 30, 90, 365, 10_000, rng.randint(0, 200)])

        expected = _reference_history(entries, source_id, as_of_date, lookback_days, source_type)
        actual = track_record.history_for(source_id, as_of_date, lookback_days, source_type)
        assert len(actual) == len(expected)
        assert all(a is e for a, e in zip(actual, expected, strict=True))

        assert track_record.source_ids_as_of(as_of_date) == _reference_source_ids(
            entries, as_of_date
        )

    for source_id in QUERY_SOURCE_IDS:
        expected = _reference_entries_for(entries, source_id)
        actual = track_record.entries_for(source_id)
        assert len(actual) == len(expected)
        assert all(a is e for a, e in zip(actual, expected, strict=True))


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("out_of_order_rate", [0.0, 0.05, 0.5])
def test_lookups_match_linear_scan_across_a_restore(seed: int, out_of_order_rate: float) -> None:
    rng = random.Random(seed)
    original = TrackRecord()
    for entry in _random_batch(rng, first_day=0, days=60, out_of_order_rate=out_of_order_rate):
        original.record(entry)
    _assert_parity(original, rng, queries=300)

    # Restore into a fresh record and into one holding unrelated history,
    # then keep recording after the restore on both.
    fresh = TrackRecord()
    fresh.restore(original.entries)
    stale = TrackRecord()
    for entry in _random_batch(rng, first_day=200, days=10, out_of_order_rate=0.3):
        stale.record(entry)
    stale.restore(original.entries)

    later = _random_batch(rng, first_day=55, days=40, out_of_order_rate=out_of_order_rate)
    for track_record in (fresh, stale):
        _assert_parity(track_record, rng, queries=150)
        for entry in later:
            track_record.record(entry)
        _assert_parity(track_record, rng, queries=300)
    assert all(a is b for a, b in zip(fresh.entries, stale.entries, strict=True))


def test_restore_of_out_of_order_history_then_in_order_appends() -> None:
    rng = random.Random(1234)
    shuffled = _random_batch(rng, first_day=0, days=30, out_of_order_rate=0.0)
    rng.shuffle(shuffled)
    track_record = TrackRecord()
    track_record.restore(shuffled)
    for entry in _random_batch(rng, first_day=30, days=30, out_of_order_rate=0.0):
        track_record.record(entry)
    _assert_parity(track_record, rng, queries=400)


def test_assigning_entries_directly_rebuilds_the_index() -> None:
    rng = random.Random(7)
    track_record = TrackRecord()
    for entry in _random_batch(rng, first_day=0, days=20, out_of_order_rate=0.1):
        track_record.record(entry)
    replacement = _random_batch(rng, first_day=100, days=20, out_of_order_rate=0.1)
    track_record._entries = replacement
    assert track_record.entries == replacement
    _assert_parity(track_record, rng, queries=200)


def test_appending_to_the_entries_list_directly_is_still_seen() -> None:
    rng = random.Random(11)
    track_record = TrackRecord()
    for entry in _random_batch(rng, first_day=0, days=20, out_of_order_rate=0.1):
        track_record.record(entry)
    _assert_parity(track_record, rng, queries=50)
    for entry in _random_batch(rng, first_day=20, days=5, out_of_order_rate=0.1):
        track_record._entries.append(entry)
    _assert_parity(track_record, rng, queries=200)


def test_returned_lists_are_fresh_copies() -> None:
    rng = random.Random(3)
    track_record = TrackRecord()
    for entry in _random_batch(rng, first_day=0, days=10, out_of_order_rate=0.0):
        track_record.record(entry)
    as_of_date = START + timedelta(days=9)

    window = track_record.history_for("analyst-a", as_of_date, 30)
    assert window
    window.clear()
    assert track_record.history_for("analyst-a", as_of_date, 30)

    full = track_record.entries_for("analyst-a")
    full.clear()
    assert track_record.entries_for("analyst-a")

    known = track_record.source_ids_as_of(as_of_date)
    known.clear()
    assert track_record.source_ids_as_of(as_of_date)


def test_source_ids_as_of_on_an_empty_record() -> None:
    assert TrackRecord().source_ids_as_of(date(2024, 1, 1)) == set()


def test_history_for_cost_does_not_grow_with_other_sources_history() -> None:
    """200k records over 50 sources: 1000 windowed lookups, timed after the
    record is built, finish within 0.2 s. A scan of the full log per call
    takes several milliseconds each, seconds in total."""
    rng = random.Random(42)
    source_ids = [f"source-{index:02d}" for index in range(50)]
    days = 1000
    per_day = 200_000 // days
    track_record = TrackRecord()
    for day in range(days):
        as_of_date = START + timedelta(days=day)
        for slot in range(per_day):
            track_record.record(
                PerformanceRecord.model_construct(
                    source_id=source_ids[slot % len(source_ids)],
                    source_type="analyst" if slot % 2 == 0 else "manager",
                    asset=ASSETS[slot % len(ASSETS)],
                    as_of_date=as_of_date,
                    attributable_return=0.0,
                    raw_attributable_return=0.0,
                    signal_score=0.0,
                    realized_return=0.0,
                )
            )
    assert len(track_record) == 200_000

    queries = [
        (
            rng.choice(source_ids),
            START + timedelta(days=rng.randint(0, days - 1)),
            rng.choice([21, 63, 252]),
            rng.choice(QUERY_SOURCE_TYPES),
        )
        for _ in range(1000)
    ]
    started = time.perf_counter()
    for source_id, as_of_date, lookback_days, source_type in queries:
        track_record.history_for(source_id, as_of_date, lookback_days, source_type)
    elapsed = time.perf_counter() - started

    assert elapsed < 0.2
