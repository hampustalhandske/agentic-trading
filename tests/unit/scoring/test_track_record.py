"""Unit tests for agentic_trading.scoring.track_record.

No mocking needed — TrackRecord operates purely on in-memory PerformanceRecord instances.
"""

from datetime import date, timedelta

from agentic_trading.agents.schemas import PerformanceRecord
from agentic_trading.scoring.track_record import TrackRecord


def _record(
    source_id: str,
    as_of_date: date,
    attributable_return: float = 0.0,
    asset: str = "AAPL",
) -> PerformanceRecord:
    return PerformanceRecord(
        source_id=source_id,
        source_type="analyst",
        asset=asset,
        as_of_date=as_of_date,
        attributable_return=attributable_return,
        raw_attributable_return=attributable_return,
    )


def test_record_appends_and_all_entries_are_retrievable() -> None:
    """Every recorded entry is retrievable via history_for once the window covers it."""
    track_record = TrackRecord()
    entries = [_record("analyst-a", date(2024, 1, day)) for day in range(1, 6)]
    for entry in entries:
        track_record.record(entry)

    result = track_record.history_for("analyst-a", date(2024, 1, 5), lookback_days=10)

    assert len(result) == len(entries)
    assert {r.as_of_date for r in result} == {e.as_of_date for e in entries}


def test_history_for_includes_entry_exactly_on_as_of_date() -> None:
    """An entry dated exactly as_of_date is inside the window's inclusive right edge."""
    track_record = TrackRecord()
    track_record.record(_record("analyst-a", date(2024, 1, 10)))

    result = track_record.history_for("analyst-a", date(2024, 1, 10), lookback_days=5)

    assert len(result) == 1
    assert result[0].as_of_date == date(2024, 1, 10)


def test_history_for_excludes_entry_after_as_of_date() -> None:
    """An entry dated after as_of_date is excluded — the no-look-ahead guarantee."""
    track_record = TrackRecord()
    track_record.record(_record("analyst-a", date(2024, 1, 11)))

    result = track_record.history_for("analyst-a", date(2024, 1, 10), lookback_days=30)

    assert result == []


def test_history_for_includes_entry_exactly_lookback_days_before() -> None:
    """The left edge is inclusive: an entry exactly lookback_days before as_of_date is returned."""
    track_record = TrackRecord()
    as_of_date = date(2024, 1, 10)
    boundary_date = as_of_date - timedelta(days=4)
    track_record.record(_record("analyst-a", boundary_date))

    # An as_of_date entry must be included even with lookback_days=0, which only holds if the
    # left boundary (as_of_date - lookback_days) is itself part of the window, not excluded.
    result = track_record.history_for("analyst-a", as_of_date, lookback_days=4)

    assert len(result) == 1
    assert result[0].as_of_date == boundary_date


def test_history_for_excludes_entry_older_than_lookback() -> None:
    """An entry dated more than lookback_days before as_of_date falls outside the window."""
    track_record = TrackRecord()
    as_of_date = date(2024, 1, 10)
    track_record.record(_record("analyst-a", as_of_date - timedelta(days=5)))

    result = track_record.history_for("analyst-a", as_of_date, lookback_days=4)

    assert result == []


def test_history_for_excludes_entries_for_a_different_source_id() -> None:
    """history_for scopes strictly to the requested source_id, ignoring other sources' entries."""
    track_record = TrackRecord()
    as_of_date = date(2024, 1, 10)
    track_record.record(_record("analyst-a", as_of_date))
    track_record.record(_record("analyst-b", as_of_date))

    result = track_record.history_for("analyst-a", as_of_date, lookback_days=5)

    assert len(result) == 1
    assert result[0].source_id == "analyst-a"


def test_history_for_unrecorded_source_id_returns_empty_list() -> None:
    """A source_id with no recorded entries yields an empty list rather than raising."""
    track_record = TrackRecord()
    track_record.record(_record("analyst-a", date(2024, 1, 10)))

    result = track_record.history_for("analyst-never-seen", date(2024, 1, 10), lookback_days=30)

    assert result == []


def test_history_for_spans_multiple_assets_for_the_same_source() -> None:
    """history_for aggregates across all assets for the requested source_id, per its docstring."""
    track_record = TrackRecord()
    as_of_date = date(2024, 1, 10)
    track_record.record(_record("analyst-a", as_of_date, asset="AAPL"))
    track_record.record(_record("analyst-a", as_of_date, asset="MSFT"))

    result = track_record.history_for("analyst-a", as_of_date, lookback_days=5)

    assert {r.asset for r in result} == {"AAPL", "MSFT"}


def test_entries_for_returns_full_unwindowed_history_sorted_chronologically() -> None:
    """entries_for is not lookback-limited — it's every entry ever recorded for a source."""
    track_record = TrackRecord()
    later, earlier = date(2024, 2, 1), date(2024, 1, 1)
    track_record.record(_record("analyst-a", later))
    track_record.record(_record("analyst-a", earlier))

    result = track_record.entries_for("analyst-a")

    assert [entry.as_of_date for entry in result] == [earlier, later]


def test_entries_for_excludes_other_sources_and_handles_unrecorded_source() -> None:
    track_record = TrackRecord()
    track_record.record(_record("analyst-a", date(2024, 1, 1)))

    assert [e.source_id for e in track_record.entries_for("analyst-a")] == ["analyst-a"]
    assert track_record.entries_for("analyst-never-seen") == []


def test_history_for_filters_by_source_type_when_given() -> None:
    track_record = TrackRecord()
    as_of_date = date(2024, 1, 10)
    track_record.record(_record("source-a", as_of_date))
    track_record.record(
        PerformanceRecord(
            source_id="source-a",
            source_type="manager",
            asset="AAPL",
            as_of_date=as_of_date,
            attributable_return=0.0,
            raw_attributable_return=0.0,
        )
    )

    assert len(track_record.history_for("source-a", as_of_date, 5)) == 2
    analysts = track_record.history_for("source-a", as_of_date, 5, source_type="analyst")
    managers = track_record.history_for("source-a", as_of_date, 5, source_type="manager")
    assert [entry.source_type for entry in analysts] == ["analyst"]
    assert [entry.source_type for entry in managers] == ["manager"]
