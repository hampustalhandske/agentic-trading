# test_trading_calendar.py

"""Tests for the NYSE session calendar: full-day closures, next_session and add_sessions."""

from datetime import date, timedelta

import pytest

from agentic_trading.data.trading_calendar import add_sessions, is_session, next_session


@pytest.mark.parametrize(
    ("day", "expected"),
    [
        (date(2024, 3, 27), True),
        (date(2023, 11, 25), False),
        (date(2023, 11, 26), False),
        (date(2023, 11, 23), False),
        (date(2024, 3, 29), False),
        (date(2022, 6, 20), False),
        (date(2023, 6, 19), False),
        (date(2021, 6, 18), True),
        (date(2012, 10, 29), False),
        (date(2012, 10, 30), False),
        (date(2012, 10, 31), True),
        (date(2018, 12, 5), False),
        (date(2025, 1, 9), False),
        (date(2004, 6, 11), False),
        (date(2007, 1, 2), False),
        (date(2001, 9, 10), True),
        (date(2001, 9, 11), False),
        (date(2001, 9, 14), False),
        (date(2001, 9, 17), True),
        (date(2021, 12, 31), True),
        (date(2023, 1, 2), False),
        (date(2024, 1, 1), False),
        (date(1997, 1, 20), True),
        (date(1998, 1, 19), False),
        (date(2024, 1, 15), False),
        (date(2024, 2, 19), False),
        (date(2024, 5, 27), False),
        (date(2024, 9, 2), False),
        (date(2020, 7, 3), False),
        (date(2021, 7, 5), False),
        (date(2023, 7, 4), False),
        (date(2024, 7, 5), True),
        (date(2021, 12, 24), False),
        (date(2022, 12, 26), False),
        (date(2024, 12, 25), False),
    ],
)
def test_is_session_follows_nyse_full_day_closures(day: date, expected: bool) -> None:
    """Weekends, rule-based holidays with observance shifts and special closures are not
    sessions."""
    assert is_session(day) is expected


def test_next_session_skips_closures_and_is_strictly_after() -> None:
    """next_session returns the first session strictly after the day, over holidays and weekends."""
    assert next_session(date(2023, 11, 22)) == date(2023, 11, 24)
    assert next_session(date(2012, 10, 26)) == date(2012, 10, 31)
    assert next_session(date(2024, 3, 28)) == date(2024, 4, 1)
    assert next_session(date(2024, 3, 27)) == date(2024, 3, 28)
    assert next_session(date(2023, 11, 25)) == date(2023, 11, 27)


def test_add_sessions_counts_sessions_strictly_after_the_day() -> None:
    """add_sessions returns the n-th session after the day, and n=1 equals next_session."""
    assert add_sessions(date(2023, 11, 15), 8) == date(2023, 11, 28)
    assert add_sessions(date(2014, 3, 14), 2) == date(2014, 3, 18)
    for day in (date(2024, 3, 28), date(2023, 11, 23), date(2012, 10, 28), date(2024, 7, 3)):
        assert add_sessions(day, 1) == next_session(day)


def test_session_invariants_over_a_range() -> None:
    """Across a multi-year range next_session lands on a later session with none skipped."""
    day = date(2019, 1, 1)
    while day < date(2025, 2, 1):
        following = next_session(day)
        assert following > day
        assert is_session(following)
        gap = (following - day).days
        assert all(not is_session(day + timedelta(days=k)) for k in range(1, gap))
        assert add_sessions(day, 3) == next_session(next_session(next_session(day)))
        day += timedelta(days=1)
