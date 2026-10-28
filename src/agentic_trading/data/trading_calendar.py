# trading_calendar.py

"""NYSE trading sessions as `datetime.date`s: `is_session`, `next_session`
and `add_sessions`.

Sessions come from exchange-calendars' `XNYS` calendar: weekdays that are
neither NYSE holidays nor unscheduled full-day closures (state funerals,
September 11, Hurricane Sandy). Early closes (half days) are sessions.

Sessions are precomputed once for `CALENDAR_START`..`CALENDAR_END`; every
function raises `ValueError` for a day, or a result, outside that range."""

from datetime import date
from functools import cache

import exchange_calendars as xcals
import pandas as pd

# Covers every source the event loaders read (FINRA, SEC EDGAR, earnings
# calendars) with room on both sides; a wider range only costs a few
# thousand more entries in the cached index.
CALENDAR_START = date(1990, 1, 1)
CALENDAR_END = date(2035, 12, 31)


@cache
def _sessions() -> pd.DatetimeIndex:
    """Every NYSE session from `CALENDAR_START` to `CALENDAR_END`, sorted,
    as timezone-naive midnight timestamps."""
    calendar = xcals.get_calendar(
        "XNYS", start=CALENDAR_START.isoformat(), end=CALENDAR_END.isoformat()
    )
    sessions = pd.DatetimeIndex(calendar.sessions)
    return sessions.tz_localize(None) if sessions.tz is not None else sessions


def _in_range(day: date) -> pd.Timestamp:
    """`day` as a midnight timestamp; raises `ValueError` outside the calendar range."""
    if not CALENDAR_START <= day <= CALENDAR_END:
        raise ValueError(
            f"{day.isoformat()} is outside the trading calendar "
            f"({CALENDAR_START.isoformat()}..{CALENDAR_END.isoformat()})"
        )
    return pd.Timestamp(day)


def is_session(day: date) -> bool:
    """Whether the NYSE trades on `day`."""
    sessions = _sessions()
    stamp = _in_range(day)
    position = int(sessions.searchsorted(stamp, side="left"))
    return position < len(sessions) and sessions[position] == stamp


def add_sessions(day: date, n: int) -> date:
    """The `n`-th session strictly after `day` (`n >= 1`), so
    `add_sessions(day, 1) == next_session(day)` whether or not `day` is
    itself a session."""
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    sessions = _sessions()
    position = int(sessions.searchsorted(_in_range(day), side="right")) + n - 1
    if position >= len(sessions):
        raise ValueError(
            f"{n} session(s) after {day.isoformat()} is past the trading calendar's end "
            f"({CALENDAR_END.isoformat()})"
        )
    return sessions[position].date()


def next_session(day: date) -> date:
    """The first session strictly after `day`."""
    return add_sessions(day, 1)
