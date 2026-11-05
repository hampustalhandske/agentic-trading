# trading_calendar.py

"""Exchange trading sessions as `datetime.date`s: `is_session`,
`next_session` and `add_sessions` on the NYSE, and `sessions_between` on
any exchange-calendars calendar (NYSE by default).

NYSE sessions come from exchange-calendars' `XNYS` calendar: weekdays
that are neither NYSE holidays nor unscheduled full-day closures (state
funerals, September 11, Hurricane Sandy). Early closes (half days) are
sessions.

Each calendar's sessions are computed once, on first use, for
`CALENDAR_START`..`CALENDAR_END`; every function raises `ValueError` for a
day, or a result, outside that range."""

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
def _sessions(calendar_name: str = "XNYS") -> pd.DatetimeIndex:
    """Every session of exchange-calendars' `calendar_name` calendar (NYSE
    by default) from `CALENDAR_START` to `CALENDAR_END`, sorted, as
    timezone-naive midnight timestamps. exchange-calendars raises for an
    unknown name or a calendar it cannot evaluate over that whole range."""
    calendar = xcals.get_calendar(
        calendar_name, start=CALENDAR_START.isoformat(), end=CALENDAR_END.isoformat()
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


def sessions_between(start: date, end: date, calendar: str = "XNYS") -> list[date]:
    """The sessions of exchange-calendars' `calendar` calendar (NYSE by
    default) from `start` to `end`, both included, in order; empty when
    `start` is after `end`."""
    sessions = _sessions(calendar)
    first = int(sessions.searchsorted(_in_range(start), side="left"))
    after_last = int(sessions.searchsorted(_in_range(end), side="right"))
    return [stamp.date() for stamp in sessions[first:after_last]]
