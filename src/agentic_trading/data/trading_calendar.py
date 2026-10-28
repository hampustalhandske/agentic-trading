# trading_calendar.py

"""NYSE trading sessions as `datetime.date`s: `is_session`, `next_session`
and `add_sessions`.

A session is a weekday that is neither a rule-based NYSE holiday
(`NYSE_HOLIDAY_RULES`) nor one of the unscheduled full-day closures in
`SPECIAL_CLOSURES`. Early closes (half days) are sessions. The rule set
is kept by hand from the NYSE holiday rules; it is not read from an
exchange feed. When the NYSE closes for a day outside its holiday rules
(a state funeral, a storm), that date must be added to `SPECIAL_CLOSURES`.

Sessions are precomputed once for `CALENDAR_START`..`CALENDAR_END`; every
function raises `ValueError` for a day, or a result, outside that range."""

from datetime import date
from functools import cache

import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    nearest_workday,
    sunday_to_monday,
)
from pandas.tseries.offsets import DateOffset

# Covers every source the event loaders read (FINRA, SEC EDGAR, earnings
# calendars) with room on both sides; a wider range only costs a few
# thousand more entries in the cached index.
CALENDAR_START = date(1990, 1, 1)
CALENDAR_END = date(2035, 12, 31)

# Full-day NYSE closures that no holiday rule produces.
SPECIAL_CLOSURES: tuple[date, ...] = (
    date(1994, 4, 27),  # Richard Nixon funeral
    date(2001, 9, 11),  # September 11 attacks
    date(2001, 9, 12),  # September 11 attacks
    date(2001, 9, 13),  # September 11 attacks
    date(2001, 9, 14),  # September 11 attacks
    date(2004, 6, 11),  # Ronald Reagan funeral
    date(2007, 1, 2),  # Gerald Ford funeral
    date(2012, 10, 29),  # Hurricane Sandy
    date(2012, 10, 30),  # Hurricane Sandy
    date(2018, 12, 5),  # George H. W. Bush funeral
    date(2025, 1, 9),  # Jimmy Carter funeral
)


# Moves a date to the first Monday (Thursday) on or after it, so "the
# third Monday of January" is the first Monday on or after January 15.
_ON_OR_AFTER_MONDAY = DateOffset(weekday=0)
_ON_OR_AFTER_THURSDAY = DateOffset(weekday=3)


# The NYSE's rule-based full-day holidays. A holiday on a Sunday is
# observed on Monday; one on a Saturday is observed on Friday, except New
# Year's Day, which is then not observed at all.
NYSE_HOLIDAY_RULES: tuple[Holiday, ...] = (
    Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
    Holiday(
        "Martin Luther King Jr. Day",
        month=1,
        day=15,
        offset=_ON_OR_AFTER_MONDAY,
        start_date=pd.Timestamp("1998-01-01"),
    ),
    Holiday("Presidents' Day", month=2, day=15, offset=_ON_OR_AFTER_MONDAY),
    GoodFriday,
    Holiday("Memorial Day", month=5, day=25, offset=_ON_OR_AFTER_MONDAY),
    Holiday(
        "Juneteenth",
        month=6,
        day=19,
        start_date=pd.Timestamp("2022-01-01"),
        observance=nearest_workday,
    ),
    Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
    Holiday("Labor Day", month=9, day=1, offset=_ON_OR_AFTER_MONDAY),
    Holiday("Thanksgiving Day", month=11, day=22, offset=_ON_OR_AFTER_THURSDAY),
    Holiday("Christmas Day", month=12, day=25, observance=nearest_workday),
)


class NYSEHolidayCalendar(AbstractHolidayCalendar):
    """A pandas holiday calendar of `NYSE_HOLIDAY_RULES`."""

    def __init__(self) -> None:
        super().__init__(name="NYSE", rules=list(NYSE_HOLIDAY_RULES))


@cache
def _sessions() -> pd.DatetimeIndex:
    """Every NYSE session from `CALENDAR_START` to `CALENDAR_END`, sorted."""
    weekdays = pd.bdate_range(CALENDAR_START, CALENDAR_END)
    holidays = NYSEHolidayCalendar().holidays(
        pd.Timestamp(CALENDAR_START), pd.Timestamp(CALENDAR_END)
    )
    closures = pd.DatetimeIndex([pd.Timestamp(day) for day in SPECIAL_CLOSURES])
    return weekdays.difference(holidays.union(closures))


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
