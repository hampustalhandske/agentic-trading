# test_trading_calendar_sessions_between.py

"""AT-14: `sessions_between(start, end, calendar)` lists an exchange-calendars calendar's
sessions from `start` to `end`, inclusive, XNYS by default."""

from datetime import date, timedelta

import exchange_calendars as xcals

from agentic_trading.data.trading_calendar import is_session, sessions_between

_START, _END = date(2023, 6, 1), date(2023, 7, 10)
_NATIONAL_DAY, _INDEPENDENCE_DAY = date(2023, 6, 6), date(2023, 7, 4)


def _xcal_sessions(name: str, start: date, end: date) -> list[date]:
    """The sessions of exchange-calendars' `name` calendar from `start` to `end`, inclusive."""
    calendar = xcals.get_calendar(name, start="2023-01-01", end="2023-12-31")
    sessions = calendar.sessions_in_range(start.isoformat(), end.isoformat())
    return [stamp.date() for stamp in sessions]


def test_sessions_between() -> None:
    """XNYS and XSTO match exchange-calendars, ends included, and a holiday-only range is empty."""
    nyse = sessions_between(_START, _END, calendar="XNYS")
    stockholm = sessions_between(_START, _END, calendar="XSTO")
    days = [_START + timedelta(days=offset) for offset in range((_END - _START).days + 1)]

    assert nyse == _xcal_sessions("XNYS", _START, _END)
    assert stockholm == _xcal_sessions("XSTO", _START, _END)
    assert sessions_between(_START, _END) == nyse
    assert nyse == [day for day in days if is_session(day)]
    assert (nyse[0], nyse[-1], stockholm[0], stockholm[-1]) == (_START, _END, _START, _END)
    assert _NATIONAL_DAY in nyse and _INDEPENDENCE_DAY not in nyse
    assert _NATIONAL_DAY not in stockholm and _INDEPENDENCE_DAY in stockholm
    assert {type(day) for day in nyse + stockholm} == {date}
    assert sessions_between(_INDEPENDENCE_DAY, _INDEPENDENCE_DAY) == []
    assert sessions_between(_INDEPENDENCE_DAY, _INDEPENDENCE_DAY, calendar="XSTO") == [
        _INDEPENDENCE_DAY
    ]
