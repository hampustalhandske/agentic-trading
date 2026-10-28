# test_trading_calendar_xnys.py

"""The NYSE session count per year, 2000-2026, so a change in the
exchange-calendars XNYS history shows up as a failing test."""

from agentic_trading.data.trading_calendar import _sessions

SESSIONS_PER_YEAR = {
    2000: 252, 2001: 248, 2002: 252, 2003: 252, 2004: 252, 2005: 252, 2006: 251,
    2007: 251, 2008: 253, 2009: 252, 2010: 252, 2011: 252, 2012: 250, 2013: 252,
    2014: 252, 2015: 252, 2016: 252, 2017: 251, 2018: 251, 2019: 252, 2020: 253,
    2021: 252, 2022: 251, 2023: 250, 2024: 252, 2025: 250, 2026: 251,
}  # fmt: skip


def test_sessions_per_year_2000_to_2026() -> None:
    """Each year from 2000 through 2026 has its recorded number of sessions."""
    sessions = _sessions()
    counts = {year: int((sessions.year == year).sum()) for year in SESSIONS_PER_YEAR}

    assert counts == SESSIONS_PER_YEAR
    assert sum(counts.values()) == 6790
