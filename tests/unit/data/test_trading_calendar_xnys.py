# test_trading_calendar_xnys.py

"""The module's NYSE sessions match exchange-calendars' XNYS sessions."""

import exchange_calendars as xcals
import pandas as pd

from agentic_trading.data.trading_calendar import _sessions


def test_sessions_match_xnys_2000_to_2026() -> None:
    """Every session from 2000 through 2026 is an XNYS session, and vice versa."""
    xnys = xcals.get_calendar("XNYS", start="2000-01-01", end="2026-12-31").sessions
    xnys = pd.DatetimeIndex(xnys).tz_localize(None) if xnys.tz is not None else xnys
    ours = _sessions()
    ours = ours[(ours >= "2000-01-01") & (ours <= "2026-12-31")]

    assert list(ours.difference(xnys)) == []
    assert list(xnys.difference(ours)) == []
    assert len(ours) == 6790
