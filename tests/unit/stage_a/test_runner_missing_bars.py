# test_runner_missing_bars.py

"""AT-14: the stage (a) runner stores an asset with no close on a day as unavailable without
asking the analyst, hands an asset with a close only the rows where it has one, and decides on
the sessions of `settings.stage_a.session_calendar`."""

import json
from datetime import UTC, date, datetime
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd
from pydantic import PrivateAttr

from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.config import settings, temporary_overrides
from agentic_trading.data.loaders.lakehouse_loader import lake_session
from agentic_trading.data.point_in_time import get_point_in_time_data
from agentic_trading.stage_a.runner import BatchResult, _day_signals, run_batch

_LISTED, _GAPPY = "LATE", "GAPPY"
_UNIVERSE = (_LISTED, _GAPPY)
_T = datetime(2026, 10, 29, tzinfo=UTC)
# The NYSE sessions of 4-15 January 2010.
_SESSIONS = [date(2010, 1, day) for day in (4, 5, 6, 7, 8, 11, 12, 13, 14, 15)]
# LATE has no row before this day, GAPPY a NaN close (its open kept) on _GAP, and the frame
# no row at all on _NO_ROW.
_LISTED_FROM, _GAP, _NO_ROW = date(2010, 1, 6), date(2010, 1, 11), date(2010, 1, 13)
_MISSING = {
    (_LISTED, date(2010, 1, 4)),
    (_LISTED, date(2010, 1, 5)),
    (_GAPPY, _GAP),
    (_LISTED, _NO_ROW),
    (_GAPPY, _NO_ROW),
}


class _Recording(AnalystAgent):
    """Returns a long 0.5 signal and records each call's asset, date and frame."""

    agent_id: str = "fake_recording"
    _calls: list[tuple[str, pd.Timestamp, pd.DataFrame]] = PrivateAttr(default_factory=list)

    def generate_signal(
        self, asset: str, market_data: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> AgentSignal:
        self._calls.append((asset, as_of_date, market_data))
        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction="long",
            confidence=0.5,
            rationale="",
        )


def _frame(fields: dict[str, dict[str, pd.Series]]) -> pd.DataFrame:
    """`fields` (asset to field to series) in `load_price_history_from_lakehouse`'s shape."""
    return pd.DataFrame(
        {(asset, field): values for asset, row in fields.items() for field, values in row.items()}
    )


def _market_data() -> pd.DataFrame:
    """Prices on `_SESSIONS` but `_NO_ROW`, with LATE listed late and GAPPY's close missing once."""
    index = pd.DatetimeIndex([day for day in _SESSIONS if day != _NO_ROW], name="date")
    close = pd.Series(np.linspace(100.0, 109.0, len(index)), index=index)
    listed = close.where(index >= pd.Timestamp(_LISTED_FROM))
    gappy = close.where(index != pd.Timestamp(_GAP))
    return _frame(
        {
            _LISTED: {
                "open": listed,
                "close": listed,
                "raw_close": listed,
                "volume": listed * 1e4,
                "log_return": np.log(listed / listed.shift(1)),
            },
            _GAPPY: {
                "open": close,
                "close": gappy,
                "raw_close": gappy,
                "volume": close * 1e4,
                "log_return": np.log(gappy / gappy.shift(1)),
            },
        }
    )


def _has_close(market: pd.DataFrame, asset: str, day: date) -> bool:
    """Whether `market` holds a row on `day` with a close for `asset`."""
    stamp = pd.Timestamp(day)
    return stamp in market.index and bool(pd.notna(market.loc[stamp, (asset, "close")]))


def _all_days(market: pd.DataFrame, analyst: _Recording) -> dict[date, list[AgentSignal]]:
    """`_day_signals` for `_UNIVERSE` on every one of `_SESSIONS`, inside one lake session."""
    with lake_session(_T, data_end=settings.stage_a.data_end) as session:
        return {
            day: _day_signals(analyst, list(_UNIVERSE), market, day, session) for day in _SESSIONS
        }


def _batch(
    analyst: AnalystAgent,
    universe: list[str],
    start: date,
    end: date,
    store_dir: Path,
    market: pd.DataFrame,
) -> BatchResult:
    """`run_batch` for one analyst as equity, with fixed provenance and resources always ok."""
    return run_batch(
        [analyst],
        universe,
        "equity",
        start,
        end,
        store_dir,
        market,
        lake_time=_T,
        lake_version="f89aef2f428a",
        quantcore_version="0.4.0",
        resources_ok=lambda _store_dir: True,
    )


def _xcal_sessions(name: str, start: date, end: date) -> list[date]:
    """The sessions of exchange-calendars' `name` calendar from `start` to `end`, inclusive."""
    calendar = xcals.get_calendar(name, start="2023-01-01", end="2023-12-31")
    sessions = calendar.sessions_in_range(start.isoformat(), end.isoformat())
    return [stamp.date() for stamp in sessions]


def test_day_signals_without_a_bar_is_unavailable() -> None:
    """An asset with a NaN close or no row on the day gets `unavailable_signal` and no call."""
    market, analyst = _market_data(), _Recording()
    by_day = _all_days(market, analyst)
    called = [(asset, stamp.date()) for asset, stamp, _ in analyst._calls]
    every = {(asset, day) for day in _SESSIONS for asset in _UNIVERSE}

    assert _MISSING == {(asset, day) for asset, day in every if not _has_close(market, asset, day)}
    assert sorted(called) == sorted(every - _MISSING)
    for day, signals in by_day.items():
        assert [signal.asset for signal in signals] == list(_UNIVERSE)
        for signal in signals:
            if (signal.asset, day) in _MISSING:
                assert signal == analyst.unavailable_signal(
                    signal.asset, pd.Timestamp(day), signal.rationale
                )
                assert signal.rationale.strip()
            else:
                assert (signal.status, signal.direction, signal.as_of_date) == (
                    "active",
                    "long",
                    day,
                )


def test_day_signals_hands_only_rows_with_the_assets_close() -> None:
    """Each call gets the point-in-time rows where its asset has a close, all columns kept."""
    market, analyst = _market_data(), _Recording()
    _all_days(market, analyst)
    frames = {(asset, stamp.date()): frame for asset, stamp, frame in analyst._calls}
    gap, before_listing = pd.Timestamp(_GAP), pd.Timestamp(2010, 1, 4)

    assert frames
    for asset, stamp, frame in analyst._calls:
        visible = get_point_in_time_data(market, stamp)
        expected = visible.loc[visible[(asset, "close")].notna()]
        pd.testing.assert_frame_equal(frame, expected, check_freq=False)
        assert frame[(asset, "close")].notna().all()
    listed_after_gap = frames[(_LISTED, date(2010, 1, 12))]
    gappy_after_gap = frames[(_GAPPY, date(2010, 1, 12))]
    assert pd.isna(listed_after_gap.loc[gap, (_GAPPY, "close")])
    assert listed_after_gap.loc[gap, (_GAPPY, "open")] == market.loc[gap, (_GAPPY, "open")]
    assert gap not in gappy_after_gap.index
    assert pd.isna(gappy_after_gap.loc[before_listing, (_LISTED, "close")])


def test_run_batch_stores_missing_bars_as_unavailable(tmp_path: Path) -> None:
    """A batch writes every session and counts each asset's no-bar days as unavailable."""
    analyst = _Recording()
    entry = _batch(
        analyst, list(_UNIVERSE), _SESSIONS[0], _SESSIONS[-1], tmp_path, _market_data()
    ).analysts[analyst.agent_id]
    counts = json.loads((tmp_path / entry.spec_hash / "counts.json").read_text())

    assert entry.days_written == len(_SESSIONS)
    assert counts["instruments"] == {
        _LISTED: {"active": 7, "unavailable": 3, "error": 0},
        _GAPPY: {"active": 8, "unavailable": 2, "error": 0},
    }


def test_run_batch_decides_on_the_session_calendar(tmp_path: Path) -> None:
    """XSTO decides 2023-07-04 and skips 2023-06-06, XNYS the reverse, and the trials hash apart."""
    index = pd.bdate_range("2023-05-01", "2023-07-14", name="date")
    close = pd.Series(np.linspace(100.0, 120.0, len(index)), index=index)
    market = _frame(
        {
            "VOLV-B.ST": {
                "open": close,
                "close": close,
                "raw_close": close,
                "volume": close * 1e4,
                "log_return": np.log(close / close.shift(1)),
            }
        }
    )
    start, end = date(2023, 6, 1), date(2023, 7, 10)
    decided, hashes = {}, {}
    for calendar in ("XNYS", "XSTO"):
        analyst = _Recording()
        with temporary_overrides({"stage_a.session_calendar": calendar}):
            result = _batch(analyst, ["VOLV-B.ST"], start, end, tmp_path / calendar, market)
        decided[calendar] = sorted({stamp.date() for _, stamp, _ in analyst._calls})
        hashes[calendar] = result.analysts[analyst.agent_id].spec_hash

    assert settings.stage_a.session_calendar == "XNYS"
    for calendar in ("XNYS", "XSTO"):
        assert decided[calendar] == _xcal_sessions(calendar, start, end), calendar
    assert date(2023, 6, 6) not in decided["XSTO"] and date(2023, 7, 4) in decided["XSTO"]
    assert date(2023, 6, 6) in decided["XNYS"] and date(2023, 7, 4) not in decided["XNYS"]
    assert hashes["XNYS"] != hashes["XSTO"]
