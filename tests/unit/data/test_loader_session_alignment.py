# test_loader_session_alignment.py

"""AT-14: `load_price_history_with_fallback` aligns tickers on a calendar's sessions by default,
recomputing each ticker's `log_return` across its own gaps, and keeps only the rows where every
ticker has a close under `align="intersection"`, which the backtest engine's callers pass."""

import inspect
from datetime import date
from pathlib import Path
from typing import Any
from unittest.mock import patch

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

import agentic_trading.data.loader as loader
import agentic_trading.gui.dashboard as dashboard
from agentic_trading.backtest.preflight import PreflightReport
from agentic_trading.cli import app, experiment_app
from agentic_trading.config import settings, temporary_overrides
from agentic_trading.data.loader import load_price_history_with_fallback
from agentic_trading.data.trading_calendar import is_session
from unit.test_cli import _fake_backtest_result
from unit.test_cli_stage_a import _args, _bound, _invoke

runner = CliRunner()
_PRICE_FIELDS = ("open", "close", "raw_close", "volume")
_EXPERIMENT_ARGS = [
    "AAPL",
    "--train-start",
    "2024-01-02",
    "--train-end",
    "2024-01-05",
    "--test-start",
    "2024-01-08",
    "--test-end",
    "2024-01-12",
    "--yes",
]


def _days(start: str, end: str, *, sessions_only: bool = True) -> pd.DatetimeIndex:
    """The NYSE sessions from `start` to `end`, or every weekday when `sessions_only` is False."""
    days = pd.bdate_range(start, end, name="date")
    return days[[is_session(day.date()) for day in days]] if sessions_only else days


def _with(days: pd.DatetimeIndex, *extra: str) -> pd.DatetimeIndex:
    """`days` with the `extra` ISO dates added, sorted."""
    return days.append(pd.DatetimeIndex(list(extra))).sort_values()


def _closes(days: pd.DatetimeIndex, seed: int) -> pd.Series:
    """A positive random-walk close on `days`, reproducible from `seed`."""
    steps = np.random.default_rng(seed).normal(0.0, 0.02, len(days))
    return pd.Series(100.0 * np.exp(np.cumsum(steps)), index=days)


def _lake(closes: dict[str, pd.Series]) -> pd.DataFrame:
    """`closes` in `load_price_history_from_lakehouse`'s shape, each ticker on its own rows."""
    per_ticker = {
        ticker: pd.DataFrame(
            {
                "open": close * 0.99,
                "close": close,
                "raw_close": close * 1.5,
                "volume": np.arange(1.0, len(close) + 1.0) * 1e3,
                "log_return": np.log(close / close.shift(1)),
            },
            index=close.index,
        )
        for ticker, close in closes.items()
    }
    frame = pd.concat(per_ticker, axis=1).sort_index()
    frame.index.name = "date"
    return frame


def _load(
    monkeypatch: pytest.MonkeyPatch,
    lake: pd.DataFrame | None,
    tickers: list[str],
    start: date,
    end: date,
    **options: Any,
) -> pd.DataFrame:
    """The frame `load_price_history_with_fallback` returns over a lake faked as `lake`."""
    monkeypatch.setattr(loader, "load_price_history_from_lakehouse", lambda *_: lake)
    frame, _ = load_price_history_with_fallback(tickers, start, end, **options)
    return frame


def _close_columns(tickers: list[str]) -> list[tuple[str, str]]:
    return [(ticker, "close") for ticker in tickers]


def _same(left: pd.Series, right: pd.Series) -> None:
    pd.testing.assert_series_equal(left, right, check_freq=False, check_names=False)


def _at(series: pd.Series, day: str) -> float:
    return float(series[pd.Timestamp(day)])


def _assert_returns_telescope(frame: pd.DataFrame, ticker: str) -> None:
    """`ticker`'s log returns are NaN off its closes and sum to log(close / first close) per row."""
    close, log_return = frame[(ticker, "close")], frame[(ticker, "log_return")]
    has_close = close.notna()
    first = has_close.idxmax()
    later = has_close & (close.index > first)

    assert log_return[~has_close].isna().all(), ticker
    assert np.isnan(log_return[first]), ticker
    assert log_return[later].notna().all(), ticker
    running = log_return.fillna(0.0).cumsum()[later].to_numpy()
    expected = np.log(close.ffill()[later] / close[first]).to_numpy()
    np.testing.assert_allclose(running, expected, rtol=0.0, atol=1e-9)


def test_late_listing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A late COIN listing keeps AAPL's earlier rows, except under "intersection"."""
    early, late = _days("2016-12-27", "2021-06-30"), _days("2021-04-14", "2021-06-30")
    lake = _lake({"AAPL": _closes(early, 1), "COIN": _closes(late, 2)})
    tickers = ["AAPL", "COIN"]
    window = (date(2016, 12, 27), date(2021, 6, 30))
    sessions = _load(monkeypatch, lake, tickers, *window, align="sessions")
    default = _load(monkeypatch, lake, tickers, *window)
    intersection = _load(monkeypatch, lake, tickers, *window, align="intersection")

    assert list(sessions.index) == list(early)
    _same(sessions[("AAPL", "close")], lake[("AAPL", "close")])
    assert sessions.loc[sessions.index < late[0], ("COIN", "close")].isna().all()
    _same(sessions.loc[late, ("COIN", "close")], lake.loc[late, ("COIN", "close")])
    for ticker in tickers:
        _assert_returns_telescope(sessions, ticker)
    pd.testing.assert_frame_equal(default, sessions)
    assert intersection.index[0] == late[0]
    expected = lake.dropna(subset=_close_columns(tickers))
    pd.testing.assert_frame_equal(intersection, expected, check_freq=False)


def test_crypto_with_equities(monkeypatch: pytest.MonkeyPatch) -> None:
    """A crypto ticker keeps every 2023 session and its returns sum to log(last/first)."""
    year = _days("2023-01-03", "2023-12-29")
    equity = _closes(year.delete([*range(40, 45), 100, 150]), 4)
    equity.iloc[200] = np.nan
    lake = _lake({"BTC-USD": _closes(year, 3), "SPY": equity})
    frame = _load(monkeypatch, lake, ["BTC-USD", "SPY"], date(2023, 1, 3), date(2023, 12, 29))
    btc = frame[("BTC-USD", "close")]

    assert list(frame.index) == list(year)
    assert frame[("BTC-USD", "log_return")].sum() == pytest.approx(
        np.log(btc.iloc[-1] / btc.iloc[0]), abs=1e-9
    )
    for ticker in ("BTC-USD", "SPY"):
        _assert_returns_telescope(frame, ticker)


def test_fx_on_an_nyse_holiday(monkeypatch: pytest.MonkeyPatch) -> None:
    """An FX bar on 2023-07-04 or a Sunday is dropped and its move lands in the next session."""
    fx = _closes(_with(_days("2023-06-26", "2023-07-14", sessions_only=False), "2023-07-09"), 5)
    equity = _closes(_days("2023-06-26", "2023-07-14"), 6)
    for lake in (_lake({"EURUSD=X": fx}), _lake({"EURUSD=X": fx, "SPY": equity})):
        tickers = list(lake.columns.get_level_values(0).unique())
        frame = _load(monkeypatch, lake, tickers, date(2023, 6, 26), date(2023, 7, 14))
        log_return = frame[("EURUSD=X", "log_return")]

        assert pd.Timestamp("2023-07-04") not in frame.index, tickers
        assert pd.Timestamp("2023-07-09") not in frame.index, tickers
        assert all(is_session(day.date()) for day in frame.index), tickers
        assert _at(log_return, "2023-07-05") == pytest.approx(
            np.log(_at(fx, "2023-07-05") / _at(fx, "2023-07-03")), abs=1e-12
        )
        assert _at(log_return, "2023-07-10") == pytest.approx(
            np.log(_at(fx, "2023-07-10") / _at(fx, "2023-07-07")), abs=1e-12
        )
        assert _at(frame[("EURUSD=X", "close")], "2023-07-05") == _at(fx, "2023-07-05")
        _assert_returns_telescope(frame, "EURUSD=X")


def test_sessions_follow_the_given_calendar(monkeypatch: pytest.MonkeyPatch) -> None:
    """Under `calendar="XSTO"` 2023-06-06 is dropped, its move carried on, and 2023-07-04 kept."""
    weekdays = _days("2023-06-01", "2023-07-10", sessions_only=False)
    close = _closes(weekdays, 11)
    window = (date(2023, 6, 1), date(2023, 7, 10))
    lake = _lake({"VOLV-B.ST": close})
    frame = _load(monkeypatch, lake, ["VOLV-B.ST"], *window, align="sessions", calendar="XSTO")
    stockholm = xcals.get_calendar("XSTO", start="2023-01-01", end="2023-12-31")
    sessions = {stamp.date() for stamp in stockholm.sessions_in_range("2023-06-01", "2023-07-10")}
    kept = [day.date() for day in weekdays if day.date() in sessions]

    assert [day.date() for day in frame.index] == kept
    assert pd.Timestamp("2023-06-06") not in frame.index
    assert _at(frame[("VOLV-B.ST", "close")], "2023-07-04") == _at(close, "2023-07-04")
    assert _at(frame[("VOLV-B.ST", "log_return")], "2023-06-07") == pytest.approx(
        np.log(_at(close, "2023-06-07") / _at(close, "2023-06-05")), abs=1e-12
    )
    _assert_returns_telescope(frame, "VOLV-B.ST")


def test_aligned_fields_match_the_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rows are sessions with some close; price fields equal the source on each ticker's rows."""
    equity = _closes(_days("2023-06-01", "2023-08-31").delete([5, 6, 30]), 8)
    equity.iloc[20] = np.nan
    fx = _closes(_with(_days("2023-06-01", "2023-08-31", sessions_only=False), "2023-07-09"), 9)
    late = _closes(_days("2023-07-17", "2023-08-31"), 10)
    lake = _lake({"SPY": equity, "EURUSD=X": fx, "LATE": late})
    tickers = ["SPY", "EURUSD=X", "LATE"]
    window = (date(2023, 6, 1), date(2023, 8, 31))
    frame = _load(monkeypatch, lake, tickers, *window)
    intersection = _load(monkeypatch, lake, tickers, *window, align="intersection")
    closes = lake[_close_columns(tickers)]
    rows = [day for day in lake.index if is_session(day.date()) and closes.loc[day].notna().any()]

    assert list(frame.index) == rows
    assert set(frame.columns) == set(lake.columns)
    for ticker in tickers:
        own = [day for day in rows if pd.notna(lake.loc[day, (ticker, "close")])]
        absent = frame.index.difference(lake[ticker].dropna(how="all").index)
        for field in _PRICE_FIELDS:
            _same(frame.loc[own, (ticker, field)], lake.loc[own, (ticker, field)])
        assert frame.loc[absent, ticker].isna().all().all(), ticker
        _assert_returns_telescope(frame, ticker)
    expected = lake.dropna(subset=_close_columns(tickers))
    pd.testing.assert_frame_equal(intersection, expected, check_freq=False)


def test_errors_in_both_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both modes raise for a ticker with no coverage and for an empty result."""
    weekdays = _days("2024-01-02", "2024-01-12")
    window = (date(2024, 1, 2), date(2024, 1, 12))
    covered = _lake({"AAPL": _closes(weekdays, 7)})
    blank = _lake({"AAPL": pd.Series(np.nan, index=weekdays)})
    off_days = pd.DatetimeIndex(["2023-07-04", "2023-07-08"])
    off_session = _lake({"EURUSD=X": pd.Series([1.1, 1.2], index=off_days)})
    for align in ("sessions", "intersection"):
        with pytest.raises(ValueError, match="AAPL"):
            _load(monkeypatch, None, ["AAPL"], *window, align=align)
        with pytest.raises(ValueError, match="MSFT"):
            _load(monkeypatch, covered, ["AAPL", "MSFT"], *window, align=align)
        with pytest.raises(ValueError, match="empty"):
            _load(monkeypatch, blank, ["AAPL"], *window, align=align)
    with pytest.raises(ValueError, match="empty"):
        _load(monkeypatch, off_session, ["EURUSD=X"], date(2023, 7, 3), date(2023, 7, 10))


def _tickers(call: Any) -> list[str]:
    """The `tickers` argument of one recorded `load_price_history_with_fallback` call."""
    bound = inspect.signature(load_price_history_with_fallback).bind(*call.args, **call.kwargs)
    return list(bound.arguments["tickers"])


def test_callers_choose_the_alignment(
    monkeypatch: pytest.MonkeyPatch, preflight_lake: Any, tmp_path: Path
) -> None:
    """Engine callers pass "intersection"; stage (a) loads "sessions" on its session calendar."""
    index = pd.date_range("2024-01-02", periods=4)
    benchmark = pd.DataFrame({("^GSPC", "close"): [100.0, 101.0, 102.0, 103.0]}, index=index)
    benchmark.columns = pd.MultiIndex.from_tuples(benchmark.columns)
    loaded = (benchmark, "lakehouse data")
    monkeypatch.setattr(
        "agentic_trading.cli.load_risk_free_rate_series", lambda *_: pd.Series(0.0, index=index)
    )
    monkeypatch.setattr(
        "agentic_trading.cli.check_data_availability",
        lambda *_, **__: PreflightReport(checked_at=date(2024, 1, 2), results=[]),
    )
    run_args = ["AAPL", "MSFT", "--start-date", "2024-01-02", "--end-date", "2024-01-05", "--yes"]
    with (
        patch("agentic_trading.cli.load_price_history_with_fallback", return_value=loaded) as load,
        patch("agentic_trading.cli.BacktestEngine") as engine,
    ):
        engine.return_value.run.return_value = _fake_backtest_result()
        ran = runner.invoke(app, run_args)
        run_calls = list(load.call_args_list)
        load.reset_mock()
        experimented = runner.invoke(experiment_app, _EXPERIMENT_ARGS)
        experiment_calls = [call for call in load.call_args_list if _tickers(call) == ["AAPL"]]
    with patch.object(
        dashboard, "load_price_history_with_fallback", return_value=loaded
    ) as dashboard_load:
        dashboard._load_basket_data(("AAPL", "MSFT"), date(2024, 1, 2), date(2024, 1, 5))
    with temporary_overrides({"stage_a.session_calendar": "XNYS"}):
        staged, staged_load, _ = _invoke(_args(tmp_path))
        overridden, overridden_load, _ = _invoke(
            _args(tmp_path, "--set", "stage_a.session_calendar=XSTO")
        )

    assert ran.exit_code == 0, ran.output
    assert [call.kwargs.get("align") for call in run_calls] == ["intersection"]
    assert experimented.exit_code == 0, experimented.output
    assert [call.kwargs.get("align") for call in experiment_calls] == ["intersection"]
    assert dashboard_load.call_args.kwargs.get("align") == "intersection"
    assert staged.exit_code == 0, staged.output
    staged_args = _bound(load_price_history_with_fallback, staged_load)
    assert (staged_args["align"], staged_args["calendar"]) == ("sessions", "XNYS")
    assert overridden.exit_code == 0, overridden.output
    overridden_args = _bound(load_price_history_with_fallback, overridden_load)
    assert (overridden_args["align"], overridden_args["calendar"]) == ("sessions", "XSTO")
    assert settings.stage_a.session_calendar == "XNYS"
