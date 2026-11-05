# loader.py

"""Market-data entry point for the CLI and GUI: price history comes from
the lakehouse (`data/loaders/lakehouse_loader.py`) and nowhere else. The
frame this module returns still has to pass through the point-in-time
boundary (`data/point_in_time.py`) before any agent sees it."""

from datetime import date, timedelta
from typing import Literal

import numpy as np
import pandas as pd

from agentic_trading.data.loaders.lakehouse_loader import (
    _guard_data_end,
    load_price_history_from_lakehouse,
)
from agentic_trading.data.trading_calendar import sessions_between
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_BASKET: list[str] = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM"]
# Calendar days of history loaded before the first simulated day so every
# rolling window is full on day 1: the longest analyst window is three
# trading years (`settings.analysts.risk_window_garch`/`hmm_window` = 750
# observations), which is ~1100 calendar days.
DEFAULT_BACKTEST_WARMUP_DAYS = 1100


def load_price_history_with_fallback(
    tickers: list[str],
    start: date,
    end: date,
    warmup_days: int = 0,
    *,
    align: Literal["sessions", "intersection"] = "sessions",
    calendar: str = "XNYS",
) -> tuple[pd.DataFrame, str]:
    """The shared entry point the CLI uses to load market data — always
    returns MultiIndex (ticker, field) columns, even for a single-ticker
    basket, so every downstream consumer only has to handle one shape.
    History starts `warmup_days` calendar days before `start` and ends at
    `end`.

    `align` chooses the rows:

    - `"sessions"` (the default): every session of the exchange-calendars
      `calendar` calendar (NYSE by default) on which at least one ticker
      has a close. A ticker with no close on a kept row has NaN there, so
      a late listing does not cut the other tickers' history, and a bar
      on a day that is not a session (an FX or crypto bar on an exchange
      holiday or a weekend) is dropped. Each ticker's `log_return` is
      recomputed as the log of its close over its previous close in the
      returned frame, so a dropped bar's move lands on the next kept bar
      and the returns sum to the log of the last close over the first.
      Every other field is the lakehouse's.
    - `"intersection"`: only the rows on which every ticker has a close,
      every field as the lakehouse gives it; `calendar` is not used. The
      backtest engine's callers pass this, because the engine fills and
      marks every ticker on every bar and so needs a close for each.

    Lakehouse-only: this project never calls a live price API or
    fabricates data during a run. Raises `ValueError` naming whichever
    tickers the lakehouse doesn't cover for the requested range —
    `backtest/preflight.py` is what reports this to the user before a
    run starts, rather than this function's raise being the first
    anyone hears of it mid-load — and `ValueError` when no row is left
    after alignment.

    Inside a `lake_session` with a `data_end`, raises `HoldoutReadError`
    before any other check or query when `end` is later than it (see
    `lakehouse_loader._guard_data_end`).

    Returns `(frame, source_label)` — `source_label` is always
    `"lakehouse data"`, kept in the return shape so existing callers
    (e.g. the CLI's log line) don't need to change.
    """
    _guard_data_end(end, "load_price_history_with_fallback")
    if warmup_days < 0:
        raise ValueError("warmup_days must be non-negative")
    data_start = start - timedelta(days=warmup_days)

    lake_frame = load_price_history_from_lakehouse(list(tickers), data_start, end)
    lake_tickers = set(lake_frame.columns.get_level_values(0)) if lake_frame is not None else set()
    missing_tickers = [ticker for ticker in tickers if ticker not in lake_tickers]
    if lake_frame is None or missing_tickers:
        raise ValueError(
            f"lakehouse has no price coverage for {missing_tickers} over "
            f"{data_start.isoformat()} to {end.isoformat()}"
        )

    close_columns = [(ticker, "close") for ticker in tickers]
    if align == "intersection":
        frame = lake_frame.dropna(subset=close_columns)
    else:
        frame = _align_on_sessions(lake_frame, tickers, calendar)
    if frame.empty:
        raise ValueError(
            f"lakehouse price data for {tickers} was empty over "
            f"{data_start.isoformat()} to {end.isoformat()}"
        )
    return frame, "lakehouse data"


def _align_on_sessions(frame: pd.DataFrame, tickers: list[str], calendar: str) -> pd.DataFrame:
    """The rows of `frame` that are `calendar` sessions on which at least
    one of `tickers` has a close, with each ticker's `log_return` the log
    of its close over its previous close among those rows (NaN where it
    has no close, and on its first). Empty when no row has a close."""
    has_close = frame[[(ticker, "close") for ticker in tickers]].notna().any(axis=1)
    closed_days = frame.index[has_close]
    if closed_days.empty:
        return frame.iloc[0:0]
    sessions = pd.DatetimeIndex(
        sessions_between(closed_days.min().date(), closed_days.max().date(), calendar)
    )
    aligned = frame.loc[has_close & frame.index.isin(sessions)].copy()
    for ticker in tickers:
        close = aligned[(ticker, "close")].dropna()
        log_close = pd.Series(np.log(close.to_numpy(dtype=float)), index=close.index)
        aligned.loc[:, (ticker, "log_return")] = log_close.diff().reindex(aligned.index)
    return aligned
