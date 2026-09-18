# loader.py

"""Market-data entry point for the CLI and GUI: price history comes from
the lakehouse (`data/loaders/lakehouse_loader.py`) and nowhere else. The
frame this module returns still has to pass through the point-in-time
boundary (`data/point_in_time.py`) before any agent sees it."""

from datetime import date, timedelta

import pandas as pd

from agentic_trading.data.loaders.lakehouse_loader import load_price_history_from_lakehouse
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
) -> tuple[pd.DataFrame, str]:
    """The shared entry point the CLI uses to load market data — always
    returns MultiIndex (ticker, field) columns, even for a single-ticker
    basket, so every downstream consumer only has to handle one shape.

    Lakehouse-only: this project never calls a live price API or
    fabricates data during a run. Raises `ValueError` naming whichever
    tickers the lakehouse doesn't cover for the requested range —
    `backtest/preflight.py` is what reports this to the user before a
    run starts, rather than this function's raise being the first
    anyone hears of it mid-load.

    Returns `(frame, source_label)` — `source_label` is always
    `"lakehouse data"`, kept in the return shape so existing callers
    (e.g. the CLI's log line) don't need to change.
    """
    if warmup_days < 0:
        raise ValueError("warmup_days must be non-negative")
    data_start = start - timedelta(days=warmup_days)

    lake_frame = load_price_history_from_lakehouse(list(tickers), data_start, end)
    lake_tickers = set(lake_frame.columns.get_level_values(0)) if lake_frame is not None else set()
    missing_tickers = [ticker for ticker in tickers if ticker not in lake_tickers]
    if missing_tickers:
        raise ValueError(
            f"lakehouse has no price coverage for {missing_tickers} over "
            f"{data_start.isoformat()} to {end.isoformat()}"
        )

    close_columns = [(ticker, "close") for ticker in tickers]
    frame = lake_frame.dropna(subset=close_columns)  # type: ignore[union-attr]
    if frame.empty:
        raise ValueError(
            f"lakehouse price data for {tickers} was empty over "
            f"{data_start.isoformat()} to {end.isoformat()}"
        )
    return frame, "lakehouse data"
