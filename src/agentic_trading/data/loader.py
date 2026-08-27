# loader.py

import math
from datetime import date

import numpy as np
import pandas as pd
import pandas_datareader.data as pdr
import yfinance as yf

from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_BASKET: list[str] = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM"]


def _resolve_tickers(asset: str | list[str] | None) -> list[str]:
    if asset is None:
        return list(DEFAULT_BASKET)
    if isinstance(asset, str):
        return [asset]
    return asset


def _reshape_single_ticker(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        raise ValueError("yfinance returned no data for the requested ticker/date range")

    close = raw["Close"]
    result = pd.DataFrame({"close": close})
    # Log returns, not simple returns: quantcore's core/sde_solver.py and
    # pricing/black_scholes.py both assume GBM log-price dynamics.
    result["log_return"] = np.log(result["close"] / result["close"].shift(1))
    return result


def _reshape_stooq_frame(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        raise ValueError("Stooq returned no data for the requested ticker/date range")
    # Stooq returns rows in descending date order, unlike yfinance.
    ordered = raw.sort_index()
    result = pd.DataFrame({"close": ordered["Close"]})
    result["log_return"] = np.log(result["close"] / result["close"].shift(1))
    return result


def _yfinance_result_is_usable(raw: pd.DataFrame) -> bool:
    return not raw.empty and not raw["Close"].isna().all()


def _load_single_ticker(ticker: str, start: date | None, end: date | None) -> pd.DataFrame:
    # yfinance 1.6.0 defaults multi_level_index=True, which returns
    # MultiIndex (Price, Ticker) columns even for a single ticker. Force
    # flat columns so raw["Close"] yields a Series, not a one-column
    # DataFrame. progress=False suppresses yfinance's own "N of M completed"
    # progress bar, which it prints straight to stdout regardless of any
    # logging configuration.
    raw = yf.download(ticker, start=start, end=end, multi_level_index=False, progress=False)
    if _yfinance_result_is_usable(raw):
        logger.info("Loaded price history for %s from yfinance", ticker)
        return _reshape_single_ticker(raw)

    logger.info("yfinance had no usable data for %s — trying Stooq fallback", ticker)
    stooq_raw = pdr.DataReader(ticker, "stooq", start, end)
    result = _reshape_stooq_frame(stooq_raw)
    logger.info("Loaded price history for %s from Stooq", ticker)
    return result


def load_price_history(
    asset: str | list[str] | None = None,
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """Loads full price history for `asset` between `start` and `end`
    (inclusive). Does not enforce point-in-time discipline — callers that
    need a no-look-ahead guarantee must pass the result through
    get_point_in_time_data (see point_in_time.py).

    `asset` may be a single ticker, an explicit list of tickers, or None
    to fall back to the project's default basket (DEFAULT_BASKET). A
    single ticker returns a frame with flat 'close'/'log_return' columns;
    multiple tickers return a frame with a MultiIndex on columns, keyed
    (ticker, field). `start`/`end` follow yfinance's own convention of
    None meaning an unbounded end of the range.

    Tries yfinance first for each ticker; if it raises, comes back empty,
    or comes back all-NaN, falls through to Stooq (an independent, no-key
    source) before giving up. Only `load_price_history_with_fallback`'s
    synthetic generator kicks in if both real sources fail.
    """
    tickers = _resolve_tickers(asset)
    logger.debug("Loading price history for %s (%s to %s)", tickers, start, end)

    if len(tickers) == 1:
        return _load_single_ticker(tickers[0], start, end)

    frames: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        frames[ticker] = _load_single_ticker(ticker, start, end)
    return pd.concat(frames, axis=1)


def _synthetic_price_history(ticker: str, start: date, end: date) -> pd.DataFrame:
    """A deterministic, offline stand-in price series for `ticker` — used
    only when a live yfinance load fails (see
    `load_price_history_with_fallback`). Not randomness in the trading
    sense (no unseeded RNG — the "seed" here is derived purely from the
    ticker's characters, so the same ticker always produces the same
    synthetic series), just enough shape (drift + a bounded oscillation) to
    let the rest of the system keep running and be visibly exercised
    offline."""
    dates = pd.bdate_range(start=start, end=end)
    if dates.empty:
        raise ValueError("no business days available for the requested date range")

    seed = sum(ord(character) for character in ticker)
    base_price = 80.0 + (seed % 40)
    drift = 0.0004 + ((seed % 7) * 0.00015)
    amplitude = 0.012 + ((seed % 5) * 0.003)

    prices: list[float] = []
    level = base_price
    for index, _timestamp in enumerate(dates):
        wave = math.sin(index / 3.0 + seed / 10.0) * amplitude
        level *= 1.0 + drift + wave
        prices.append(max(level, 1.0))

    frame = pd.DataFrame({"close": prices}, index=dates)
    frame["log_return"] = np.log(frame["close"] / frame["close"].shift(1))
    return frame


def load_price_history_with_fallback(
    tickers: list[str], start: date, end: date
) -> tuple[pd.DataFrame, str]:
    """The shared entry point both the CLI and the GUI use to load market
    data — always returns MultiIndex (ticker, field) columns, even for a
    single-ticker basket, so every downstream consumer only has to handle
    one shape. Falls back to a synthetic series per-ticker (logged as a
    WARNING — this is exactly a "gave a different output than intended"
    moment, not a silent one) if the live yfinance load fails for any
    reason, so a network hiccup degrades the run instead of crashing it.

    Returns `(frame, source_label)`, where `source_label` is
    "live yfinance data" or "synthetic fallback data" — surfaced to the
    user (CLI log line / GUI caption) so it's always clear which one a
    given run actually used.
    """
    try:
        frame = load_price_history(list(tickers), start, end)
        if frame.empty:
            raise ValueError("loaded market data was empty")
        if not isinstance(frame.columns, pd.MultiIndex):
            frame = pd.concat({tickers[0]: frame}, axis=1)
        return frame, "live yfinance data"
    except Exception as exc:
        logger.warning(
            "Live yfinance load failed for %s (%s to %s): %s — using synthetic fallback "
            "data for every ticker in the basket.",
            tickers,
            start,
            end,
            exc,
        )
        synthetic = {ticker: _synthetic_price_history(ticker, start, end) for ticker in tickers}
        return pd.concat(synthetic, axis=1), "synthetic fallback data"
