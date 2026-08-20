from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MarketSlice:
    close: float | None
    log_returns: np.ndarray


def _slice_history(
    market_data: pd.DataFrame,
    as_of_date: pd.Timestamp,
    asset: str | None = None,
) -> pd.DataFrame:
    history = market_data.loc[market_data.index <= as_of_date]
    if isinstance(history, pd.Series):
        history = history.to_frame().T
    if asset is None or not isinstance(history.columns, pd.MultiIndex):
        return history.copy()

    if asset not in history.columns.get_level_values(0):
        return pd.DataFrame(index=history.index)

    sliced = history.xs(asset, axis=1, level=0)
    if isinstance(sliced, pd.Series):
        return sliced.to_frame()
    return sliced.copy()


def market_slice(
    market_data: pd.DataFrame,
    as_of_date: pd.Timestamp,
    asset: str | None = None,
) -> MarketSlice:
    history = _slice_history(market_data, as_of_date, asset)

    close: float | None = None
    if "close" in history.columns:
        close_series = history["close"].dropna()
        if not close_series.empty:
            close = float(close_series.iloc[-1])
    elif "Close" in history.columns:
        close_series = history["Close"].dropna()
        if not close_series.empty:
            close = float(close_series.iloc[-1])

    if "log_return" in history.columns:
        return_series: pd.Series = history["log_return"].dropna()
    elif close is not None and "close" in history.columns:
        close_series = history["close"].dropna()
        return_series = pd.Series(
            np.log(close_series / close_series.shift(1)), index=close_series.index
        ).dropna()
    elif close is not None and "Close" in history.columns:
        close_series = history["Close"].dropna()
        return_series = pd.Series(
            np.log(close_series / close_series.shift(1)), index=close_series.index
        ).dropna()
    else:
        return_series = pd.Series(dtype=float)

    return MarketSlice(close=close, log_returns=return_series.to_numpy(dtype=np.float64))


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return float(max(lower, min(upper, value)))
