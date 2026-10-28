# macro_loader.py

"""FRED series identifiers shared across analysts and the backtest's
risk-free-rate report, plus the risk-free rate series itself, read from
the lakehouse's `fred_vintages` table."""

from datetime import date

import pandas as pd

from agentic_trading.data.loaders.lakehouse_loader import load_fred_series_from_lakehouse

DGS3MO = "DGS3MO"
DGS2 = "DGS2"
DGS10 = "DGS10"
VIXCLS = "VIXCLS"
CPIAUCSL = "CPIAUCSL"
# ICE BofA option-adjusted-spread indices — high yield and investment
# grade. Under an ICE licensing change FRED only serves a rolling 3-year
# window for these two series regardless of the requested range — the
# full 1996+ history is ICE Data Indices' property, available only via a
# commercial feed. A real, source-side limitation.
BAMLH0A0HYM2 = "BAMLH0A0HYM2"
BAMLC0A0CM = "BAMLC0A0CM"

SUPPORTED_SERIES: frozenset[str] = frozenset(
    {DGS3MO, DGS2, DGS10, VIXCLS, CPIAUCSL, BAMLH0A0HYM2, BAMLC0A0CM}
)

TRADING_DAYS_PER_YEAR = 252


def load_risk_free_rate_series(start_date: date, end_date: date) -> pd.Series:
    """Daily-equivalent risk-free rate for the backtest engine's cash
    interest, margin interest and excess-return performance metrics
    (Sharpe, Sortino), derived from FRED's 3-month T-bill
    secondary-market rate (`DGS3MO`, quoted as an annualized percent)
    as stored in the lakehouse.

    Returned as a decimal daily rate (`annualized_percent / 100 / 252`),
    indexed by calendar date from `start_date` to `end_date` and
    forward-filled across non-publishing days (weekends/holidays) so
    callers can align it against any trading-day index without gaps. It
    is never filled backwards: every day carries a rate observed on or
    before it. This is a backtest's own after-the-fact performance report,
    not a trading signal `generate_signal` reads live — the rate in force
    on a given day is that day's real rate, so the series is read as
    published by `end_date`.

    Raises `ValueError` if the lakehouse has no `DGS3MO` observations, or
    if `start_date` precedes the first observation (the message names
    that observation's date)."""
    annualized_percent = load_fred_series_from_lakehouse(DGS3MO, end_date)
    if annualized_percent is None or annualized_percent.empty:
        raise ValueError(
            f"lakehouse has no FRED {DGS3MO} observations up to {end_date.isoformat()}; "
            "the risk-free rate series cannot be built without it"
        )
    first_observation = annualized_percent.index.min()
    if pd.Timestamp(start_date) < first_observation:
        raise ValueError(
            f"start date {start_date.isoformat()} precedes the first FRED {DGS3MO} "
            f"observation on {first_observation.date().isoformat()}; the risk-free rate "
            "is never filled backwards, so the window must start on or after that date"
        )
    daily_rate = annualized_percent / 100.0 / TRADING_DAYS_PER_YEAR
    calendar = pd.date_range(start_date, end_date, freq="D")
    return daily_rate.reindex(calendar).ffill()
