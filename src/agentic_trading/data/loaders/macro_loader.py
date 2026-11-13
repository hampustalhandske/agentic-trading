# macro_loader.py

"""FRED series identifiers shared across analysts and the backtest's
risk-free-rate report, plus the risk-free rate series itself, read from
the lakehouse's `fred_vintages` table, and the BIS policy rates the
backtest accrues spot FX carry from."""

import heapq
import math
from datetime import date

import pandas as pd

from agentic_trading.data.instruments import currency_pair
from agentic_trading.data.loaders.lakehouse_loader import (
    _FRED_VERSIONS,
    _guard_data_end,
    load_policy_rates_from_lakehouse,
)

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
    indexed by every calendar date from `start_date` to `end_date`. Each
    day d carries the DGS3MO value as known on d: the value of the latest
    observation with a value, among the versions known strictly before d
    (`lakehouse_loader._select_fred_vintages`, the selection
    `load_fred_vintages_from_lakehouse` makes), so a revision published
    after d never reaches d, and a day without a new observation
    (a weekend, a holiday, the same-day H.15 release published after the
    close) keeps the previous value. Never filled backwards. The versions
    are fetched once, as known before `end_date`, through the FRED window
    store source.

    Raises `ValueError` if the lakehouse has no `DGS3MO` observations
    known before `end_date`, or if no value is known on `start_date` (the
    message names the first observation's date)."""
    _guard_data_end(end_date, "load_risk_free_rate_series")
    versions = _FRED_VERSIONS.read([DGS3MO], end_date, end_date)[DGS3MO]
    if versions["value"].isna().all():
        raise ValueError(
            f"lakehouse has no FRED {DGS3MO} observations up to {end_date.isoformat()}; "
            "the risk-free rate series cannot be built without it"
        )
    calendar = pd.date_range(start_date, end_date, freq="D")
    annualized_percent = known_value_by_day(versions, calendar)
    if annualized_percent[0] is None:
        first_observation = pd.Timestamp(versions["date"].min()).date()
        raise ValueError(
            f"start date {start_date.isoformat()} precedes the first day a FRED {DGS3MO} "
            f"value is known (observation {first_observation.isoformat()}); the "
            "risk-free rate is never filled backwards, so the window must start after it"
        )
    return pd.Series(annualized_percent, index=calendar) / 100.0 / TRADING_DAYS_PER_YEAR


def known_value_by_day(versions: pd.DataFrame, days: pd.DatetimeIndex) -> list[float | None]:
    """For each of `days` (ascending), the value of the latest observation
    with a value in `lakehouse_loader._select_fred_vintages(versions, day)`,
    or `None` when it holds none: one ordered pass over `versions`, the
    `_fetch_fred_versions` rows (`date`, `value`, `known_at`, in that
    fetch's order), instead of one selection per day.

    A version is in force for its observation date from the first day
    after its `known_at` until a later version of the date is; of versions
    becoming known together, the last in `versions`' order is. An
    observation whose version in force has no value is skipped, so the
    previous observation's value carries."""
    observed = pd.to_datetime(versions["date"]).tolist()
    known_at = pd.to_datetime(versions["known_at"]).tolist()
    values = versions["value"].astype("float64").tolist()
    # Positions by known_at; the sort is stable, so versions becoming known
    # on the same date stay in `versions`' order.
    arrival = sorted(range(len(versions)), key=known_at.__getitem__)
    in_force: dict[pd.Timestamp, float] = {}
    # Max-heap (by negated nanoseconds) of observation dates whose version in
    # force had a value when pushed; an entry whose date has since lost it is
    # discarded on reaching the top.
    dated: list[tuple[int, pd.Timestamp]] = []
    result: list[float | None] = []
    next_arrival = 0
    for day in days:
        while next_arrival < len(arrival) and known_at[arrival[next_arrival]] < day:
            position = arrival[next_arrival]
            in_force[observed[position]] = values[position]
            if not math.isnan(values[position]):
                heapq.heappush(dated, (-observed[position].value, observed[position]))
            next_arrival += 1
        while dated and math.isnan(in_force[dated[0][1]]):
            heapq.heappop(dated)
        result.append(in_force[dated[0][1]] if dated else None)
    return result


def load_fx_policy_rates(
    assets: list[str], start_date: date, end_date: date
) -> pd.DataFrame | None:
    """The BIS central-bank policy rates of every currency of the spot FX
    pairs among `assets` (`data/instruments.currency_pair`), as
    `load_policy_rates_from_lakehouse` reads them for `start_date` to
    `end_date`: annual decimals, one column per currency, one row per
    observation date. This is the backtest engine's `policy_rates`; a
    pair's carry on a day is its base currency's rate minus its quote
    currency's, each the latest observation on or before that day.

    Returns `None` when `assets` holds no FX pair. Raises `ValueError`
    naming every currency with no observation in the window, so a pair is
    never carried at an invented rate."""
    currencies = sorted(
        {
            currency
            for asset in assets
            if (pair := currency_pair(asset)) is not None
            for currency in (pair.base, pair.quote)
        }
    )
    if not currencies:
        return None
    rates = load_policy_rates_from_lakehouse(currencies, start_date, end_date)
    missing = [
        currency
        for currency in currencies
        if rates is None or currency not in rates.columns or rates[currency].isna().all()
    ]
    if rates is None or missing:
        raise ValueError(
            f"lakehouse has no BIS policy rate for {missing} over {start_date.isoformat()} to "
            f"{end_date.isoformat()}; FX carry cannot be accrued without it"
        )
    return rates
