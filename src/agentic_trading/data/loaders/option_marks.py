# option_marks.py

"""Daily marks for listed option contracts, from the lakehouse option
chain.

A contract is marked from the latest chain snapshot on or before the
mark date within `settings.options.max_snapshot_age_days`. A same-day
snapshot is used as quoted; an older one is re-marked with quantcore
Black-Scholes at the snapshot's implied volatility and today's spot,
keeping the snapshot's relative bid/ask spread. On or after expiry the
mark is intrinsic value against the underlying's close on the last
session on or before the expiry. Chain queries are cached in-process per
(underlying, as_of_date) so marking several contracts on one underlying
costs one lakehouse query per day.

Marks read the chain as of the mark date. Contract selection
(`select_expiry`, `nearest_strike`, `list_expiries`) reads it as of
whatever date the caller passes; a decision-time caller passes
`published_chain_date(day)` so it only sees chains already published,
and `after=day` so it never picks an expiry on or before its decision
day.
"""

import math
from datetime import date, timedelta
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel
from quantcore.pricing.black_scholes import black_scholes_call, black_scholes_put
from quantcore.pricing.greeks import bs_delta, bs_gamma, bs_theta, bs_vega
from quantcore.pricing.options_positions import option_roll_schedule

from agentic_trading.agents.schemas import OptionContract
from agentic_trading.config import settings
from agentic_trading.data.loaders.lakehouse_loader import (
    STRIKE_MATCH_TOLERANCE,
    load_fred_series_from_lakehouse,
    load_option_quote_from_lakehouse,
    load_options_snapshot_from_lakehouse,
    load_raw_close_from_lakehouse,
)
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

_RATE_SERIES_ID = "DGS3MO"
_DAYS_PER_YEAR = 365.0
# An expiry settles against the last close at most this many calendar days
# before it: an expiry falling on a holiday or a long market closure still
# finds the prior session, while a ticker whose prices stopped weeks
# earlier falls back to the caller's spot instead of a stale print.
SETTLEMENT_CLOSE_LOOKBACK_DAYS = 10

# Latest chain snapshot per (underlying, as_of_date). A backtest walks
# forward, reading each day's chain for marks and the published (lagged)
# chain for selection, so querying a new as_of_date drops that
# underlying's entries more than the publication lag older than it; a
# day's selection then reuses the chain cached by an earlier day's marks.
_CHAIN_CACHE: dict[tuple[str, date], pd.DataFrame | None] = {}
_RATE_CACHE: dict[date, float] = {}
# Settlement close per (underlying, expiry); `None` when the lake has none.
_SETTLEMENT_CLOSE_CACHE: dict[tuple[str, date], float | None] = {}


class OptionMark(BaseModel):
    """One contract's mark on one day: quotes, greeks and how the mark was
    produced. `tradeable` is `True` only when the mark rests on a live
    quote with a positive bid inside the snapshot-age window."""

    contract: OptionContract
    as_of_date: date
    snapshot_date: date | None
    mid: float
    bid: float
    ask: float
    implied_volatility: float
    delta: float
    gamma: float
    vega: float
    theta: float
    is_model_mark: bool
    is_expired: bool
    tradeable: bool


def clear_cache() -> None:
    _QUOTE_CACHE.clear()
    _CHAIN_CACHE.clear()
    _RATE_CACHE.clear()
    _SETTLEMENT_CLOSE_CACHE.clear()


def published_chain_date(day: date) -> date:
    """The latest chain snapshot date an agent deciding on `day` can see:
    `day` minus `settings.options.publication_lag_days` calendar days."""
    return day - timedelta(days=settings.options.publication_lag_days)


def _query_chain(underlying: str, as_of_date: date) -> pd.DataFrame | None:
    """Every row of the latest chain snapshot for `underlying` dated on or
    before `as_of_date` within the snapshot-age window, or `None`."""
    return load_options_snapshot_from_lakehouse(
        underlying, as_of_date, lookback_days=settings.options.max_snapshot_age_days
    )


def _query_rate(as_of_date: date) -> float:
    """Continuously-compounded-style annual risk-free rate from the latest
    DGS3MO observation on or before `as_of_date`; 0.0 when none exists."""
    series = load_fred_series_from_lakehouse(_RATE_SERIES_ID, as_of_date)
    if series is None:
        return 0.0
    series = series.dropna()
    if series.empty:
        return 0.0
    return float(series.iloc[-1]) / 100.0


def _rate(as_of_date: date) -> float:
    if as_of_date not in _RATE_CACHE:
        _RATE_CACHE.clear()
        _RATE_CACHE[as_of_date] = _query_rate(as_of_date)
    return _RATE_CACHE[as_of_date]


def _snapshot(underlying: str, as_of_date: date) -> pd.DataFrame | None:
    """The cached latest snapshot for `underlying` as of `as_of_date`,
    `None` when there is none or it is older than the age window."""
    key = (underlying, as_of_date)
    if key not in _CHAIN_CACHE:
        oldest_kept = published_chain_date(as_of_date)
        for stale_key in [k for k in _CHAIN_CACHE if k[0] == underlying and k[1] < oldest_kept]:
            del _CHAIN_CACHE[stale_key]
        frame = _query_chain(underlying, as_of_date)
        if frame is not None and not frame.empty:
            frame = frame.copy()
            frame["date"] = pd.to_datetime(frame["date"])
            frame["expiration"] = pd.to_datetime(frame["expiration"])
        else:
            frame = None
        _CHAIN_CACHE[key] = frame
    frame = _CHAIN_CACHE[key]
    if frame is None:
        return None
    snapshot_date = frame["date"].max().date()
    age = (as_of_date - snapshot_date).days
    if age < 0 or age > settings.options.max_snapshot_age_days:
        return None
    return frame


def _finite(value: Any, default: float) -> float:
    if value is None:
        return default
    number = float(value)
    return number if math.isfinite(number) else default


def _intrinsic(contract: OptionContract, spot: float) -> float:
    if contract.option_type == "call":
        return max(spot - contract.strike, 0.0)
    return max(contract.strike - spot, 0.0)


def _settlement_close(underlying: str, expiry: date) -> float | None:
    """The cached unadjusted close of `underlying` on its last session on
    or before `expiry` within `SETTLEMENT_CLOSE_LOOKBACK_DAYS`, or `None`."""
    key = (underlying, expiry)
    if key not in _SETTLEMENT_CLOSE_CACHE:
        _SETTLEMENT_CLOSE_CACHE[key] = load_raw_close_from_lakehouse(
            underlying, expiry, SETTLEMENT_CLOSE_LOOKBACK_DAYS
        )
    return _SETTLEMENT_CLOSE_CACHE[key]


def _expired_mark(contract: OptionContract, as_of_date: date, spot: float) -> OptionMark:
    """Intrinsic value (bid = ask = mid, zero greeks, untradeable) against
    the settlement close from `_settlement_close`, or against `spot` when
    the lake has no close near the expiry."""
    settlement = _settlement_close(contract.underlying, contract.expiry)
    intrinsic = _intrinsic(contract, spot if settlement is None else settlement)
    return OptionMark(
        contract=contract,
        as_of_date=as_of_date,
        snapshot_date=None,
        mid=intrinsic,
        bid=intrinsic,
        ask=intrinsic,
        implied_volatility=0.0,
        delta=0.0,
        gamma=0.0,
        vega=0.0,
        theta=0.0,
        is_model_mark=False,
        is_expired=True,
        tradeable=False,
    )


def _contract_row(snapshot: pd.DataFrame, contract: OptionContract) -> pd.Series | None:
    """The first `snapshot` row with `contract`'s expiration and option
    type and a strike within `STRIKE_MATCH_TOLERANCE` of its strike, or
    `None` when no row matches."""
    rows = snapshot[
        (snapshot["expiration"] == pd.Timestamp(contract.expiry))
        & (snapshot["option_type"] == contract.option_type)
        & ((snapshot["strike"] - contract.strike).abs() < STRIKE_MATCH_TOLERANCE)
    ]
    if rows.empty:
        return None
    return rows.iloc[0]


def _model_mark(
    contract: OptionContract,
    as_of_date: date,
    spot: float,
    snapshot_date: date,
    implied_volatility: float,
    quoted_bid: float,
    quoted_ask: float,
) -> OptionMark:
    """Black-Scholes re-mark of a stale quote at today's `spot`, with the
    quoted bid/ask spread carried over as a relative spread around the
    model mid. A quoted zero bid stays a zero bid."""
    time_to_expiry = (contract.expiry - as_of_date).days / _DAYS_PER_YEAR
    rate = _rate(as_of_date)
    pricer = black_scholes_call if contract.option_type == "call" else black_scholes_put
    mid = pricer(spot, contract.strike, rate, implied_volatility, time_to_expiry)
    quoted_mid = (quoted_bid + quoted_ask) / 2.0
    relative_spread = (quoted_ask - quoted_bid) / quoted_mid if quoted_mid > 0.0 else 0.0
    bid = 0.0 if quoted_bid <= 0.0 else max(mid * (1.0 - relative_spread / 2.0), 0.0)
    ask = mid * (1.0 + relative_spread / 2.0)
    option_type: Literal["call", "put"] = contract.option_type
    return OptionMark(
        contract=contract,
        as_of_date=as_of_date,
        snapshot_date=snapshot_date,
        mid=mid,
        bid=bid,
        ask=ask,
        implied_volatility=implied_volatility,
        delta=bs_delta(
            spot, contract.strike, rate, implied_volatility, time_to_expiry, option_type
        ),
        gamma=bs_gamma(spot, contract.strike, rate, implied_volatility, time_to_expiry),
        vega=bs_vega(spot, contract.strike, rate, implied_volatility, time_to_expiry),
        theta=bs_theta(
            spot, contract.strike, rate, implied_volatility, time_to_expiry, option_type
        ),
        is_model_mark=True,
        is_expired=False,
        tradeable=bid > 0.0,
    )


_QUOTE_CACHE: dict[tuple[str, date], pd.Series | None] = {}


def _last_quote_row(contract: OptionContract, as_of_date: date) -> pd.Series | None:
    """The contract's own latest quote within `settings.options.max_mark_age_days`
    when the latest snapshot no longer lists its expiry (the historical
    chain carries three expiries per snapshot and rotates the near ones)."""
    key = (contract.contract_id, as_of_date)
    if key not in _QUOTE_CACHE:
        if len(_QUOTE_CACHE) > 4096:
            _QUOTE_CACHE.clear()
        frame = load_option_quote_from_lakehouse(
            contract.underlying,
            contract.expiry,
            contract.option_type,
            contract.strike,
            as_of_date,
            settings.options.max_mark_age_days,
        )
        _QUOTE_CACHE[key] = None if frame is None or frame.empty else frame.iloc[0]
    return _QUOTE_CACHE[key]


def load_option_mark(contract: OptionContract, as_of_date: date, spot: float) -> OptionMark | None:
    """`contract`'s mark on `as_of_date` at underlying price `spot`.

    On or after `contract.expiry` the mark is intrinsic value against the
    underlying's close on the last session on or before the expiry,
    falling back to `spot` when the lake has none (no chain lookup).
    Otherwise the contract is looked up in the latest snapshot
    within the age window: a same-day snapshot is used as quoted, an
    older one is re-marked with Black-Scholes at the snapshot's implied
    volatility and today's `spot`. `None` when no snapshot in the window
    quotes the contract."""
    if as_of_date >= contract.expiry:
        return _expired_mark(contract, as_of_date, spot)
    snapshot = _snapshot(contract.underlying, as_of_date)
    row = _contract_row(snapshot, contract) if snapshot is not None else None
    if row is None:
        row = _last_quote_row(contract, as_of_date)
    if row is None:
        return None
    snapshot_date = pd.Timestamp(row["date"]).date()
    bid = max(_finite(row["bid"], 0.0), 0.0)
    ask = max(_finite(row["ask"], bid), bid)
    implied_volatility = _finite(row["implied_volatility"], 0.0)
    if snapshot_date < as_of_date and implied_volatility > 0.0 and spot > 0.0:
        mark = _model_mark(contract, as_of_date, spot, snapshot_date, implied_volatility, bid, ask)
        # A quote older than the trading window still marks the position
        # but cannot be traded against.
        if (as_of_date - snapshot_date).days > settings.options.max_snapshot_age_days:
            return mark.model_copy(update={"tradeable": False})
        return mark
    return OptionMark(
        contract=contract,
        as_of_date=as_of_date,
        snapshot_date=snapshot_date,
        mid=(bid + ask) / 2.0,
        bid=bid,
        ask=ask,
        implied_volatility=implied_volatility,
        delta=_finite(row["delta"], 0.0),
        gamma=_finite(row["gamma"], 0.0),
        vega=_finite(row["vega"], 0.0),
        theta=_finite(row["theta"], 0.0),
        is_model_mark=False,
        is_expired=False,
        tradeable=bid > 0.0,
    )


def list_expiries(underlying: str, as_of_date: date, after: date | None = None) -> list[date]:
    """Expiries strictly after `after` (default `as_of_date`) listed in
    the latest snapshot on or before `as_of_date` within the age window,
    ascending; empty when there is no usable snapshot."""
    snapshot = _snapshot(underlying, as_of_date)
    if snapshot is None:
        return []
    earliest_excluded = as_of_date if after is None else after
    expiries = {pd.Timestamp(value).date() for value in snapshot["expiration"].unique()}
    return sorted(expiry for expiry in expiries if expiry > earliest_excluded)


def nearest_strike(
    underlying: str,
    expiry: date,
    option_type: Literal["call", "put"],
    target_strike: float,
    as_of_date: date,
) -> float | None:
    """The listed strike with a positive bid closest to `target_strike`
    for `expiry`/`option_type`, or `None` when none is quoted."""
    snapshot = _snapshot(underlying, as_of_date)
    if snapshot is None:
        return None
    rows = snapshot[
        (snapshot["expiration"] == pd.Timestamp(expiry))
        & (snapshot["option_type"] == option_type)
        & (snapshot["bid"].fillna(0.0) > 0.0)
    ]
    if rows.empty:
        return None
    distance = (rows["strike"] - target_strike).abs()
    return _finite(rows.loc[distance.idxmin(), "strike"], target_strike)


def select_expiry(
    underlying: str, as_of_date: date, target_tenor_days: int, after: date | None = None
) -> date | None:
    """The listed expiry nearest `as_of_date + target_tenor_days`
    (quantcore `option_roll_schedule` over `list_expiries`) among those
    strictly after `after` (default `as_of_date`), or `None` when nothing
    qualifies. A caller reading a lagged chain passes its decision day as
    `after` so a contract expiring on or before that day is never chosen."""
    expiries = list_expiries(underlying, as_of_date, after)
    if not expiries:
        return None
    selected = option_roll_schedule(as_of_date, target_tenor_days, expiries)
    return date(selected.year, selected.month, selected.day)
