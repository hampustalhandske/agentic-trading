# option_marks.py

"""Daily marks for listed option contracts, from the lakehouse option
chain.

A contract is marked from the latest chain snapshot on or before the
mark date within `settings.options.max_snapshot_age_days`. A same-day
snapshot is used as quoted; an older one is re-marked with quantcore
Black-Scholes at the snapshot's implied volatility and today's spot,
keeping the snapshot's relative bid/ask spread. On or after expiry the
mark is intrinsic value. Chain queries are cached in-process per
(underlying, as_of_date) so marking several contracts on one underlying
costs one lakehouse query per day.
"""

import math
from datetime import date
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel
from quantcore.pricing.black_scholes import black_scholes_call, black_scholes_put
from quantcore.pricing.greeks import bs_delta, bs_gamma, bs_theta, bs_vega
from quantcore.pricing.options_positions import option_roll_schedule

from agentic_trading.agents.schemas import OptionContract
from agentic_trading.config import settings
from agentic_trading.data.loaders.lakehouse_loader import (
    load_fred_series_from_lakehouse,
    load_option_quote_from_lakehouse,
    load_options_snapshot_from_lakehouse,
)
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

_RATE_SERIES_ID = "DGS3MO"
_DAYS_PER_YEAR = 365.0

# Latest chain snapshot per (underlying, as_of_date); only the most recent
# as_of_date per underlying is kept, since a backtest walks forward.
_CHAIN_CACHE: dict[tuple[str, date], pd.DataFrame | None] = {}
_RATE_CACHE: dict[date, float] = {}


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
        for stale_key in [k for k in _CHAIN_CACHE if k[0] == underlying]:
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


def _expired_mark(contract: OptionContract, as_of_date: date, spot: float) -> OptionMark:
    intrinsic = _intrinsic(contract, spot)
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
    rows = snapshot[
        (snapshot["expiration"] == pd.Timestamp(contract.expiry))
        & (snapshot["option_type"] == contract.option_type)
        & ((snapshot["strike"] - contract.strike).abs() < 1e-9)
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

    On or after `contract.expiry` the mark is intrinsic value (no chain
    lookup). Otherwise the contract is looked up in the latest snapshot
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


def list_expiries(underlying: str, as_of_date: date) -> list[date]:
    """Expiries after `as_of_date` listed in the latest snapshot within
    the age window, ascending; empty when there is no usable snapshot."""
    snapshot = _snapshot(underlying, as_of_date)
    if snapshot is None:
        return []
    expiries = {pd.Timestamp(value).date() for value in snapshot["expiration"].unique()}
    return sorted(expiry for expiry in expiries if expiry > as_of_date)


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


def select_expiry(underlying: str, as_of_date: date, target_tenor_days: int) -> date | None:
    """The listed expiry nearest `as_of_date + target_tenor_days`
    (quantcore `option_roll_schedule` over `list_expiries`), or `None`
    when nothing is listed."""
    expiries = list_expiries(underlying, as_of_date)
    if not expiries:
        return None
    selected = option_roll_schedule(as_of_date, target_tenor_days, expiries)
    return date(selected.year, selected.month, selected.day)
