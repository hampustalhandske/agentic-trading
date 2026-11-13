# instruments.py

"""Instrument conventions read from a ticker, shared by the price loader,
the portfolio and the analysts.

A ticker ending in `=F` is a future on the lake root before the suffix
(`CL=F` is root `CL`). A six-letter ticker ending in `=X` is a spot FX pair,
its first three letters the base currency and the next three the quote
currency, the price being quote currency per unit of base (`USDJPY=X` is JPY
per USD). Both are marked instruments: the book holds them without paying
cash for them and settles every price move into cash (see
`backtest/portfolio.py::MarkedPosition`). Every other ticker is a cash
security.

The account is kept in `ACCOUNT_CURRENCY`. A pair is valued in it only when
one of its two currencies is the account currency."""

import re

from pydantic import BaseModel

ACCOUNT_CURRENCY = "USD"

_FUTURES_SUFFIX = "=F"
_FX_PATTERN = re.compile(r"^([A-Z]{3})([A-Z]{3})=X$")


class CurrencyPair(BaseModel):
    """An FX pair's base and quote currency (ISO codes); its price is quote
    currency per unit of base."""

    base: str
    quote: str


def futures_root(ticker: str) -> str | None:
    """The lake root of futures ticker `ticker` (`"CL"` for `"CL=F"`), or
    `None` when `ticker` is not a future."""
    if ticker.endswith(_FUTURES_SUFFIX) and len(ticker) > len(_FUTURES_SUFFIX):
        return ticker[: -len(_FUTURES_SUFFIX)]
    return None


def currency_pair(ticker: str) -> CurrencyPair | None:
    """The base and quote currency of spot FX ticker `ticker`, or `None`
    when `ticker` is not one."""
    match = _FX_PATTERN.match(ticker)
    if match is None:
        return None
    return CurrencyPair(base=match.group(1), quote=match.group(2))


def is_marked(ticker: str) -> bool:
    """Whether `ticker` is a future or a spot FX pair."""
    return futures_root(ticker) is not None or currency_pair(ticker) is not None


def _account_pair(ticker: str) -> CurrencyPair | None:
    """`ticker`'s pair, `None` for a future; raises `ValueError` for a pair
    with no leg in `ACCOUNT_CURRENCY`."""
    pair = currency_pair(ticker)
    if pair is not None and ACCOUNT_CURRENCY not in (pair.base, pair.quote):
        raise ValueError(
            f"{ticker} has no {ACCOUNT_CURRENCY} leg; only pairs against the account currency "
            "can be valued"
        )
    return pair


def unit_value_usd(ticker: str, price: float) -> float:
    """The USD value of one unit of a marked position in `ticker` at
    `price`: `price` for a future and for a USD-quoted pair (one unit is
    one unit of base currency, worth `price` USD), and 1.0 for a USD-base
    pair (one unit is one USD). Raises `ValueError` for a pair with no USD
    leg."""
    pair = _account_pair(ticker)
    return 1.0 if pair is not None and pair.base == ACCOUNT_CURRENCY else price


def quote_unit_value_usd(ticker: str, price: float) -> float:
    """The USD value of one unit of `ticker`'s quote currency at `price`:
    1.0 for a future and for a USD-quoted pair, `1 / price` for a USD-base
    pair. A position's profit in quote currency times this is its profit in
    USD. Raises `ValueError` for a pair with no USD leg."""
    pair = _account_pair(ticker)
    return 1.0 / price if pair is not None and pair.base == ACCOUNT_CURRENCY else 1.0
