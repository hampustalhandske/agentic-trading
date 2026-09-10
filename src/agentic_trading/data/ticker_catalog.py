# ticker_catalog.py

"""Curated, extendable ticker categorization lookup.

yfinance has no endpoint that enumerates "all tickers," so this module takes
the opposite approach: a small, hand-picked seed set of tickers, each
classified once via yfinance's own per-ticker metadata (`Ticker.info`'s
`quoteType` field) into one of five asset classes, and stored below as a
static table rather than re-queried from yfinance on every import or app
start — that would make every dashboard boot slow, flaky, and rate-limit
prone for data that essentially never changes. `CATALOG` is the source of
truth; `classify_via_yfinance()` is a dev-time helper for extending it.

Options are deliberately excluded from this catalog. An option is a
contract on an underlying ticker (a call/put with its own strike and
expiry) — yfinance reports `quoteType == "OPTION"` for one specific
contract's symbol (e.g. "AAPL250117C00150000"), not for "the options on
AAPL" as a durable, catalogable identity the way a stock/index/commodity
future/forex pair/crypto pair is. There is no stable ticker to classify the
way there is for the other five classes, so forcing an "option" row into
this table would misrepresent a derivative as a sixth parallel asset class
instead of what it actually is: a contract layered on top of one of the
other five.

This catalog is what will let later analysts restrict themselves to a
single asset class (e.g. an FX-only or crypto-only analyst) by filtering
`CATALOG` rather than hardcoding ticker lists.
"""

from __future__ import annotations

from typing import Literal

import yfinance as yf
from pydantic import BaseModel

AssetClass = Literal["stock"]

# yfinance's Ticker.info["quoteType"] values, as observed for each class in
# this catalog, mapped to our five asset classes. Any quoteType not listed
# here (e.g. "OPTION", "MUTUALFUND") has no home in this catalog — see the
# module docstring for why options specifically are excluded.
_QUOTE_TYPE_TO_ASSET_CLASS: dict[str, AssetClass] = {"EQUITY": "stock"}


class TickerInfo(BaseModel):
    """One catalog entry: a ticker symbol, its display name, and the asset
    class it was classified into from yfinance's `quoteType`."""

    symbol: str
    name: str
    asset_class: AssetClass
    quote_type: str


class TickerGroup(BaseModel):
    """A named, curated basket of stock tickers — a preset for the GUI's
    ticker-group selector (see gui/dashboard.py)."""

    name: str
    description: str
    symbols: list[str]


# Classified from live yfinance Ticker.info lookups (quoteType field) at the
# time this catalog was built. Re-run classify_via_yfinance() on a symbol to
# verify or refresh its class before editing an entry below.
CATALOG: tuple[TickerInfo, ...] = (
    # --- stock (quoteType: EQUITY) ---
    TickerInfo(symbol="AAPL", name="Apple Inc.", asset_class="stock", quote_type="EQUITY"),
    TickerInfo(
        symbol="MSFT", name="Microsoft Corporation", asset_class="stock", quote_type="EQUITY"
    ),
    TickerInfo(symbol="GOOGL", name="Alphabet Inc.", asset_class="stock", quote_type="EQUITY"),
    TickerInfo(symbol="AMZN", name="Amazon.com, Inc.", asset_class="stock", quote_type="EQUITY"),
    TickerInfo(
        symbol="JPM", name="JP Morgan Chase & Co.", asset_class="stock", quote_type="EQUITY"
    ),
    TickerInfo(symbol="NVDA", name="NVIDIA Corporation", asset_class="stock", quote_type="EQUITY"),
    TickerInfo(symbol="TSLA", name="Tesla, Inc.", asset_class="stock", quote_type="EQUITY"),
    TickerInfo(
        symbol="META", name="Meta Platforms, Inc.", asset_class="stock", quote_type="EQUITY"
    ),
    TickerInfo(
        symbol="BAC", name="Bank of America Corporation", asset_class="stock", quote_type="EQUITY"
    ),
    TickerInfo(
        symbol="GS",
        name="Goldman Sachs Group, Inc. (The)",
        asset_class="stock",
        quote_type="EQUITY",
    ),
    TickerInfo(symbol="JNJ", name="Johnson & Johnson", asset_class="stock", quote_type="EQUITY"),
    TickerInfo(symbol="PFE", name="Pfizer, Inc.", asset_class="stock", quote_type="EQUITY"),
    TickerInfo(
        symbol="XOM",
        name="Exxon Mobil Corporation",
        asset_class="stock",
        quote_type="EQUITY",
    ),
    TickerInfo(symbol="CVX", name="Chevron Corporation", asset_class="stock", quote_type="EQUITY"),
    # --- stock: consumer discretionary ---
    TickerInfo(
        symbol="HD", name="Home Depot, Inc. (The)", asset_class="stock", quote_type="EQUITY"
    ),
    TickerInfo(symbol="NKE", name="Nike, Inc.", asset_class="stock", quote_type="EQUITY"),
    TickerInfo(
        symbol="MCD", name="McDonald's Corporation", asset_class="stock", quote_type="EQUITY"
    ),
    TickerInfo(
        symbol="SBUX", name="Starbucks Corporation", asset_class="stock", quote_type="EQUITY"
    ),
    # --- stock: communication services ---
    TickerInfo(
        symbol="DIS", name="Walt Disney Company (The)", asset_class="stock", quote_type="EQUITY"
    ),
    TickerInfo(symbol="NFLX", name="Netflix, Inc.", asset_class="stock", quote_type="EQUITY"),
    TickerInfo(
        symbol="VZ",
        name="Verizon Communications Inc.",
        asset_class="stock",
        quote_type="EQUITY",
    ),
    # --- stock: industrials ---
    TickerInfo(symbol="BA", name="Boeing Company (The)", asset_class="stock", quote_type="EQUITY"),
    TickerInfo(symbol="CAT", name="Caterpillar, Inc.", asset_class="stock", quote_type="EQUITY"),
    TickerInfo(
        symbol="UPS",
        name="United Parcel Service, Inc.",
        asset_class="stock",
        quote_type="EQUITY",
    ),
    # --- stock: consumer staples ---
    TickerInfo(
        symbol="PG",
        name="Procter & Gamble Company (The)",
        asset_class="stock",
        quote_type="EQUITY",
    ),
    TickerInfo(
        symbol="KO", name="Coca-Cola Company (The)", asset_class="stock", quote_type="EQUITY"
    ),
    TickerInfo(symbol="WMT", name="Walmart Inc.", asset_class="stock", quote_type="EQUITY"),
    TickerInfo(
        symbol="COST",
        name="Costco Wholesale Corporation",
        asset_class="stock",
        quote_type="EQUITY",
    ),
    # --- stock: utilities ---
    TickerInfo(symbol="NEE", name="NextEra Energy, Inc.", asset_class="stock", quote_type="EQUITY"),
    TickerInfo(
        symbol="DUK", name="Duke Energy Corporation", asset_class="stock", quote_type="EQUITY"
    ),
    # --- stock: real estate (REITs) ---
    TickerInfo(
        symbol="AMT", name="American Tower Corporation", asset_class="stock", quote_type="EQUITY"
    ),
    TickerInfo(symbol="PLD", name="Prologis, Inc.", asset_class="stock", quote_type="EQUITY"),
    # --- stock: materials ---
    TickerInfo(symbol="LIN", name="Linde plc", asset_class="stock", quote_type="EQUITY"),
    # --- stock: more tech / semiconductors ---
    TickerInfo(
        symbol="AMD", name="Advanced Micro Devices, Inc.", asset_class="stock", quote_type="EQUITY"
    ),
    TickerInfo(symbol="INTC", name="Intel Corporation", asset_class="stock", quote_type="EQUITY"),
    TickerInfo(symbol="AVGO", name="Broadcom Inc.", asset_class="stock", quote_type="EQUITY"),
    TickerInfo(symbol="ORCL", name="Oracle Corporation", asset_class="stock", quote_type="EQUITY"),
    TickerInfo(symbol="CRM", name="Salesforce, Inc.", asset_class="stock", quote_type="EQUITY"),
    TickerInfo(symbol="ADBE", name="Adobe Inc.", asset_class="stock", quote_type="EQUITY"),
    # --- stock: payments / more financials ---
    TickerInfo(symbol="V", name="Visa Inc.", asset_class="stock", quote_type="EQUITY"),
    TickerInfo(
        symbol="MA", name="Mastercard Incorporated", asset_class="stock", quote_type="EQUITY"
    ),
)

_BY_SYMBOL: dict[str, TickerInfo] = {entry.symbol: entry for entry in CATALOG}

# Preset baskets for the GUI's ticker-group selector (see gui/dashboard.py).
# Most are stock-only desk baskets; the multi-asset basket is intended for
# diversified portfolio experiments rather than a single-sector desk.
TICKER_GROUPS: tuple[TickerGroup, ...] = (
    TickerGroup(
        name="Original Five",
        description="The desk's original five-name basket.",
        symbols=["AAPL", "MSFT", "GOOGL", "AMZN", "JPM"],
    ),
    TickerGroup(
        name="Mega-Cap Tech",
        description="Large-cap technology and growth names.",
        symbols=["AAPL", "MSFT", "GOOGL", "META", "NVDA", "TSLA"],
    ),
    TickerGroup(
        name="Financials",
        description="Money-center banks and payment networks.",
        symbols=["JPM", "BAC", "GS", "V", "MA"],
    ),
    TickerGroup(
        name="Value & Defensive",
        description="Healthcare and energy blue chips.",
        symbols=["JNJ", "PFE", "XOM", "CVX"],
    ),
    TickerGroup(
        name="Consumer & Retail",
        description="Consumer discretionary and staples retail names.",
        symbols=["HD", "NKE", "MCD", "SBUX", "WMT", "COST"],
    ),
    TickerGroup(
        name="Communication & Media",
        description="Streaming, telecom, and social/search platforms.",
        symbols=["DIS", "NFLX", "VZ", "GOOGL", "META"],
    ),
    TickerGroup(
        name="Industrials & Materials",
        description="Manufacturing, logistics, and basic materials.",
        symbols=["BA", "CAT", "UPS", "LIN"],
    ),
    TickerGroup(
        name="Semiconductors",
        description="Chipmakers powering the AI/compute buildout.",
        symbols=["NVDA", "AMD", "INTC", "AVGO"],
    ),
    TickerGroup(
        name="Enterprise Software",
        description="Cloud, CRM, and creative-software platforms.",
        symbols=["MSFT", "ORCL", "CRM", "ADBE"],
    ),
    TickerGroup(
        name="Dividend & Income",
        description="Utilities, REITs, and consumer staples with steady payouts.",
        symbols=["NEE", "DUK", "AMT", "PLD", "PG", "KO"],
    ),
    TickerGroup(
        name="Broad Diversified 15",
        description="A wide, 15-name spread across every sector in the catalog.",
        symbols=[
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "JPM",
            "JNJ",
            "XOM",
            "HD",
            "DIS",
            "CAT",
            "PG",
            "NEE",
            "AMT",
            "LIN",
            "V",
        ],
    ),
    TickerGroup(
        name="Diversified Multi-Asset 20",
        description="Equities, indices, commodities, FX, and crypto for portfolio experiments.",
        symbols=[
            "AAPL",
            "MSFT",
            "JPM",
            "JNJ",
            "XOM",
            "CAT",
            "^GSPC",
            "^RUT",
            "GC=F",
            "CL=F",
            "SI=F",
            "NG=F",
            "EURUSD=X",
            "GBPUSD=X",
            "JPY=X",
            "AUDUSD=X",
            "BTC-USD",
            "ETH-USD",
            "NEE",
            "PG",
        ],
    ),
)


def get_ticker_info(symbol: str) -> TickerInfo | None:
    """Look up one symbol's catalog entry, or None if it isn't catalogued."""
    return _BY_SYMBOL.get(symbol)


def tickers_by_asset_class(asset_class: AssetClass) -> list[TickerInfo]:
    """Every catalogued ticker in one asset class, in catalog order."""
    return [entry for entry in CATALOG if entry.asset_class == asset_class]


def classify_via_yfinance(symbol: str) -> AssetClass | None:
    """Dev-time helper for extending the catalog: looks up `symbol`'s live
    quoteType via yfinance and maps it to an AssetClass. Returns None if the
    lookup fails or the quoteType has no mapped class (e.g. "OPTION" — see
    the module docstring). Not called anywhere at import time or app
    startup; the catalog above is the static source of truth. Use this to
    find a new symbol's class, then add a TickerInfo entry for it."""
    try:
        quote_type = yf.Ticker(symbol).info.get("quoteType")
    except Exception:
        return None
    if quote_type is None:
        return None
    return _QUOTE_TYPE_TO_ASSET_CLASS.get(quote_type)
