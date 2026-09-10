"""Unit tests for agentic_trading.data.ticker_catalog.

classify_via_yfinance() is mocked at the module boundary (agentic_trading.
data.ticker_catalog.yf.Ticker), mirroring test_loader.py's convention — no
real network calls in this suite.
"""

from unittest.mock import MagicMock, patch

from agentic_trading.data.ticker_catalog import (
    CATALOG,
    TICKER_GROUPS,
    AssetClass,
    classify_via_yfinance,
    get_ticker_info,
    tickers_by_asset_class,
)

_ALL_ASSET_CLASSES: tuple[AssetClass, ...] = ("stock", "index", "commodity", "forex", "crypto")


def test_catalog_is_non_empty() -> None:
    assert len(CATALOG) > 0


def test_catalog_has_no_duplicate_symbols() -> None:
    symbols = [entry.symbol for entry in CATALOG]
    assert len(symbols) == len(set(symbols))


def test_every_asset_class_is_represented_at_least_once() -> None:
    """Part A's seed set must cover all five classes, not just stocks."""
    represented = {entry.asset_class for entry in CATALOG}
    assert represented == set(_ALL_ASSET_CLASSES)


def test_get_ticker_info_returns_the_matching_entry() -> None:
    entry = get_ticker_info("AAPL")

    assert entry is not None
    assert entry.symbol == "AAPL"
    assert entry.asset_class == "stock"


def test_get_ticker_info_returns_none_for_unknown_symbol() -> None:
    assert get_ticker_info("NOT_IN_THE_CATALOG") is None


def test_tickers_by_asset_class_filters_to_exactly_that_class() -> None:
    for asset_class in _ALL_ASSET_CLASSES:
        entries = tickers_by_asset_class(asset_class)

        assert entries, f"expected at least one seed ticker for {asset_class}"
        assert all(entry.asset_class == asset_class for entry in entries)


def test_options_are_not_a_represented_asset_class() -> None:
    """See the module docstring: options are contracts on an underlying
    ticker, not a standalone catalogable asset class."""
    assert "option" not in {entry.asset_class for entry in CATALOG}


def test_ticker_groups_are_non_empty_and_reference_stock_symbols() -> None:
    catalog_symbols = {entry.symbol for entry in CATALOG}

    assert len(TICKER_GROUPS) > 0
    for group in TICKER_GROUPS:
        assert group.symbols
        assert set(group.symbols) <= catalog_symbols


@patch("agentic_trading.data.ticker_catalog.yf.Ticker")
def test_classify_via_yfinance_maps_known_quote_types(mock_ticker_cls: MagicMock) -> None:
    mock_ticker_cls.return_value.info = {"quoteType": "CRYPTOCURRENCY"}

    result = classify_via_yfinance("BTC-USD")

    assert result == "crypto"


@patch("agentic_trading.data.ticker_catalog.yf.Ticker")
def test_classify_via_yfinance_returns_none_for_unmapped_quote_type(
    mock_ticker_cls: MagicMock,
) -> None:
    """OPTION has no mapped class — see the module docstring on why options are excluded."""
    mock_ticker_cls.return_value.info = {"quoteType": "OPTION"}

    result = classify_via_yfinance("AAPL250117C00150000")

    assert result is None


@patch("agentic_trading.data.ticker_catalog.yf.Ticker")
def test_classify_via_yfinance_returns_none_on_lookup_failure(mock_ticker_cls: MagicMock) -> None:
    mock_ticker_cls.side_effect = Exception("network error")

    result = classify_via_yfinance("NOT_A_REAL_TICKER")

    assert result is None
