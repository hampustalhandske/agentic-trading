"""Unit tests for agentic_trading.data.ticker_catalog."""

from agentic_trading.data.ticker_catalog import (
    CATALOG,
    TICKER_GROUPS,
    AssetClass,
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
