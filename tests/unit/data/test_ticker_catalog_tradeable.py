"""Every catalog entry and preset symbol is something a backtest can trade on lake data."""

import pytest

from agentic_trading.data.ticker_catalog import CATALOG, TICKER_GROUPS, get_ticker_info


def test_catalog_lists_no_untradeable_index() -> None:
    """Indices are offered through tradeable ETFs, never as `^` symbols with INDEX quote type."""
    for entry in CATALOG:
        assert not entry.symbol.startswith("^"), entry.symbol
        assert entry.quote_type != "INDEX", entry.symbol


@pytest.mark.parametrize(("symbol", "asset_class"), [("SPY", "index"), ("IWM", "index")])
def test_index_exposure_is_through_etfs(symbol: str, asset_class: str) -> None:
    """S&P 500 and Russell 2000 exposure is offered as SPY and IWM."""
    entry = get_ticker_info(symbol)

    assert entry is not None
    assert entry.asset_class == asset_class
    assert entry.quote_type == "ETF"


def test_yen_pair_uses_the_lake_symbol() -> None:
    """USD/JPY is catalogued as `USDJPY=X`, the symbol the lake carries."""
    assert get_ticker_info("JPY=X") is None
    entry = get_ticker_info("USDJPY=X")
    assert entry is not None
    assert entry.asset_class == "forex"


def test_presets_hold_only_tradeable_symbols() -> None:
    """No preset references an index symbol or `JPY=X`."""
    for group in TICKER_GROUPS:
        for symbol in group.symbols:
            assert not symbol.startswith("^"), (group.name, symbol)
            assert symbol != "JPY=X", group.name
