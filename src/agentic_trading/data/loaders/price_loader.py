# price_loader.py

"""Price history entry point for the `data/loaders/` package.

Re-exports `agentic_trading.data.loader`, which already implements the
yfinance-primary/Stooq-fallback/synthetic-fallback chain — kept in its
original module rather than duplicated here so there remains exactly one
implementation and one set of unit tests for it.
"""

from agentic_trading.data.loader import (
    load_price_history,
    load_price_history_with_fallback,
)

__all__ = ["load_price_history", "load_price_history_with_fallback"]
