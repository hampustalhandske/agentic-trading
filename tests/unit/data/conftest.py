# conftest.py

"""Resets `lakehouse_loader`'s cached per-thread DuckDB connection before
every test under this directory, so a test that monkeypatches
`duckdb.connect` always sees its own fake connection rather than one left
over (real or fake) from an earlier test on the same thread."""

import pytest

from agentic_trading.data.loaders.lakehouse_loader import reset_connection_cache


@pytest.fixture(autouse=True)
def _reset_lakehouse_connection_cache() -> None:
    reset_connection_cache()
