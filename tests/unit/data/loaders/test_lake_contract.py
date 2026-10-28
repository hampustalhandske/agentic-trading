# test_lake_contract.py

"""Conformance of the lakehouse loaders with the lake's frozen data contract.

`tests/fixtures/lakehouse_contract_v1.json` lists every view the loaders read
and each view's column names and DuckDB types. Registering those views as
empty typed tables and running every loader proves each SQL statement binds
against the contract's names and types.
"""

import inspect
import json
import re
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from agentic_trading.data.loaders import lakehouse_loader

_CONTRACT_PATH = Path(__file__).parents[3] / "fixtures" / "lakehouse_contract_v1.json"
_CONTRACT = json.loads(_CONTRACT_PATH.read_text())
_AS_OF = date(2024, 3, 15)
_START = date(2024, 1, 2)
_END = date(2024, 3, 15)

_EMPTY_CASES: dict[str, Callable[[], Any]] = {
    "load_sp500_constituents_asof": lambda: lakehouse_loader.load_sp500_constituents_asof(_AS_OF),
    "load_options_snapshot_from_lakehouse": lambda: (
        lakehouse_loader.load_options_snapshot_from_lakehouse("AAPL", _AS_OF)
    ),
    "load_cot_positioning_from_lakehouse[disaggregated]": lambda: (
        lakehouse_loader.load_cot_positioning_from_lakehouse("GC=F", _AS_OF)
    ),
    "load_cot_positioning_from_lakehouse[tff]": lambda: (
        lakehouse_loader.load_cot_positioning_from_lakehouse("EURUSD=X", _AS_OF)
    ),
    "load_dividends_from_lakehouse": lambda: lakehouse_loader.load_dividends_from_lakehouse(
        "AAPL", _AS_OF
    ),
    "load_earnings_surprise_from_lakehouse": lambda: (
        lakehouse_loader.load_earnings_surprise_from_lakehouse("AAPL", _AS_OF)
    ),
    "load_sic_code_from_lakehouse": lambda: lakehouse_loader.load_sic_code_from_lakehouse("AAPL"),
    "load_price_history_from_lakehouse": lambda: lakehouse_loader.load_price_history_from_lakehouse(
        ["AAPL", "MSFT"], _START, _END
    ),
    "load_fred_series_from_lakehouse": lambda: lakehouse_loader.load_fred_series_from_lakehouse(
        "DGS3MO", _AS_OF
    ),
    "load_fama_french_factors_from_lakehouse[3factor]": lambda: (
        lakehouse_loader.load_fama_french_factors_from_lakehouse(_AS_OF, "3factor")
    ),
    "load_fama_french_factors_from_lakehouse[5factor]": lambda: (
        lakehouse_loader.load_fama_french_factors_from_lakehouse(_AS_OF, "5factor")
    ),
    "load_company_facts_from_lakehouse": lambda: lakehouse_loader.load_company_facts_from_lakehouse(
        "AAPL", _AS_OF
    ),
    "load_short_interest_from_lakehouse": lambda: (
        lakehouse_loader.load_short_interest_from_lakehouse("AAPL", _AS_OF)
    ),
    "load_insider_transactions_from_lakehouse": lambda: (
        lakehouse_loader.load_insider_transactions_from_lakehouse("AAPL", _AS_OF)
    ),
    "load_option_quote_from_lakehouse": lambda: lakehouse_loader.load_option_quote_from_lakehouse(
        "AAPL", date(2024, 6, 21), "call", 190.0, _AS_OF, 10
    ),
}


def _register_contract(fake_lake: Any) -> None:
    """Creates every contract view as an empty table with its exact column types."""
    for view, columns in _CONTRACT["views"].items():
        quoted = {
            f'"{name}"': ("VARCHAR" if kind == '"NULL"' else kind) for name, kind in columns.items()
        }
        fake_lake.add_empty(view, quoted)


@pytest.mark.parametrize("case", list(_EMPTY_CASES), ids=list(_EMPTY_CASES))
def test_loader_sql_binds_against_contract_and_returns_empty_value(
    fake_lake: Any, case: str
) -> None:
    """Each loader's SQL runs on empty contract-typed views and yields its empty value."""
    _register_contract(fake_lake)

    assert _EMPTY_CASES[case]() is None
    assert fake_lake.calls


def test_every_public_loader_has_a_conformance_case() -> None:
    """No public `load_*` function in the loader module escapes the conformance cases."""
    public = {
        name
        for name, member in inspect.getmembers(lakehouse_loader, inspect.isfunction)
        if name.startswith("load_") and member.__module__ == lakehouse_loader.__name__
    }
    covered = {case.split("[")[0] for case in _EMPTY_CASES}

    assert public == covered


def test_every_table_in_loader_sql_is_a_contract_view() -> None:
    """Each identifier after FROM or JOIN in the loader's SQL is a view in the frozen contract."""
    assert _CONTRACT["contract"] == "lakehouse-contract-v1"
    source = Path(inspect.getsourcefile(lakehouse_loader) or "").read_text()
    tables = set(re.findall(r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)", source))

    assert tables
    assert tables <= set(_CONTRACT["views"])
