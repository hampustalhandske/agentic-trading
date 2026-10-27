"""A failed lake request propagates; `None` means only "query succeeded
with zero rows"."""

from datetime import date
from typing import Any

import pandas as pd
import pytest

from agentic_trading.data.lakehouse_client import LakehouseError
from agentic_trading.data.loaders import lakehouse_loader


def test_lakehouse_error_propagates_instead_of_becoming_none(fake_lake: Any) -> None:
    fake_lake.error = LakehouseError("HTTP 400: bad column")
    with pytest.raises(LakehouseError):
        lakehouse_loader.load_dividends_from_lakehouse("AAPL", date(2024, 1, 15))
    with pytest.raises(LakehouseError):
        lakehouse_loader.load_price_history_from_lakehouse(
            ["AAPL"], date(2024, 1, 2), date(2024, 1, 5)
        )


def test_connection_error_propagates(fake_lake: Any) -> None:
    fake_lake.error = ConnectionError("refused")
    with pytest.raises(ConnectionError):
        lakehouse_loader._split_adjustment_factor("AAPL", date(2024, 1, 5))


def test_zero_rows_is_none(fake_lake: Any) -> None:
    fake_lake.add(
        "dolthub_stock_dividends",
        pd.DataFrame({"act_symbol": ["MSFT"], "ex_date": ["2024-01-10"], "amount": [0.5]}),
        date_columns=("ex_date",),
    )
    assert lakehouse_loader.load_dividends_from_lakehouse("AAPL", date(2024, 1, 15)) is None
