"""Unit tests for agentic_trading.agents._market_math's shared reshaping
helpers used by every pod manager's construct_portfolio()."""

import pandas as pd
import pytest

from agentic_trading.agents._market_math import single_asset_from_attrs, wide_log_returns


def _multi_index_market_data() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    return pd.concat(
        {
            "AAPL": pd.DataFrame({"log_return": [0.01, 0.02, -0.01]}, index=index),
            "MSFT": pd.DataFrame({"log_return": [0.00, 0.01, 0.02]}, index=index),
        },
        axis=1,
    )


def test_wide_log_returns_returns_empty_frame_for_no_assets() -> None:
    """A scoped pod with zero assets in scope on a given day (e.g. a
    partial universe that includes none of its asset class) must not
    crash — see the bug this guards against in agents/base.py's
    construct_portfolio()."""
    result = wide_log_returns([], _multi_index_market_data())

    assert result.empty


def test_wide_log_returns_reshapes_multi_index_columns() -> None:
    result = wide_log_returns(["AAPL", "MSFT"], _multi_index_market_data())

    assert list(result.columns) == ["AAPL", "MSFT"]
    assert len(result) == 3


def test_single_asset_from_attrs_reads_the_set_attr() -> None:
    market_data = pd.DataFrame({"log_return": [0.01]})
    market_data.attrs["asset"] = "TLT"

    assert single_asset_from_attrs(market_data) == "TLT"


def test_single_asset_from_attrs_raises_when_unset() -> None:
    """Must fail loudly rather than silently defaulting to some other
    ticker — a silent default here would attribute this asset's real
    return to the wrong pod-return history entry."""
    market_data = pd.DataFrame({"log_return": [0.01]})

    with pytest.raises(ValueError, match="attrs"):
        single_asset_from_attrs(market_data)
