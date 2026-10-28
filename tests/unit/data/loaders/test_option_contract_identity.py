# test_option_contract_identity.py

"""Option contract identity: same expiration and type, strike within half a cent."""

from datetime import date
from typing import Any

import pandas as pd
import pytest

from agentic_trading.agents.schemas import OptionContract
from agentic_trading.data.loaders import lakehouse_loader
from agentic_trading.data.loaders.option_marks import _contract_row

_EXPIRATION = date(2022, 9, 16)


def _chain_row(day: str, option_type: str, strike: float, bid: float) -> dict[str, Any]:
    return {
        "underlying": "TSLA",
        "date": day,
        "expiration": "2022-09-16",
        "option_type": option_type,
        "strike": strike,
        "bid": bid,
        "ask": bid + 0.1,
        "implied_volatility": 0.6,
        "delta": 0.5,
        "gamma": 0.01,
        "theta": -0.1,
        "vega": 0.2,
        "rho": 0.05,
    }


def _seed_split_chain(fake_lake: Any) -> None:
    """TSLA 3-for-1 split on 2022-08-25: a pre-split row at 625.0 and post-split
    rows at 208.33 and 210.0."""
    rows = [
        _chain_row("2022-08-24", "call", 625.0, 30.0),
        _chain_row("2022-08-26", "call", 208.33, 10.0),
        _chain_row("2022-08-26", "call", 210.0, 9.0),
        _chain_row("2022-08-26", "put", 208.33, 7.0),
    ]
    fake_lake.add("dolthub_option_chain", pd.DataFrame(rows), date_columns=("date", "expiration"))
    splits = pd.DataFrame(
        [{"ticker": "TSLA", "ex_date": "2022-08-25", "ratio": 1 / 3, "in_close_basis": True}]
    )
    fake_lake.add("yahoo_split_close_basis", splits, date_columns=("ex_date",))


@pytest.mark.parametrize(
    ("strike", "matches"),
    [(208.333333, True), (208.334, True), (208.34, False), (208.32, False)],
)
def test_load_option_quote_matches_strike_within_half_a_cent(
    fake_lake: Any, strike: float, matches: bool
) -> None:
    """The newest same-type row within 0.005 of the strike is returned, never a
    stale or other-type row."""
    _seed_split_chain(fake_lake)

    result = lakehouse_loader.load_option_quote_from_lakehouse(
        "TSLA", _EXPIRATION, "call", strike, date(2022, 8, 26), lookback_days=10
    )

    if not matches:
        assert result is None
        return
    assert result is not None
    assert len(result) == 1
    assert result["date"].iloc[0] == pd.Timestamp("2022-08-26")
    assert result["strike"].iloc[0] == pytest.approx(208.33)
    assert result["bid"].iloc[0] == pytest.approx(10.0)


@pytest.mark.parametrize(
    ("strike", "matches"), [(208.333333, True), (208.334, True), (208.32, False)]
)
def test_contract_row_matches_strike_within_half_a_cent(strike: float, matches: bool) -> None:
    """`_contract_row` picks the 208.33 row for strikes within half a cent and None otherwise."""
    snapshot = pd.DataFrame(
        {
            "expiration": pd.to_datetime(["2022-09-16", "2022-09-16"]),
            "option_type": ["call", "call"],
            "strike": [208.33, 210.0],
            "bid": [10.0, 9.0],
        }
    )
    contract = OptionContract(
        underlying="TSLA", expiry=_EXPIRATION, strike=strike, option_type="call"
    )

    row = _contract_row(snapshot, contract)

    if not matches:
        assert row is None
        return
    assert row is not None
    assert row["strike"] == pytest.approx(208.33)
    assert row["bid"] == 10.0
