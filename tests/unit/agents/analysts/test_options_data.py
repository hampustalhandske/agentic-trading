"""Unit tests for `_options_data.real_atm_options_by_expiration` — shared
real-chain reader for every CVaR-pod analyst that needs a market quote.

Regression coverage for a boundary bug found live: this function takes
`as_of_date` as a `pd.Timestamp` (the analyst layer's convention) but
must convert to a plain `datetime.date` before calling into
`data/loaders/lakehouse_loader.py`, which is typed and self-asserts on
`datetime.date`. Passing the `Timestamp` straight through worked by
accident until `assert_point_in_time` was added there, at which point it
raised `TypeError: Cannot compare Timestamp with datetime.date` on every
call — silently zeroing out `black_scholes_valuation_analyst`,
`greeks_delta_hedge_analyst` and `heston_vol_surface_analyst` for an
entire live run before it was caught.
"""

from datetime import date

import pandas as pd
import pytest

from agentic_trading.agents.analysts.cvar_pod._options_data import (
    real_atm_options_by_expiration,
)
from agentic_trading.config import temporary_overrides


def _chain_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-04-01")] * 2,
            "expiration": [pd.Timestamp("2024-04-26")] * 2,
            "option_type": ["call", "put"],
            "strike": [180.0, 180.0],
            "bid": [5.0, 4.8],
            "ask": [5.2, 5.0],
            "implied_volatility": [0.25, 0.24],
            "delta": [0.5, -0.5],
            "gamma": [0.01, 0.01],
        }
    )


def test_real_atm_options_converts_timestamp_to_date_before_the_lakehouse_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The loader is typed `as_of_date: date`; passing a `pd.Timestamp`
    straight through must not happen even though both types share an
    `.isoformat()` (which is why this bug went unnoticed for a while)."""
    seen: dict[str, object] = {}

    def _fake_loader(ticker: str, as_of_date: object) -> pd.DataFrame:
        seen["ticker"] = ticker
        seen["as_of_date"] = as_of_date
        seen["as_of_date_type"] = type(as_of_date)
        return _chain_frame()

    monkeypatch.setattr(
        "agentic_trading.agents.analysts.cvar_pod._options_data."
        "load_options_snapshot_from_lakehouse",
        _fake_loader,
    )

    with temporary_overrides({"options.publication_lag_days": 0}):
        result = real_atm_options_by_expiration("AAPL", pd.Timestamp("2024-04-01"), spot=180.0)

    assert seen["as_of_date_type"] is date
    assert seen["as_of_date"] == date(2024, 4, 1)
    assert result is not None
    assert result.iloc[0]["strike"] == pytest.approx(180.0)


def test_real_atm_options_returns_none_without_a_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "agentic_trading.agents.analysts.cvar_pod._options_data."
        "load_options_snapshot_from_lakehouse",
        lambda _ticker, _as_of_date: None,
    )

    result = real_atm_options_by_expiration("AAPL", pd.Timestamp("2024-04-01"), spot=180.0)

    assert result is None
