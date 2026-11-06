# test_futures_and_policy_rates.py

"""The continuous futures price history (`futures_continuous_daily` with roll
days from `futures_roll_calendar`) and the BIS central-bank policy rates
(`bis_observations`, dataflow `WS_CBPOL`), read through a `FakeLake`.

The futures level chains the ratio-adjusted close's daily returns from the
first bar's unadjusted close, with a roll day whose gap is unknown set to a
zero return; `roll` counts the front-contract changes of the roll
calendar."""

from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import pytest

from agentic_trading.data.loaders import lakehouse_loader

_DATES = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"]


def _continuous(root: str = "CL") -> pd.DataFrame:
    """Six CL bars: a roll with a known gap on 01-04, one with an unknown gap on 01-05
    (its ratio-adjusted close still jumps by the gap), and a bar on 01-09 after the read."""
    return pd.DataFrame(
        {
            "root": root,
            "date": _DATES,
            "front_contract": ["CLG24", "CLG24", "CLH24", "CLJ24", "CLJ24", "CLJ24"],
            "unadjusted_close": [80.0, 81.6, 85.0, 96.0, 96.96, 97.0],
            "ratio_adjusted_close": [100.0, 102.0, 101.0, 120.0, 121.2, 122.0],
            "diff_adjusted_close": [0.0] * 6,
            "is_roll_day": [False, False, True, True, False, False],
            "roll_return": [None, None, 0.04, 0.13, None, None],
            "roll_gap_known": pd.array([None, None, True, False, None, None], dtype="boolean"),
        }
    )


def _calendar(first_front: str = "CLG24", root: str = "CL") -> pd.DataFrame:
    """Roll-calendar rows from 2023-12-29, the session before the read starts; the front
    changes on 01-04 and 01-05."""
    dates = ["2023-12-29", *_DATES]
    fronts = [first_front, "CLG24", "CLG24", "CLH24", "CLJ24", "CLJ24", "CLJ24"]
    return pd.DataFrame(
        {
            "root": root,
            "date": dates,
            "front_contract": fronts,
            "front_expiry": "2024-02-20",
            "second_contract": "CLK24",
            "second_expiry": "2024-03-20",
            "roll_rule": "five calendar days before the last trading day",
        }
    )


def _add_futures(fake_lake: Any, first_front: str = "CLG24") -> None:
    fake_lake.add("futures_continuous_daily", _continuous(), date_columns=("date",))
    fake_lake.add(
        "futures_roll_calendar",
        _calendar(first_front),
        date_columns=("date", "front_expiry", "second_expiry"),
    )


def test_futures_level_chains_returns_and_zeroes_an_unknown_gap_roll(fake_lake: Any) -> None:
    _add_futures(fake_lake)

    frame = lakehouse_loader.load_futures_price_history_from_lakehouse(
        ["CL=F"], date(2024, 1, 2), date(2024, 1, 8)
    )

    assert frame is not None
    cl = frame["CL=F"]
    returns = [0.0, 102.0 / 100.0 - 1.0, 101.0 / 102.0 - 1.0, 0.0, 121.2 / 120.0 - 1.0]
    expected_level = 80.0 * np.cumprod(1.0 + np.array(returns))
    assert list(cl.index) == list(pd.to_datetime(_DATES[:5]))
    assert cl["close"].to_numpy() == pytest.approx(expected_level, rel=1e-12)
    assert cl.loc["2024-01-05", "close"] == cl.loc["2024-01-04", "close"]
    assert cl["open"].to_numpy() == pytest.approx(cl["close"].to_numpy())
    assert cl["raw_close"].tolist() == [80.0, 81.6, 85.0, 96.0, 96.96]
    assert cl["volume"].isna().all()
    assert np.isnan(cl["log_return"].iloc[0])
    assert cl["log_return"].iloc[1:].to_numpy() == pytest.approx(np.log1p(returns[1:]))
    assert cl["roll"].tolist() == [0.0, 0.0, 1.0, 1.0, 0.0]


def test_a_roll_on_the_first_bar_is_read_from_the_calendar_before_the_start(
    fake_lake: Any,
) -> None:
    _add_futures(fake_lake, first_front="CLF24")

    frame = lakehouse_loader.load_futures_price_history_from_lakehouse(
        ["CL=F"], date(2024, 1, 2), date(2024, 1, 8)
    )

    assert frame is not None
    assert frame[("CL=F", "roll")].tolist() == [1.0, 0.0, 1.0, 1.0, 0.0]


def test_futures_loader_returns_none_without_rows(fake_lake: Any) -> None:
    _add_futures(fake_lake)

    assert (
        lakehouse_loader.load_futures_price_history_from_lakehouse(
            ["GC=F"], date(2024, 1, 2), date(2024, 1, 8)
        )
        is None
    )


def test_futures_loader_refuses_a_ticker_that_is_not_a_future() -> None:
    with pytest.raises(ValueError, match="AAPL"):
        lakehouse_loader.load_futures_price_history_from_lakehouse(
            ["AAPL"], date(2024, 1, 2), date(2024, 1, 8)
        )


def _bis(rows: list[tuple[str, str, str, str, float]]) -> pd.DataFrame:
    """`bis_observations` rows from (dataflow, freq, ref_area, date, value)."""
    return pd.DataFrame(
        {
            "dataflow": [row[0] for row in rows],
            "series_key": [f"{row[1]}.{row[2]}" for row in rows],
            "period": [row[3] for row in rows],
            "date": [row[3] for row in rows],
            "ref_area": [row[2] for row in rows],
            "freq": [row[1] for row in rows],
            "value": [row[4] for row in rows],
            "obs_status": "A",
            "pre_break_value": pd.array([None] * len(rows), dtype="Float64"),
        }
    )


def test_policy_rates_are_daily_ws_cbpol_values_as_decimals(fake_lake: Any) -> None:
    """Daily WS_CBPOL observations from the lookback before `start` to `end`, one column per
    currency, per cent divided by 100; monthly rows, other dataflows and rows after `end`
    are not read."""
    fake_lake.add(
        "bis_observations",
        _bis(
            [
                ("WS_CBPOL", "D", "US", "2023-12-28", 5.5),
                ("WS_CBPOL", "D", "US", "2024-01-03", 5.25),
                ("WS_CBPOL", "D", "XM", "2024-01-03", 4.5),
                ("WS_CBPOL", "D", "JP", "2024-01-03", -0.1),
                ("WS_CBPOL", "M", "US", "2024-01-03", 9.0),
                ("WS_EER", "D", "US", "2024-01-03", 101.0),
                ("WS_CBPOL", "D", "US", "2024-01-10", 4.0),
                ("WS_CBPOL", "D", "US", "2023-06-01", 1.0),
            ]
        ),
        date_columns=("date",),
    )

    rates = lakehouse_loader.load_policy_rates_from_lakehouse(
        ["USD", "EUR"], date(2024, 1, 2), date(2024, 1, 8)
    )

    assert rates is not None
    assert list(rates.columns) == ["USD", "EUR"]
    assert list(rates.index) == list(pd.to_datetime(["2023-12-28", "2024-01-03"]))
    assert rates.loc["2023-12-28", "USD"] == pytest.approx(0.055)
    assert np.isnan(rates.loc["2023-12-28", "EUR"])
    assert rates.loc["2024-01-03", "USD"] == pytest.approx(0.0525)
    assert rates.loc["2024-01-03", "EUR"] == pytest.approx(0.045)


def test_policy_rates_return_none_without_rows(fake_lake: Any) -> None:
    fake_lake.add(
        "bis_observations", _bis([("WS_CBPOL", "D", "US", "2024-01-03", 5.25)]), ("date",)
    )

    assert (
        lakehouse_loader.load_policy_rates_from_lakehouse(
            ["JPY"], date(2024, 1, 2), date(2024, 1, 8)
        )
        is None
    )


def test_policy_rates_refuse_a_currency_without_a_bis_area() -> None:
    with pytest.raises(ValueError, match="XYZ"):
        lakehouse_loader.load_policy_rates_from_lakehouse(
            ["XYZ"], date(2024, 1, 2), date(2024, 1, 8)
        )
