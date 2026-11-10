"""Unit tests for agentic_trading.data.loaders.macro_loader.

`load_risk_free_rate_series` reads the `fred_vintages` table of the in-memory fake lake
(`fake_lake`); no lakehouse API is reached.
"""

from datetime import date
from typing import Any

import pandas as pd
import pytest

from agentic_trading.data.loaders.macro_loader import (
    DGS3MO,
    TRADING_DAYS_PER_YEAR,
    load_risk_free_rate_series,
)

_VINTAGE_DATES = ("date", "realtime_start")


def _published(series_id: str, days: pd.DatetimeIndex, values: list[float | None]) -> pd.DataFrame:
    """`fred_vintages` rows of `series_id`, each value published on its own observation date."""
    return pd.DataFrame(
        {"series_id": series_id, "date": days, "realtime_start": days, "value": values}
    )


def test_risk_free_rate_series_is_a_calendar_indexed_decimal_daily_rate(fake_lake: Any) -> None:
    """A 5.04% annualized T-bill quote becomes `0.0504 / 252` per day,
    reindexed onto every calendar day of the window with weekends and
    holidays forward-filled from the last published observation."""
    published = pd.bdate_range("2024-05-31", "2024-06-07")
    fake_lake.add(
        "fred_vintages",
        _published(DGS3MO, published, [5.04] * len(published)),
        date_columns=_VINTAGE_DATES,
    )

    result = load_risk_free_rate_series(date(2024, 6, 3), date(2024, 6, 9))

    expected_calendar = pd.date_range("2024-06-03", "2024-06-09", freq="D")
    assert result.index.equals(expected_calendar)
    assert not result.isna().any()
    assert result.iloc[0] == pytest.approx(5.04 / 100.0 / TRADING_DAYS_PER_YEAR)
    assert result.loc["2024-06-08"] == pytest.approx(result.loc["2024-06-07"])


def test_risk_free_rate_series_raises_for_a_window_starting_before_the_first_observation(
    fake_lake: Any,
) -> None:
    """A window opening before the first observation would need a
    backward-filled rate, so it raises naming that observation's date."""
    published = pd.bdate_range("2024-06-05", "2024-06-07")
    fake_lake.add(
        "fred_vintages",
        _published(DGS3MO, published, [4.0, 4.5, 5.0]),
        date_columns=_VINTAGE_DATES,
    )

    with pytest.raises(ValueError, match="2024-06-05"):
        load_risk_free_rate_series(date(2024, 6, 3), date(2024, 6, 7))


def test_risk_free_rate_series_raises_when_lakehouse_has_no_dgs3mo(fake_lake: Any) -> None:
    published = pd.bdate_range("2024-05-27", "2024-06-07")
    fake_lake.add(
        "fred_vintages",
        _published("DGS10", published, [4.0] * len(published)),
        date_columns=_VINTAGE_DATES,
    )

    with pytest.raises(ValueError, match=DGS3MO):
        load_risk_free_rate_series(date(2024, 6, 3), date(2024, 6, 7))


def test_risk_free_rate_series_raises_when_lakehouse_series_is_empty(fake_lake: Any) -> None:
    """A DGS3MO series whose every observation has no value holds no rate."""
    published = pd.bdate_range("2024-05-27", "2024-06-07")
    fake_lake.add(
        "fred_vintages",
        _published(DGS3MO, published, [None] * len(published)),
        date_columns=_VINTAGE_DATES,
    )

    with pytest.raises(ValueError, match=DGS3MO):
        load_risk_free_rate_series(date(2024, 6, 3), date(2024, 6, 7))
