"""Unit tests for agentic_trading.data.loaders.macro_loader.

The lakehouse query is mocked at the module boundary
(`agentic_trading.data.loaders.macro_loader.load_fred_series_from_lakehouse`)
— no DuckDB/parquet access in this suite.
"""

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from agentic_trading.data.loaders.macro_loader import (
    DGS3MO,
    TRADING_DAYS_PER_YEAR,
    load_risk_free_rate_series,
)


@patch("agentic_trading.data.loaders.macro_loader.load_fred_series_from_lakehouse")
def test_risk_free_rate_series_is_a_calendar_indexed_decimal_daily_rate(
    mock_load: MagicMock,
) -> None:
    """A 5.04% annualized T-bill quote becomes `0.0504 / 252` per day,
    reindexed onto every calendar day of the window with weekends and
    holidays forward-filled from the last published observation."""
    published = pd.bdate_range("2024-06-03", "2024-06-07")
    mock_load.return_value = pd.Series([5.04] * len(published), index=published)

    result = load_risk_free_rate_series(date(2024, 6, 3), date(2024, 6, 9))

    mock_load.assert_called_once_with(DGS3MO, date(2024, 6, 9))
    expected_calendar = pd.date_range("2024-06-03", "2024-06-09", freq="D")
    assert result.index.equals(expected_calendar)
    assert not result.isna().any()
    assert result.iloc[0] == pytest.approx(5.04 / 100.0 / TRADING_DAYS_PER_YEAR)
    assert result.loc["2024-06-08"] == pytest.approx(result.loc["2024-06-07"])


@patch("agentic_trading.data.loaders.macro_loader.load_fred_series_from_lakehouse")
def test_risk_free_rate_series_raises_for_a_window_starting_before_the_first_observation(
    mock_load: MagicMock,
) -> None:
    """A window opening before the first observation would need a
    backward-filled rate, so it raises naming that observation's date."""
    published = pd.bdate_range("2024-06-05", "2024-06-07")
    mock_load.return_value = pd.Series([4.0, 4.5, 5.0], index=published)

    with pytest.raises(ValueError, match="2024-06-05"):
        load_risk_free_rate_series(date(2024, 6, 3), date(2024, 6, 7))


@patch("agentic_trading.data.loaders.macro_loader.load_fred_series_from_lakehouse")
def test_risk_free_rate_series_raises_when_lakehouse_has_no_dgs3mo(
    mock_load: MagicMock,
) -> None:
    mock_load.return_value = None

    with pytest.raises(ValueError, match=DGS3MO):
        load_risk_free_rate_series(date(2024, 6, 3), date(2024, 6, 7))


@patch("agentic_trading.data.loaders.macro_loader.load_fred_series_from_lakehouse")
def test_risk_free_rate_series_raises_when_lakehouse_series_is_empty(
    mock_load: MagicMock,
) -> None:
    mock_load.return_value = pd.Series(dtype="float64")

    with pytest.raises(ValueError, match=DGS3MO):
        load_risk_free_rate_series(date(2024, 6, 3), date(2024, 6, 7))
