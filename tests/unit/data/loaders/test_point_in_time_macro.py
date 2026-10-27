# test_point_in_time_macro.py

"""Point-in-time tests for the FRED, Fama-French and risk-free-rate loaders.

FRED reads only the `fred_vintages` table (one row per published version of
an observation); Fama-French daily values become public at the end of the
month after their month; the risk-free series is never filled backwards.
"""

from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import pytest

from agentic_trading.data.loaders import lakehouse_loader, macro_loader

_VINTAGE_DATES = ("date", "realtime_start")


def _vintages(rows: list[tuple[str, str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["series_id", "date", "realtime_start", "value"])


def _cpi_vintages() -> pd.DataFrame:
    return _vintages(
        [
            ("CPIAUCSL", "2022-05-01", "2022-06-10", 292.296),
            ("CPIAUCSL", "2022-06-01", "2022-07-13", 296.311),
            ("CPIAUCSL", "2022-06-01", "2023-02-14", 295.328),
            ("OTHER", "2022-06-01", "2022-07-13", 1.0),
        ]
    )


@pytest.mark.parametrize(
    ("as_of", "expected"),
    [
        (date(2022, 7, 12), None),
        (date(2022, 7, 13), 296.311),
        (date(2023, 2, 13), 296.311),
        (date(2023, 2, 14), 295.328),
        (date(2023, 3, 1), 295.328),
    ],
)
def test_fred_series_uses_latest_version_published_by_as_of(
    fake_lake: Any, as_of: date, expected: float | None
) -> None:
    """Each observation carries the value of its latest version with realtime_start <= as_of."""
    fake_lake.add("fred_vintages", _cpi_vintages(), date_columns=_VINTAGE_DATES)

    result = lakehouse_loader.load_fred_series_from_lakehouse("CPIAUCSL", as_of)

    assert result is not None
    assert result.loc[pd.Timestamp("2022-05-01")] == pytest.approx(292.296)
    june = pd.Timestamp("2022-06-01")
    if expected is None:
        assert june not in result.index
    else:
        assert result.loc[june] == pytest.approx(expected)
    assert result.index.is_monotonic_increasing
    assert result.index.is_unique


@pytest.mark.parametrize(
    ("as_of", "present"), [(date(2010, 3, 1), False), (date(2010, 3, 2), True)]
)
def test_fred_series_excludes_observation_until_published(
    fake_lake: Any, as_of: date, present: bool
) -> None:
    """An observation appears only once its first version's realtime_start has been reached."""
    fake_lake.add(
        "fred_vintages",
        _vintages(
            [
                ("VIXCLS", "2010-02-01", "2010-02-02", 20.0),
                ("VIXCLS", "2010-03-01", "2010-03-02", 25.0),
            ]
        ),
        date_columns=_VINTAGE_DATES,
    )

    result = lakehouse_loader.load_fred_series_from_lakehouse("VIXCLS", as_of)

    assert result is not None
    assert (pd.Timestamp("2010-03-01") in result.index) is present
    if present:
        assert result.loc[pd.Timestamp("2010-03-01")] == 25.0


def test_fred_series_returns_none_when_nothing_published(fake_lake: Any) -> None:
    """No version published by as_of, or an unknown series, yields None."""
    fake_lake.add("fred_vintages", _cpi_vintages(), date_columns=_VINTAGE_DATES)

    assert lakehouse_loader.load_fred_series_from_lakehouse("CPIAUCSL", date(2022, 6, 9)) is None
    assert lakehouse_loader.load_fred_series_from_lakehouse("MISSING", date(2023, 3, 1)) is None


def _french_table() -> pd.DataFrame:
    days = pd.bdate_range("2017-05-01", "2017-08-31")
    rows = [
        ("F-F_Research_Data_Factors_daily", "daily", day, portfolio, 1.0)
        for day in days
        for portfolio in ("Mkt-RF", "SMB", "HML", "RF")
    ]
    return pd.DataFrame(rows, columns=["dataset", "frequency", "date", "portfolio", "value"])


@pytest.mark.parametrize(
    ("as_of", "last_public"),
    [
        (date(2017, 7, 15), "2017-05-31"),
        (date(2017, 7, 31), "2017-06-30"),
        (date(2017, 8, 30), "2017-06-30"),
        (date(2017, 8, 31), "2017-07-31"),
    ],
)
def test_fama_french_values_public_after_month_end_lag(
    fake_lake: Any, as_of: date, last_public: str
) -> None:
    """Daily factors for a month are returned only from the end of the following month."""
    fake_lake.add("famafrench_returns", _french_table(), date_columns=("date",))

    result = lakehouse_loader.load_fama_french_factors_from_lakehouse(as_of)

    assert result is not None
    assert result.index.max() == pd.Timestamp(last_public)
    assert result.index.min() == pd.Timestamp("2017-05-01")


def test_fama_french_3factor_columns_and_percent_scaling(fake_lake: Any) -> None:
    """The 3-factor result has Mkt-RF, SMB, HML and RF columns, divided by 100."""
    fake_lake.add("famafrench_returns", _french_table(), date_columns=("date",))

    result = lakehouse_loader.load_fama_french_factors_from_lakehouse(date(2017, 8, 31))

    assert result is not None
    assert set(result.columns) == {"Mkt-RF", "SMB", "HML", "RF"}
    assert np.allclose(result.to_numpy(), 0.01)


def _dgs3mo_vintages() -> pd.DataFrame:
    return _vintages(
        [
            ("DGS3MO", "2010-01-04", "2010-01-04", 4.0),
            ("DGS3MO", "2010-01-06", "2010-01-06", 5.0),
        ]
    )


def test_risk_free_series_forward_fills_without_backfilling(fake_lake: Any) -> None:
    """From the first observation on, gaps are forward-filled and values are percent/100/252."""
    fake_lake.add("fred_vintages", _dgs3mo_vintages(), date_columns=_VINTAGE_DATES)

    result = macro_loader.load_risk_free_rate_series(date(2010, 1, 4), date(2010, 1, 8))

    assert list(result.index) == list(pd.date_range("2010-01-04", "2010-01-08", freq="D"))
    assert result.loc["2010-01-04"] == pytest.approx(0.04 / 252)
    assert result.loc["2010-01-05"] == pytest.approx(0.04 / 252)
    assert result.loc["2010-01-06"] == pytest.approx(0.05 / 252)
    assert result.loc["2010-01-08"] == pytest.approx(0.05 / 252)
    assert not result.isna().any()


def test_risk_free_series_raises_when_start_precedes_first_observation(fake_lake: Any) -> None:
    """A start date before the first DGS3MO observation raises ValueError naming that date."""
    fake_lake.add("fred_vintages", _dgs3mo_vintages(), date_columns=_VINTAGE_DATES)

    with pytest.raises(ValueError, match="2010-01-04"):
        macro_loader.load_risk_free_rate_series(date(2010, 1, 1), date(2010, 1, 8))
