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

pytestmark = pytest.mark.unpinned_lake_reads

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
        (date(2022, 7, 13), None),
        (date(2022, 7, 14), 296.311),
        (date(2023, 2, 14), 296.311),
        (date(2023, 2, 15), 295.328),
        (date(2023, 3, 1), 295.328),
    ],
)
def test_fred_series_uses_latest_version_published_by_as_of(
    fake_lake: Any, as_of: date, expected: float | None
) -> None:
    """Each observation carries the value of its latest version with realtime_start < as_of."""
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
    ("as_of", "present"), [(date(2010, 3, 2), False), (date(2010, 3, 3), True)]
)
def test_fred_series_excludes_observation_until_published(
    fake_lake: Any, as_of: date, present: bool
) -> None:
    """An observation appears only from the day after its first version's realtime_start."""
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
    """From the first day a value is known on, gaps are forward-filled and values are
    percent/100/252; a value published on its own date is known from the next day."""
    fake_lake.add("fred_vintages", _dgs3mo_vintages(), date_columns=_VINTAGE_DATES)

    result = macro_loader.load_risk_free_rate_series(date(2010, 1, 5), date(2010, 1, 8))

    assert list(result.index) == list(pd.date_range("2010-01-05", "2010-01-08", freq="D"))
    assert result.loc["2010-01-05"] == pytest.approx(0.04 / 252)
    assert result.loc["2010-01-06"] == pytest.approx(0.04 / 252)
    assert result.loc["2010-01-07"] == pytest.approx(0.05 / 252)
    assert result.loc["2010-01-08"] == pytest.approx(0.05 / 252)
    assert not result.isna().any()


def test_risk_free_series_accrues_each_day_at_the_value_known_that_day(fake_lake: Any) -> None:
    """A revision published later does not reach back: each day carries the latest DGS3MO value
    known before it, the vintage of that day, not the window end's."""
    fake_lake.add(
        "fred_vintages",
        _vintages(
            [
                ("DGS3MO", "2010-01-04", "2010-01-04", 4.0),
                ("DGS3MO", "2010-01-04", "2010-01-07", 4.8),
                ("DGS3MO", "2010-01-08", "2010-01-08", 5.0),
            ]
        ),
        date_columns=_VINTAGE_DATES,
    )

    result = macro_loader.load_risk_free_rate_series(date(2010, 1, 5), date(2010, 1, 10))

    annual = {
        "2010-01-05": 4.0,
        "2010-01-06": 4.0,
        "2010-01-07": 4.0,
        "2010-01-08": 4.8,
        "2010-01-09": 5.0,
        "2010-01-10": 5.0,
    }
    assert list(result.index) == list(pd.date_range("2010-01-05", "2010-01-10", freq="D"))
    for day, percent in annual.items():
        assert result.loc[day] == pytest.approx(percent / 100 / 252), day


def test_risk_free_series_reads_the_versions_once(fake_lake: Any) -> None:
    """The versions are fetched in one query, then selected for every day of the window."""
    fake_lake.add("fred_vintages", _dgs3mo_vintages(), date_columns=_VINTAGE_DATES)
    reads: list[Any] = []
    fetch = lakehouse_loader._FRED_VERSIONS.fetch

    def _counting(*args: Any) -> Any:
        reads.append(args)
        return fetch(*args)

    lakehouse_loader._FRED_VERSIONS.fetch = _counting
    try:
        macro_loader.load_risk_free_rate_series(date(2010, 1, 5), date(2010, 1, 20))
    finally:
        lakehouse_loader._FRED_VERSIONS.fetch = fetch

    assert len(reads) == 1


def _random_versions(seed: int) -> pd.DataFrame:
    """`_fetch_fred_versions` rows, in its order, for 60 business days of observations: some
    revised (up to three versions, some on the same date and with the same value), some with a
    null value or a null `realtime_start`."""
    rng = np.random.default_rng(seed)
    rows = []
    for observed in pd.bdate_range("2012-03-01", periods=60):
        for _ in range(int(rng.integers(1, 4))):
            lag = int(rng.integers(-2, 15))
            realtime_start = None if rng.random() < 0.05 else observed + pd.Timedelta(days=lag)
            value = None if rng.random() < 0.1 else float(rng.choice([1.0, 1.5, 2.0, 2.5]))
            rows.append((observed, value, realtime_start))
    frame = pd.DataFrame(rows, columns=["date", "value", "realtime_start"])
    frame["realtime_start"] = pd.to_datetime(frame["realtime_start"])
    frame["known_at"] = frame[["date", "realtime_start"]].max(axis=1)
    frame = frame.sort_values(
        ["date", "realtime_start", "value"], na_position="first", kind="stable"
    )
    return frame.reset_index(drop=True)


@pytest.mark.parametrize("seed", range(5))
def test_known_value_by_day_is_the_last_value_of_each_days_selection(seed: int) -> None:
    """Each day's value is the last non-null value of `_select_fred_vintages(versions, day)`."""
    versions = _random_versions(seed)
    days = pd.date_range("2012-02-25", "2012-06-30", freq="D")

    result = macro_loader.known_value_by_day(versions, days)

    expected = []
    for day in days:
        selected = lakehouse_loader._select_fred_vintages(versions, day.date())
        known = None if selected is None else selected["value"].dropna()
        expected.append(None if known is None or known.empty else float(known.iloc[-1]))
    assert result == expected


@pytest.mark.parametrize("start", [date(2010, 1, 1), date(2010, 1, 4)])
def test_risk_free_series_raises_when_start_precedes_first_observation(
    fake_lake: Any, start: date
) -> None:
    """A start date on or before the first DGS3MO observation, before any value is known,
    raises ValueError naming that observation's date."""
    fake_lake.add("fred_vintages", _dgs3mo_vintages(), date_columns=_VINTAGE_DATES)

    with pytest.raises(ValueError, match="2010-01-04"):
        macro_loader.load_risk_free_rate_series(start, date(2010, 1, 8))
