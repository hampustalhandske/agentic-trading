# test_fred_same_day.py

"""AT-118 item 2: FRED can publish after the 16:00 ET close (the H.15 daily rates post around
16:15 ET, VIXCLS the next morning), so `SOURCES` marks it as an after-close source and
`load_fred_series_from_lakehouse` uses only vintages whose `realtime_start` is strictly before
`as_of_date`. A vintage first published on `as_of_date` leaves the previous vintage's value in
place, or the observation absent when there was none."""

from datetime import date
from typing import Any

import pandas as pd
import pytest

from agentic_trading.data.loaders import lakehouse_loader
from agentic_trading.stage_a.decisions import SOURCES

_VINTAGE_DATES = ("date", "realtime_start")


def _vintages(rows: list[tuple[str, str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["series_id", "date", "realtime_start", "value"])


def test_fred_is_an_after_close_source() -> None:
    """SOURCES classifies FRED as a source that can publish after the close."""
    assert SOURCES["fred"] is True


def test_fred_series_uses_vintages_published_before_as_of(fake_lake: Any) -> None:
    """A vintage whose realtime_start is as_of_date is not used; the one before it applies."""
    fake_lake.add(
        "fred_vintages",
        _vintages(
            [
                ("DGS10", "2024-01-09", "2024-01-10", 4.00),
                ("DGS10", "2024-01-10", "2024-01-11", 4.10),
                ("DGS10", "2024-01-10", "2024-01-16", 4.15),
                ("DGS10", "2024-01-11", "2024-01-12", 4.20),
                ("OTHER", "2024-01-09", "2024-01-09", 9.99),
            ]
        ),
        date_columns=_VINTAGE_DATES,
    )
    expected: dict[date, dict[str, float] | None] = {
        date(2024, 1, 10): None,
        date(2024, 1, 11): {"2024-01-09": 4.00},
        date(2024, 1, 12): {"2024-01-09": 4.00, "2024-01-10": 4.10},
        date(2024, 1, 16): {"2024-01-09": 4.00, "2024-01-10": 4.10, "2024-01-11": 4.20},
        date(2024, 1, 17): {"2024-01-09": 4.00, "2024-01-10": 4.15, "2024-01-11": 4.20},
    }

    for as_of, values in expected.items():
        result = lakehouse_loader.load_fred_series_from_lakehouse("DGS10", as_of)
        if values is None:
            assert result is None, as_of
            continue
        assert result is not None, as_of
        assert result.index.is_monotonic_increasing and result.index.is_unique
        read = {stamp.date().isoformat(): value for stamp, value in result.items()}
        assert read == pytest.approx(values), as_of


def test_fred_value_is_known_from_the_later_of_its_vintage_and_its_observation(
    fake_lake: Any,
) -> None:
    """A FRED value is known at `known_at = max(realtime_start, date)` and is used only when
    `known_at` is strictly before `as_of_date`: a vintage stamped before its own observation date
    (as on some VIXCLS rows) cannot reveal that day's value early, and a same-day vintage counts
    from the next day. `known_at` is returned beside `realtime_start`."""
    fake_lake.add(
        "fred_vintages",
        _vintages(
            [
                ("VIXCLS", "2020-03-09", "2020-03-09", 54.5),
                ("VIXCLS", "2020-03-12", "2020-03-09", 75.5),
            ]
        ),
        date_columns=_VINTAGE_DATES,
    )
    for as_of in (date(2020, 3, 9), date(2020, 3, 10), date(2020, 3, 11), date(2020, 3, 12)):
        vintages = lakehouse_loader.load_fred_vintages_from_lakehouse("VIXCLS", as_of)
        observed = set() if vintages is None else {s.date() for s in vintages.index}
        assert date(2020, 3, 12) not in observed, as_of
        if as_of > date(2020, 3, 9):
            assert vintages is not None
            assert observed == {date(2020, 3, 9)}, as_of
        else:
            assert vintages is None, as_of

    vintages = lakehouse_loader.load_fred_vintages_from_lakehouse("VIXCLS", date(2020, 3, 13))
    assert vintages is not None
    known = {s.date(): k.date() for s, k in vintages["known_at"].items()}
    assert known == {date(2020, 3, 9): date(2020, 3, 9), date(2020, 3, 12): date(2020, 3, 12)}
    series = lakehouse_loader.load_fred_series_from_lakehouse("VIXCLS", date(2020, 3, 12))
    assert series is not None and list(series.index.date) == [date(2020, 3, 9)]
