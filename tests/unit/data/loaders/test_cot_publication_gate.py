# test_cot_publication_gate.py

"""`load_cot_positioning_from_lakehouse` and the `cftc_cot` coverage gate on
`cftc_cot_publication`: a timed release is usable on its published date, an
untimed one the next day, and a report without a published date never."""

from datetime import date
from typing import Any

import pandas as pd
import pytest

from agentic_trading.data.loaders import lakehouse_loader

ZN, GC, ZB = "043602", "088691", "020601"


def _publication(rows: list[tuple[str, str | None, str | None, str | None]]) -> pd.DataFrame:
    """Publication rows as (report_date, published_date, published_at UTC, rule_date)."""
    return pd.DataFrame(
        {
            "report_date": [r[0] for r in rows],
            "published_date": [r[1] for r in rows],
            "published_at": pd.to_datetime([r[2] for r in rows], utc=True),
            "rule_date": [r[3] for r in rows],
            "basis": "test",
        }
    )


def _tff(report_dates: list[str], code: str = ZN) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cftc_contract_market_code": code,
            "report_date": report_dates,
            "open_interest_all": 1000,
            "leveraged_funds_long": 300,
            "leveraged_funds_short": 100,
        }
    )


def _seed(lake: Any, reports: list[str], publication: list, table: str = "cftc_cot_tff") -> None:
    frame = _tff(reports, ZN)
    if table == "cftc_cot_disaggregated":
        frame = frame.rename(
            columns={
                "leveraged_funds_long": "managed_money_long",
                "leveraged_funds_short": "managed_money_short",
            }
        ).assign(cftc_contract_market_code=GC)
    lake.add(table, frame, date_columns=("report_date",))
    lake.add(
        "cftc_cot_publication",
        _publication(publication),
        date_columns=("report_date", "published_date", "rule_date"),
    )


def _dates(ticker: str, as_of: date, **kwargs: Any) -> list[date]:
    result = lakehouse_loader.load_cot_positioning_from_lakehouse(ticker, as_of, **kwargs)
    return [] if result is None else [pd.Timestamp(i).date() for i in result.index]


def test_timed_publication_is_available_on_its_published_date(fake_lake: Any) -> None:
    """A report with a release time is usable on its published date, not before."""
    _seed(
        fake_lake,
        ["2018-12-11"],
        [("2018-12-11", "2018-12-14", "2018-12-14 20:30", "2018-12-14")],
    )

    assert date(2018, 12, 11) not in _dates("ZN=F", date(2018, 12, 13))
    assert _dates("ZN=F", date(2018, 12, 14)) == [date(2018, 12, 11)]


def test_untimed_publication_is_available_from_the_next_day(fake_lake: Any) -> None:
    """A report without a release time is usable only after its published date."""
    _seed(fake_lake, ["2018-12-31"], [("2018-12-31", "2019-02-05", None, "2019-01-04")])

    assert _dates("ZN=F", date(2019, 1, 7)) == []
    assert _dates("ZN=F", date(2019, 2, 5)) == []
    assert _dates("ZN=F", date(2019, 2, 6)) == [date(2018, 12, 31)]


def test_report_without_published_date_is_never_available(fake_lake: Any) -> None:
    """A null published_date, or a missing publication row, never yields a report."""
    _seed(
        fake_lake,
        ["1999-06-01", "2020-03-03"],
        [("1999-06-01", None, None, None)],
    )

    assert _dates("ZN=F", date(2001, 1, 1), lookback_days=10_000) == []
    assert _dates("ZN=F", date(2021, 1, 1), lookback_days=10_000) == []


def test_disaggregated_table_uses_the_same_gate(fake_lake: Any) -> None:
    """Commodity reports in cftc_cot_disaggregated are gated by publication too."""
    _seed(
        fake_lake,
        ["2024-01-09"],
        [("2024-01-09", "2024-01-12", "2024-01-12 20:30", "2024-01-12")],
        table="cftc_cot_disaggregated",
    )

    assert _dates("GC=F", date(2024, 1, 11)) == []
    assert _dates("GC=F", date(2024, 1, 12)) == [date(2024, 1, 9)]


def test_positioning_value_is_net_leveraged_funds_over_open_interest(fake_lake: Any) -> None:
    """The TFF value is (leveraged_funds_long - leveraged_funds_short) / open_interest_all."""
    _seed(
        fake_lake,
        ["2018-12-11"],
        [("2018-12-11", "2018-12-14", "2018-12-14 20:30", "2018-12-14")],
    )

    result = lakehouse_loader.load_cot_positioning_from_lakehouse("ZN=F", date(2018, 12, 14))

    assert result is not None
    assert list(result) == [pytest.approx(0.2)]


def test_source_coverage_spans_published_dates_of_qualifying_rows(fake_lake: Any) -> None:
    """Coverage is the published_date span over rows with positive open interest and a date."""
    frame = pd.concat(
        [
            _tff(["2018-12-11", "2019-01-08", "1999-06-01", "2018-12-04"]),
            _tff(["2025-01-01"], code=ZB),
        ],
        ignore_index=True,
    )
    frame.loc[3, "open_interest_all"] = 0
    fake_lake.add("cftc_cot_tff", frame, date_columns=("report_date",))
    fake_lake.add(
        "cftc_cot_publication",
        _publication(
            [
                ("2018-12-11", "2018-12-14", "2018-12-14 20:30", "2018-12-14"),
                ("2019-01-08", "2019-01-11", "2019-01-11 20:30", "2019-01-11"),
                ("1999-06-01", None, None, None),
                ("2018-12-04", "2018-11-01", "2018-11-01 20:30", "2018-11-01"),
                ("2025-01-01", "2025-01-03", "2025-01-03 20:30", "2025-01-03"),
            ]
        ),
        date_columns=("report_date", "published_date", "rule_date"),
    )

    assert lakehouse_loader.source_coverage("cftc_cot", "ZN=F") == (
        date(2018, 12, 14),
        date(2019, 1, 11),
    )
    assert lakehouse_loader.source_coverage("cftc_cot", "TLT") == (
        date(2025, 1, 3),
        date(2025, 1, 3),
    )


def test_source_coverage_is_none_without_qualifying_rows(fake_lake: Any) -> None:
    """Only null-published rows leave no coverage."""
    _seed(fake_lake, ["1999-06-01"], [("1999-06-01", None, None, None)])

    assert lakehouse_loader.source_coverage("cftc_cot", "ZN=F") is None
