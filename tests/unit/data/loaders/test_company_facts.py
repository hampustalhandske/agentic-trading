# test_company_facts.py

"""Tests for `load_company_facts_from_lakehouse`: the `sec_company_facts_log` source, DISTINCT
de-duplication and keeping only the latest available filing per reporting period."""

from datetime import date
from typing import Any

import pandas as pd

from agentic_trading.data.loaders import lakehouse_loader

_Q2_START, _Q2_END = "2023-01-01", "2023-04-01"


def _fact(filed: str, start: str, end: str, value: float, **overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "ticker": "AAPL",
        "cik": 320193,
        "concept": "EarningsPerShareDiluted",
        "unit": "USD/shares",
        "period_start": start,
        "period_end": end,
        "value": value,
        "fiscal_year": 2023,
        "fiscal_period": "Q2",
        "form": "10-Q",
        "accession_number": f"ACC-{filed}",
        "filed_date": filed,
        "scraped_at": pd.Timestamp("2024-07-01", tz="UTC"),
        "dt": "2024-07-01",
        "table": "sec_company_facts_log",
    }
    row.update(overrides)
    return row


def _seed(fake_lake: Any, rows: list[dict[str, Any]]) -> None:
    """Registers the facts log and a filings table accepted 12:00 UTC on each filed date."""
    fake_lake.add(
        "sec_company_facts_log",
        pd.DataFrame(rows),
        date_columns=("dt", "filed_date", "period_start", "period_end"),
    )
    accessions = {r["accession_number"]: r["filed_date"] for r in rows}
    fake_lake.add(
        "sec_filings",
        pd.DataFrame(
            {
                "accession_number": list(accessions),
                "accepted_at": [pd.Timestamp(f"{d} 12:00", tz="UTC") for d in accessions.values()],
            }
        ),
    )


def test_company_facts_keep_only_latest_filing_per_period(fake_lake: Any) -> None:
    """A period restated in a later 10-Q yields the later filing once it is available."""
    _seed(
        fake_lake,
        [
            _fact("2023-05-05", _Q2_START, _Q2_END, 1.52),
            _fact("2024-05-03", _Q2_START, _Q2_END, 1.52, fiscal_year=2024),
        ],
    )

    late = lakehouse_loader.load_company_facts_from_lakehouse("AAPL", date(2024, 6, 30))
    early = lakehouse_loader.load_company_facts_from_lakehouse("AAPL", date(2023, 6, 1))

    assert late is not None
    assert len(late) == 1
    assert late["filed"].iloc[0] == pd.Timestamp("2024-05-03")
    assert late["available_date"].iloc[0] == pd.Timestamp("2024-05-03")
    assert early is not None
    assert len(early) == 1
    assert early["filed"].iloc[0] == pd.Timestamp("2023-05-05")
    assert lakehouse_loader.load_company_facts_from_lakehouse("AAPL", date(2023, 5, 4)) is None


def test_company_facts_output_shape(fake_lake: Any) -> None:
    """The output has the documented columns and datetime64 date columns."""
    _seed(fake_lake, [_fact("2023-05-05", _Q2_START, _Q2_END, 1.52)])

    result = lakehouse_loader.load_company_facts_from_lakehouse("AAPL", date(2023, 6, 1))

    assert result is not None
    assert {
        "concept",
        "unit",
        "period_start",
        "period_end",
        "value",
        "fiscal_year",
        "fiscal_period",
        "form",
        "filed",
        "available_date",
    } <= set(result.columns)
    for column in ("period_start", "period_end", "filed", "available_date"):
        assert str(result[column].dtype).startswith("datetime64")
    assert lakehouse_loader.load_company_facts_from_lakehouse("ZZZZ", date(2023, 6, 1)) is None


def test_company_facts_different_periods_are_all_returned(fake_lake: Any) -> None:
    """A YTD fact and the quarter inside it are different periods and both survive."""
    _seed(
        fake_lake,
        [
            _fact("2023-05-05", _Q2_START, _Q2_END, 1.52),
            _fact("2023-05-05", "2022-09-25", _Q2_END, 3.40, accession_number="ACC-YTD"),
        ],
    )

    result = lakehouse_loader.load_company_facts_from_lakehouse("AAPL", date(2023, 6, 1))

    assert result is not None
    spans = (result["period_end"] - result["period_start"]).dt.days
    assert dict(zip(spans, result["value"], strict=True)) == {90: 1.52, 188: 3.40}


def test_company_facts_distinct_over_every_column_but_scrape_metadata(fake_lake: Any) -> None:
    """Facts differing only in period_start both survive; repeated scrapes collapse to one row."""
    rescrape = {"scraped_at": pd.Timestamp("2024-07-02", tz="UTC"), "dt": "2024-07-02"}
    _seed(
        fake_lake,
        [
            _fact("2023-05-05", "2023-01-01", _Q2_END, 1.52),
            _fact("2023-05-05", "2023-01-01", _Q2_END, 1.52, **rescrape),
            _fact("2023-05-05", "2022-12-31", _Q2_END, 1.52),
        ],
    )

    result = lakehouse_loader.load_company_facts_from_lakehouse("AAPL", date(2023, 6, 1))

    assert result is not None
    starts = sorted(result["period_start"])
    assert starts == [pd.Timestamp("2022-12-31"), pd.Timestamp("2023-01-01")]
