# test_point_in_time_events.py

"""Point-in-time tests for the short-interest, earnings, insider and company-facts loaders.

Decisions are taken at the 16:00 America/New_York close of `as_of_date`. FINRA short
interest is gated on the loader's own dissemination estimate, earnings on the
announcement session implied by the `when` field, and SEC data on the filing's
acceptance time.
"""

from datetime import date
from typing import Any

import pandas as pd
import pytest

from agentic_trading.data.loaders import lakehouse_loader


def _short_interest(rows: list[tuple[str, str, str, str | None]]) -> pd.DataFrame:
    """Rows of (symbol, settlement, lake dissemination, revision_flag)."""
    return pd.DataFrame(
        [(s, st, d, 1_000_000 + i, 2.0, 500_000, f) for i, (s, st, d, f) in enumerate(rows)],
        columns=[
            "symbol",
            "settlement_date",
            "dissemination_date",
            "current_short_position",
            "days_to_cover",
            "average_daily_volume",
            "revision_flag",
        ],
    )


def _seed_short_interest(fake_lake: Any, rows: list[tuple[str, str, str, str | None]]) -> None:
    fake_lake.add(
        "finra_short_interest",
        _short_interest(rows),
        date_columns=("settlement_date", "dissemination_date"),
    )


@pytest.mark.parametrize(
    ("as_of", "present"),
    [(date(2023, 11, 27), False), (date(2023, 11, 28), False), (date(2023, 11, 29), True)],
)
def test_short_interest_uses_own_dissemination_estimate(
    fake_lake: Any, as_of: date, present: bool
) -> None:
    """A settlement is available at next_session(settlement + 8 sessions), ignoring the
    lake's column."""
    _seed_short_interest(fake_lake, [("AAPL", "2023-11-15", "2023-11-27", None)])

    result = lakehouse_loader.load_short_interest_from_lakehouse("AAPL", as_of)

    if not present:
        assert result is None
        return
    assert result is not None
    assert list(result.columns) == [
        "settlement_date",
        "dissemination_date",
        "short_interest_shares",
        "days_to_cover",
        "avg_daily_volume",
    ]
    assert result["settlement_date"].iloc[0] == pd.Timestamp("2023-11-15")
    assert result["dissemination_date"].iloc[0] == pd.Timestamp("2023-11-29")


def test_short_interest_revision_waits_for_next_settlement_availability(fake_lake: Any) -> None:
    """An 'R' row is returned only from the availability date of the symbol's next
    settlement row."""
    _seed_short_interest(
        fake_lake,
        [
            ("AAPL", "2023-11-30", "2023-12-11", "R"),
            ("AAPL", "2023-12-15", "2023-12-27", None),
            ("MSFT", "2023-12-01", "2023-12-12", None),
        ],
    )

    for as_of in (date(2023, 12, 13), date(2023, 12, 28)):
        assert lakehouse_loader.load_short_interest_from_lakehouse("AAPL", as_of) is None

    result = lakehouse_loader.load_short_interest_from_lakehouse("AAPL", date(2023, 12, 29))

    assert result is not None
    assert list(result["settlement_date"]) == [
        pd.Timestamp("2023-11-30"),
        pd.Timestamp("2023-12-15"),
    ]


def test_short_interest_revision_without_later_settlement_is_never_returned(fake_lake: Any) -> None:
    """An 'R' row that is the symbol's latest settlement is withheld; unknown tickers give None."""
    _seed_short_interest(
        fake_lake,
        [("AAPL", "2023-11-15", "2023-11-27", None), ("AAPL", "2023-11-30", "2023-12-11", "R")],
    )

    result = lakehouse_loader.load_short_interest_from_lakehouse("AAPL", date(2024, 6, 1))

    assert result is not None
    assert list(result["settlement_date"]) == [pd.Timestamp("2023-11-15")]
    assert lakehouse_loader.load_short_interest_from_lakehouse("ZZZZ", date(2024, 6, 1)) is None


def _seed_earnings(fake_lake: Any, calendar_when: list[str | None]) -> None:
    fake_lake.add(
        "dolthub_eps_history",
        pd.DataFrame(
            [
                ("AAPL", "2024-03-30", 1.53, 1.50),
                ("AAPL", "2024-06-29", 1.40, 1.35),
            ],
            columns=["act_symbol", "period_end_date", "reported", "estimate"],
        ),
        date_columns=("period_end_date",),
    )
    fake_lake.add(
        "dolthub_earnings_calendar",
        pd.DataFrame(
            [("AAPL", "2024-05-02", calendar_when[0]), ("AAPL", "2024-08-01", calendar_when[1])],
            columns=["act_symbol", "date", "when"],
        ),
        date_columns=("date",),
    )


@pytest.mark.parametrize(
    ("as_of", "expected"),
    [
        (date(2024, 5, 2), None),
        (date(2024, 5, 3), (1.53, 1.50, pd.Timestamp("2024-05-02"))),
        (date(2024, 8, 1), (1.53, 1.50, pd.Timestamp("2024-05-02"))),
        (date(2024, 8, 2), (1.40, 1.35, pd.Timestamp("2024-08-01"))),
    ],
)
def test_earnings_surprise_after_close_available_next_session(
    fake_lake: Any, as_of: date, expected: tuple[float, float, pd.Timestamp] | None
) -> None:
    """After-close announcements are available the next session; the date returned is the
    announcement date."""
    _seed_earnings(fake_lake, ["After market close", "After market close"])

    result = lakehouse_loader.load_earnings_surprise_from_lakehouse("AAPL", as_of)

    assert result == expected


def test_earnings_surprise_before_open_available_same_day_and_null_is_after_close(
    fake_lake: Any,
) -> None:
    """Before-open announcements are available on their date; a NULL `when` waits a session."""
    _seed_earnings(fake_lake, ["Before market open", None])

    assert lakehouse_loader.load_earnings_surprise_from_lakehouse("AAPL", date(2024, 5, 2)) == (
        1.53,
        1.50,
        pd.Timestamp("2024-05-02"),
    )
    assert lakehouse_loader.load_earnings_surprise_from_lakehouse("AAPL", date(2024, 8, 1)) == (
        1.53,
        1.50,
        pd.Timestamp("2024-05-02"),
    )
    assert lakehouse_loader.load_earnings_surprise_from_lakehouse("AAPL", date(2024, 8, 2)) == (
        1.40,
        1.35,
        pd.Timestamp("2024-08-01"),
    )
    assert lakehouse_loader.load_earnings_surprise_from_lakehouse("ZZZZ", date(2024, 8, 2)) is None


_UTC = "UTC"
_EARLY = "2024-03-12 14:00:00"
_LATE = "2024-03-12 21:30:00"


def _seed_filings(fake_lake: Any, accepted: dict[str, str]) -> None:
    """Registers `sec_filings` with one row per (accession_number, UTC acceptance time)."""
    fake_lake.add(
        "sec_filings",
        pd.DataFrame(
            {
                "accession_number": list(accepted),
                "accepted_at": [pd.Timestamp(v, tz=_UTC) for v in accepted.values()],
                "cik": 320193,
                "dt": pd.Timestamp("2024-03-13"),
                "filing_date": pd.Timestamp("2024-03-12"),
                "form": "4",
                "items": None,
                "primary_doc_description": None,
                "primary_document": None,
                "report_date": pd.Timestamp("2024-03-12"),
                "scraped_at": pd.Timestamp("2024-03-13", tz=_UTC),
                "size": 1000,
                "table": "sec_filings",
                "ticker": "AAPL",
            }
        ),
        date_columns=("dt", "filing_date", "report_date"),
    )


def _insider_row(accession: str, **overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "accession_number": accession,
        "acquired_disposed": "D",
        "dt": "2024-03-13",
        "filed_date": "2024-03-12",
        "is_director": False,
        "is_officer": True,
        "is_ten_percent_owner": False,
        "issuer_cik": 320193,
        "issuer_ticker": "AAPL",
        "officer_title": None,
        "owner_cik": 1214156,
        "owner_name": "COOK TIMOTHY D",
        "ownership_nature": "D",
        "price_per_share": 170.0,
        "scraped_at": pd.Timestamp("2024-03-13 06:00:00", tz=_UTC),
        "security_title": "Common Stock",
        "shares": 100.0,
        "shares_owned_after": 5000.0,
        "table": "sec_insider_transactions_log",
        "transaction_code": "S",
        "transaction_date": "2024-03-11",
    }
    row.update(overrides)
    return row


def _seed_insiders(fake_lake: Any, rows: list[dict[str, Any]]) -> None:
    fake_lake.add(
        "sec_insider_transactions_log",
        pd.DataFrame(rows),
        date_columns=("dt", "filed_date", "transaction_date"),
    )


@pytest.mark.parametrize(
    ("accepted_utc", "available", "same_day_visible"),
    [
        (_EARLY, "2024-03-12", True),
        (_LATE, "2024-03-13", False),
        ("2024-03-12 19:59:59", "2024-03-12", True),
        ("2024-03-12 20:00:00", "2024-03-13", False),
        (None, "2024-03-13", False),
    ],
    ids=["early", "late", "just-before-close", "at-close", "no-filing-row"],
)
def test_insider_transactions_gate_on_acceptance_time(
    fake_lake: Any, accepted_utc: str | None, available: str, same_day_visible: bool
) -> None:
    """Filings accepted before 16:00 ET are available on the filed date, others the next session."""
    _seed_filings(fake_lake, {"0001-24-000001": accepted_utc or _EARLY, "OTHER": _EARLY})
    accession = "0001-24-000001" if accepted_utc else "0001-24-NOFILING"
    _seed_insiders(fake_lake, [_insider_row(accession)])

    same_day = lakehouse_loader.load_insider_transactions_from_lakehouse("AAPL", date(2024, 3, 12))
    next_day = lakehouse_loader.load_insider_transactions_from_lakehouse("AAPL", date(2024, 3, 13))

    assert (same_day is not None) is same_day_visible
    assert next_day is not None
    assert len(next_day) == 1
    assert next_day["available_date"].iloc[0] == pd.Timestamp(available)
    assert str(next_day["available_date"].dtype).startswith("datetime64")
    assert next_day["filed_date"].iloc[0] == pd.Timestamp("2024-03-12")
    assert {"owner_cik", "shares", "price_per_share", "transaction_date"} <= set(next_day.columns)


def test_insider_transactions_over_weekend_roll_to_next_session(fake_lake: Any) -> None:
    """A late Friday filing becomes available on Monday."""
    _seed_filings(fake_lake, {"A1": "2024-03-15 21:30:00"})
    _seed_insiders(fake_lake, [_insider_row("A1", filed_date="2024-03-15")])

    assert (
        lakehouse_loader.load_insider_transactions_from_lakehouse("AAPL", date(2024, 3, 15)) is None
    )
    result = lakehouse_loader.load_insider_transactions_from_lakehouse("AAPL", date(2024, 3, 18))

    assert result is not None
    assert result["available_date"].iloc[0] == pd.Timestamp("2024-03-18")


def test_insider_transactions_filing_listed_under_two_tickers_uses_latest_acceptance(
    fake_lake: Any,
) -> None:
    """A filing listed once per issuer ticker yields one row, gated on its latest acceptance."""
    fake_lake.add(
        "sec_filings",
        pd.DataFrame(
            {
                "accession_number": ["A1", "A1"],
                "ticker": ["AAPL", "MSFT"],
                "accepted_at": [pd.Timestamp(_EARLY, tz=_UTC), pd.Timestamp(_LATE, tz=_UTC)],
            }
        ),
    )
    _seed_insiders(fake_lake, [_insider_row("A1")])

    same_day = lakehouse_loader.load_insider_transactions_from_lakehouse("AAPL", date(2024, 3, 12))
    result = lakehouse_loader.load_insider_transactions_from_lakehouse("AAPL", date(2024, 3, 13))

    assert same_day is None
    assert result is not None
    assert len(result) == 1
    assert result["available_date"].iloc[0] == pd.Timestamp("2024-03-13")


def test_insider_transactions_deduplicate_scrapes_but_keep_distinct_lines(fake_lake: Any) -> None:
    """Lines differing in shares or price are kept; the same line from two scrapes appears once."""
    _seed_filings(fake_lake, {"A1": _EARLY})
    later_scrape = pd.Timestamp("2024-03-14 06:00:00", tz=_UTC)
    _seed_insiders(
        fake_lake,
        [
            _insider_row("A1", shares=100.0),
            _insider_row("A1", shares=250.0),
            _insider_row("A1", shares=100.0, price_per_share=171.0),
            _insider_row("A1", shares=100.0, scraped_at=later_scrape, dt="2024-03-14"),
            _insider_row("A1", shares=250.0, scraped_at=later_scrape, dt="2024-03-14"),
            _insider_row("B2", issuer_ticker="MSFT"),
        ],
    )

    result = lakehouse_loader.load_insider_transactions_from_lakehouse("AAPL", date(2024, 3, 20))

    assert result is not None
    assert len(result) == 3
    assert sorted(zip(result["shares"], result["price_per_share"], strict=True)) == [
        (100.0, 170.0),
        (100.0, 171.0),
        (250.0, 170.0),
    ]
    assert (
        lakehouse_loader.load_insider_transactions_from_lakehouse("ZZZZ", date(2024, 3, 20)) is None
    )


def _fact_row(accession: str) -> dict[str, Any]:
    return {
        "accession_number": accession,
        "cik": 320193,
        "concept": "Revenues",
        "dt": "2024-03-13",
        "filed_date": "2024-03-12",
        "fiscal_period": "Q1",
        "fiscal_year": 2024,
        "form": "10-Q",
        "period_end": "2023-12-30",
        "period_start": "2023-10-01",
        "scraped_at": pd.Timestamp("2024-03-13", tz=_UTC),
        "table": "sec_company_facts",
        "ticker": "AAPL",
        "unit": "USD",
        "value": 119.0e9,
    }


@pytest.mark.parametrize(
    ("accepted_utc", "available", "same_day_visible"),
    [(_EARLY, "2024-03-12", True), (_LATE, "2024-03-13", False)],
    ids=["before-close", "after-close"],
)
def test_company_facts_gate_on_acceptance_time(
    fake_lake: Any, accepted_utc: str, available: str, same_day_visible: bool
) -> None:
    """Facts from a filing accepted before 16:00 ET are available that day, otherwise the
    next session."""
    _seed_filings(fake_lake, {"F1": accepted_utc})
    fake_lake.add(
        "sec_company_facts",
        pd.DataFrame([_fact_row("F1")]),
        date_columns=("dt", "filed_date", "period_end", "period_start"),
    )

    same_day = lakehouse_loader.load_company_facts_from_lakehouse("AAPL", date(2024, 3, 12))
    next_day = lakehouse_loader.load_company_facts_from_lakehouse("AAPL", date(2024, 3, 13))

    assert (same_day is not None) is same_day_visible
    assert next_day is not None
    assert next_day["available_date"].iloc[0] == pd.Timestamp(available)
    assert str(next_day["available_date"].dtype).startswith("datetime64")
    assert {"concept", "value", "fiscal_period", "filed"} <= set(next_day.columns)
    assert next_day["filed"].iloc[0] == pd.Timestamp("2024-03-12")
