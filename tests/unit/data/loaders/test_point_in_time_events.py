# test_point_in_time_events.py

"""Point-in-time tests for the short-interest, earnings, insider and company-facts loaders.

Decisions are taken at the 16:00 America/New_York close of `as_of_date`. FINRA short
interest is gated on the loader's own dissemination estimate, earnings on the
announcement session implied by the `when` field, and SEC data on the next NYSE session
after the later of the filing's filed date and its New York acceptance date.
"""

from datetime import UTC, date, datetime, timedelta
from typing import Any

import pandas as pd
import pytest

from agentic_trading.agents.roster import FIRM_V1_EQUITIES
from agentic_trading.data.loaders import lakehouse_loader

pytestmark = pytest.mark.unpinned_lake_reads


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


def test_earnings_surprise_before_open_and_null_wait_a_session(fake_lake: Any) -> None:
    """Before-open announcements, like a NULL `when`, are available from the next session."""
    _seed_earnings(fake_lake, ["Before market open", None])

    assert lakehouse_loader.load_earnings_surprise_from_lakehouse("AAPL", date(2024, 5, 2)) is None
    assert lakehouse_loader.load_earnings_surprise_from_lakehouse("AAPL", date(2024, 5, 3)) == (
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
    ("accepted_utc", "available"),
    [
        (_EARLY, date(2024, 3, 13)),
        (_LATE, date(2024, 3, 13)),
        ("2024-03-13 03:59:59", date(2024, 3, 13)),
        ("2024-03-13 04:00:00", date(2024, 3, 14)),
        (None, date(2024, 3, 13)),
    ],
    ids=["before-close", "after-close", "ny-day-end", "ny-next-day", "no-filing-row"],
)
def test_insider_transactions_gate_on_acceptance_time(
    fake_lake: Any, accepted_utc: str | None, available: date
) -> None:
    """A filing filed 2024-03-12 is available from the session after the later of its filed
    date and its New York acceptance date, and not on the filed date or the day before that."""
    _seed_filings(fake_lake, {"0001-24-000001": accepted_utc or _EARLY, "OTHER": _EARLY})
    accession = "0001-24-000001" if accepted_utc else "0001-24-NOFILING"
    _seed_insiders(fake_lake, [_insider_row(accession)])

    for as_of in (date(2024, 3, 12), available - timedelta(days=1)):
        assert lakehouse_loader.load_insider_transactions_from_lakehouse("AAPL", as_of) is None
    result = lakehouse_loader.load_insider_transactions_from_lakehouse("AAPL", available)

    assert result is not None
    assert len(result) == 1
    assert result["available_date"].iloc[0] == pd.Timestamp(available)
    assert str(result["available_date"].dtype).startswith("datetime64")
    assert result["filed_date"].iloc[0] == pd.Timestamp("2024-03-12")
    assert {"owner_cik", "shares", "price_per_share", "transaction_date"} <= set(result.columns)


def test_insider_transactions_over_weekend_roll_to_next_session(fake_lake: Any) -> None:
    """A Friday filing accepted before the close becomes available on Monday."""
    _seed_filings(fake_lake, {"A1": "2024-03-15 14:00:00"})
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
                "accepted_at": [
                    pd.Timestamp(_EARLY, tz=_UTC),
                    pd.Timestamp("2024-03-13 14:00:00", tz=_UTC),
                ],
            }
        ),
    )
    _seed_insiders(fake_lake, [_insider_row("A1")])

    first_session = lakehouse_loader.load_insider_transactions_from_lakehouse(
        "AAPL", date(2024, 3, 13)
    )
    result = lakehouse_loader.load_insider_transactions_from_lakehouse("AAPL", date(2024, 3, 14))

    assert first_session is None
    assert result is not None
    assert len(result) == 1
    assert result["available_date"].iloc[0] == pd.Timestamp("2024-03-14")


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
            _insider_row("B2", issuer_ticker="MSFT", issuer_cik=789019),
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


_GOOGLE_INC = 1288776
_ALPHABET = 1652044
_AFTER_FILINGS = date(2018, 1, 2)
_INSIDER_COLUMNS = [
    "owner_cik",
    "owner_name",
    "is_officer",
    "is_director",
    "is_ten_percent_owner",
    "transaction_date",
    "transaction_code",
    "shares",
    "price_per_share",
    "acquired_disposed",
    "filed_date",
    "available_date",
]


def _issuer_rows() -> list[dict[str, Any]]:
    """One Form 4 line per owner, stored out of filed order: Google Inc. (owner 1) and Alphabet
    (owner 2) printed as GOOG, another issuer printed as GOOG (owner 3), Apple printed as AAPL
    (owner 4) and as APPLE (owner 5), Microsoft (owner 6), an issuer outside the reviewed
    map printed as GOOGL (owner 7), another issuer printed as C (owner 8) and Citigroup printed
    as C (owner 9)."""
    lines = [
        ("G2", "GOOG", _ALPHABET, 2, "2016-03-01"),
        ("G1", "GOOG", _GOOGLE_INC, 1, "2015-03-02"),
        ("G3", "GOOG", 999999, 3, "2015-06-01"),
        ("L1", "GOOGL", 7777777, 7, "2017-06-01"),
        ("A1", "AAPL", 320193, 4, "2015-04-01"),
        ("A2", "APPLE", 320193, 5, "2015-05-01"),
        ("M1", "MSFT", 789019, 6, "2015-07-01"),
        ("C1", "C", 1043325, 8, "2015-08-03"),
        ("C2", "C", 831001, 9, "2015-09-01"),
    ]
    return [
        _insider_row(
            accession,
            issuer_ticker=ticker,
            issuer_cik=cik,
            owner_cik=owner,
            filed_date=filed,
            transaction_date=filed,
        )
        for accession, ticker, cik, owner, filed in lines
    ]


def _seed_issuer_rows(fake_lake: Any, rows: list[dict[str, Any]]) -> None:
    """Seeds `rows` with each filing accepted before the close on its filed date."""
    _seed_filings(
        fake_lake, {row["accession_number"]: f"{row['filed_date']} 14:00:00" for row in rows}
    )
    _seed_insiders(fake_lake, rows)


def test_issuer_ciks_by_ticker_lists_one_cik_per_firm_equity() -> None:
    """The reviewed map covers exactly firm-v1's equities, one issuer CIK each, except GOOGL,
    which has Google Inc. and Alphabet."""
    ciks = lakehouse_loader.ISSUER_CIKS_BY_TICKER
    assert set(ciks) == set(FIRM_V1_EQUITIES)
    assert ciks["GOOGL"] == (_GOOGLE_INC, _ALPHABET)
    assert ciks["C"] == (831001,)
    assert all(len(found) == 1 for ticker, found in ciks.items() if ticker != "GOOGL")


@pytest.mark.parametrize(
    ("ticker", "owners"),
    [
        ("GOOGL", [1, 2]),
        ("GOOG", None),
        ("AAPL", [4, 5]),
        ("MSFT", [6]),
        ("C", [9]),
        ("ZZZZ", None),
    ],
    ids=[
        "mapped-ciks-other-print",
        "unlisted-ticker",
        "listed-cik-other-print",
        "single-cik",
        "foreign-cik-printed-as-ticker",
        "unknown",
    ],
)
def test_insider_transactions_select_rows_by_issuer_cik(
    fake_lake: Any, ticker: str, owners: list[int] | None
) -> None:
    """A ticker gets the lines of its listed issuer CIKs whatever their printed ticker, and no
    line of any other CIK, even one printed with the ticker."""
    _seed_issuer_rows(fake_lake, _issuer_rows())

    result = lakehouse_loader.load_insider_transactions_from_lakehouse(ticker, _AFTER_FILINGS)

    if owners is None:
        assert result is None
        return
    assert result is not None
    assert list(result.columns) == _INSIDER_COLUMNS
    assert list(result["owner_cik"]) == owners
    assert result["filed_date"].is_monotonic_increasing


@pytest.mark.parametrize(
    ("accepted_utc", "available"),
    [("2016-03-01 14:00:00", date(2016, 3, 2)), ("2016-03-02 14:00:00", date(2016, 3, 3))],
    ids=["accepted-on-filed-date", "accepted-day-after-filed"],
)
def test_insider_transactions_through_ciks_gate_on_filing_not_trade_date(
    fake_lake: Any, accepted_utc: str, available: date
) -> None:
    """A GOOG line filed 2016-03-01 but dated 2027-01-25 is gated for GOOGL on its filing alone."""
    _seed_filings(fake_lake, {"G2": accepted_utc})
    _seed_insiders(
        fake_lake,
        [
            _insider_row(
                "G2",
                issuer_ticker="GOOG",
                issuer_cik=_ALPHABET,
                filed_date="2016-03-01",
                transaction_date="2027-01-25",
            )
        ],
    )

    for as_of in (date(2016, 3, 1), available - timedelta(days=1)):
        assert lakehouse_loader.load_insider_transactions_from_lakehouse("GOOGL", as_of) is None
    result = lakehouse_loader.load_insider_transactions_from_lakehouse("GOOGL", available)

    assert result is not None
    assert list(result["available_date"]) == [pd.Timestamp(available)]
    assert list(result["filed_date"]) == [pd.Timestamp("2016-03-01")]
    assert list(result["transaction_date"]) == [pd.Timestamp("2027-01-25")]


def test_insider_transactions_through_ciks_count_each_line_once(fake_lake: Any) -> None:
    """A line from two scrapes counts once, reached through a mapped CIK or its own ticker."""
    later_scrape = pd.Timestamp("2024-03-14 06:00:00", tz=_UTC)
    alphabet: dict[str, Any] = {
        "issuer_ticker": "GOOG",
        "issuer_cik": _ALPHABET,
        "filed_date": "2016-03-01",
        "transaction_date": "2016-02-26",
    }
    _seed_filings(fake_lake, {"G2": "2016-03-01 14:00:00", "A1": _EARLY})
    _seed_insiders(
        fake_lake,
        [
            _insider_row("G2", shares=100.0, **alphabet),
            _insider_row("G2", shares=100.0, scraped_at=later_scrape, dt="2024-03-14", **alphabet),
            _insider_row("G2", shares=250.0, **alphabet),
            _insider_row("A1"),
            _insider_row("A1", scraped_at=later_scrape, dt="2024-03-14"),
        ],
    )

    googl = lakehouse_loader.load_insider_transactions_from_lakehouse("GOOGL", date(2024, 3, 20))
    aapl = lakehouse_loader.load_insider_transactions_from_lakehouse("AAPL", date(2024, 3, 20))

    assert googl is not None
    assert sorted(googl["shares"]) == [100.0, 250.0]
    assert aapl is not None
    assert len(aapl) == 1


def test_insider_transactions_through_ciks_read_scrapes_up_to_lake_time(fake_lake: Any) -> None:
    """Inside a lake session a line reached through a CIK is returned only if scraped by T."""
    rows = _issuer_rows()[:2]
    rows[0]["scraped_at"] = pd.Timestamp("2021-06-01", tz=_UTC)
    rows[1]["scraped_at"] = pd.Timestamp("2020-06-01", tz=_UTC)
    _seed_issuer_rows(fake_lake, rows)

    with lakehouse_loader.lake_session(datetime(2021, 1, 1, tzinfo=UTC)):
        pinned = lakehouse_loader.load_insider_transactions_from_lakehouse("GOOGL", _AFTER_FILINGS)
    unpinned = lakehouse_loader.load_insider_transactions_from_lakehouse("GOOGL", _AFTER_FILINGS)

    assert pinned is not None
    assert list(pinned["owner_cik"]) == [1]
    assert unpinned is not None
    assert list(unpinned["owner_cik"]) == [1, 2]


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
        "table": "sec_company_facts_log",
        "ticker": "AAPL",
        "unit": "USD",
        "value": 119.0e9,
    }


@pytest.mark.parametrize(
    ("accepted_utc", "available"),
    [
        (_EARLY, date(2024, 3, 13)),
        (_LATE, date(2024, 3, 13)),
        ("2024-03-13 14:00:00", date(2024, 3, 14)),
    ],
    ids=["before-close", "after-close", "accepted-day-after-filed"],
)
def test_company_facts_gate_on_acceptance_time(
    fake_lake: Any, accepted_utc: str, available: date
) -> None:
    """Facts from a filing filed 2024-03-12 are available from the session after the later of
    its filed date and its New York acceptance date, and not on the filed date or the day
    before that."""
    _seed_filings(fake_lake, {"F1": accepted_utc})
    fake_lake.add(
        "sec_company_facts_log",
        pd.DataFrame([_fact_row("F1")]),
        date_columns=("dt", "filed_date", "period_end", "period_start"),
    )

    for as_of in (date(2024, 3, 12), available - timedelta(days=1)):
        assert lakehouse_loader.load_company_facts_from_lakehouse("AAPL", as_of) is None
    next_day = lakehouse_loader.load_company_facts_from_lakehouse("AAPL", available)

    assert next_day is not None
    assert next_day["available_date"].iloc[0] == pd.Timestamp(available)
    assert str(next_day["available_date"].dtype).startswith("datetime64")
    assert {"concept", "value", "fiscal_period", "filed"} <= set(next_day.columns)
    assert next_day["filed"].iloc[0] == pd.Timestamp("2024-03-12")


_NY = "America/New_York"


@pytest.mark.parametrize(
    ("filed", "accepted_at", "available"),
    [
        ("2024-03-12", pd.Timestamp("2024-03-12 10:00", tz=_NY), "2024-03-13"),
        ("2024-03-12", pd.Timestamp("2024-03-12 18:00", tz=_NY), "2024-03-13"),
        ("2024-03-12", pd.Timestamp("2024-03-13 09:00", tz=_NY), "2024-03-14"),
        ("2024-03-12", pd.Timestamp("2024-03-11 18:00", tz=_NY), "2024-03-13"),
        ("2024-03-15", pd.Timestamp("2024-03-15 10:00", tz=_NY), "2024-03-18"),
        ("2024-02-16", pd.Timestamp("2024-02-16 10:00", tz=_NY), "2024-02-20"),
        ("2024-03-28", pd.Timestamp("2024-03-28 10:00", tz=_NY), "2024-04-01"),
        ("2024-03-12", None, "2024-03-13"),
        ("2024-03-15", None, "2024-03-18"),
        ("2024-03-12", pd.Timestamp("2024-03-13 02:00", tz=_UTC), "2024-03-13"),
        ("2024-03-12", pd.Timestamp("2024-03-13 05:00", tz=_UTC), "2024-03-14"),
        ("2024-01-16", pd.Timestamp("2024-01-17 04:30", tz=_UTC), "2024-01-17"),
        ("2024-03-08", pd.Timestamp("2024-03-10 06:30", tz=_UTC), "2024-03-11"),
        ("2024-11-01", pd.Timestamp("2024-11-03 05:30", tz=_UTC), "2024-11-04"),
        ("2024-11-01", pd.Timestamp("2024-11-04 04:30", tz=_UTC), "2024-11-04"),
    ],
    ids=[
        "pre-close-on-filed-date",
        "after-close-on-filed-date",
        "accepted-day-after-filed",
        "accepted-day-before-filed",
        "friday-to-monday",
        "friday-over-presidents-day",
        "thursday-over-good-friday",
        "missing-acceptance",
        "missing-acceptance-friday",
        "utc-next-day-is-ny-filed-date",
        "utc-next-day-is-ny-next-day",
        "winter-utc-next-day-is-est-filed-date",
        "dst-start-day",
        "dst-end-day",
        "first-est-night-after-dst-end",
    ],
)
def test_sec_available_dates_is_session_after_later_of_filed_and_ny_acceptance(
    filed: str, accepted_at: pd.Timestamp | None, available: str
) -> None:
    """A filing is known at the close of the next NYSE session after the later of its filed
    date and the New York calendar date of its acceptance; with no acceptance, after the
    filed date."""
    result = lakehouse_loader._sec_available_dates(
        pd.Series([pd.Timestamp(filed)]), pd.Series([accepted_at], dtype=object)
    )

    assert result.iloc[0] == pd.Timestamp(available)
    assert str(result.dtype).startswith("datetime64")


def test_sec_available_dates_keeps_filed_index_and_mixes_missing_acceptance() -> None:
    """The result is aligned with `filed`; present and missing acceptances resolve per row."""
    filed = pd.Series(pd.to_datetime(["2024-03-12", "2024-03-15"]), index=[7, 3])
    accepted = pd.Series([pd.Timestamp("2024-03-12 21:30", tz=_UTC), None], index=[7, 3])

    result = lakehouse_loader._sec_available_dates(filed, accepted)

    assert list(result.index) == [7, 3]
    assert list(result) == [pd.Timestamp("2024-03-13"), pd.Timestamp("2024-03-18")]


@pytest.mark.parametrize("unit", ["ns", "us", "s"])
def test_sec_available_dates_keeps_the_resolution_of_filed(unit: str) -> None:
    """The result has the datetime64 resolution of `filed`."""
    filed = pd.Series(pd.to_datetime(["2024-03-12", "2024-03-15"]).as_unit(unit))
    accepted = pd.Series([pd.Timestamp("2024-03-12 21:30", tz=_UTC), None])

    result = lakehouse_loader._sec_available_dates(filed, accepted)

    assert result.dtype == filed.dtype
    assert list(result) == [pd.Timestamp("2024-03-13"), pd.Timestamp("2024-03-18")]


def _fetched_insider_frame(accepted_at: pd.Timestamp | None) -> pd.DataFrame:
    """One `_fetch_insider_transactions` row of a Form 4 purchase filed 2024-03-12."""
    return pd.DataFrame(
        {
            "owner_cik": [1214156],
            "owner_name": ["COOK TIMOTHY D"],
            "is_officer": [True],
            "is_director": [False],
            "is_ten_percent_owner": [False],
            "transaction_date": [pd.Timestamp("2024-03-11")],
            "transaction_code": ["P"],
            "shares": [100.0],
            "price_per_share": [170.0],
            "acquired_disposed": ["A"],
            "filed_date": [pd.Timestamp("2024-03-12")],
            "accepted_at": pd.Series([accepted_at], dtype="datetime64[ns, UTC]"),
        }
    )


def _fetched_fact_frame(accepted_at: pd.Timestamp | None) -> pd.DataFrame:
    """One `_fetch_company_facts` row of a 10-Q filed 2024-03-12."""
    return pd.DataFrame(
        {
            "concept": ["Revenues"],
            "unit": ["USD"],
            "period_start": [pd.Timestamp("2023-10-01")],
            "period_end": [pd.Timestamp("2023-12-30")],
            "value": [119.0e9],
            "fiscal_year": [2024],
            "fiscal_period": ["Q1"],
            "form": ["10-Q"],
            "filed": [pd.Timestamp("2024-03-12")],
            "accepted_at": pd.Series([accepted_at], dtype="datetime64[ns, UTC]"),
        }
    )


_PRE_CLOSE = pd.Timestamp("2024-03-12 10:00", tz=_NY).tz_convert(_UTC)


@pytest.mark.parametrize("accepted_at", [_PRE_CLOSE, None], ids=["pre-close", "missing"])
def test_select_insider_transactions_excludes_filed_date_includes_next_session(
    accepted_at: pd.Timestamp | None,
) -> None:
    """A Form 4 filed and accepted before the close on D is not returned at D, only from the
    next session."""
    frame = _fetched_insider_frame(accepted_at)

    assert lakehouse_loader._select_insider_transactions(frame, date(2024, 3, 12)) is None
    result = lakehouse_loader._select_insider_transactions(frame, date(2024, 3, 13))

    assert result is not None
    assert list(result["available_date"]) == [pd.Timestamp("2024-03-13")]
    assert "accepted_at" not in result.columns


@pytest.mark.parametrize("accepted_at", [_PRE_CLOSE, None], ids=["pre-close", "missing"])
def test_select_company_facts_excludes_filed_date_includes_next_session(
    accepted_at: pd.Timestamp | None,
) -> None:
    """Facts of a filing filed and accepted before the close on D are not returned at D, only
    from the next session."""
    frame = _fetched_fact_frame(accepted_at)

    assert lakehouse_loader._select_company_facts(frame, date(2024, 3, 12)) is None
    result = lakehouse_loader._select_company_facts(frame, date(2024, 3, 13))

    assert result is not None
    assert list(result["available_date"]) == [pd.Timestamp("2024-03-13")]
    assert "accepted_at" not in result.columns


def _restated_fact_frame() -> pd.DataFrame:
    """Two `_fetch_company_facts` rows of one period, in fetch order (`filed` first): Y, filed
    2024-03-12 but accepted 2024-03-20 10:00 ET (value 100), and X, filed and accepted
    2024-03-15 10:00 ET (value 200)."""
    y = _fetched_fact_frame(pd.Timestamp("2024-03-20 10:00", tz=_NY).tz_convert(_UTC))
    x = _fetched_fact_frame(pd.Timestamp("2024-03-15 10:00", tz=_NY).tz_convert(_UTC))
    y["value"] = 100.0
    x["value"] = 200.0
    x["filed"] = pd.Timestamp("2024-03-15")
    return pd.concat([y, x], ignore_index=True)


@pytest.mark.parametrize("reverse", [False, True], ids=["fetch-order", "reversed"])
def test_select_company_facts_keeps_latest_filed_known_row_per_period(reverse: bool) -> None:
    """At 2024-03-22 both rows are known (Y from 2024-03-21, X from 2024-03-18); X, the later
    filing, is kept although Y became available later."""
    frame = _restated_fact_frame()
    if reverse:
        frame = frame.iloc[::-1].reset_index(drop=True)

    result = lakehouse_loader._select_company_facts(frame, date(2024, 3, 22))

    assert result is not None
    assert list(result["value"]) == [200.0]
    assert list(result["filed"]) == [pd.Timestamp("2024-03-15")]
    assert list(result["available_date"]) == [pd.Timestamp("2024-03-18")]


def test_select_company_facts_keeps_known_row_before_restatement_is_available() -> None:
    """At 2024-03-20 only X is known; at 2024-03-15 neither is."""
    frame = _restated_fact_frame()

    assert lakehouse_loader._select_company_facts(frame, date(2024, 3, 15)) is None
    result = lakehouse_loader._select_company_facts(frame, date(2024, 3, 20))

    assert result is not None
    assert list(result["value"]) == [200.0]
