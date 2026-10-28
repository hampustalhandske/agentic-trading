# test_sp500_membership.py

"""`load_sp500_constituents_asof` returns one current ticker per company for a date."""

from datetime import date
from typing import Any

import pytest

from agentic_trading.data.loaders.lakehouse_loader import load_sp500_constituents_asof

# (ticker, start_date, end_date, renamed_from, renamed_to)
_MEMBERSHIP = [
    ("FB", "2013-12-23", "2022-06-09", None, "META"),
    ("META", "2022-06-09", None, "FB", None),
    ("X", "2015-01-02", "2020-01-02", None, None),
    ("GOOG", "2014-04-03", None, None, None),
    ("GOOGL", "2006-03-31", None, None, None),
    ("NOHIST", "2010-01-04", None, None, None),
]

# (security_id, ticker, valid_from, valid_to)
_HISTORY = [
    (1001, "FB", "2012-05-21", "2022-06-09"),
    (1001, "META", "2022-06-09", None),
    (2002, "X", "2000-01-03", None),
    (3001, "GOOG", "2014-04-03", None),
    (3002, "GOOGL", "2004-08-19", None),
]


def _seed(fake_lake: Any) -> None:
    fake_lake.add_empty(
        "sp500_membership",
        {
            "ticker": "VARCHAR",
            "start_date": "DATE",
            "end_date": "DATE",
            "scraped_at": "TIMESTAMP",
            "renamed_from": "VARCHAR",
            "renamed_to": "VARCHAR",
            "dt": "DATE",
            '"table"': "VARCHAR",
        },
    )
    fake_lake.add_empty(
        "equity_ticker_history",
        {"security_id": "BIGINT", "ticker": "VARCHAR", "valid_from": "DATE", "valid_to": "DATE"},
    )
    connection = fake_lake._connection
    for ticker, start, end, renamed_from, renamed_to in _MEMBERSHIP:
        connection.execute(
            "INSERT INTO sp500_membership VALUES (?, ?, ?, NULL, ?, ?, NULL, NULL)",
            [ticker, start, end, renamed_from, renamed_to],
        )
    for row in _HISTORY:
        connection.execute("INSERT INTO equity_ticker_history VALUES (?, ?, ?, ?)", list(row))


@pytest.fixture
def lake(fake_lake: Any) -> Any:
    _seed(fake_lake)
    return fake_lake


def test_rename_resolves_to_the_ticker_valid_on_the_date(lake: Any) -> None:
    """On the rename day META is returned and FB is not; the day before, the reverse."""
    on_rename = load_sp500_constituents_asof(date(2022, 6, 9))
    before_rename = load_sp500_constituents_asof(date(2022, 6, 8))

    assert on_rename is not None and before_rename is not None
    assert "META" in on_rename and "FB" not in on_rename
    assert "FB" in before_rename and "META" not in before_rename


def test_end_date_is_exclusive_and_start_date_inclusive(lake: Any) -> None:
    """A removed member is absent on its end date, present the day before and on its start."""
    assert "X" not in (load_sp500_constituents_asof(date(2020, 1, 2)) or [])
    assert "X" in (load_sp500_constituents_asof(date(2020, 1, 1)) or [])
    assert "X" in (load_sp500_constituents_asof(date(2015, 1, 2)) or [])
    assert "X" not in (load_sp500_constituents_asof(date(2015, 1, 1)) or [])


def test_members_without_history_and_distinct_securities_are_kept_sorted(lake: Any) -> None:
    """Members without ticker history are kept, GOOG and GOOGL both appear, sorted and unique."""
    result = load_sp500_constituents_asof(date(2024, 1, 2))

    assert result is not None
    assert {"NOHIST", "GOOG", "GOOGL", "META"} <= set(result)
    assert len(result) == len(set(result))
    assert result == sorted(result)
    assert "X" not in result and "FB" not in result


def test_one_entry_per_security_prefers_later_start_date(lake: Any) -> None:
    """Two membership rows resolving to one security yield only the later-started ticker."""
    lake._connection.execute("DELETE FROM equity_ticker_history WHERE security_id = 1001")
    lake._connection.execute(
        "INSERT INTO equity_ticker_history VALUES (1001, 'FB', '2012-05-21', NULL), "
        "(1001, 'META', '2012-05-21', NULL)"
    )
    lake._connection.execute("UPDATE sp500_membership SET end_date = NULL WHERE ticker = 'FB'")

    result = load_sp500_constituents_asof(date(2023, 1, 3))

    assert result is not None
    assert "META" in result and "FB" not in result
