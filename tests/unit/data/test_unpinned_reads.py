# test_unpinned_reads.py

"""A loader read with no `lake_session` open is refused with `UnpinnedLakeReadError`, naming
the loader and the table, before any query is sent. A caller opts in to unpinned reads with
`unpinned_lake_reads(reason)`: inside that block, with no session, a read names its table bare
and sees the lake's current views; inside a session every read stays pinned."""

import threading
from datetime import UTC, date, datetime
from typing import Any

import pandas as pd
import pytest

from agentic_trading.data.loaders import lakehouse_loader as ll

_T = datetime(2024, 8, 1, tzinfo=UTC)


def _add_dividends(lake: Any) -> None:
    lake.add(
        "equity_distributions",
        pd.DataFrame(
            {
                "ticker": ["AAA"],
                "ex_date": ["2024-01-10"],
                "amount_split_adjusted": [0.5],
                "distribution_type": ["regular_cash"],
            }
        ),
        ("ex_date",),
        key_columns=["ticker", "ex_date", "distribution_type"],
    )


def test_a_read_with_no_session_is_refused_naming_the_loader_and_table(fake_lake: Any) -> None:
    """A public loader called with no session open raises before sending a query; the message
    names the loader and the table it reads."""
    _add_dividends(fake_lake)

    with pytest.raises(ll.UnpinnedLakeReadError) as refused:
        ll.load_dividends_from_lakehouse("AAA", date(2024, 2, 1))

    assert "load_dividends_from_lakehouse" in str(refused.value)
    assert "equity_distributions" in str(refused.value)
    assert fake_lake.calls == []


def test_a_table_named_outside_any_loader_is_refused_naming_it() -> None:
    """`_table_sql` called outside any public loader names the table and says no loader was in
    progress."""
    with pytest.raises(ll.UnpinnedLakeReadError, match=r"outside any public loader.*sp500"):
        ll._table_sql("sp500_membership")


def test_the_opt_in_reads_the_current_table_bare(fake_lake: Any) -> None:
    """Inside `unpinned_lake_reads`, with no session, the loader reads the bare table and
    returns its rows; the refusal is back once the block closes."""
    _add_dividends(fake_lake)

    with ll.unpinned_lake_reads("a test of the opt-in"):
        assert ll._table_sql("equity_distributions") == "equity_distributions"
        dividends = ll.load_dividends_from_lakehouse("AAA", date(2024, 2, 1))

    assert dividends is not None
    assert dividends.tolist() == [0.5]
    with pytest.raises(ll.UnpinnedLakeReadError):
        ll.load_dividends_from_lakehouse("AAA", date(2024, 2, 1))


def test_the_opt_in_reaches_worker_threads_and_nests(fake_lake: Any) -> None:
    """The opt-in is process-wide, as a session is: a worker thread started inside the block
    reads too, and an inner block closing leaves the outer one open."""
    outcomes: list[str] = []

    def read() -> None:
        outcomes.append(ll._table_sql("instrument_prices"))

    with ll.unpinned_lake_reads("outer"):
        with ll.unpinned_lake_reads("inner"):
            pass
        worker = threading.Thread(target=read)
        worker.start()
        worker.join()

    assert outcomes == ["instrument_prices"]


def test_inside_a_session_the_opt_in_changes_nothing(fake_lake: Any) -> None:
    """A read inside a session is pinned whether or not an opt-in block is open."""
    _add_dividends(fake_lake)

    with ll.lake_session(_T), ll.unpinned_lake_reads("a test of the opt-in"):
        relation = ll._table_sql("equity_distributions")

    assert relation != "equity_distributions"
    assert "equity_distributions_log" in relation


def test_the_opt_in_needs_a_reason() -> None:
    with pytest.raises(ValueError, match="reason"), ll.unpinned_lake_reads("  "):
        pass
