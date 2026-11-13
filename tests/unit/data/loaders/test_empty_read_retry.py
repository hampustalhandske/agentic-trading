# test_empty_read_retry.py

"""A table-level read of a table that always holds rows (`ALWAYS_POPULATED_TABLES`) that comes
back with no rows is retried once after `EMPTY_READ_RETRY_SECONDS`; a second empty read raises
`EmptyTableReadError`. A read filtered to one name is never retried, a failed request is never
retried, and HTTP errors from the lake propagate with the lake's message."""

import math
import time
from collections.abc import Callable
from contextlib import nullcontext
from datetime import UTC, date, datetime
from typing import Any

import pandas as pd
import pytest

from agentic_trading.data.lakehouse_client import LakehouseError
from agentic_trading.data.loaders import lakehouse_loader as ll

pytestmark = pytest.mark.unpinned_lake_reads

_WAIT = 0.25
_T = datetime(2024, 6, 1, 12, tzinfo=UTC)
_MON, _TUE = date(2024, 3, 4), date(2024, 3, 5)
_COT_DAY = date(2018, 12, 14)
_ZN, _CL, _GC = "043602", "067651", "088691"
_REQUIRED = (
    "cftc_cot_publication",
    "equity_adjustment_factors",
    "yahoo_split_close_basis",
    "sp500_membership",
)


class _Clock:
    """Stands in for `time.sleep`: records each wait in seconds, then runs the `on_wait` hooks."""

    def __init__(self) -> None:
        self.waits: list[float] = []
        self.on_wait: list[Callable[[], None]] = []

    def sleep(self, seconds: float) -> None:
        self.waits.append(seconds)
        for hook in self.on_wait:
            hook()


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    """Replaces `time.sleep` with a `_Clock` and sets the retry wait to `_WAIT` seconds."""
    fake = _Clock()
    monkeypatch.setattr(time, "sleep", fake.sleep)
    monkeypatch.setattr(ll, "EMPTY_READ_RETRY_SECONDS", _WAIT, raising=False)
    return fake


class _ScriptedQuery:
    """Stands in for `_query`: answers each call with the next scripted frame (or raises it when
    it is an exception) and records every `(query, params)`."""

    def __init__(self, *answers: pd.DataFrame | Exception) -> None:
        self.answers = list(answers)
        self.calls: list[tuple[str, list[Any]]] = []

    def __call__(self, query: str, params: list[Any]) -> pd.DataFrame:
        self.calls.append((query, list(params)))
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def _seed(lake: Any) -> None:
    """Populates every table the tests read: security 1 has two bars and a factor row, AAPL a
    4-for-1 split, ZN=F and CL=F one COT report published 2018-12-14, and MSFT prices, earnings
    and a dividend; AAPL, NVDA, GC=F and security 9 have no rows of their own."""
    lake.add(
        "equity_daily_from_bars",
        pd.DataFrame(
            {
                "ticker": ["AAA", "AAA"],
                "date": [_MON, _TUE],
                "open": [100.0, 101.0],
                "close": [100.0, 101.0],
                "volume": [1_000.0, 1_000.0],
                "official_close": [100.0, 101.0],
                "security_id": [1, 1],
            }
        ),
        date_columns=("date",),
    )
    lake.add(
        "equity_adjustment_factors",
        pd.DataFrame(
            {
                "ticker": ["AAA", "BBB", "CCC"],
                "effective_date": ["2024-06-03"] * 3,
                "split_factor": [2.0] * 3,
                "cum_factor": [0.5] * 3,
                "security_id": [1, 2, 3],
            }
        ),
        date_columns=("effective_date",),
    )
    lake.add(
        "yahoo_split_close_basis",
        pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT"],
                "ex_date": ["2024-06-10", "2024-06-10"],
                "ratio": [0.25, 0.5],
                "in_close_basis": [True, True],
            }
        ),
        date_columns=("ex_date",),
    )
    positions = {"report_date": ["2018-12-11"], "open_interest_all": [1000]}
    lake.add(
        "cftc_cot_tff",
        pd.DataFrame(
            {
                "cftc_contract_market_code": [_ZN],
                **positions,
                "leveraged_funds_long": [300],
                "leveraged_funds_short": [100],
            }
        ),
        date_columns=("report_date",),
    )
    lake.add(
        "cftc_cot_disaggregated",
        pd.DataFrame(
            {
                "cftc_contract_market_code": [_CL],
                **positions,
                "managed_money_long": [300],
                "managed_money_short": [100],
            }
        ),
        date_columns=("report_date",),
    )
    lake.add(
        "cftc_cot_publication",
        pd.DataFrame(
            {
                "report_date": ["2018-12-11"],
                "published_date": ["2018-12-14"],
                "published_at": pd.to_datetime(["2018-12-14 20:30"], utc=True),
                "rule_date": ["2018-12-14"],
                "basis": ["test"],
            }
        ),
        date_columns=("report_date", "published_date", "rule_date"),
    )
    lake.add(
        "sp500_membership",
        pd.DataFrame(
            {
                "ticker": ["AAA", "BBB"],
                "start_date": ["2000-01-03", "2005-03-01"],
                "end_date": ["2010-06-30", "2015-12-31"],
            }
        ),
        date_columns=("start_date", "end_date"),
    )
    lake.add(
        "instrument_prices",
        pd.DataFrame(
            {
                "ticker": ["MSFT"],
                "date": [_MON],
                "open": [10.0],
                "high": [11.0],
                "low": [9.0],
                "close": [10.0],
                "adj_close": [10.0],
                "volume": [100],
            }
        ),
        date_columns=("date",),
    )
    lake.add(
        "dolthub_eps_history",
        pd.DataFrame(
            {
                "act_symbol": ["MSFT"],
                "period_end_date": ["2023-12-31"],
                "reported": [1.0],
                "estimate": [0.9],
            }
        ),
        date_columns=("period_end_date",),
    )
    lake.add(
        "dolthub_earnings_calendar",
        pd.DataFrame({"act_symbol": ["MSFT"], "date": ["2024-01-25"]}),
        date_columns=("date",),
    )
    lake.add(
        "equity_distributions",
        pd.DataFrame(
            {
                "ticker": ["MSFT"],
                "ex_date": ["2024-02-14"],
                "distribution_type": ["regular_cash"],
                "amount_split_adjusted": [0.75],
            }
        ),
        date_columns=("ex_date",),
    )


def _rebuild(lake: Any, clock: _Clock, table: str, *, finishes: bool) -> None:
    """Empties `table` as a whole-table rebuild does while it runs; its rows come back at the
    loader's first wait when `finishes`, and never otherwise."""
    held = f"{table}_held"
    lake._connection.execute(f"CREATE TABLE {held} AS SELECT * FROM {table}")
    lake._connection.execute(f"DELETE FROM {table}")
    if finishes:
        pending = [held]

        def restore() -> None:
            if pending:
                lake._connection.execute(f"INSERT INTO {table} SELECT * FROM {pending.pop()}")

        clock.on_wait.append(restore)


def _ratios(reports: Any) -> list[float]:
    """The positioning values of a COT read: a reports frame's ratio column, or a series."""
    values = reports["net_positioning_ratio"] if isinstance(reports, pd.DataFrame) else reports
    return [float(value) for value in values]


# Each public read that reads a whole always-populated table, as (table, call, expected result).
_TABLE_LEVEL_READS = [
    pytest.param(
        "equity_adjustment_factors",
        lambda: ll.adjustment_table_row_count("equity_adjustment_factors"),
        3,
        id="factors_row_count",
    ),
    pytest.param(
        "yahoo_split_close_basis",
        lambda: ll.adjustment_table_row_count("yahoo_split_close_basis"),
        2,
        id="close_basis_row_count",
    ),
    pytest.param(
        "yahoo_split_close_basis",
        lambda: ll.split_adjustment_factor("AAPL", date(2024, 1, 2)),
        4.0,
        id="split_adjustment_factor",
    ),
    pytest.param(
        "equity_adjustment_factors",
        lambda: float(ll.load_security_log_returns([1], _MON, _TUE)[1].iloc[0]),
        pytest.approx(math.log(101 / 100)),
        id="security_log_returns",
    ),
    pytest.param(
        "equity_adjustment_factors",
        lambda: ll.lake_provenance().adjustment_table_rows,
        {"equity_adjustment_factors": 3, "yahoo_split_close_basis": 2},
        id="lake_provenance",
    ),
    pytest.param(
        "cftc_cot_publication",
        lambda: _ratios(ll.load_cot_reports_from_lakehouse("ZN=F", _COT_DAY)),
        pytest.approx([0.2]),
        id="cot_reports",
    ),
    pytest.param(
        "cftc_cot_publication",
        lambda: _ratios(ll.load_cot_positioning_from_lakehouse("ZN=F", _COT_DAY)),
        pytest.approx([0.2]),
        id="cot_positioning",
    ),
    pytest.param(
        "sp500_membership",
        lambda: ll.source_coverage("sp500_membership", None),
        (date(2000, 1, 3), date(2015, 12, 31)),
        id="sp500_membership_coverage",
    ),
]


def test_always_populated_tables_and_retry_settings() -> None:
    """The set holds the four rebuilt-whole tables, the wait is 1 s, the error a RuntimeError."""
    assert isinstance(ll.ALWAYS_POPULATED_TABLES, frozenset)
    assert set(_REQUIRED) <= ll.ALWAYS_POPULATED_TABLES
    assert ll.EMPTY_READ_RETRY_SECONDS == 1.0
    assert issubclass(ll.EmptyTableReadError, RuntimeError)


@pytest.mark.parametrize("table", _REQUIRED)
def test_query_table_retries_one_empty_read(
    table: str, clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty read is sent again, unchanged, after one wait; a non-empty read is not."""
    query, params = f"SELECT * FROM {table}", ["p"]
    empty, full = pd.DataFrame({"x": []}), pd.DataFrame({"x": [1]})
    scripted = _ScriptedQuery(empty, full)
    monkeypatch.setattr(ll, "_query", scripted)

    result = ll._query_table(table, query, params)

    pd.testing.assert_frame_equal(result, full)
    assert scripted.calls == [(query, params), (query, params)]
    assert clock.waits == [_WAIT]

    scripted = _ScriptedQuery(full, full)
    monkeypatch.setattr(ll, "_query", scripted)
    pd.testing.assert_frame_equal(ll._query_table(table, query, params), full)
    assert len(scripted.calls) == 1
    assert clock.waits == [_WAIT]


@pytest.mark.parametrize("table", _REQUIRED)
def test_query_table_raises_when_empty_twice(
    table: str, clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two empty reads raise `EmptyTableReadError` naming the table, with no third read."""
    empty = pd.DataFrame({"x": []})
    scripted = _ScriptedQuery(empty, empty, pd.DataFrame({"x": [1]}))
    monkeypatch.setattr(ll, "_query", scripted)

    with pytest.raises(ll.EmptyTableReadError) as error:
        ll._query_table(table, f"SELECT * FROM {table}", [])

    message = str(error.value)
    assert table in message
    assert "empty" in message.lower() and "twice" in message.lower()
    assert len(scripted.calls) == 2
    assert clock.waits == [_WAIT]


@pytest.mark.parametrize("table", ["instrument_prices", "cftc_cot_tff", "sp500_price_coverage"])
def test_query_table_returns_an_empty_read_of_other_tables(
    table: str, clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A table outside the set returns its empty frame from one read, with no wait."""
    assert table not in ll.ALWAYS_POPULATED_TABLES
    scripted = _ScriptedQuery(pd.DataFrame({"x": []}), pd.DataFrame({"x": [1]}))
    monkeypatch.setattr(ll, "_query", scripted)

    result = ll._query_table(table, f"SELECT x FROM {table}", [])

    assert result.empty and list(result.columns) == ["x"]
    assert len(scripted.calls) == 1
    assert clock.waits == []


@pytest.mark.parametrize(
    "error",
    [
        LakehouseError("HTTP 400: Binder Error"),
        LakehouseError("HTTP 409: rebuilt"),
        TimeoutError("read timed out"),
    ],
    ids=["http_400", "http_409", "timeout"],
)
def test_query_table_does_not_retry_a_failed_read(
    error: Exception, clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed `_query` propagates as is from one read; after an empty read it ends the retry."""
    scripted = _ScriptedQuery(error, pd.DataFrame({"x": [1]}))
    monkeypatch.setattr(ll, "_query", scripted)
    with pytest.raises(type(error)) as caught:
        ll._query_table("cftc_cot_publication", "SELECT 1", [])
    assert caught.value is error
    assert len(scripted.calls) == 1
    assert clock.waits == []

    scripted = _ScriptedQuery(pd.DataFrame({"x": []}), error, pd.DataFrame({"x": [1]}))
    monkeypatch.setattr(ll, "_query", scripted)
    with pytest.raises(type(error)):
        ll._query_table("cftc_cot_publication", "SELECT 1", [])
    assert len(scripted.calls) == 2
    assert clock.waits == [_WAIT]


@pytest.mark.parametrize(("table", "call", "expected"), _TABLE_LEVEL_READS)
def test_table_level_reads_survive_a_rebuild_with_one_retry(
    table: str, call: Callable[[], Any], expected: Any, fake_lake: Any, clock: _Clock
) -> None:
    """A public read that finds its whole table empty waits once and returns the rebuilt data."""
    _seed(fake_lake)
    _rebuild(fake_lake, clock, table, finishes=True)

    assert call() == expected
    assert clock.waits == [_WAIT]


@pytest.mark.parametrize(("table", "call", "expected"), _TABLE_LEVEL_READS)
def test_table_level_reads_raise_when_the_table_stays_empty(
    table: str, call: Callable[[], Any], expected: Any, fake_lake: Any, clock: _Clock
) -> None:
    """A public read that finds its whole table empty twice raises instead of returning none."""
    _seed(fake_lake)
    _rebuild(fake_lake, clock, table, finishes=False)

    with pytest.raises(ll.EmptyTableReadError, match=table):
        call()
    assert clock.waits == [_WAIT]


@pytest.mark.parametrize(
    ("call", "name", "expected"),
    [
        pytest.param(lambda: ll.split_adjustment_factor("NVDA", _MON), "NVDA", 1.0, id="split"),
        pytest.param(lambda: ll.load_security_log_returns([9], _MON, _TUE), 9, {}, id="returns"),
        pytest.param(lambda: ll.load_security_daily_bars([9], _MON, _TUE), 9, None, id="bars"),
        pytest.param(
            lambda: ll.load_cot_reports_from_lakehouse("GC=F", _COT_DAY), _GC, None, id="cot"
        ),
        pytest.param(
            lambda: ll.load_price_history_from_lakehouse(["AAPL"], _MON, _TUE),
            "AAPL",
            None,
            id="prices",
        ),
        pytest.param(
            lambda: ll.load_earnings_surprise_from_lakehouse("AAPL", _TUE),
            "AAPL",
            None,
            id="earnings",
        ),
        pytest.param(
            lambda: ll.load_dividends_from_lakehouse("AAPL", _TUE), "AAPL", None, id="dividends"
        ),
        pytest.param(lambda: ll.source_coverage("cftc_cot", "GC=F"), _GC, None, id="cot_coverage"),
        pytest.param(
            lambda: ll.source_coverage("dividends", "AAPL"), "AAPL", None, id="dividend_coverage"
        ),
    ],
)
def test_reads_for_one_name_with_no_rows_are_not_retried(
    call: Callable[[], Any], name: Any, expected: Any, fake_lake: Any, clock: _Clock
) -> None:
    """A read filtered to one security, ticker or market with no rows is sent once, unretried."""
    _seed(fake_lake)

    assert call() == expected
    assert sum(name in params for _, params in fake_lake.calls) == 1
    assert clock.waits == []


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda: ll.load_cot_reports_from_lakehouse("ZN=F", _COT_DAY), id="cot"),
        pytest.param(
            lambda: ll.load_cot_positioning_from_lakehouse("ZN=F", _COT_DAY), id="cot_positioning"
        ),
        pytest.param(
            lambda: ll.adjustment_table_row_count("equity_adjustment_factors"), id="factors_count"
        ),
        pytest.param(lambda: ll.split_adjustment_factor("AAPL", _MON), id="split_factor"),
        pytest.param(lambda: ll.load_security_log_returns([1], _MON, _TUE), id="log_returns"),
        pytest.param(lambda: ll.source_coverage("sp500_membership", None), id="sp500_coverage"),
    ],
)
def test_http_errors_propagate_with_the_lake_message(
    call: Callable[[], Any], fake_lake: Any, clock: _Clock
) -> None:
    """HTTP 409 and 400 raise from one request with the lake's detail, in and out of a session."""
    _seed(fake_lake)
    conflict = LakehouseError("HTTP 409: cftc_cot_publication was rebuilt after the pinned time")
    bad_request = LakehouseError("HTTP 400: Binder Error: table is being replaced")
    scenarios = [
        (False, conflict, LakehouseError),
        (False, bad_request, LakehouseError),
        (True, conflict, ll.LakeTimeError),
        (True, bad_request, LakehouseError),
    ]
    for in_session, error, raised in scenarios:
        fake_lake.calls.clear()
        session = ll.lake_session(_T) if in_session else nullcontext()
        with pytest.raises(raised) as caught, session:
            fake_lake.error = error
            call()
        fake_lake.error = None

        assert str(error) in str(caught.value)
        if raised is LakehouseError:
            assert not isinstance(caught.value, ll.LakeTimeError)
        else:
            assert _T.isoformat() in str(caught.value)
        assert len(fake_lake.calls) == 1
        assert clock.waits == []


def test_require_rows_reads_a_populated_table_once_per_session(
    monkeypatch: pytest.MonkeyPatch, clock: _Clock
) -> None:
    """Inside a session, a table `_require_rows` finds holding rows is not read again; outside a
    session every call reads it."""
    populated = pd.DataFrame({"present": [1]})
    scripted = _ScriptedQuery(*([populated] * 4))
    monkeypatch.setattr(ll, "_query", scripted)
    session = ll.LakeSession(lake_time=_T)
    monkeypatch.setattr(ll, "_current_session", lambda: session)
    monkeypatch.setattr(ll, "_table_sql", lambda table: table)

    ll._require_rows("cftc_cot_publication")
    ll._require_rows("cftc_cot_publication")
    assert len(scripted.calls) == 1

    monkeypatch.setattr(ll, "_current_session", lambda: None)
    ll._require_rows("cftc_cot_publication")
    ll._require_rows("cftc_cot_publication")
    assert len(scripted.calls) == 3
    assert clock.waits == []


def test_require_rows_never_caches_an_empty_table(
    monkeypatch: pytest.MonkeyPatch, clock: _Clock
) -> None:
    """A table read empty twice raises and is not recorded as populated, so the next call in the
    same session reads it again."""
    empty = pd.DataFrame({"present": pd.Series([], dtype="int64")})
    populated = pd.DataFrame({"present": [1]})
    scripted = _ScriptedQuery(empty, empty, populated)
    monkeypatch.setattr(ll, "_query", scripted)
    session = ll.LakeSession(lake_time=_T)
    monkeypatch.setattr(ll, "_current_session", lambda: session)
    monkeypatch.setattr(ll, "_table_sql", lambda table: table)

    with pytest.raises(ll.EmptyTableReadError, match="cftc_cot_publication"):
        ll._require_rows("cftc_cot_publication")
    ll._require_rows("cftc_cot_publication")
    assert len(scripted.calls) == 3
    assert clock.waits == [_WAIT]
