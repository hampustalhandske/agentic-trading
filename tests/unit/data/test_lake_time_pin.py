# test_lake_time_pin.py

"""AT-106: every lakehouse read inside a `lake_session` sees the lake as it stood at one lake
time T. Tables read through `<t>_log` filtered on `scraped_at <= T` and deduplicated on their key
columns, append-only tables filtered only, `crypto_klines_1m` on its own `open_time`, and
rebuilt-whole tables unpinned with their build recorded."""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from agentic_trading.data.lakehouse_client import LakehouseError
from agentic_trading.data.loaders import lakehouse_loader as ll

_T = datetime(2024, 6, 1, 12, tzinfo=UTC)
_B1, _B2, _AFTER = "2024-04-01T00:00:00Z", "2024-05-01T00:00:00Z", "2024-07-01T00:00:00Z"
_REBUILT = frozenset(
    {
        "cftc_cot_publication",
        "equity_adjustment_factors",
        "equity_daily_from_bars",
        "equity_distributions",
        "sp500_price_coverage",
        "yahoo_split_close_basis",
    }
)
_CONTRACT = json.loads(
    (Path(__file__).parents[2] / "fixtures" / "lakehouse_contract_v1.json").read_text()
)["views"]
_SPLIT_COLUMNS = ["ticker", "ex_date", "ratio", "in_close_basis"]
_FRED_COLUMNS = ["series_id", "date", "realtime_start", "value"]
_DIST_COLUMNS = ["ticker", "ex_date", "distribution_type", "amount_split_adjusted"]


def _seed_versions(
    lake: Any,
    name: str,
    versions: list[tuple[str, pd.DataFrame]],
    key_columns: list[str],
    date_columns: tuple[str, ...] = (),
    kind: str = "current",
) -> None:
    """Writes `versions` (scrape time, frame) to the log and sets the current table to the
    lake's present state: the newest version per key, or every row for an append-only table."""
    first_at, first = versions[0]
    lake.add(name, first, date_columns, scraped_at=first_at, kind=kind, key_columns=key_columns)
    for scraped_at, frame in versions[1:]:
        lake.add_log_rows(name, frame, scraped_at)
    if kind == "current":
        keys = ", ".join(key_columns)
        body = (
            f"* EXCLUDE (scraped_at) FROM {name}_log QUALIFY "
            f"row_number() OVER (PARTITION BY {keys} ORDER BY scraped_at DESC) = 1"
        )
    else:
        body = f"* EXCLUDE (scraped_at) FROM {name}_log"
    lake._connection.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT {body}")


def _splits(rows: list[tuple[str, str, float, bool]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=_SPLIT_COLUMNS)


def _seed_rebuilt(lake: Any, name: str, frame: pd.DataFrame, build: str, **kwargs: Any) -> None:
    lake.add(name, frame, scraped_at=build, rebuilt_whole=True, **kwargs)


def _seed_splits(lake: Any, rows: list[tuple[str, str, float, bool]], build: str) -> None:
    _seed_rebuilt(lake, "yahoo_split_close_basis", _splits(rows), build, date_columns=("ex_date",))


def _seed_fred(lake: Any) -> None:
    before = pd.DataFrame(
        [("DGS10", "2024-01-10", "2024-01-11", 5.25), ("DGS10", "2024-01-11", "2024-01-12", 5.30)],
        columns=_FRED_COLUMNS,
    )
    after = pd.DataFrame(
        [("DGS10", "2024-01-10", "2024-01-11", 5.60), ("DGS10", "2024-01-12", "2024-01-13", 5.40)],
        columns=_FRED_COLUMNS,
    )
    _seed_versions(
        lake,
        "fred_vintages",
        [(_B1, before), (_AFTER, after)],
        ["series_id", "date", "realtime_start"],
        ("date", "realtime_start"),
    )


def _seed_klines(lake: Any) -> None:
    """Two BTCUSDT minutes on 2023-01-17: 09:30 ET (14:30 UTC) and 15:59 ET (20:59 UTC)."""
    rows = []
    for minute, open_, close in (("14:30", 21_000.0, 21_010.0), ("20:59", 21_100.0, 21_123.45)):
        stamp = pd.Timestamp(f"2023-01-17 {minute}")
        rows.append(
            {
                "symbol": "BTCUSDT",
                "open_time": stamp,
                "open": open_,
                "high": max(open_, close),
                "low": min(open_, close),
                "close": close,
                "volume": 1.0,
                "quote_volume": close,
                "n_trades": 10,
                "taker_buy_base_volume": 0.5,
                "taker_buy_quote_volume": close / 2,
                "dt": stamp.normalize(),
                "table": "crypto_klines_1m",
            }
        )
    lake.add(
        "crypto_klines_1m",
        pd.DataFrame(rows),
        ("dt",),
        kind="append-only",
        pin_column="open_time",
    )


def test_interface_names_and_read_tables() -> None:
    """READ_TABLES lists the 23 tables the loader reads and LakeTimeError is a RuntimeError."""
    names = set(ll.READ_TABLES)
    base = {
        "sp500_membership",
        "sp500_price_coverage",
        "equity_daily_from_bars",
        "equity_adjustment_factors",
        "yahoo_split_close_basis",
        "dolthub_option_chain",
        "option_chain_history",
        "cftc_cot_disaggregated",
        "cftc_cot_tff",
        "cftc_cot_publication",
        "equity_distributions",
        "dolthub_eps_history",
        "dolthub_earnings_calendar",
        "sec_sic_classification",
        "instrument_prices",
        "crypto_klines_1m",
        "fred_vintages",
        "famafrench_returns",
        "sec_filings",
        "finra_short_interest",
        "sec_company_facts_log",
        "sec_insider_transactions_log",
    }

    assert issubclass(ll.LakeTimeError, RuntimeError)
    assert len(ll.READ_TABLES) == len(names) == 23
    assert base <= names
    assert names & {"equity_ticker_history", "security_source_map"}


def test_lake_session_explicit_time_nesting_and_validation(fake_lake: Any) -> None:
    """An open session exposes T, nested sessions join it, a conflicting T or a naive T raises."""
    assert ll.current_lake_time() is None
    with ll.lake_session(_T) as session:
        assert session.lake_time == _T
        assert ll.current_lake_time() == _T
        with ll.lake_session() as joined:
            assert joined.lake_time == _T
        with ll.lake_session(_T):
            pass
        with pytest.raises(ll.LakeTimeError), ll.lake_session(_T + timedelta(days=1)):
            pass
        assert ll.current_lake_time() == _T
    assert ll.current_lake_time() is None
    with pytest.raises(ValueError), ll.lake_session(datetime(2024, 6, 1, 12)):
        pass


def test_default_lake_time_is_newest_write_plus_one_second(fake_lake: Any) -> None:
    """The default T is 1 s after the newest last_write of the lake's READ_TABLES, in UTC."""
    fred = pd.DataFrame(
        {"series_id": ["S"], "date": ["2024-01-02"], "realtime_start": ["2024-01-03"]}
    )
    fake_lake.add(
        "fred_vintages", fred, ("date", "realtime_start"), scraped_at="2024-05-01T12:00:00.5Z"
    )
    fake_lake.add("instrument_prices", pd.DataFrame({"ticker": ["A"]}), scraped_at=_B1)
    fake_lake.add("unrelated_table", pd.DataFrame({"x": [1]}), scraped_at="2025-01-01T00:00:00Z")

    with ll.lake_session() as session:
        assert session.lake_time == datetime(2024, 5, 1, 12, 0, 2, tzinfo=UTC)
        assert session.lake_time.utcoffset() == timedelta(0)


def test_default_lake_time_never_precedes_a_rebuilt_whole_build(fake_lake: Any) -> None:
    """A lagging last_write cannot put the default T before a rebuilt-whole table's build."""
    fred = pd.DataFrame(
        {"series_id": ["S"], "date": ["2024-01-02"], "realtime_start": ["2024-01-03"]}
    )
    fake_lake.add(
        "fred_vintages", fred, ("date", "realtime_start"), scraped_at="2024-05-01T12:00:00Z"
    )
    _seed_splits(fake_lake, [("A", "2024-01-02", 0.5, True)], "2024-05-10T00:00:00Z")
    fake_lake.last_write_overrides["yahoo_split_close_basis"] = "2024-04-01T00:00:00Z"

    with ll.lake_session() as session:
        assert session.lake_time == datetime(2024, 5, 10, 0, 0, 1, tzinfo=UTC)


def test_current_table_reads_each_key_as_of_t_even_when_the_new_version_fails_the_where(
    fake_lake: Any,
) -> None:
    """Dedupe precedes the loader's WHERE: a key reclassified before T is dropped."""
    v1 = pd.DataFrame(
        [
            ("X", "2024-01-10", "regular_cash", 1.0),
            ("X", "2024-02-10", "special_cash", 2.0),
            ("X", "2024-03-10", "regular_cash", 3.0),
            ("X", "2024-03-20", "regular_cash", 5.0),
        ],
        columns=_DIST_COLUMNS,
    )
    reclassified = pd.DataFrame([("X", "2024-01-10", "special_cash", 1.0)], columns=_DIST_COLUMNS)
    after_t = pd.DataFrame(
        [
            ("X", "2024-02-10", "regular_cash", 2.0),
            ("X", "2024-03-10", "regular_cash", 3.5),
            ("X", "2024-03-15", "regular_cash", 4.0),
        ],
        columns=_DIST_COLUMNS,
    )
    _seed_versions(
        fake_lake,
        "equity_distributions",
        [(_B1, v1), (_B2, reclassified), (_AFTER, after_t)],
        ["ticker", "ex_date"],
        ("ex_date",),
    )
    as_of = date(2024, 4, 1)

    current = ll.load_dividends_from_lakehouse("X", as_of)
    assert not any("scraped_at" in query or "_log" in query for query, _ in fake_lake.calls)
    with ll.lake_session(_T):
        pinned = ll.load_dividends_from_lakehouse("X", as_of)

    assert current is not None and pinned is not None
    assert current.tolist() == [2.0, 3.5, 4.0, 5.0]
    assert pinned.index.strftime("%Y-%m-%d").tolist() == ["2024-03-10", "2024-03-20"]
    assert pinned.tolist() == [3.0, 5.0]


def test_append_only_table_keeps_every_row_up_to_t(fake_lake: Any) -> None:
    """An append-only table is filtered on scraped_at only, so same-key rows before T all return."""
    versions = [
        (at, pd.DataFrame([("X", "2024-01-10", "regular_cash", amount)], columns=_DIST_COLUMNS))
        for at, amount in ((_B1, 1.0), (_B2, 1.1), (_AFTER, 9.0))
    ]
    _seed_versions(
        fake_lake,
        "equity_distributions",
        versions,
        ["ticker", "ex_date"],
        ("ex_date",),
        "append-only",
    )

    with ll.lake_session(_T):
        pinned = ll.load_dividends_from_lakehouse("X", date(2024, 4, 1))

    assert pinned is not None
    assert sorted(pinned.tolist()) == [1.0, 1.1]


def test_price_history_returns_the_bar_as_of_t(fake_lake: Any) -> None:
    """A bar rewritten after T returns its pre-T value and a bar first written after T is absent."""
    columns = ["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]

    def bars(rows: list[tuple[str, float]]) -> pd.DataFrame:
        return pd.DataFrame(
            [("AAPL", day, c, c + 1, c - 1, c, c, 1000) for day, c in rows], columns=columns
        )

    _seed_versions(
        fake_lake,
        "instrument_prices",
        [
            (_B1, bars([("2024-01-02", 100.0), ("2024-01-03", 101.0)])),
            (_AFTER, bars([("2024-01-03", 150.0), ("2024-01-04", 103.0)])),
        ],
        ["ticker", "date"],
        ("date",),
    )
    args = (["AAPL"], date(2024, 1, 2), date(2024, 1, 5))

    current = ll.load_price_history_from_lakehouse(*args)
    with ll.lake_session(_T):
        pinned = ll.load_price_history_from_lakehouse(*args)

    assert current is not None and pinned is not None
    assert current["AAPL"]["close"].tolist() == [100.0, 150.0, 103.0]
    assert pinned["AAPL"]["close"].tolist() == [100.0, 101.0]
    assert [stamp.date() for stamp in pinned.index] == [date(2024, 1, 2), date(2024, 1, 3)]


def test_fred_series_is_pinned_including_reads_from_worker_threads(fake_lake: Any) -> None:
    """FRED values rewritten after T return their pre-T value, in the caller and in pool workers."""
    _seed_fred(fake_lake)
    as_of = date(2024, 1, 20)

    def read(_: int) -> pd.Series | None:
        return ll.load_fred_series_from_lakehouse("DGS10", as_of)

    current = ll.load_fred_series_from_lakehouse("DGS10", as_of)
    with ll.lake_session(_T):
        pinned = ll.load_fred_series_from_lakehouse("DGS10", as_of)
        with ThreadPoolExecutor(max_workers=2) as pool:
            threaded = list(pool.map(read, range(2)))

    assert current is not None and current.tolist() == [5.60, 5.30, 5.40]
    assert pinned is not None and pinned.tolist() == [5.25, 5.30]
    assert [None if s is None else s.tolist() for s in threaded] == [[5.25, 5.30]] * 2


def test_company_facts_direct_log_read_is_filtered_without_dedupe(fake_lake: Any) -> None:
    """The facts log is filtered on scraped_at <= T: later restatements are invisible."""

    def fact(filed: str, value: float, scraped_at: str) -> dict[str, Any]:
        return {
            "ticker": "AAPL",
            "cik": 320193,
            "concept": "EarningsPerShareDiluted",
            "unit": "USD/shares",
            "period_start": "2023-01-01",
            "period_end": "2023-04-01",
            "value": value,
            "fiscal_year": 2023,
            "fiscal_period": "Q2",
            "form": "10-Q",
            "accession_number": f"ACC-{filed}",
            "filed_date": filed,
            "scraped_at": pd.Timestamp(scraped_at),
            "dt": "2024-07-01",
            "table": "sec_company_facts_log",
        }

    rows = [
        fact("2023-05-05", 1.52, "2024-05-01"),
        fact("2023-05-05", 1.52, "2024-05-15"),
        fact("2024-05-03", 1.60, "2024-07-01"),
    ]
    fake_lake.add(
        "sec_company_facts_log",
        pd.DataFrame(rows),
        date_columns=("dt", "filed_date", "period_start", "period_end"),
    )
    accepted = [
        pd.Timestamp("2023-05-05 12:00", tz="UTC"),
        pd.Timestamp("2024-05-03 12:00", tz="UTC"),
    ]
    fake_lake.add(
        "sec_filings",
        pd.DataFrame(
            {"accession_number": ["ACC-2023-05-05", "ACC-2024-05-03"], "accepted_at": accepted}
        ),
    )
    as_of = date(2024, 6, 30)

    current = ll.load_company_facts_from_lakehouse("AAPL", as_of)
    with ll.lake_session(_T):
        pinned = ll.load_company_facts_from_lakehouse("AAPL", as_of)

    assert current is not None and current["value"].tolist() == [1.60]
    assert pinned is not None
    assert pinned["value"].tolist() == [1.52]
    assert pinned["filed"].tolist() == [pd.Timestamp("2023-05-05")]


def test_crypto_klines_are_pinned_on_their_own_row_time(fake_lake: Any) -> None:
    """A kline opening after T is invisible, so a bar missing its closing minute is absent."""
    _seed_klines(fake_lake)
    args = (["BTC-USD"], date(2023, 1, 17), date(2023, 1, 17))

    with ll.lake_session(datetime(2023, 1, 17, 20, 0, tzinfo=UTC)):
        early = ll.load_price_history_from_lakehouse(*args)
    with ll.lake_session(datetime(2023, 1, 18, tzinfo=UTC)):
        late = ll.load_price_history_from_lakehouse(*args)

    assert early is None
    assert late is not None and late["BTC-USD"]["close"].tolist() == [21_123.45]


def test_ticker_history_link_rewritten_inactive_before_t_is_absent(fake_lake: Any) -> None:
    """is_active applies after the dedupe, so a link deactivated before T stays off."""
    columns = ["security_id", "source", "source_symbol", "valid_from", "valid_to", "is_active"]
    source = "us_equity_bars_5m"

    def link(symbol: str, active: bool) -> pd.DataFrame:
        row = (1, source, symbol, "2000-01-01", "2100-01-01", active)
        return pd.DataFrame([row], columns=columns)

    _seed_versions(
        fake_lake,
        "security_source_map",
        [
            (_B1, pd.concat([link("OLD", True), link("NEW", True)])),
            (_B2, link("OLD", False)),
            (_AFTER, link("OLD", True)),
        ],
        ["source", "source_symbol", "valid_from"],
        ("valid_from", "valid_to"),
    )
    fake_lake.add_view(
        "equity_ticker_history",
        "SELECT security_id, source_symbol AS ticker, valid_from, valid_to "
        f"FROM security_source_map WHERE source = '{source}' AND is_active",
    )
    membership = pd.DataFrame(
        {"ticker": ["OLD", "NEW"], "start_date": ["2010-01-01"] * 2, "end_date": ["2100-01-01"] * 2}
    )
    fake_lake.add("sp500_membership", membership, ("start_date", "end_date"))
    as_of = date(2024, 1, 2)

    current = ll.load_sp500_constituents_asof(as_of)
    with ll.lake_session(_T):
        pinned = ll.load_sp500_constituents_asof(as_of)

    assert current == ["NEW"]
    assert pinned == ["NEW", "OLD"]


def test_rebuilt_whole_table_reads_unpinned_and_records_its_build(fake_lake: Any) -> None:
    """A rebuilt-whole table is read unpinned and its build recorded; an earlier T raises."""
    _seed_splits(fake_lake, [("AAPL", "2024-08-01", 0.25, True)], "2024-05-20T00:00:00Z")

    with ll.lake_session(_T) as session:
        factor = ll.split_adjustment_factor("AAPL", date(2024, 1, 2))
        builds = dict(session.rebuilt_whole_builds)
    assert factor == pytest.approx(4.0)
    assert builds["yahoo_split_close_basis"] == datetime(2024, 5, 20, tzinfo=UTC)
    assert not any("yahoo_split_close_basis_log" in query for query, _ in fake_lake.calls)

    with (
        pytest.raises(ll.LakeTimeError, match="yahoo_split_close_basis") as error,
        ll.lake_session(datetime(2024, 5, 1, tzinfo=UTC)),
    ):
        ll.split_adjustment_factor("AAPL", date(2024, 1, 2))
    assert "2024-05-20" in str(error.value)


def _pin_values(query: str, params: list[Any]) -> list[datetime]:
    found: list[datetime] = []
    for value in params:
        if isinstance(value, datetime):
            found.append(value)
        elif isinstance(value, str):
            try:
                found.append(datetime.fromisoformat(value))
            except ValueError:
                continue
    literal = r"(?:scraped_at|open_time)\s*<=\s*(?:CAST\()?\s*'([^']+)'"
    found.extend(datetime.fromisoformat(text) for text in re.findall(literal, query, re.I))
    return found


def _pin_problems(query: str, params: list[Any]) -> list[str]:
    ctes = {c.lower() for c in re.findall(r"(?:\bWITH|,)\s*(\w+)\s+AS\s*\(", query, re.I)}
    refs = re.findall(r"\b(?:FROM|JOIN)\s+([A-Za-z_]\w*)", query, re.I)
    refs = [ref for ref in refs if ref.lower() not in ctes]
    values = _pin_values(query, params)
    problems: list[str] = []
    logs = [ref for ref in refs if ref.endswith("_log")]
    for ref in refs:
        if ref in _REBUILT or ref.endswith("_log"):
            continue
        if ref != "crypto_klines_1m":
            problems.append(f"reads the current view {ref}")
            continue
        if not re.search(r"open_time\s*<=", query, re.I):
            problems.append("crypto_klines_1m has no open_time <= pin")
        if _T not in values and _T.replace(tzinfo=None) not in values:
            problems.append("crypto_klines_1m pin does not carry T")
    if logs:
        if len(re.findall(r"scraped_at\s*<=", query, re.I)) < len(logs):
            problems.append(f"a _log reference in {logs} has no scraped_at <= pin")
        if _T not in values:
            problems.append("the pin does not carry T")
    return problems


def _sweep_lake(lake: Any) -> None:
    """Every contract view as an empty table with a one-row log, the rebuilt six flagged, and a
    security_source_map behind equity_ticker_history."""
    unkeyed = {"scraped_at", "dt", "table", "when"}
    for view, columns in _CONTRACT.items():
        if view == "equity_ticker_history":
            continue
        typed = {f'"{n}"': ("VARCHAR" if k == '"NULL"' else k) for n, k in columns.items()}
        row_time = view == "crypto_klines_1m"
        lake.add_empty(
            view,
            typed,
            kind="append-only" if row_time else "current",
            key_columns=[n for n in columns if n not in unkeyed],
            rebuilt_whole=view in _REBUILT,
            pin_column="open_time" if row_time else None,
        )
        if not row_time:
            lake._connection.execute(
                f"INSERT INTO {view}_log (scraped_at) VALUES (CAST(? AS TIMESTAMPTZ))", [_B1]
            )
    source_map = {
        "security_id": "BIGINT",
        "source": "VARCHAR",
        "source_symbol": "VARCHAR",
        "valid_from": "DATE",
        "valid_to": "DATE",
        "is_active": "BOOLEAN",
    }
    lake.add_empty(
        "security_source_map", source_map, key_columns=["source", "source_symbol", "valid_from"]
    )
    lake.add_view(
        "equity_ticker_history",
        "SELECT security_id, source_symbol AS ticker, valid_from, valid_to "
        "FROM security_source_map",
    )
    factors = pd.DataFrame(
        {
            "ticker": ["ZZ"],
            "effective_date": ["2024-01-02"],
            "cum_factor": [1.0],
            "security_id": [99],
        }
    )
    _seed_rebuilt(lake, "equity_adjustment_factors", factors, _B1, date_columns=("effective_date",))
    _seed_splits(lake, [("ZZ", "2000-01-03", 0.5, True)], _B1)


def test_every_public_loader_sql_is_pinned_inside_a_session(fake_lake: Any) -> None:
    """Every SQL a loader sends in a session reads a pinned _log, a rebuilt table or klines."""
    _sweep_lake(fake_lake)
    day, window = date(2024, 3, 15), (date(2024, 3, 1), date(2024, 3, 15))
    calls = {
        "constituents": lambda: ll.load_sp500_constituents_asof(day),
        "member_securities": lambda: ll.load_sp500_member_securities_asof(day),
        "log_returns": lambda: ll.load_security_log_returns([1], *window),
        "split_factor": lambda: ll.split_adjustment_factor("AAPL", day),
        "factors_count": lambda: ll.adjustment_table_row_count("equity_adjustment_factors"),
        "splits_count": lambda: ll.adjustment_table_row_count("yahoo_split_close_basis"),
        "snapshot_spy": lambda: ll.load_options_snapshot_from_lakehouse("SPY", day),
        "snapshot_aapl": lambda: ll.load_options_snapshot_from_lakehouse("AAPL", day),
        "quote_spy": lambda: ll.load_option_quote_from_lakehouse(
            "SPY", day, "call", 100.0, day, 10
        ),
        "cot_commodity": lambda: ll.load_cot_positioning_from_lakehouse("GC=F", day),
        "cot_tff": lambda: ll.load_cot_positioning_from_lakehouse("EURUSD=X", day),
        "dividends": lambda: ll.load_dividends_from_lakehouse("AAPL", day),
        "earnings": lambda: ll.load_earnings_surprise_from_lakehouse("AAPL", day),
        "sic": lambda: ll.load_sic_code_from_lakehouse("AAPL"),
        "sic_pairs": lambda: ll.same_sic_candidate_pairs(["AAPL", "MSFT"]),
        "prices_equity": lambda: ll.load_price_history_from_lakehouse(["AAPL"], *window),
        "prices_crypto": lambda: ll.load_price_history_from_lakehouse(["BTC-USD"], *window),
        "raw_close": lambda: ll.load_raw_close_from_lakehouse("AAPL", day, 10),
        "fred": lambda: ll.load_fred_series_from_lakehouse("DGS10", day),
        "ff3": lambda: ll.load_fama_french_factors_from_lakehouse(day, "3factor"),
        "ff5": lambda: ll.load_fama_french_factors_from_lakehouse(day, "5factor"),
        "company_facts": lambda: ll.load_company_facts_from_lakehouse("AAPL", day),
        "short_interest": lambda: ll.load_short_interest_from_lakehouse("AAPL", day),
        "insider": lambda: ll.load_insider_transactions_from_lakehouse("AAPL", day),
        "cov_options_spy": lambda: ll.source_coverage("options_chain", "SPY"),
        "cov_options_aapl": lambda: ll.source_coverage("options_chain", "AAPL"),
        "cov_cot": lambda: ll.source_coverage("cftc_cot", "GC=F"),
        "cov_cot_etf": lambda: ll.source_coverage("cftc_cot", "SHY"),
        "cov_earnings_all": lambda: ll.source_coverage("earnings", None),
        "cov_earnings": lambda: ll.source_coverage("earnings", "AAPL"),
        "cov_dividends_all": lambda: ll.source_coverage("dividends", None),
        "cov_dividends": lambda: ll.source_coverage("dividends", "AAPL"),
        "cov_membership": lambda: ll.source_coverage("sp500_membership", None),
    }

    problems: list[str] = []
    with ll.lake_session(_T):
        for label, call in calls.items():
            start = len(fake_lake.calls)
            call()
            for query, params in fake_lake.calls[start:]:
                problems += [f"{label}: {problem}" for problem in _pin_problems(query, params)]

    assert len(fake_lake.calls) > len(calls) // 2
    assert not problems


def test_http_409_inside_a_session_raises_lake_time_error_without_an_unpinned_retry(
    fake_lake: Any,
) -> None:
    """A 409 raises LakeTimeError naming T and quoting the detail; other errors pass through."""
    _seed_fred(fake_lake)
    detail = "table fred_vintages has no build at that time"
    real_sql = fake_lake.sql

    def refuse_pinned(query: str, params: list[Any] | None = None) -> pd.DataFrame:
        if "scraped_at" in query:
            raise LakehouseError(f"HTTP 409: {detail}")
        return real_sql(query, params)

    def fail_server(query: str, params: list[Any] | None = None) -> pd.DataFrame:
        raise LakehouseError("HTTP 500: boom")

    fake_lake.sql = refuse_pinned
    with ll.lake_session(_T), pytest.raises(ll.LakeTimeError) as error:
        ll.load_fred_series_from_lakehouse("DGS10", date(2024, 1, 20))

    assert "2024-06-01T12:00:00" in str(error.value)
    assert detail in str(error.value)
    assert fake_lake.calls == []

    fake_lake.sql = fail_server
    with ll.lake_session(_T), pytest.raises(LakehouseError, match="boom") as other:
        ll.load_fred_series_from_lakehouse("DGS10", date(2024, 1, 20))
    assert not isinstance(other.value, ll.LakeTimeError)


def test_lake_version_change_during_a_session_raises(fake_lake: Any) -> None:
    """The read after X-Lakehouse-Version changes mid-session raises LakeTimeError."""
    _seed_fred(fake_lake)

    with ll.lake_session(_T):
        assert ll.load_fred_series_from_lakehouse("DGS10", date(2024, 1, 20)) is not None
        fake_lake.version = "fake-lake-2"
        with pytest.raises(ll.LakeTimeError):
            ll.load_fred_series_from_lakehouse("DGS10", date(2024, 1, 20))


def test_adjustment_row_count_cache_does_not_cross_lake_times(fake_lake: Any) -> None:
    """A row count read outside a session or under one T is re-read under another T."""

    def rebuild(rows: int, build: str) -> None:
        _seed_splits(fake_lake, [("A", "2024-01-02", 0.5, True)] * rows, build)

    rebuild(1, "2024-05-10T00:00:00Z")
    outside_first = ll.adjustment_table_row_count("yahoo_split_close_basis")
    rebuild(3, "2024-05-15T00:00:00Z")
    with ll.lake_session(_T):
        first_t = ll.adjustment_table_row_count("yahoo_split_close_basis")
    rebuild(5, "2024-05-20T00:00:00Z")
    with ll.lake_session(_T + timedelta(days=1)):
        second_t = ll.adjustment_table_row_count("yahoo_split_close_basis")
    outside_last = ll.adjustment_table_row_count("yahoo_split_close_basis")

    assert [outside_first, first_t, second_t, outside_last] == [1, 3, 5, 5]


def test_first_dolthub_chain_date_cache_does_not_cross_lake_times(fake_lake: Any) -> None:
    """Each T, and the unpinned read, picks the chain table from its own first date."""
    greeks = {
        "implied_volatility": 0.2,
        "delta": 0.5,
        "gamma": 0.01,
        "theta": -0.1,
        "vega": 0.2,
        "rho": 0.05,
    }

    def chain(day: str, bid: float, history: bool = False) -> pd.DataFrame:
        row: dict[str, Any] = {
            "underlying": "SPY",
            "expiration": "2019-04-01",
            "option_type": "call",
            "strike": 100.0,
            "bid": bid,
            "ask": bid + 0.2,
            **greeks,
        }
        if history:
            row |= {"snapshot_date": day, "option_symbol": "S", "mark": 9.0}
        else:
            row["date"] = day
        return pd.DataFrame([row])

    _seed_versions(
        fake_lake,
        "dolthub_option_chain",
        [(_B1, chain("2019-03-01", 2.0)), (_AFTER, chain("2019-02-09", 1.0))],
        ["underlying", "date", "expiration", "option_type", "strike"],
        ("date", "expiration"),
    )
    _seed_versions(
        fake_lake,
        "option_chain_history",
        [(_B1, chain("2019-02-09", 7.0, history=True))],
        ["underlying", "snapshot_date", "expiration", "option_type", "strike"],
        ("snapshot_date", "expiration"),
    )
    _seed_splits(fake_lake, [("ZZUNREL", "2000-01-03", 0.5, True)], _B1)
    as_of = date(2019, 2, 9)

    def bids() -> list[float]:
        frame = ll.load_options_snapshot_from_lakehouse("SPY", as_of, 10)
        assert frame is not None
        return list(frame["bid"].tolist())

    with ll.lake_session(_T):
        at_t = bids()
    outside = bids()
    with ll.lake_session(datetime(2024, 8, 1, tzinfo=UTC)):
        later = bids()

    assert (at_t, outside, later) == ([7.0], [1.0], [1.0])


def test_lake_provenance_carries_lake_time_builds_and_row_time_tables(fake_lake: Any) -> None:
    """In a session, lake_provenance records T, the rebuilt builds and the row-time tables."""
    _seed_klines(fake_lake)
    factors = pd.DataFrame({"ticker": ["A"], "security_id": [1]})
    _seed_rebuilt(fake_lake, "equity_adjustment_factors", factors, "2024-05-01T00:00:00Z")
    _seed_splits(fake_lake, [("A", "2024-01-02", 0.5, True)], "2024-05-02T00:00:00Z")
    default = ll.LakeProvenance(lake_version="x", read_at=_T, status_generated_at=None)

    outside = ll.lake_provenance()
    with ll.lake_session(_T):
        ll.load_price_history_from_lakehouse(["BTC-USD"], date(2023, 1, 17), date(2023, 1, 17))
        inside = ll.lake_provenance()

    for bare in (default, outside):
        assert bare.lake_time is None
        assert bare.rebuilt_whole_builds == {}
        assert bare.row_time_pinned_tables == []
    assert inside.lake_time == _T
    assert inside.rebuilt_whole_builds == {
        "equity_adjustment_factors": datetime(2024, 5, 1, tzinfo=UTC),
        "yahoo_split_close_basis": datetime(2024, 5, 2, tzinfo=UTC),
    }
    assert inside.row_time_pinned_tables == ["crypto_klines_1m"]
