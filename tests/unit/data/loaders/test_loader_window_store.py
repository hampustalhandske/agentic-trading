# test_loader_window_store.py

"""The loaders' window store: every stored loader answers exactly as the
per-call reference loaders (`scripts/at126_legacy.py`) do, on the direct
path and on the store path, over a synthetic lake holding each source's
point-in-time traps; the store's own mechanics (one fetch per key, the
data_end guard, copies, eviction, threads); and the A/B harness's
comparison logic and probe guards (`scripts/at126_ab.py`)."""

import importlib
import json
import logging
import math
import sys
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel
from typer.testing import CliRunner

import agentic_trading.cli as cli
from agentic_trading.agents.roster import FIRM_V1_EQUITIES, FIRM_V1_UNIVERSE
from agentic_trading.agents.schemas import OptionContract, UnfactoredSplitMove
from agentic_trading.backtest.engine import BacktestEngine
from agentic_trading.cli import (
    LOADER_QUERY_COUNTS_PREFIX,
    LOADER_STORE_USAGE_PREFIX,
    firm_record_app,
)
from agentic_trading.config import settings, temporary_overrides
from agentic_trading.data.loaders import lakehouse_loader as ll
from agentic_trading.data.loaders import option_marks
from agentic_trading.data.trading_calendar import sessions_between
from agentic_trading.firm_record.runner import FirmRecordResult

_SCRIPTS = Path(__file__).resolve().parents[4] / "scripts"
_SCRAPED_AT = pd.Timestamp("2020-01-01T00:00:00Z")


@pytest.fixture(scope="module")
def ab() -> Iterator[ModuleType]:
    """`scripts/at126_ab.py`, imported with `scripts/` on the path so it
    finds `at126_legacy`."""
    sys.path.insert(0, str(_SCRIPTS))
    try:
        yield importlib.import_module("at126_ab")
    finally:
        sys.path.remove(str(_SCRIPTS))


# ---------------------------------------------------------------- the synthetic lake

_E1, _E2 = date(2019, 6, 21), date(2019, 9, 20)
_SPLIT_DAY = date(2019, 4, 10)
_SPY_FIRST_DOLTHUB = date(2019, 4, 1)
_AAA_CHAIN_GAP = (date(2019, 5, 3), date(2019, 5, 14))
_BBB_UNFACTORED_SPLIT = date(2019, 3, 12)
_BBB_NEW_SECURITY = date(2019, 4, 15)
_MISSING_AAA_BAR = date(2019, 3, 6)
_BBB_LAST_PRICE = date(2019, 5, 10)

_DAYS = [
    date(2019, 3, 1),
    date(2019, 3, 8),
    date(2019, 3, 12),
    date(2019, 3, 13),
    date(2019, 3, 29),
    date(2019, 4, 1),
    date(2019, 4, 2),
    date(2019, 4, 10),
    date(2019, 4, 11),
    date(2019, 4, 16),
    date(2019, 4, 30),
    date(2019, 5, 1),
    date(2019, 5, 13),
    date(2019, 5, 15),
    date(2019, 5, 31),
    date(2019, 6, 3),
]
# The days the whole-lake equality runs on: a month boundary, the chain
# source switch, the split, the chain gap and the last days.
_EQUALITY_DAYS = [
    date(2019, 3, 1),
    date(2019, 3, 12),
    date(2019, 3, 29),
    date(2019, 4, 1),
    date(2019, 4, 10),
    date(2019, 4, 16),
    date(2019, 5, 1),
    date(2019, 5, 13),
    date(2019, 5, 31),
    date(2019, 6, 3),
]
_TICKERS = ["AAA", "BBB", "ZZZ", "BTC-USD", "SPY"]
_PRICE_LOOKBACK = 60
_QUOTE_LOOKBACK = 20


def _sessions(start: date, end: date) -> list[date]:
    return sessions_between(start, end)


def _put(lake: Any, table: str, rows: list[dict[str, Any]]) -> None:
    """Inserts `rows` into `table` and into its log, scraped at `_SCRAPED_AT`
    (a relation without a `scraped_at` column takes the rows without it). A `<t>_log` name
    stands for table `<t>`."""
    table = table.removesuffix("_log")
    frame = pd.DataFrame(rows)
    frame["scraped_at"] = _SCRAPED_AT
    connection = lake._connection
    connection.register("_staged_rows", frame)
    try:
        for relation in (table, f"{table}_log"):
            existing = set(lake._columns(relation))
            columns = ", ".join(f'"{column}"' for column in frame.columns if column in existing)
            connection.execute(f"INSERT INTO {relation} BY NAME SELECT {columns} FROM _staged_rows")
    finally:
        connection.unregister("_staged_rows")


def _coverage_rows() -> list[dict[str, Any]]:
    def piece(ticker: str, start: str, sid: int | None, p0: str, p1: str | None) -> dict[str, Any]:
        return {
            "ticker": ticker,
            "start_date": date.fromisoformat(start),
            "end_date": None,
            "security_id": sid,
            "piece_start": date.fromisoformat(p0),
            "piece_end": None if p1 is None else date.fromisoformat(p1),
        }

    return [
        piece("AAA", "2010-01-01", 1, "2010-01-01", None),
        # A rename recorded as an overlapping span: the later start names the ticker.
        piece("AAA2", "2019-04-01", 1, "2019-04-01", None),
        # One span whose member security changes on 2019-04-15.
        piece("BBB", "2015-01-01", 2, "2015-01-01", "2019-04-15"),
        piece("BBB", "2015-01-01", 3, "2019-04-15", None),
        piece("CCC", "2012-01-01", 4, "2012-01-01", "2019-03-20"),
        # Two spans of one security starting the same day: the ticker breaks the tie.
        piece("DDD", "2016-01-01", 5, "2016-01-01", None),
        piece("DDA", "2016-01-01", 5, "2016-01-01", None),
        # A span with no security is left out (and warned about).
        piece("UNR", "2019-01-01", None, "2019-01-01", None),
    ]


def _bar_rows() -> list[dict[str, Any]]:
    rows = []
    for index, day in enumerate(_sessions(date(2018, 11, 1), date(2019, 6, 28))):
        if day != _MISSING_AAA_BAR:
            close = 100.0 * math.exp(0.002 * index) * (0.5 if day >= _SPLIT_DAY else 1.0) * 2
            rows.append(
                {
                    "ticker": "AAA",
                    "security_id": 1,
                    "date": day,
                    "open": None if index % 17 == 0 else close * 0.99,
                    "high": close * 1.01,
                    "low": close * 0.98,
                    "close": close,
                    # Official close missing on some days (the close is used),
                    # zero on one (the bar is missing).
                    "official_close": (
                        0.0 if day == date(2019, 2, 5) else None if index % 5 == 0 else close + 0.01
                    ),
                    "volume": 0.0 if index % 23 == 0 else 1000.0 + index,
                    "n_bars": 390,
                }
            )
        if day < _BBB_NEW_SECURITY:
            close = 50.0 + 0.1 * index
            if day >= _BBB_UNFACTORED_SPLIT:
                close /= 2.0
            sid = 2
        else:
            close = 25.0 + 0.05 * index
            sid = 3
        rows.append(
            {
                "ticker": "BBB",
                "security_id": sid,
                "date": day,
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "official_close": None,
                "volume": None if index % 11 == 0 else 500.0,
                "n_bars": 390,
            }
        )
    return rows


def _factor_rows() -> list[dict[str, Any]]:
    return [
        {"ticker": "AAA", "security_id": 1, "effective_date": date(2018, 12, 14),
         "split_factor": 1.0, "dividend_factor": 0.99, "cum_factor": 0.495},
        {"ticker": "AAA", "security_id": 1, "effective_date": _SPLIT_DAY,
         "split_factor": 2.0, "dividend_factor": 1.0, "cum_factor": 0.5},
    ]  # fmt: skip


def _split_rows() -> list[dict[str, Any]]:
    return [
        {"ticker": "SPY", "ex_date": _SPLIT_DAY, "ratio": 2.0, "in_close_basis": True},
        {"ticker": "SPY", "ex_date": date(2019, 5, 20), "ratio": 3.0, "in_close_basis": False},
        {"ticker": "SPY", "ex_date": date(2019, 6, 17), "ratio": 1.5, "in_close_basis": None},
        {"ticker": "AAA", "ex_date": _SPLIT_DAY, "ratio": 2.0, "in_close_basis": True},
    ]


def _chain_contracts(day: date, pre_split: bool) -> list[tuple[date, str, float]]:
    """The contracts each snapshot lists: three strikes of the near call,
    one of the near put (the quoted contract), two of the far call and
    one far put, at raw strikes that halve across the split."""
    scale = 0.5 if pre_split else 1.0
    return [
        (_E1, "call", 90.0 * scale),
        (_E1, "call", 100.0 * scale),
        (_E1, "call", 110.0 * scale),
        (_E1, "put", 100.0 * scale),
        (_E2, "call", 100.0 * scale),
        (_E2, "call", 120.0 * scale),
        (_E2, "put", 80.0 * scale),
    ]


def _chain_row(underlying: str, day: date, contract: tuple[date, str, float], n: int) -> dict:
    expiration, option_type, strike = contract
    bid = 1.0 + 0.01 * n + 0.001 * strike
    return {
        "underlying": underlying,
        "expiration": expiration,
        "option_type": option_type,
        "strike": strike,
        "bid": bid,
        "ask": bid + 0.1,
        "implied_volatility": 0.2 + 0.0001 * n,
        "delta": 0.5 if option_type == "call" else -0.5,
        "gamma": 0.01,
        "theta": -0.02,
        "vega": 0.1,
        "rho": 0.05,
    }


def _history_rows() -> list[dict[str, Any]]:
    rows = []
    for n, day in enumerate(_sessions(date(2019, 1, 2), date(2019, 6, 28))):
        if n % 3 == 2:
            continue
        for contract in _chain_contracts(day, pre_split=True):
            row = _chain_row("SPY", day, contract, n)
            row.update(snapshot_date=day, option_symbol=f"SPY{n}", mark=row["bid"] + 0.05,
                       bid_size=10, ask_size=12, open_interest=100, volume=5,
                       last_trade_price=row["bid"])  # fmt: skip
            rows.append(row)
    return rows


def _dolthub_rows() -> list[dict[str, Any]]:
    rows = []
    for n, day in enumerate(_sessions(date(2019, 2, 1), date(2019, 6, 28))):
        for underlying in ("SPY", "AAA"):
            if underlying == "SPY" and (day < _SPY_FIRST_DOLTHUB or n % 2):
                continue
            if underlying == "AAA" and _AAA_CHAIN_GAP[0] <= day <= _AAA_CHAIN_GAP[1]:
                continue
            for contract in _chain_contracts(day, pre_split=day < _SPLIT_DAY):
                rows.append({**_chain_row(underlying, day, contract, n), "date": day})
    return rows


def _cot_rows() -> tuple[list[dict], list[dict], list[dict]]:
    disaggregated, tff, publication = [], [], []
    for n, day in enumerate(pd.date_range("2018-01-02", "2019-06-25", freq="7D")):
        report = day.date()
        open_interest = 0 if n == 40 else 100_000 + 100 * n
        disaggregated.append({"cftc_contract_market_code": "088691", "report_date": report,
                              "managed_money_long": 30_000 + 37 * n,
                              "managed_money_short": 10_000 + 11 * n,
                              "open_interest_all": open_interest})  # fmt: skip
        tff.append({"cftc_contract_market_code": "043602", "report_date": report,
                    "leveraged_funds_long": 20_000 - 13 * n,
                    "leveraged_funds_short": 25_000 + 7 * n,
                    "open_interest_all": 300_000})  # fmt: skip
        if n == 55:
            continue  # a report with no publication row is never usable
        published = report + timedelta(days=3)
        if date(2018, 12, 25) <= report <= date(2019, 1, 22):
            published = date(2019, 3, 8)  # a delayed release
        untimed = n % 4 == 0
        publication.append({
            "report_date": report,
            "published_date": published,
            "published_at": None if untimed else pd.Timestamp(
                f"{published.isoformat()} 15:30", tz="America/New_York"
            ),
        })  # fmt: skip
    return disaggregated, tff, publication


def _distribution_rows() -> list[dict[str, Any]]:
    rows = [
        {"ticker": "AAA", "ex_date": date(y, m, 8), "distribution_type": "regular_cash",
         "amount_split_adjusted": 0.20 + 0.01 * m, "security_id": 1}
        for y, m in ((2018, 2), (2018, 5), (2018, 8), (2018, 11), (2019, 2), (2019, 5))
    ]  # fmt: skip
    rows.append({"ticker": "AAA", "ex_date": date(2019, 3, 1), "distribution_type": "special",
                 "amount_split_adjusted": 5.0, "security_id": 1})  # fmt: skip
    rows.append({"ticker": "BBB", "ex_date": date(2019, 3, 13),
                 "distribution_type": "regular_cash", "amount_split_adjusted": 0.5,
                 "security_id": 2})  # fmt: skip
    return rows


def _earnings_rows() -> tuple[list[dict], list[dict]]:
    history = [
        {"act_symbol": "AAA", "period_end_date": date(2018, 9, 30), "reported": 0.9,
         "estimate": 0.95},
        {"act_symbol": "AAA", "period_end_date": date(2018, 12, 31), "reported": 1.1,
         "estimate": 1.0},
        {"act_symbol": "AAA", "period_end_date": date(2019, 3, 31), "reported": 1.3,
         "estimate": None},
        {"act_symbol": "BBB", "period_end_date": date(2019, 1, 31), "reported": 0.4,
         "estimate": 0.5},
    ]  # fmt: skip
    calendar = [
        {"act_symbol": "AAA", "date": date(2018, 10, 30), "when": "after"},
        {"act_symbol": "AAA", "date": date(2019, 1, 30), "when": "after"},
        {"act_symbol": "AAA", "date": date(2019, 4, 30), "when": "after"},
        # Announced on a Friday: known the next session, the Monday.
        {"act_symbol": "BBB", "date": date(2019, 3, 29), "when": "after"},
    ]
    return history, calendar


def _fred_rows() -> list[dict[str, Any]]:
    rows = []
    for n, day in enumerate(_sessions(date(2018, 12, 3), date(2019, 6, 3))):
        rows.append({"series_id": "DGS10", "date": day, "value": 2.5 + 0.01 * n,
                     "realtime_start": day + timedelta(days=1)})  # fmt: skip
        if n % 10 == 0:
            # A revision released two weeks later.
            rows.append({"series_id": "DGS10", "date": day, "value": 2.6 + 0.01 * n,
                         "realtime_start": day + timedelta(days=14)})  # fmt: skip
        if n % 7 == 0:
            # No realtime_start: known from the observation date.
            rows.append({"series_id": "BAMLH0A0HYM2", "date": day, "value": 4.0 - 0.01 * n,
                         "realtime_start": None})  # fmt: skip
        else:
            rows.append({"series_id": "BAMLH0A0HYM2", "date": day, "value": 4.0 + 0.01 * n,
                         "realtime_start": day})  # fmt: skip
    return rows


def _fama_french_rows() -> list[dict[str, Any]]:
    rows = []
    for n, day in enumerate(_sessions(date(2018, 12, 3), date(2019, 5, 31))):
        for k, portfolio in enumerate(("Mkt-RF", "SMB", "HML", "RF")):
            rows.append({"dataset": "F-F_Research_Data_Factors_daily", "frequency": "daily",
                         "portfolio": portfolio, "date": day, "value": 0.1 * k + 0.01 * n,
                         "weighting": "value"})  # fmt: skip
        if n == 3:
            rows.append({"dataset": "F-F_Research_Data_Factors_daily", "frequency": "daily",
                         "portfolio": "SMB", "date": day, "value": 9.0,
                         "weighting": "equal"})  # fmt: skip
    return rows


def _company_fact_rows() -> tuple[list[dict], list[dict]]:
    def fact(accession: str, concept: str, start: str, end: str, value: float, year: Any,
             period: str, form: str, filed: str) -> dict[str, Any]:  # fmt: skip
        return {
            "ticker": "AAA",
            "cik": 100,
            "accession_number": accession,
            "concept": concept,
            "unit": "USD",
            "period_start": date.fromisoformat(start),
            "period_end": date.fromisoformat(end),
            "value": value,
            "fiscal_year": year,
            "fiscal_period": period,
            "form": form,
            "filed_date": date.fromisoformat(filed),
        }

    facts = [
        fact("a1", "Revenues", "2018-07-01", "2018-09-30", 10.0, 2018, "Q3", "10-Q", "2018-11-01"),
        fact("a2", "Revenues", "2018-10-01", "2018-12-31", 12.0, 2018, "FY", "10-K", "2019-02-20"),
        # An amendment of the same period, accepted after the close.
        fact("a3", "Revenues", "2018-10-01", "2018-12-31", 12.5, 2018, "FY", "10-K/A",
             "2019-03-12"),
        fact("a3", "NetIncome", "2018-10-01", "2018-12-31", 2.0, 2018, "FY", "10-K/A",
             "2019-03-12"),
        # A filing with no acceptance record: known the next session.
        fact("a4", "Revenues", "2019-01-01", "2019-03-31", 13.0, 2019, "Q1", "10-Q",
             "2019-04-30"),
        # A later filing with no fiscal year.
        fact("a5", "Revenues", "2019-01-01", "2019-03-31", 13.1, None, "Q1", "10-Q/A",
             "2019-05-31"),
    ]  # fmt: skip
    filings = [
        {"accession_number": "a1", "ticker": "AAA",
         "accepted_at": pd.Timestamp("2018-11-01 09:00", tz="America/New_York")},
        {"accession_number": "a2", "ticker": "AAA",
         "accepted_at": pd.Timestamp("2019-02-20 12:00", tz="America/New_York")},
        {"accession_number": "a3", "ticker": "AAA",
         "accepted_at": pd.Timestamp("2019-03-12 17:30", tz="America/New_York")},
        {"accession_number": "a5", "ticker": "AAA",
         "accepted_at": pd.Timestamp("2019-05-31 08:00", tz="America/New_York")},
    ]  # fmt: skip
    return facts, filings


def _short_interest_rows() -> list[dict[str, Any]]:
    rows = []
    settlements = [date(2018, 11, 15), date(2018, 11, 30), date(2018, 12, 14), date(2018, 12, 31),
                   date(2019, 1, 15), date(2019, 1, 31), date(2019, 2, 15), date(2019, 2, 28),
                   date(2019, 3, 15), date(2019, 3, 29), date(2019, 4, 15), date(2019, 4, 30),
                   date(2019, 5, 15), date(2019, 5, 31)]  # fmt: skip
    # Revised settlements (published with the next settlement, none for the
    # last) hold only their revised row: the reference loader's sort leaves
    # the order of two rows of one settlement to the lake.
    revised = {date(2019, 3, 15), date(2019, 5, 31)}
    for n, settlement in enumerate(settlements):
        rows.append({"symbol": "AAA", "settlement_date": settlement,
                     "revision_flag": "R" if settlement in revised else None,
                     "current_short_position": 2_000_000 if settlement in revised
                     else 1_000_000 + 1000 * n,
                     "days_to_cover": 1.5 + n,
                     "average_daily_volume": None if n == 2 else 600_000})  # fmt: skip
    return rows


def _insider_rows() -> list[dict[str, Any]]:
    def trade(accession: str, filed: str, traded: str, cik: Any, director: Any) -> dict:
        return {
            "issuer_ticker": "AAA",
            "issuer_cik": 100,
            "accession_number": accession,
            "owner_cik": cik,
            "owner_name": f"owner {accession}",
            "is_officer": True,
            "is_director": director,
            "is_ten_percent_owner": False,
            "transaction_date": date.fromisoformat(traded),
            "transaction_code": "P",
            "shares": 100.0,
            "price_per_share": 10.0,
            "acquired_disposed": "A",
            "filed_date": date.fromisoformat(filed),
        }

    return [
        trade("a1", "2018-11-01", "2018-10-30", 7, False),
        trade("a2", "2019-02-20", "2019-02-18", 8, True),
        trade("a3", "2019-03-12", "2019-03-11", 9, False),
        # Later filings with a missing owner and a missing director flag.
        trade("a5", "2019-05-31", "2019-05-29", None, None),
    ]


def _price_rows() -> list[dict[str, Any]]:
    rows = []
    for n, day in enumerate(_sessions(date(2018, 11, 1), date(2019, 6, 28))):
        for ticker, base in (("AAA", 200.0), ("BBB", 50.0), ("SPY", 280.0)):
            if ticker == "BBB" and day > _BBB_LAST_PRICE:
                continue
            close = base + 0.3 * n
            row = {"ticker": ticker, "date": day, "open": close - 0.5, "high": close + 1.0,
                   "low": close - 1.0, "close": close, "adj_close": close * 0.98,
                   "volume": 0 if n % 13 == 0 else 10_000 + n, "asset_class": "equity"}  # fmt: skip
            if ticker == "AAA" and n == 30:
                row["open"] = close + 5.0  # above the high: the open is dropped
            if ticker == "BBB" and n == 40:
                row["close"] = 0.0  # not positive: no raw close that day
            rows.append(row)
    return rows


def _kline_rows() -> list[dict[str, Any]]:
    rows = []
    days = pd.date_range("2018-12-01", "2019-06-05", freq="D")
    for n, day in enumerate(days):
        for minute, price in (("09:30", 3500.0 + n), ("12:00", 3510.0 + n), ("15:59", 3520.0 + n)):
            local = pd.Timestamp(f"{day.date().isoformat()} {minute}", tz="America/New_York")
            rows.append({"symbol": "BTCUSDT", "open": price, "high": price + 1, "low": price - 1,
                         "close": price + 0.5, "volume": 2.0 + n,
                         "open_time": local.tz_convert("UTC").tz_localize(None)})  # fmt: skip
    return rows


_CL_ROLL_BEFORE_START = date(2019, 3, 1)


def _futures_rows() -> list[dict[str, Any]]:
    """CL and NG continuous rows. CL rolls on 2019-03-01 (the first bar of
    a read starting there compares against the last kept bar before it,
    on 2019-02-27, because the bar of 2019-02-28 has no adjusted close),
    rolls with an unknown gap on 2019-04-10, and rolls with a known gap on
    2019-05-15."""
    rows = []
    for n, day in enumerate(_sessions(date(2018, 11, 1), date(2019, 6, 28))):
        if day < _CL_ROLL_BEFORE_START:
            front = "CLH19"
        elif day < date(2019, 4, 10):
            front = "CLK19"
        elif day < date(2019, 5, 15):
            front = "CLN19"
        else:
            front = "CLQ19"
        adjusted = None if day == date(2019, 2, 28) else 60.0 + 0.1 * n
        rows.append({
            "root": "CL", "date": day, "front_contract": front,
            "unadjusted_close": None if n % 19 == 0 else 55.0 + 0.1 * n,
            "ratio_adjusted_close": adjusted, "diff_adjusted_close": 0.0,
            "is_roll_day": day in (_CL_ROLL_BEFORE_START, date(2019, 4, 10), date(2019, 5, 15)),
            "roll_return": None,
            "roll_gap_known": False if day == date(2019, 4, 10)
            else True if day == date(2019, 5, 15) else None,
        })  # fmt: skip
        rows.append({
            "root": "NG", "date": day, "front_contract": "NGK19" if n < 80 else "NGM19",
            "unadjusted_close": 2.5 + 0.01 * n, "ratio_adjusted_close": 2.6 + 0.01 * n,
            "diff_adjusted_close": 0.0, "is_roll_day": n == 80, "roll_return": None,
            "roll_gap_known": True if n == 80 else None,
        })  # fmt: skip
    return rows


def _bis_rows() -> list[dict[str, Any]]:
    """Daily WS_CBPOL rates for the US and the euro area on sessions only
    (no weekend rows), a monthly row that is not read, a null value, and
    another dataflow's row."""
    rows = []
    for n, day in enumerate(_sessions(date(2018, 11, 1), date(2019, 6, 28))):
        rows.append({"dataflow": "WS_CBPOL", "freq": "D", "ref_area": "US", "date": day,
                     "value": 2.25 if day < date(2018, 12, 20) else 2.5})  # fmt: skip
        if n % 2 == 0:
            rows.append({"dataflow": "WS_CBPOL", "freq": "D", "ref_area": "XM", "date": day,
                         "value": None if n == 40 else 0.0})  # fmt: skip
    rows.append({"dataflow": "WS_CBPOL", "freq": "M", "ref_area": "US",
                 "date": date(2019, 3, 1), "value": 9.0})  # fmt: skip
    rows.append({"dataflow": "WS_OTHER", "freq": "D", "ref_area": "US",
                 "date": date(2019, 3, 4), "value": 7.0})  # fmt: skip
    return rows


@pytest.fixture
def ab_lake(preflight_lake: Any) -> Any:
    """`preflight_lake` filled with every stored loader's synthetic source rows."""
    lake = preflight_lake
    _put(lake, "sp500_price_coverage", _coverage_rows())
    _put(lake, "equity_daily_from_bars", _bar_rows())
    _put(lake, "equity_adjustment_factors", _factor_rows())
    _put(lake, "yahoo_split_close_basis", _split_rows())
    _put(lake, "option_chain_history", _history_rows())
    _put(lake, "dolthub_option_chain", _dolthub_rows())
    disaggregated, tff, publication = _cot_rows()
    _put(lake, "cftc_cot_disaggregated", disaggregated)
    _put(lake, "cftc_cot_tff", tff)
    _put(lake, "cftc_cot_publication", publication)
    _put(lake, "equity_distributions", _distribution_rows())
    history, calendar = _earnings_rows()
    _put(lake, "dolthub_eps_history", history)
    _put(lake, "dolthub_earnings_calendar", calendar)
    _put(lake, "sec_sic_classification", [
        {"ticker": "AAA", "cik": 100, "sic": "3571"},
        {"ticker": "BBB", "cik": 200, "sic": None},
    ])  # fmt: skip
    _put(lake, "instrument_prices", _price_rows())
    _put(lake, "crypto_klines_1m", _kline_rows())
    _put(lake, "fred_vintages", _fred_rows())
    _put(lake, "famafrench_returns", _fama_french_rows())
    facts, filings = _company_fact_rows()
    _put(lake, "sec_company_facts_log", facts)
    _put(lake, "sec_filings", filings)
    _put(lake, "finra_short_interest", _short_interest_rows())
    _put(lake, "sec_insider_transactions_log", _insider_rows())
    _put(lake, "futures_continuous_daily", _futures_rows())
    _put(lake, "bis_observations", _bis_rows())
    return lake


@pytest.fixture(autouse=True)
def _fresh_loader_state(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    # The live insider loader reads only the issuer CIKs listed for a ticker;
    # the synthetic issuer AAA files its Form 4 lines under CIK 100.
    monkeypatch.setitem(ll.ISSUER_CIKS_BY_TICKER, "AAA", (100,))
    ll.clear_loader_window()
    ll.clear_unfactored_split_move_registry()
    ll.reset_loader_query_counts()
    yield
    ll.clear_loader_window()
    ll.clear_unfactored_split_move_registry()


def _plan(ab: ModuleType, days: list[date] = _DAYS) -> Any:
    return ab.CallPlan(
        days=days,
        tickers=_TICKERS,
        security_ids=[1, 2, 3, 77],
        fred_series=["DGS10", "BAMLH0A0HYM2", "DGS3MO"],
        fama_french_models=["3factor", "5factor"],
        cot_tickers=["GC=F", "ZN=F", "ZZZ"],
        option_underlyings=["SPY", "AAA", "ZZZ"],
        contracts=[
            ("SPY", _E1, "put", 150.0),
            ("AAA", _E1, "put", 100.0),
        ],
        futures_tickers=["CL=F", "NG=F"],
        currencies=["EUR", "USD", "JPY"],
        price_lookback_days=_PRICE_LOOKBACK,
        quote_lookback_days=_QUOTE_LOOKBACK,
    )


def _window(ab: ModuleType) -> Any:
    """A run from the first test day to the end of the synthetic data, so
    every trap call is served from the store on the store path."""
    return ab.RunWindow(start=_DAYS[0], end=date(2019, 6, 28))


def _three_ways(ab: ModuleType, loader: str, *args: Any) -> tuple[Any, Any, Any]:
    """The reference, direct and store answers to one call, each from a
    fresh registry and (for the store) a freshly declared window; the
    session must be open."""
    window = _window(ab)
    call = ab.LoaderCall(loader=loader, args=args, day=_DAYS[0])
    passes = [ab.run_pass(path, [call], window) for path in ("legacy", "direct", "store")]
    return passes[0].outputs[0], passes[1].outputs[0], passes[2].outputs[0]


def _agree(ab: ModuleType, loader: str, *args: Any) -> Any:
    """Asserts the three paths give the same answer to one call and returns it."""
    reference, direct, stored = _three_ways(ab, loader, *args)
    assert ab.compare(reference, direct) is None
    assert ab.compare(reference, stored) is None
    return reference


def _queries(loader: str, path: str) -> int:
    return sum(
        count.queries
        for count in ll.loader_query_counts()
        if count.loader == loader and count.path == path
    )


# ---------------------------------------------------------------- equality over the whole lake


def test_every_stored_loader_answers_exactly_as_the_reference_on_both_paths(
    ab: ModuleType, ab_lake: Any
) -> None:
    """Every loader, every day and key of the plan: the direct and store
    answers equal the reference, the split-move registries match, every
    loader gives at least one non-`None` answer, and the store sends far
    fewer queries than the direct path."""
    plan = _plan(ab, _EQUALITY_DAYS)
    calls = sorted(ab.build_calls(plan), key=lambda call: call.day)
    window = ab.plan_window(plan)
    with ll.lake_session():
        passes = {path: ab.run_pass(path, calls, window) for path in ab.PATHS}
    report = ab.compare_passes(calls, passes, window)

    assert not report.mismatches, "\n\n".join(
        f"{m.loader}{m.key} on {m.day} {m.path}:\n{m.difference}" for m in report.mismatches
    )
    assert report.verdict == "PASS"
    assert all(summary.legacy_self_mismatches == 0 for summary in report.loaders)
    assert report.registry_differences == {"legacy_repeat": None, "direct": None, "store": None}
    assert passes["legacy"].registry, "the plan should flag an unfactored split move"
    answered = {
        call.loader
        for call, output in zip(calls, passes["legacy"].outputs, strict=True)
        if output is not None and not isinstance(output, ab.RaisedError)
    }
    assert answered == {call.loader for call in calls}
    assert not any(isinstance(output, ab.RaisedError) for output in passes["legacy"].outputs)
    totals = {path: sum(summary.queries[path] for summary in report.loaders) for path in ab.PATHS}
    assert totals["store"] * 5 < totals["direct"]
    assert answered == set(ab.STORED_LOADERS)
    assert {summary.loader for summary in report.loaders} == set(ab.STORED_LOADERS)
    # The run's window ends a day before the last call day, as the CLI declares it.
    assert window.end < _EQUALITY_DAYS[-1]
    assert report.window.start == _EQUALITY_DAYS[0] - timedelta(days=window.warmup_days)
    assert any(count.path == "direct" and count.calls for count in passes["store"].counts)
    assert passes["store"].usage is not None and passes["store"].usage.peak_bytes > 0


# ---------------------------------------------------------------- per trap


def test_first_bar_has_no_return_even_when_the_store_holds_the_bar_before(
    ab: ModuleType, ab_lake: Any
) -> None:
    """A call's first bar gets no return though the stored rows hold the
    session before it, and a split move before the call's start is
    neither returned nor registered."""
    start, end = date(2019, 3, 13), date(2019, 3, 29)
    with ll.lake_session():
        returns = _agree(ab, "load_security_log_returns", [1, 2], start, end)
        assert ll.unfactored_split_move_registry() == []
        bars = _agree(ab, "load_security_daily_bars", [2], start, end)
        moved = _agree(ab, "load_security_log_returns", [2], date(2019, 3, 11), end)
        (move,) = ll.unfactored_split_move_registry()
    assert isinstance(move, UnfactoredSplitMove)
    assert (move.security_id, move.as_of_date) == (2, _BBB_UNFACTORED_SPLIT)
    assert move.ratio == pytest.approx(0.5, abs=0.01)
    assert returns[1].index[0] > pd.Timestamp(start)
    assert math.isnan(bars[(2, "log_return")].iloc[0])
    assert moved[2].loc[pd.Timestamp(_BBB_UNFACTORED_SPLIT)] < -0.6


def test_price_level_takes_the_factor_after_each_bar_whatever_the_window_start(
    ab: ModuleType, ab_lake: Any
) -> None:
    """A call starting after a factor date reads the same adjusted levels
    from the store (fetched from the window start) as its own query."""
    with ll.lake_session():
        bars = _agree(ab, "load_security_daily_bars", [1], date(2019, 1, 2), date(2019, 4, 30))
    frame = bars[1]
    before, after = frame.loc["2019-04-09"], frame.loc["2019-04-10"]
    assert before["close"] == pytest.approx(before["raw_close"] * 0.5)
    assert after["close"] == after["raw_close"]
    assert math.isnan(
        frame.loc[pd.Timestamp(_MISSING_AAA_BAR) + pd.Timedelta(days=1), "log_return"]
    )


def test_fred_versions_are_known_from_the_later_of_vintage_and_date(
    ab: ModuleType, ab_lake: Any
) -> None:
    """A vintage is used only once its known_at is before the day; a
    revision replaces it from then on; a null realtime_start is known
    from the observation date."""
    observed = date(2018, 12, 3)
    with ll.lake_session():
        early = _agree(ab, "load_fred_vintages_from_lakehouse", "DGS10", date(2018, 12, 4))
        unrevised = _agree(ab, "load_fred_vintages_from_lakehouse", "DGS10", date(2018, 12, 17))
        revised = _agree(ab, "load_fred_vintages_from_lakehouse", "DGS10", date(2018, 12, 18))
        spread = _agree(ab, "load_fred_series_from_lakehouse", "BAMLH0A0HYM2", date(2018, 12, 4))
        assert _agree(ab, "load_fred_series_from_lakehouse", "DGS10", observed) is None
    assert early is None
    assert unrevised.loc[pd.Timestamp(observed), "value"] == pytest.approx(2.5)
    assert revised.loc[pd.Timestamp(observed), "value"] == pytest.approx(2.6)
    assert spread.loc[pd.Timestamp(observed)] == pytest.approx(4.0)


def test_fred_versions_released_the_same_day_resolve_to_the_same_value_on_both_paths(
    ab: ModuleType, ab_lake: Any
) -> None:
    """Two versions of one observation with one realtime_start: both
    current paths keep the same one (the larger value), whatever order
    the lake stores them in."""
    _put(ab_lake, "fred_vintages", [
        {"series_id": "TIE", "date": date(2019, 1, 2), "value": 2.0,
         "realtime_start": date(2019, 1, 3)},
        {"series_id": "TIE", "date": date(2019, 1, 2), "value": 1.0,
         "realtime_start": date(2019, 1, 3)},
    ])  # fmt: skip
    with ll.lake_session():
        _, direct, stored = _three_ways(ab, "load_fred_series_from_lakehouse", "TIE", _DAYS[0])
    assert ab.compare(direct, stored) is None
    assert direct.tolist() == [2.0]


def test_sec_facts_slice_keeps_the_read_dtypes_and_order(ab: ModuleType, ab_lake: Any) -> None:
    """Before the filing with no fiscal year, the stored slice types
    `fiscal_year` as the per-call read does; acceptance after the close
    moves availability to the next session; amendments win per period."""
    with ll.lake_session():
        on_filing_day = _agree(ab, "load_company_facts_from_lakehouse", "AAA", date(2019, 3, 12))
        next_day = _agree(ab, "load_company_facts_from_lakehouse", "AAA", date(2019, 3, 13))
        later = _agree(ab, "load_company_facts_from_lakehouse", "AAA", date(2019, 6, 3))
        insiders = _agree(ab, "load_insider_transactions_from_lakehouse", "AAA", date(2019, 4, 1))
    assert on_filing_day["value"].tolist() == [10.0, 12.0]
    assert next_day["value"].tolist() == [10.0, 2.0, 12.5]
    assert next_day["fiscal_year"].dtype == np.int64
    assert later["value"].tolist()[-1] == pytest.approx(13.1)
    assert insiders["owner_cik"].dtype == np.int64
    assert insiders["is_director"].dtype == np.bool_


def test_chain_source_switches_to_dolthub_on_its_first_date(ab: ModuleType, ab_lake: Any) -> None:
    """SPY reads the history table before its first dolthub snapshot and
    dolthub from then on, on every path, with the split product of every
    later split applied in a fixed order."""
    with ll.lake_session():
        history = _agree(ab, "load_options_snapshot_from_lakehouse", "SPY", date(2019, 3, 29))
        dolthub = _agree(ab, "load_options_snapshot_from_lakehouse", "SPY", date(2019, 4, 2))
        factor = _agree(ab, "split_adjustment_factor", "SPY", date(2019, 3, 29))
    assert history["date"].iloc[0] < pd.Timestamp(_SPY_FIRST_DOLTHUB)
    assert dolthub["date"].iloc[0] == pd.Timestamp(_SPY_FIRST_DOLTHUB)
    assert factor == 1.0 / 2.0 / 1.5
    assert history["strike"].max() == pytest.approx(60.0 / factor)


def test_quote_takes_the_latest_quoted_date_across_months_and_the_split(
    ab: ModuleType, ab_lake: Any
) -> None:
    """The quote spans a month boundary and the split: the latest row of
    the contract in the adjusted basis, labelled by its position, latest
    date first; a strike no chain lists gives `None`."""
    with ll.lake_session():
        quote = _agree(ab, "load_option_quote_from_lakehouse", "AAA", _E1, "put", 100.0,
                       date(2019, 5, 13), 45)  # fmt: skip
        assert (
            _agree(ab, "load_option_quote_from_lakehouse", "AAA", _E1, "put", 101.0,
                   date(2019, 5, 13), 45)
            is None
        )  # fmt: skip
    assert quote["date"].iloc[0] == pd.Timestamp(date(2019, 5, 2))
    assert quote.index.tolist() == [0]


def test_short_interest_availability_uses_the_full_settlement_history(
    ab: ModuleType, ab_lake: Any
) -> None:
    """A revised row is known with the next settlement's publication, and
    one with no next settlement never is."""
    with ll.lake_session():
        frame = _agree(ab, "load_short_interest_from_lakehouse", "AAA", date(2019, 6, 28))
    revised = frame[frame["short_interest_shares"] == 2_000_000]
    assert revised["settlement_date"].tolist() == [pd.Timestamp(2019, 3, 15)]
    assert revised["dissemination_date"].iloc[0] == pd.Timestamp(
        ll._finra_available_date(date(2019, 3, 29))
    )


def test_short_interest_rows_of_one_settlement_keep_a_fixed_order(
    ab: ModuleType, ab_lake: Any
) -> None:
    """An original and a revised row of one settlement come back original
    first on both current paths, whichever order the lake stores them in."""
    _put(ab_lake, "finra_short_interest", [
        {"symbol": "TIE", "settlement_date": date(2019, 1, 15), "revision_flag": "R",
         "current_short_position": 2, "days_to_cover": 1.0, "average_daily_volume": 1},
        {"symbol": "TIE", "settlement_date": date(2019, 1, 15), "revision_flag": None,
         "current_short_position": 1, "days_to_cover": 1.0, "average_daily_volume": 1},
        {"symbol": "TIE", "settlement_date": date(2019, 1, 31), "revision_flag": None,
         "current_short_position": 3, "days_to_cover": 1.0, "average_daily_volume": 1},
    ])  # fmt: skip
    with ll.lake_session():
        _, direct, stored = _three_ways(ab, "load_short_interest_from_lakehouse", "TIE", _DAYS[-1])
    assert ab.compare(direct, stored) is None
    assert direct["short_interest_shares"].tolist() == [1, 2, 3]


def test_ken_french_rows_appear_at_their_public_date(ab: ModuleType, ab_lake: Any) -> None:
    with ll.lake_session():
        march_end = _agree(ab, "load_fama_french_factors_from_lakehouse", date(2019, 3, 29))
        april = _agree(ab, "load_fama_french_factors_from_lakehouse", date(2019, 4, 1))
        assert _agree(ab, "load_fama_french_factors_from_lakehouse", date(2019, 4, 1),
                      "5factor") is None  # fmt: skip
    assert march_end.index.max() == pd.Timestamp(2019, 1, 31)
    assert april.index.max() == pd.Timestamp(2019, 2, 28)
    assert april.loc[pd.Timestamp(2018, 12, 7), "SMB"] == pytest.approx(0.09)


def test_earnings_are_known_the_session_after_the_announcement(
    ab: ModuleType, ab_lake: Any
) -> None:
    with ll.lake_session():
        on_day = _agree(ab, "load_earnings_surprise_from_lakehouse", "AAA", date(2019, 4, 30))
        next_day = _agree(ab, "load_earnings_surprise_from_lakehouse", "AAA", date(2019, 5, 1))
        weekend = _agree(ab, "load_earnings_surprise_from_lakehouse", "BBB", date(2019, 3, 29))
        monday = _agree(ab, "load_earnings_surprise_from_lakehouse", "BBB", date(2019, 4, 1))
    assert on_day[2] == pd.Timestamp(2019, 1, 30)
    assert next_day[2] == pd.Timestamp(2019, 1, 30)  # the Q1 row has no estimate
    assert weekend is None
    assert monday == (0.4, 0.5, pd.Timestamp(2019, 3, 29))


def test_cot_reports_gate_on_their_release(ab: ModuleType, ab_lake: Any) -> None:
    """An untimed release is usable the next day, a timed one on its day;
    a delayed release holds its reports back; a report with no
    publication row or no open interest is never returned."""
    with ll.lake_session():
        gold = _agree(ab, "load_cot_reports_from_lakehouse", "GC=F", date(2019, 3, 8))
        gold_next = _agree(ab, "load_cot_reports_from_lakehouse", "GC=F", date(2019, 3, 9))
        notes = _agree(ab, "load_cot_positioning_from_lakehouse", "ZN=F", date(2019, 6, 3))
    held_back = [day for day in gold.index if pd.Timestamp(2018, 12, 25) <= day]
    assert all(gold.loc[day, "published_date"] <= pd.Timestamp(2019, 3, 8) for day in held_back)
    assert len(gold_next) >= len(gold)
    assert notes.index.is_monotonic_increasing


def test_dividends_raw_close_and_members(ab: ModuleType, ab_lake: Any) -> None:
    with ll.lake_session():
        dividends = _agree(ab, "load_dividends_from_lakehouse", "BBB", date(2019, 3, 13))
        close = _agree(ab, "load_raw_close_from_lakehouse", "BBB", date(2019, 5, 13), 10)
        stale = _agree(ab, "load_raw_close_from_lakehouse", "BBB", date(2019, 5, 31), 10)
        before = _agree(ab, "load_sp500_member_securities_asof", date(2019, 3, 29))
        after = _agree(ab, "load_sp500_member_securities_asof", date(2019, 4, 16))
    assert dividends.tolist() == [0.5]
    assert close is not None and stale is None
    assert [(m.security_id, m.ticker) for m in before] == [(1, "AAA"), (2, "BBB"), (5, "DDA")]
    assert [(m.security_id, m.ticker) for m in after] == [(1, "AAA2"), (3, "BBB"), (5, "DDA")]


def test_futures_first_bar_and_level_are_computed_per_request_window(
    ab: ModuleType, ab_lake: Any
) -> None:
    """The store holds CL from long before each start, yet each call's
    first bar compares fronts against the last kept bar before its own
    start (skipping the bar with no adjusted close), its first return is
    zero, and its level starts from its own first unadjusted close; an
    unknown-gap roll's return is zeroed and a known-gap one kept."""
    end = date(2019, 6, 3)
    with ll.lake_session():
        rolled = _agree(ab, "load_futures_price_history_from_lakehouse", ["CL=F", "NG=F"],
                        _CL_ROLL_BEFORE_START, end)  # fmt: skip
        later = _agree(ab, "load_futures_price_history_from_lakehouse", ["CL=F"],
                       date(2019, 3, 4), end)  # fmt: skip
        assert _agree(ab, "load_futures_price_history_from_lakehouse", ["ZZ=F"],
                      _CL_ROLL_BEFORE_START, end) is None  # fmt: skip
    first = rolled["CL=F"].iloc[0]
    assert first["roll"] == 1.0
    assert later["CL=F"]["roll"].iloc[0] == 0.0
    assert rolled["CL=F"]["log_return"].isna().iloc[0]
    assert first["close"] == first["raw_close"]
    assert later["CL=F"]["close"].iloc[0] == later["CL=F"]["raw_close"].iloc[0]
    unknown_gap = rolled["CL=F"].loc[pd.Timestamp(2019, 4, 10)]
    known_gap = rolled["CL=F"].loc[pd.Timestamp(2019, 5, 15)]
    assert unknown_gap["log_return"] == 0.0 and unknown_gap["roll"] == 1.0
    assert known_gap["log_return"] > 0.0 and known_gap["roll"] == 1.0
    assert list(rolled.columns.get_level_values(0).unique()) == ["CL=F", "NG=F"]


def test_policy_rates_read_the_lookback_before_start(ab: ModuleType, ab_lake: Any) -> None:
    """A Saturday start still sees the rate observed on the Friday before
    it; a currency with no observations is an all-NaN column; monthly and
    other-dataflow rows are never read."""
    with ll.lake_session():
        rates = _agree(ab, "load_policy_rates_from_lakehouse", ["USD", "EUR", "JPY"],
                       date(2019, 3, 2), date(2019, 3, 8))  # fmt: skip
        assert _agree(ab, "load_policy_rates_from_lakehouse", ["JPY"], date(2019, 3, 2),
                      date(2019, 3, 8)) is None  # fmt: skip
    assert list(rates.columns) == ["USD", "EUR", "JPY"]
    assert rates.index[0] < pd.Timestamp(2019, 3, 2)
    assert rates["JPY"].isna().all()
    assert (rates["USD"] == 0.025).all()


def test_no_rows_is_none_on_every_path(ab: ModuleType, ab_lake: Any) -> None:
    with ll.lake_session():
        for loader, args in (
            ("load_dividends_from_lakehouse", ("ZZZ", _DAYS[-1])),
            ("load_company_facts_from_lakehouse", ("ZZZ", _DAYS[-1])),
            ("load_short_interest_from_lakehouse", ("ZZZ", _DAYS[-1])),
            ("load_insider_transactions_from_lakehouse", ("ZZZ", _DAYS[-1])),
            ("load_earnings_surprise_from_lakehouse", ("ZZZ", _DAYS[-1])),
            ("load_fred_vintages_from_lakehouse", ("DGS3MO", _DAYS[-1])),
            ("load_options_snapshot_from_lakehouse", ("ZZZ", _DAYS[-1])),
            ("load_cot_reports_from_lakehouse", ("ZZZ", _DAYS[-1])),
            ("load_price_history_from_lakehouse", (["ZZZ"], _DAYS[0], _DAYS[-1])),
            ("load_security_daily_bars", ([77], _DAYS[0], _DAYS[-1])),
            ("load_sic_code_from_lakehouse", ("BBB",)),
        ):
            assert _agree(ab, loader, *args) is None, loader
        assert _agree(ab, "load_security_log_returns", [77], _DAYS[0], _DAYS[-1]) == {}


# ---------------------------------------------------------------- store mechanics


def test_a_key_is_fetched_once_and_later_calls_send_no_query(ab_lake: Any) -> None:
    with ll.lake_session():
        ll.declare_loader_window(date(2019, 1, 2), date(2019, 6, 3))
        for day in _DAYS:
            ll.load_dividends_from_lakehouse("AAA", day)
    counts = {(c.loader, c.path): c for c in ll.loader_query_counts()}
    stored = counts[("load_dividends_from_lakehouse", "store")]
    assert stored.calls == len(_DAYS)
    assert stored.queries == 1


def test_a_call_after_the_window_end_fetches_for_itself(ab_lake: Any) -> None:
    with ll.lake_session():
        ll.declare_loader_window(date(2019, 1, 2), date(2019, 3, 29))
        ll.load_dividends_from_lakehouse("AAA", date(2019, 3, 29))
        ll.load_dividends_from_lakehouse("AAA", date(2019, 4, 1))
        ll.load_dividends_from_lakehouse("AAA", date(2019, 4, 2))
    counts = {(c.loader, c.path): (c.calls, c.queries) for c in ll.loader_query_counts()}
    assert counts[("load_dividends_from_lakehouse", "store")] == (1, 1)
    assert counts[("load_dividends_from_lakehouse", "direct")] == (2, 2)


def test_a_ranged_call_starting_before_the_window_refetches_from_its_start(ab_lake: Any) -> None:
    with ll.lake_session():
        ll.declare_loader_window(date(2019, 3, 1), date(2019, 6, 3))
        late = ll.load_security_daily_bars([1], date(2019, 3, 1), date(2019, 4, 1))
        early = ll.load_security_daily_bars([1], date(2019, 1, 2), date(2019, 4, 1))
        again = ll.load_security_daily_bars([1], date(2019, 2, 1), date(2019, 4, 1))
        window = ll.loader_window()
    assert early.index[0] < late.index[0]
    assert again.index[0] >= pd.Timestamp(2019, 2, 1)
    assert window == ll.LoaderWindow(start=date(2019, 3, 1), end=date(2019, 6, 3))
    assert _queries("load_security_daily_bars", "store") == 3  # 2 fetches, 1 table check


def test_the_data_end_guard_runs_before_the_store(ab_lake: Any) -> None:
    with pytest.raises(ll.HoldoutReadError), ll.lake_session(data_end=date(2019, 4, 1)):
        ll.declare_loader_window(date(2019, 1, 2), date(2019, 4, 1))
        ll.load_dividends_from_lakehouse("AAA", date(2019, 4, 1))
        sent = len(ab_lake.calls)
        with pytest.raises(ll.HoldoutReadError):
            ll.load_dividends_from_lakehouse("AAA", date(2019, 4, 2))
        assert len(ab_lake.calls) == sent


def test_a_window_cannot_end_after_data_end(ab_lake: Any) -> None:
    with pytest.raises(ll.HoldoutReadError), ll.lake_session(data_end=date(2019, 4, 1)):
        with pytest.raises(ll.HoldoutReadError):
            ll.declare_loader_window(date(2019, 1, 2), date(2019, 4, 2))
        assert ll.loader_window() is None


def test_declaring_needs_a_session_and_an_ordered_window(ab_lake: Any) -> None:
    with pytest.raises(RuntimeError):
        ll.declare_loader_window(date(2019, 1, 2), date(2019, 4, 2))
    with ll.lake_session(), pytest.raises(ValueError):
        ll.declare_loader_window(date(2019, 4, 2), date(2019, 1, 2))


def test_the_store_is_dropped_when_its_session_closes(ab_lake: Any) -> None:
    with ll.lake_session():
        ll.declare_loader_window(date(2019, 1, 2), date(2019, 6, 3))
        ll.load_dividends_from_lakehouse("AAA", date(2019, 4, 1))
        assert ll.loader_window() is not None
    assert ll.loader_window() is None
    assert ll._STORE is None
    with ll.lake_session():
        assert ll.loader_window() is None
        ll.reset_loader_query_counts()
        ll.load_dividends_from_lakehouse("AAA", date(2019, 4, 1))
    assert _queries("load_dividends_from_lakehouse", "direct") == 1


def test_changing_an_answer_never_changes_the_store(ab_lake: Any) -> None:
    with ll.lake_session():
        ll.declare_loader_window(date(2019, 1, 2), date(2019, 6, 3))
        loaders: list[Callable[[], Any]] = [
            lambda: ll.load_dividends_from_lakehouse("AAA", date(2019, 6, 3)),
            lambda: ll.load_fred_vintages_from_lakehouse("DGS10", date(2019, 6, 3)),
            lambda: ll.load_company_facts_from_lakehouse("AAA", date(2019, 6, 3)),
            lambda: ll.load_short_interest_from_lakehouse("AAA", date(2019, 6, 3)),
            lambda: ll.load_options_snapshot_from_lakehouse("AAA", date(2019, 6, 3)),
            lambda: ll.load_security_daily_bars([1], date(2019, 1, 2), date(2019, 6, 3)),
            lambda: ll.load_price_history_from_lakehouse(["AAA"], date(2019, 1, 2),
                                                         date(2019, 6, 3)),
            lambda: ll.load_cot_reports_from_lakehouse("GC=F", date(2019, 6, 3)),
        ]  # fmt: skip
        for load in loaders:
            first = load()
            expected = first.copy(deep=True)
            if isinstance(first, pd.Series):
                first.iloc[0] = -999.0
            else:
                for position, dtype in enumerate(first.dtypes):
                    if dtype.kind == "f":
                        first.iloc[0, position] = -999.0
            first.drop(first.index[:1], inplace=True)
            if isinstance(expected, pd.Series):
                pd.testing.assert_series_equal(load(), expected, check_exact=True)
            else:
                pd.testing.assert_frame_equal(load(), expected, check_exact=True)


def test_concurrent_calls_fetch_a_key_once(ab_lake: Any) -> None:
    with ll.lake_session():
        ll.declare_loader_window(date(2019, 1, 2), date(2019, 6, 3))
        barrier = threading.Barrier(8)

        def load(day: date) -> pd.Series | None:
            barrier.wait()
            return ll.load_dividends_from_lakehouse("AAA", day)

        with ThreadPoolExecutor(max_workers=8) as pool:
            answers = list(pool.map(load, _DAYS[:8]))
        expected = [ll.load_dividends_from_lakehouse("AAA", day) for day in _DAYS[:8]]
    for answer, wanted in zip(answers, expected, strict=True):
        pd.testing.assert_series_equal(answer, wanted, check_exact=True)
    assert _queries("load_dividends_from_lakehouse", "store") == 1


def test_chain_months_are_loaded_as_reached_and_evicted_when_out_of_every_lookback(
    ab_lake: Any,
) -> None:
    with ll.lake_session():
        store_window = ll.declare_loader_window(date(2019, 1, 2), date(2019, 6, 28))
        store = ll._active_store()
        assert store is not None and store.window == store_window
        ll.load_options_snapshot_from_lakehouse("AAA", date(2019, 3, 1))
        march = {key[3:] for key in store.keys(ll._CHAIN_MONTHS)}
        ll.load_options_snapshot_from_lakehouse("AAA", date(2019, 6, 3))
        june = {key[3:] for key in store.keys(ll._CHAIN_MONTHS)}
    assert march == {(2019, 2), (2019, 3)}
    assert (2019, 2) not in june and (2019, 3) not in june
    assert (2019, 6) in june


def test_option_marks_hand_out_copies_of_their_cached_quote_and_snapshot(
    ab_lake: Any,
) -> None:
    contract = OptionContract(underlying="AAA", expiry=_E1, strike=100.0, option_type="put")
    with ll.lake_session():
        first = option_marks._last_quote_row(contract, date(2019, 5, 13))
        assert first is not None
        bid = first["bid"]
        first["bid"] = -1.0
        assert option_marks._last_quote_row(contract, date(2019, 5, 13))["bid"] == bid
        snapshot = option_marks._snapshot("AAA", date(2019, 4, 2))
        assert snapshot is not None
        snapshot.loc[:, "bid"] = -1.0
        again = option_marks._snapshot("AAA", date(2019, 4, 2))
        assert again is not None and (again["bid"] > 0).all()


# ---------------------------------------------------------------- harness logic


def test_compare_is_exact(ab: ModuleType) -> None:
    frame = pd.DataFrame({"a": [1.0, np.nan]}, index=pd.DatetimeIndex(["2019-01-01", "2019-01-02"]))
    assert ab.compare(None, None) is None
    assert "expected None" in ab.compare(None, frame)
    assert ab.compare(frame, frame.copy()) is None
    assert ab.compare(frame, frame.astype({"a": "float32"})) is not None
    assert ab.compare(frame, frame + 1e-15) is not None
    assert ab.compare(frame["a"], frame["a"].rename("b")) is not None
    assert ab.compare(float("nan"), float("nan")) is None
    assert ab.compare(0.1 + 0.2, 0.3) is not None
    assert ab.compare(0.0, -0.0) is not None
    assert ab.compare(1, 1.0) is not None
    assert ab.compare({1: frame["a"]}, {1: frame["a"]}) is None
    assert ab.compare({1: 1.0, 2: 2.0}, {2: 2.0, 1: 1.0}) is not None
    assert ab.compare((1.0, 2.0), (1.0, 2.5)) == "value[1]: 2.0 != 2.5"
    assert (
        ab.compare(
            [ll.IndexMember(security_id=1, ticker="A")], [ll.IndexMember(security_id=1, ticker="B")]
        )
        is not None
    )
    stamp = pd.Timestamp("2019-01-01")
    assert ab.compare(stamp, stamp.as_unit("ns")) is not None
    assert (
        ab.compare(
            ab.RaisedError(type="ValueError", message="x"),
            ab.RaisedError(type="ValueError", message="x"),
        )
        is None
    )


def _short_interest(*rows: tuple[str, int]) -> pd.DataFrame:
    """A short-interest answer, one row per `(settlement_date, shares)`."""
    return pd.DataFrame(
        {
            "settlement_date": pd.to_datetime([day for day, _ in rows]),
            "short_interest_shares": [shares for _, shares in rows],
        }
    )


_SHORT_INTEREST = "load_short_interest_from_lakehouse"


def test_trace_tie_accepts_rows_reordered_or_picked_within_a_tie(ab: ModuleType) -> None:
    columns = ab.TIE_COLUMNS[_SHORT_INTEREST]
    legacy = _short_interest(("2019-01-15", 1), ("2019-01-15", 2), ("2019-01-31", 3))
    swapped = _short_interest(("2019-01-15", 2), ("2019-01-15", 1), ("2019-01-31", 3))
    assert ab.trace_tie(legacy, swapped, columns) is None
    # No ORDER BY at all: every row ties, so any equal-shaped difference traces.
    assert ab.trace_tie(0.5, 0.5000000000000001, ()) is None


def test_trace_tie_rejects_a_difference_outside_a_tie(ab: ModuleType) -> None:
    columns = ab.TIE_COLUMNS[_SHORT_INTEREST]
    legacy = _short_interest(("2019-01-15", 1), ("2019-01-31", 3))
    changed = _short_interest(("2019-01-15", 1), ("2019-01-31", 4))
    reordered = _short_interest(("2019-01-31", 3), ("2019-01-15", 1))
    assert ab.trace_tie(legacy, legacy, columns) is None
    assert ab.trace_tie(legacy, changed, columns) is None  # same key: a picked tie
    assert "ORDER BY column 'settlement_date'" in ab.trace_tie(legacy, reordered, columns)
    assert "row count" in ab.trace_tie(legacy, legacy.iloc[:1], columns)
    assert ab.trace_tie(legacy, None, columns) is not None
    assert ab.trace_tie(1.0, 2.0, None) is not None
    dividends = pd.Series([0.2, 0.3], index=pd.to_datetime(["2019-02-08", "2019-05-08"]))
    shifted = pd.Series([0.2, 0.3], index=pd.to_datetime(["2019-02-08", "2019-05-09"]))
    assert ab.trace_tie(dividends, shifted, ab.TIE_COLUMNS["load_dividends_from_lakehouse"])


def _pass(ab: ModuleType, path: str, outputs: list[Any]) -> Any:
    counted = "store" if path == "store" else "direct"
    return ab.PassResult(
        path=path,
        outputs=outputs,
        registry=[],
        counts=[ll.LoaderQueryCount(loader=_SHORT_INTEREST, path=counted, calls=len(outputs),
                                    queries=len(outputs))],
        seconds={_SHORT_INTEREST: 0.1},
    )  # fmt: skip


def test_a_tie_difference_passes_only_where_the_reference_varies_too(ab: ModuleType) -> None:
    window = ab.RunWindow(start=date(2019, 2, 1), end=date(2019, 2, 28))
    days = [date(2019, 2, 1), date(2019, 2, 4)]
    calls = [ab.LoaderCall(loader=_SHORT_INTEREST, args=("AAA", day), day=day) for day in days]
    tied = _short_interest(("2019-01-15", 1), ("2019-01-15", 2))
    swapped = _short_interest(("2019-01-15", 2), ("2019-01-15", 1))

    varies = {
        "legacy": _pass(ab, "legacy", [tied, tied]),
        "legacy_repeat": _pass(ab, "legacy_repeat", [swapped, swapped]),
        "direct": _pass(ab, "direct", [tied, swapped]),
        "store": _pass(ab, "store", [swapped, swapped]),
    }
    report = ab.compare_passes(calls, varies, window, required=[_SHORT_INTEREST])
    (summary,) = report.loaders
    assert summary.verdict == "PASS" and report.verdict == "PASS"
    assert summary.legacy_self_mismatches == 2
    assert summary.mismatches == {"direct": 1, "store": 2}
    assert summary.allowed == {"direct": 1, "store": 2}
    assert [(tie.path, tie.key, tie.count) for tie in report.allowed_ties] == [
        ("direct", "('AAA',)", 1),
        ("store", "('AAA',)", 2),
    ]

    stable = {**varies, "legacy_repeat": _pass(ab, "legacy_repeat", [tied, tied])}
    report = ab.compare_passes(calls, stable, window, required=[_SHORT_INTEREST])
    assert report.loaders[0].verdict == "FAIL" and report.verdict == "FAIL"
    assert report.allowed_ties == []
    assert all(not m.legacy_varies and m.tie is None for m in report.mismatches)


def test_a_difference_that_is_not_a_tie_fails_even_where_the_reference_varies(
    ab: ModuleType,
) -> None:
    window = ab.RunWindow(start=date(2019, 2, 1), end=date(2019, 2, 28))
    calls = [ab.LoaderCall(loader=_SHORT_INTEREST, args=("AAA", _DAYS[0]), day=_DAYS[0])]
    legacy = _short_interest(("2019-01-15", 1), ("2019-01-31", 3))
    passes = {
        "legacy": _pass(ab, "legacy", [legacy]),
        "legacy_repeat": _pass(ab, "legacy_repeat", [legacy.iloc[::-1].reset_index(drop=True)]),
        "direct": _pass(ab, "direct", [legacy]),
        "store": _pass(ab, "store", [_short_interest(("2019-01-15", 1), ("2019-02-15", 3))]),
    }
    report = ab.compare_passes(calls, passes, window, required=[_SHORT_INTEREST])
    (mismatch,) = report.mismatches
    assert mismatch.legacy_varies and mismatch.tie is not None and not mismatch.allowed
    assert report.verdict == "FAIL"
    assert report.loaders[0].queries == {"legacy": 1, "legacy_repeat": 1, "direct": 1, "store": 1}


def test_an_unexercised_stored_loader_fails_the_verdict(ab: ModuleType) -> None:
    """A required loader that was never called, answered only `None`, or
    raised in the reference fails, though no answer mismatched."""
    window = ab.RunWindow(start=date(2019, 2, 1), end=date(2019, 2, 28))
    calls = [
        ab.LoaderCall(loader=_SHORT_INTEREST, args=("AAA", _DAYS[0]), day=_DAYS[0]),
        ab.LoaderCall(loader="load_dividends_from_lakehouse", args=("ZZZ", _DAYS[0]),
                      day=_DAYS[0]),
        ab.LoaderCall(loader="load_sic_code_from_lakehouse", args=("AAA",), day=_DAYS[0]),
    ]  # fmt: skip
    error = ab.RaisedError(type="ValueError", message="boom")
    outputs = [_short_interest(("2019-01-15", 1)), None, error]
    passes = {path: _pass(ab, path, outputs) for path in ab.PATHS}
    required = [_SHORT_INTEREST, "load_dividends_from_lakehouse",
                "load_sic_code_from_lakehouse", "load_cot_reports_from_lakehouse"]  # fmt: skip
    report = ab.compare_passes(calls, passes, window, required=required)
    verdicts = {s.loader: (s.verdict, s.calls, s.reference_answers, s.reference_errors)
                for s in report.loaders}  # fmt: skip
    assert verdicts == {
        _SHORT_INTEREST: ("PASS", 1, 1, 0),
        "load_dividends_from_lakehouse": ("FAIL", 1, 0, 0),
        "load_sic_code_from_lakehouse": ("FAIL", 1, 0, 1),
        "load_cot_reports_from_lakehouse": ("FAIL", 0, 0, 0),
    }
    assert report.mismatches == [] and report.verdict == "FAIL"
    assert set(ab.STORED_LOADERS) == set(ab.TIE_COLUMNS)


def _sec_fetch(filed: str, accepted_ny: str | None, filed_column: str = "filed") -> Any:
    """A stand-in for an SEC `_fetch_*` function returning one row filed on `filed` and
    accepted at `accepted_ny` (America/New_York, or missing)."""
    accepted = None if accepted_ny is None else pd.Timestamp(accepted_ny, tz="America/New_York")
    frame = pd.DataFrame(
        {
            filed_column: [pd.Timestamp(filed)],
            "accepted_at": pd.Series([accepted], dtype="datetime64[ns, UTC]"),
        }
    )
    return lambda tickers, _start, _end: {ticker: frame for ticker in tickers}


def test_coverage_finds_each_firm_v1_event(ab: ModuleType, monkeypatch) -> None:
    monkeypatch.setattr(ll, "_fetch_company_facts", _sec_fetch("2017-07-27", "2017-07-27 16:05"))
    day = date(2017, 8, 1)
    index = pd.DatetimeIndex([pd.Timestamp(2017, 7, 31), pd.Timestamp(day)], name="date")
    futures = pd.concat(
        {"CL=F": pd.DataFrame({"close": [50.0, 51.0], "roll": [0.0, 1.0]}, index=index)}, axis=1
    )
    prices = pd.concat(
        {
            "EURUSD=X": pd.DataFrame({"close": [1.17, 1.18]}, index=index),
            "BTC-USD": pd.DataFrame({"close": [np.nan, np.nan]}, index=index),
        },
        axis=1,
    )
    vintages = pd.DataFrame(
        {"value": [1.0], "realtime_start": [pd.Timestamp(2017, 7, 20)]},
        index=pd.DatetimeIndex([pd.Timestamp(2017, 7, 3)]),
    )
    facts = pd.DataFrame(
        {"filed": [pd.Timestamp(2017, 7, 27)], "available_date": [pd.Timestamp(2017, 7, 28)]}
    )
    cot = pd.DataFrame({"net_positioning_ratio": [0.1]}, index=index[:1])
    quote = pd.DataFrame({"date": [pd.Timestamp(day)]})

    def call(loader: str, *args: Any, on: date = day) -> Any:
        return ab.LoaderCall(loader=loader, args=args, day=on)

    calls = [
        call("load_futures_price_history_from_lakehouse", ["CL=F"], day, day),
        call("load_price_history_from_lakehouse", ["EURUSD=X", "BTC-USD"], day, day),
        call("load_fred_vintages_from_lakehouse", "DGS10", day),
        call("load_company_facts_from_lakehouse", "AAA", day),
        call("load_cot_reports_from_lakehouse", "GC=F", date(2017, 7, 28), on=date(2017, 7, 28)),
        call("load_cot_reports_from_lakehouse", "GC=F", day),
        call("load_option_quote_from_lakehouse", "SPY", day, "put", 240.0, day, 45),
    ]
    outputs = [futures, prices, vintages, facts, cot, pd.concat([cot, cot]), quote]
    assert ab.coverage(calls, outputs) == dict.fromkeys(
        ["futures_roll", "fx_pair", "cot_release", "fred_revision", "sec_filing_after_close",
         "spy_option_quote", "missing_bar"],
        True,
    )  # fmt: skip
    assert not any(ab.coverage(calls, [None] * len(calls)).values())


@pytest.mark.parametrize(
    ("filed", "accepted_ny", "day", "expected"),
    [
        ("2017-07-27", "2017-07-27 15:59", date(2017, 8, 1), False),
        ("2017-07-27", "2017-07-27 16:00", date(2017, 8, 1), True),
        ("2017-07-27", "2017-07-28 09:00", date(2017, 8, 1), True),
        ("2017-07-27", None, date(2017, 8, 1), False),
        ("2017-07-27", "2017-07-27 18:00", date(2017, 7, 27), False),
    ],
    ids=["pre-close", "at-close", "later-date", "missing", "not-yet-known"],
)
@pytest.mark.parametrize(
    ("loader", "fetch", "filed_column"),
    [
        ("load_company_facts_from_lakehouse", "_fetch_company_facts", "filed"),
        ("load_insider_transactions_from_lakehouse", "_fetch_insider_transactions", "filed_date"),
    ],
    ids=["facts", "insiders"],
)
def test_sec_accepted_after_close_reads_ny_acceptance_against_the_close(
    ab: ModuleType,
    monkeypatch,
    loader: str,
    fetch: str,
    filed_column: str,
    filed: str,
    accepted_ny: str | None,
    day: date,
    expected: bool,
) -> None:
    """A row counts when it is known by the day and EDGAR accepted it at or after 16:00 New
    York time on its filed date or on a later date."""
    monkeypatch.setattr(ll, fetch, _sec_fetch(filed, accepted_ny, filed_column))

    assert ab.sec_accepted_after_close(loader, "AAA", day) is expected


def test_firm_v1_days_lie_in_2010_to_2017_and_end_past_the_window(ab: ModuleType) -> None:
    days = ab.firm_v1_days()
    assert days == sorted(days) and len(days) == len(set(days))
    assert date(2010, 1, 1) <= days[0] and days[-1] <= date(2017, 12, 31)
    # Days on both sides of the first crypto klines (2017-08-17).
    assert any(day < date(2017, 8, 17) for day in days[3:])
    assert any(day > date(2017, 8, 17) for day in days)
    universe = ab.firm_v1_universe([{"security_id": 1, "ticker": "AAA"}])
    assert len(universe) == 1 + 6 + 5 + 4 + 2
    assert ab.cot_tickers_for(universe) == [
        "ZT=F", "ZN=F", "ZB=F", "GC=F", "CL=F", "SI=F", "NG=F", "HG=F",
        "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X",
    ]  # fmt: skip


def test_the_store_reports_its_memory_and_logs_it_when_dropped(
    ab_lake: Any, caplog: pytest.LogCaptureFixture
) -> None:
    with ll.lake_session():
        ll.declare_loader_window(date(2019, 1, 2), date(2019, 6, 28))
        ll.load_dividends_from_lakehouse("AAA", date(2019, 4, 1))
        ll.load_options_snapshot_from_lakehouse("AAA", date(2019, 3, 1))
        before = ll.loader_store_usage()
        ll.load_options_snapshot_from_lakehouse("AAA", date(2019, 6, 3))
        after = ll.loader_store_usage()
        with caplog.at_level("INFO", logger=ll.logger.name):
            ll.clear_loader_window()
    assert before is not None and after is not None
    sources = {source.source: source for source in before.sources}
    assert sources["equity_distributions"].keys == 1 and sources["equity_distributions"].rows == 6
    assert after.peak_bytes >= before.held_bytes > 0
    assert after.peak_bytes > after.held_bytes  # the March months were evicted
    assert ll.loader_store_usage() is None
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        m.startswith("Window store held equity_distributions: 1 key(s), 6 rows") for m in messages
    )
    assert any("peak ~" in m for m in messages)


def test_a_float_only_difference_is_flagged(ab: ModuleType) -> None:
    call = ab.LoaderCall(loader="split_adjustment_factor", args=("SPY", _DAYS[0]), day=_DAYS[0])
    mismatch = ab.judge(call, "store", 1.0 / 3.0, 1.0 / 3.0, (1.0 / 1.5) / 2.0 + 1e-16)
    assert mismatch is not None and mismatch.float_only
    assert not mismatch.legacy_varies and not mismatch.allowed
    assert ab.judge(call, "store", 0.5, 0.5, 0.5) is None


def test_backtest_comparison_reads_the_result_and_counts(ab: ModuleType) -> None:
    payload = '{\n  "provenance": {\n    "args": 1\n  },\n  "result": {\n    "x": 1.5\n  }\n}'
    stdout = "\n  Assets: SPY\n\n" + payload + "\n"
    counts = [{"loader": "load_x", "path": "store", "calls": 3, "queries": 1}]
    stderr = "noise\n" + ab.LOADER_QUERY_COUNTS_PREFIX + __import__("json").dumps(counts) + "\n"
    assert ab.result_text(stdout) == '{\n  "x": 1.5\n}'
    assert ab.query_counts(stderr) == counts
    assert ab.query_counts("nothing") == []
    with pytest.raises(ValueError):
        ab.result_text("no payload")
    assert ab.first_difference("a\nb", "a\nb") is None
    assert ab.first_difference("a\nb", "a\nc") == "line 2: 'b' != 'c'"
    assert ab.first_difference("a", "a\nb").startswith("line 2")

    def run(store: bool, out: str, code: int = 0) -> Any:
        return ab.CliRun(store=store, returncode=code, stdout=out, stderr=stderr, seconds=1.0)

    same = ab.compare_backtests(run(False, stdout), run(True, stdout))
    assert same["identical"] and same["store"]["queries"] == 1
    differs = ab.compare_backtests(run(False, stdout), run(True, stdout.replace("1.5", "1.6")))
    assert not differs["identical"] and "1.6" in differs["difference"]
    failed = ab.compare_backtests(run(False, stdout), run(True, "", code=1))
    assert not failed["identical"]
    command = ab.backtest_command(["SPY"], "2019-01-02", "2019-02-01", "T", True, ["--set", "a=b"])
    assert command[-3:] == ["--loader-store", "--set", "a=b"]
    assert "--lake-time" in command and "--yes" in command


# ---------------------------------------------------------------- the firm-record path


@pytest.mark.parametrize("store", [True, False])
def test_firm_record_declares_the_run_window(
    preflight_lake: Any, tmp_path: Path, store: bool
) -> None:
    """The firm-record command declares the window store from --start minus
    stage_a.warmup_days to --end unless --no-loader-store is given, and
    writes the query counts and the store's usage to stderr."""
    market = pd.DataFrame(
        {("SPY", "close"): [100.0, 101.0]}, index=pd.DatetimeIndex(["2010-01-04", "2010-01-05"])
    )
    seen: list[ll.LoaderWindow | None] = []

    def record(*_args: Any, **_kwargs: Any) -> FirmRecordResult:
        seen.append(ll.loader_window())
        return FirmRecordResult(
            spec_hash="0" * 64,
            days_written=2,
            last_completed=date(2010, 1, 5),
            complete=True,
            counted=False,
        )

    args = ["--start", "2010-01-04", "--end", "2010-01-05", "--store-dir", str(tmp_path)]
    with (
        patch(
            "agentic_trading.backtest.engine_inputs.load_price_history_with_fallback",
            return_value=(market, "lakehouse data"),
        ),
        patch(
            "agentic_trading.backtest.engine_inputs.load_risk_free_rate_series", return_value=None
        ),
        patch(
            "agentic_trading.backtest.engine_inputs.load_fx_policy_rates",
            return_value=pd.DataFrame({"USD": [0.0]}),
        ),
        patch("agentic_trading.cli.run_firm_record", side_effect=record),
        patch("agentic_trading.cli.check_price_coverage"),
    ):
        result = CliRunner().invoke(
            firm_record_app, [*args, "--loader-store" if store else "--no-loader-store"]
        )

    assert result.exit_code == 0, result.output
    expected = ll.LoaderWindow(
        start=date(2010, 1, 4) - timedelta(days=settings.stage_a.warmup_days),
        end=date(2010, 1, 5),
    )
    assert seen == [expected if store else None]
    stderr = result.stderr.splitlines()
    assert any(line.startswith(LOADER_QUERY_COUNTS_PREFIX) for line in stderr)
    usage = [line for line in stderr if line.startswith(LOADER_STORE_USAGE_PREFIX)]
    assert len(usage) == 1 and usage[0].endswith("null") != store
    assert ll.loader_window() is None


def test_the_firm_v1_preset_is_the_registered_universe(ab: ModuleType) -> None:
    equities = [{"security_id": n, "ticker": t} for n, t in enumerate(FIRM_V1_EQUITIES)]
    assert ab.firm_v1_universe(equities) == list(FIRM_V1_UNIVERSE)


def test_the_firm_record_driver_writes_the_engine_result(
    ab: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The driver runs firm_record_main on its options and writes what
    BacktestEngine.run returns to its result path."""

    class Result(BaseModel):
        x: float = 1.5

    calls: list[tuple[list[str], Any]] = []
    monkeypatch.setattr(BacktestEngine, "run", lambda _self, *_a, **_k: Result())
    monkeypatch.setattr(
        cli, "firm_record_main", lambda args: calls.append((args, BacktestEngine.run(None)))
    )
    path = tmp_path / ab.FIRM_RECORD_RESULT_FILE
    monkeypatch.setattr(sys, "argv", ["-c", str(path), "--start", "2010-01-04"])
    exec(ab.FIRM_RECORD_DRIVER, {"__name__": "__main__"})

    assert calls == [(["--start", "2010-01-04"], Result())]
    assert path.read_text() == Result().model_dump_json(indent=2)


def _firm_record_run(
    ab: ModuleType,
    run_dir: Path,
    store: bool,
    *,
    weight: float = 0.25,
    sharpe: float = 0.5,
    complete: bool = True,
    code: int = 0,
    readme: str = "data\n",
) -> Any:
    """A firm-record run directory holding a one-trial store (with a
    nested `data/README.md` holding `readme`) and a BacktestResult, and
    the CliRun that printed its FirmRecordResult."""
    spec_hash = "a" * 64
    trial = run_dir / "store" / spec_hash
    trial.mkdir(parents=True)
    pd.DataFrame(
        {
            "decision_date": pd.to_datetime(["2010-01-04", "2010-01-04"]),
            "instrument_id": ["SPY", "TLT"],
            "pod_id": ["risk_manager", "rates_manager"],
            "target_weight": [weight, 0.1],
        }
    ).to_parquet(trial / "decisions.parquet")
    (trial / "progress.json").write_text('{"last_completed": "2010-01-04"}\n')
    (trial / "data").mkdir()
    (trial / "data" / "README.md").write_text(readme)
    (run_dir / ab.FIRM_RECORD_RESULT_FILE).write_text(
        f'{{\n  "start_date": "2010-01-04",\n  "final_sharpe_ratio": {sharpe}\n}}'
    )
    payload = (
        f'{{\n  "spec_hash": "{spec_hash}",\n  "days_written": 1,\n'
        f'  "last_completed": "2010-01-04",\n  "complete": {str(complete).lower()},\n'
        '  "counted": false\n}'
    )
    stderr = ab.LOADER_QUERY_COUNTS_PREFIX + '[{"loader": "x", "path": "store", "queries": 2}]'
    run = ab.CliRun(
        store=store, returncode=code, stdout="Reading\n" + payload + "\n", stderr=stderr, seconds=1
    )
    return ab.FirmRecordRun(run=run, run_dir=run_dir)


def test_firm_record_comparison_needs_identical_files_and_result(
    ab: ModuleType, tmp_path: Path
) -> None:
    direct = _firm_record_run(ab, tmp_path / "direct", False)
    same = ab.compare_firm_records(direct, _firm_record_run(ab, tmp_path / "same", True))
    assert same["identical"] and same["store"]["queries"] == 2
    assert same["direct"]["firm_record"]["spec_hash"] == "a" * 64
    assert set(same["trial_dir"]["files"]) == {
        "data/README.md",
        "decisions.parquet",
        "progress.json",
    }

    nested = ab.compare_firm_records(
        direct, _firm_record_run(ab, tmp_path / "nested", True, readme="other\n")
    )
    assert not nested["identical"] and nested["backtest_result"]["identical"]
    assert not nested["trial_dir"]["files"]["data/README.md"]["identical"]

    weights = ab.compare_firm_records(
        direct, _firm_record_run(ab, tmp_path / "weights", True, weight=0.3)
    )
    assert not weights["identical"] and weights["backtest_result"]["identical"]
    assert weights["trial_dir"]["files"]["decisions.parquet"]["difference"] == (
        "row 0 {'decision_date': '2010-01-04 00:00:00', 'instrument_id': 'SPY', "
        "'pod_id': 'risk_manager'}: columns ['target_weight'] differ"
    )

    sharpe = ab.compare_firm_records(
        direct, _firm_record_run(ab, tmp_path / "sharpe", True, sharpe=0.75)
    )
    assert not sharpe["identical"] and sharpe["trial_dir"]["identical"]
    difference = sharpe["backtest_result"]["difference"]
    assert difference == 'line 3 (key "final_sharpe_ratio")'
    assert "0.5" not in difference and "0.75" not in difference

    incomplete = ab.compare_firm_records(
        direct, _firm_record_run(ab, tmp_path / "incomplete", True, complete=False)
    )
    assert not incomplete["identical"]
    failed = ab.compare_firm_records(
        direct, _firm_record_run(ab, tmp_path / "failed", True, code=1)
    )
    assert not failed["identical"] and "stderr_tail" in failed["store"]


def test_the_firm_record_command_is_uncounted_and_switches_the_store(
    ab: ModuleType, tmp_path: Path
) -> None:
    command = ab.firm_record_command(
        "2010-01-04", "2017-12-29", "T", False, tmp_path, ["--set", "a=b"]
    )
    assert command[1:4] == ["-c", ab.FIRM_RECORD_DRIVER, str(tmp_path / "backtest_result.json")]
    assert "--no-loader-store" in command and command[-2:] == ["--set", "a=b"]
    assert command[command.index("--store-dir") + 1] == str(tmp_path / "store")
    assert f"trials.ledger_path={tmp_path / 'ledger.duckdb'}" in command
    assert f"trials.registry_dir={tmp_path / 'registry'}" in command
    assert "--loader-store" in ab.firm_record_command("s", "e", "T", True, tmp_path, [])


# ---------------------------------------------------------------- probes and counts-only output

_EARNINGS = "load_earnings_surprise_from_lakehouse"
_PROBED = (_EARNINGS, _SHORT_INTEREST)
# A short position `_short_interest_rows` gives AAA (its 2018-11-30
# settlement), in every short-interest answer from 2019 on.
_FIXTURE_SHORT_POSITION = "1001000"
# The short position the poisoned reference loader reports instead.
_POISONED_SHORT_POSITION = 987654321


def _probe_args(*extra: str, ticker: str = "AAA") -> list[str]:
    """`loaders` arguments probing the two loaders on two fixture days."""
    only = [arg for name in _PROBED for arg in ("--only-loader", name)]
    return ["loaders", "--day", "2019-03-15", "--day", "2019-04-16", "--ticker", ticker,
            *only, *extra]  # fmt: skip


def _poison_short_interest(ab: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    """Makes the reference short-interest loader report
    `_POISONED_SHORT_POSITION` for every row, so the direct and store
    answers mismatch it."""
    reference = ab.LEGACY_LOADERS[_SHORT_INTEREST]

    def poisoned(*args: Any) -> Any:
        frame = reference(*args)
        if frame is not None:
            shares = frame["short_interest_shares"]
            frame["short_interest_shares"] = shares * 0 + _POISONED_SHORT_POSITION
        return frame

    monkeypatch.setitem(ab.LEGACY_LOADERS, _SHORT_INTEREST, poisoned)


def test_only_loader_builds_the_named_loaders_calls_only(ab: ModuleType) -> None:
    plan = _plan(ab, _EQUALITY_DAYS)
    every = ab.build_calls(plan)
    limited = ab.build_calls(plan, _PROBED)
    assert {call.loader for call in limited} == set(_PROBED)
    assert limited == [call for call in every if call.loader in _PROBED]
    assert len(limited) == 2 * len(_TICKERS) * len(_EQUALITY_DAYS)


def test_only_loader_runs_and_judges_the_named_loaders_and_skips_coverage(
    ab: ModuleType, ab_lake: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "probe.json"
    assert ab.main(_probe_args("--out", str(out))) == 0
    payload = json.loads(out.read_text())
    assert [summary["loader"] for summary in payload["loaders"]] == list(_PROBED)
    assert all(summary["calls"] == 3 for summary in payload["loaders"])
    assert payload["coverage"] == "skipped: --only-loader given"
    assert payload["days"] == ["2019-03-15", "2019-04-16", "2019-04-17"]
    printed = capsys.readouterr().out
    assert "SKIPPED coverage (--only-loader given)" in printed
    assert "OVERALL PASS" in printed
    # No lake read for the other loaders' plan inputs (members, option snapshots).
    queries = " ".join(query for query, _ in ab_lake.calls)
    assert "sp500" not in queries and "option_chain" not in queries


def test_counts_only_output_holds_no_answer_value(
    ab: ModuleType,
    ab_lake: Any,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With a reference answer that mismatches, the full output shows the
    fixture's values in the mismatch; the counts-only output (stdout,
    stderr, the --out file and the log) holds none of them, only counts."""
    _poison_short_interest(ab, monkeypatch)
    assert ab.main(_probe_args("--out", str(tmp_path / "full.json"))) == 1
    full = capsys.readouterr()
    assert str(_POISONED_SHORT_POSITION) in full.out
    assert _FIXTURE_SHORT_POSITION in full.out

    out = tmp_path / "counts.json"
    with caplog.at_level(logging.DEBUG):
        assert ab.main(_probe_args("--counts-only", "--out", str(out))) == 1
    counted = capsys.readouterr()
    written = out.read_text()
    for text in (counted.out, counted.err, written):
        assert str(_POISONED_SHORT_POSITION) not in text
        assert _FIXTURE_SHORT_POSITION not in text
        assert "short_interest_shares" not in text
    assert caplog.records == []
    assert logging.root.manager.disable == logging.NOTSET

    payload = json.loads(written)
    assert set(payload) == {"lake_time", "verdict", "days", "tickers", "coverage", "loaders"}
    assert payload["verdict"] == "FAIL" and payload["coverage"] == "skipped"
    assert payload["tickers"] == ["AAA"]
    assert all(tuple(summary) == ab.COUNT_ONLY_FIELDS for summary in payload["loaders"])
    short = next(s for s in payload["loaders"] if s["loader"] == _SHORT_INTEREST)
    assert short["mismatches"] == {"direct": 3, "store": 3}
    assert short["allowed"] == {"direct": 0, "store": 0}
    assert short["verdict"] == "FAIL"
    lines = counted.out.splitlines()
    assert lines[-1] == "OVERALL FAIL"
    assert f"FAIL {_SHORT_INTEREST} (3 calls)" in lines


def test_a_probed_loader_with_no_real_answer_still_fails(
    ab: ModuleType, ab_lake: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    assert ab.main(_probe_args("--counts-only", ticker="ZZZ")) == 1
    payload, _ = json.JSONDecoder().raw_decode(capsys.readouterr().out)
    assert {(s["loader"], s["reference_answers"], s["verdict"]) for s in payload["loaders"]} == {
        (_EARNINGS, 0, "FAIL"),
        (_SHORT_INTEREST, 0, "FAIL"),
    }


def test_a_day_from_the_holdout_start_is_refused_before_any_query(
    ab: ModuleType, ab_lake: Any
) -> None:
    holdout = settings.stage_a.data_end + timedelta(days=1)
    assert ab.holdout_start() == holdout
    ab.refuse_holdout_days([settings.stage_a.data_end])
    sent = len(ab_lake.calls)
    with pytest.raises(ab.HoldoutDayError, match="holdout"):
        ab.main(["loaders", "--day", "2019-03-15", "--day", "2025-01-02", "--ticker", "AAA",
                 "--counts-only"])  # fmt: skip
    with pytest.raises(ab.HoldoutDayError):
        ab.main(["loaders", "--start", "2024-12-02", "--end", holdout.isoformat(),
                 "--ticker", "AAA"])  # fmt: skip
    # The session after the last day is refused too, against data_end read at call time.
    with temporary_overrides({"stage_a.data_end": date(2019, 4, 16)}):
        with pytest.raises(ab.HoldoutDayError, match="2019-04-17"):
            ab.main(_probe_args())
        with pytest.raises(ab.HoldoutDayError):
            ab.run_loaders_ab(_plan(ab))
    assert len(ab_lake.calls) == sent


def test_an_unknown_loader_name_is_refused(ab: ModuleType, ab_lake: Any) -> None:
    with pytest.raises(SystemExit, match="load_nothing_from_lakehouse"):
        ab.main(_probe_args("--only-loader", "load_nothing_from_lakehouse"))
    assert ab.select_loaders([]) is None
    assert ab.select_loaders([_SHORT_INTEREST, _EARNINGS]) == _PROBED
    assert ab_lake.calls == []
