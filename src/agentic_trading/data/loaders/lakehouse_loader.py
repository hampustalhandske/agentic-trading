# lakehouse_loader.py

"""Every data read a run makes, from the lakehouse's HTTP API (`POST
/v1/sql`, through `data/lakehouse_client.py`): equities/ETFs/indices/
commodities/FX/crypto prices, point-in-time S&P 500 membership, option
chains, dividends, CFTC positioning, earnings, SIC codes, FRED series, Ken
French factors, FINRA short interest and SEC company facts. The service
URL and token come from `LAKEHOUSE_API_URL` and `LAKEHOUSE_API_TOKEN`
(process environment or the repo-root `.env`). Each table the API serves
is already deduplicated (latest write wins), and every query passes its
values as `?` parameters.

This is the only market/alternative-data source a run reads:
`data/loader.py::load_price_history_with_fallback` raises for any ticker
the lake doesn't cover rather than falling back to a live API.

A loader returns `None` only when its query succeeds and finds no rows. A
failed request (`LakehouseError`, connection or timeout errors) propagates
to the caller."""

from datetime import UTC, date, datetime
from functools import cache
from typing import Any, Literal, cast
from weakref import WeakKeyDictionary

import numpy as np
import pandas as pd
from pydantic import BaseModel

from agentic_trading.config import _env_value
from agentic_trading.data.lakehouse_client import Lake
from agentic_trading.data.point_in_time import assert_point_in_time
from agentic_trading.data.trading_calendar import add_sessions, is_session, next_session
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

_URL_ENV_VAR = "LAKEHOUSE_API_URL"
_TOKEN_ENV_VAR = "LAKEHOUSE_API_TOKEN"


class LakeProvenance(BaseModel):
    """Which lake build a run read, and when: `lake_version` is the git
    commit of the running service, `read_at` the UTC time the status was
    read, and `status_generated_at` the timestamp the lake stamped on its
    own status document (`None` when the lake has not written one yet)."""

    lake_version: str
    read_at: datetime
    status_generated_at: datetime | None


def _required_env(name: str) -> str:
    """The value of environment variable `name`; raises naming the
    variable (never its value) when it is unset or empty."""
    value = _env_value(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set; add it to the process environment or the repo-root .env"
        )
    return value


@cache
def _lake() -> Lake:
    """The process-wide lake client, built on first use from
    `LAKEHOUSE_API_URL` and `LAKEHOUSE_API_TOKEN`. The underlying HTTP
    client is safe to share across threads."""
    return Lake(base_url=_required_env(_URL_ENV_VAR), token=_required_env(_TOKEN_ENV_VAR))


def _query(query: str, params: list[Any]) -> pd.DataFrame:
    """Runs one read-only `SELECT` with `?` placeholders bound to `params`
    and returns every row as a DataFrame. Request failures propagate."""
    return cast(pd.DataFrame, _lake().sql(query, params))


def lake_provenance() -> LakeProvenance:
    """The lake's version and the UTC time its status was read, from
    `Lake.status()` and the service's health payload."""
    lake = _lake()
    status = lake.status()
    read_at = datetime.now(UTC)
    generated = status.get("generated_at")
    return LakeProvenance(
        lake_version=str(lake.health()["version"]),
        read_at=read_at,
        status_generated_at=datetime.fromisoformat(generated) if generated else None,
    )


def load_sp500_constituents_asof(as_of_date: date) -> list[str] | None:
    """Point-in-time S&P 500 membership on `as_of_date`, one ticker per
    company, sorted.

    A `sp500_membership` row counts when `start_date <= as_of_date <
    end_date`; a null `end_date` means still a member. Each counted row is
    keyed by the security its ticker belongs to on that date, from the
    `equity_ticker_history` row with `valid_from <= as_of_date < valid_to`
    (null `valid_to` means still valid), so a renamed company yields only
    the ticker it traded under on the date. A ticker with no such history
    row is kept and keyed by itself. When several counted rows resolve to
    one security, the row with the latest `start_date` wins. Returns
    `None` when no row counts; callers fall back to a smaller static
    universe rather than treating this as an empty membership list.
    Calls `assert_point_in_time` on the result for consistency with every
    other loader in this module; it is a no-op on a list of tickers."""
    query = """
        SELECT DISTINCT ticker
        FROM (
            SELECT
                m.ticker,
                ROW_NUMBER() OVER (
                    PARTITION BY h.security_id,
                        CASE WHEN h.security_id IS NULL THEN m.ticker END
                    ORDER BY m.start_date DESC, m.ticker
                ) AS security_rank
            FROM sp500_membership AS m
            LEFT JOIN equity_ticker_history AS h
              ON h.ticker = m.ticker
             AND h.valid_from <= ?
             AND (h.valid_to IS NULL OR h.valid_to > ?)
            WHERE m.start_date <= ?
              AND (m.end_date IS NULL OR m.end_date > ?)
        ) AS ranked
        WHERE security_rank = 1
    """
    as_of = as_of_date.isoformat()
    result = _query(query, [as_of, as_of, as_of, as_of])
    if result.empty:
        return None
    tickers = sorted(result["ticker"].tolist())
    assert_point_in_time(tickers, as_of_date)
    return tickers


class IndexMember(BaseModel):
    """One S&P 500 member on a date: the lake's `security_id` for the
    company and the `ticker` it traded under on that date."""

    security_id: int
    ticker: str


def load_sp500_member_securities_asof(as_of_date: date) -> list[IndexMember] | None:
    """Point-in-time S&P 500 membership on `as_of_date` as securities, one
    `IndexMember` per `security_id`, sorted by ticker.

    Reads `sp500_price_coverage`, whose spans are keyed by `security_id`:
    a span counts when `start_date <= as_of_date < end_date`, and a null
    `end_date` means still a member. When several counted spans share one
    security (a rename recorded as two overlapping spans), the span with
    the latest `start_date` names the ticker. A counted span without a
    `security_id` is left out and named in one WARNING. Returns `None`
    when no span counts."""
    query = """
        SELECT security_id, ticker
        FROM (
            SELECT
                security_id,
                ticker,
                ROW_NUMBER() OVER (
                    PARTITION BY security_id,
                        CASE WHEN security_id IS NULL THEN ticker END
                    ORDER BY start_date DESC, ticker
                ) AS security_rank
            FROM sp500_price_coverage
            WHERE start_date <= ?
              AND (end_date IS NULL OR end_date > ?)
        ) AS ranked
        WHERE security_rank = 1
        ORDER BY ticker
    """
    as_of = as_of_date.isoformat()
    result = _query(query, [as_of, as_of])
    if result.empty:
        return None
    unresolved = result["security_id"].isna()
    if unresolved.any():
        logger.warning(
            "S&P 500 members on %s without a security_id, left out: %s",
            as_of,
            sorted(result.loc[unresolved, "ticker"].tolist()),
        )
    resolved = result.loc[~unresolved]
    members = [
        IndexMember(security_id=int(security_id), ticker=str(ticker))
        for security_id, ticker in zip(resolved["security_id"], resolved["ticker"], strict=True)
    ]
    return members or None


def load_security_log_returns(
    security_ids: list[int], start: date, end: date
) -> dict[int, pd.Series]:
    """Daily split- and distribution-adjusted log returns per security,
    keyed by `security_id`, from `equity_daily_from_bars` bars dated
    `start` to `end` inclusive. One query serves every id in
    `security_ids`.

    A bar's price is `official_close`, or `close` when that is null,
    times the `cum_factor` of the security's `equity_adjustment_factors`
    row with the smallest `effective_date` strictly after the bar's date
    (1.0 when there is none); bars and factors are joined on
    `security_id`, never on ticker. A bar whose price is null or not
    positive counts as missing. The return on day d is the log of d's
    price over the previous bar's price, and exists only when the
    security has bars on d and on the NYSE session immediately before d,
    both inside the range; a gap never yields a multi-day return, and the
    first bar in the range has none.

    Each series has an ascending `DatetimeIndex`. A security with no
    returns in the range has no key. Factor rows dated after `end` set
    the price level but cancel in every return, so no return depends on
    data after `end`. Self-asserts point-in-time on every series against
    `end` before returning."""
    if not security_ids:
        return {}
    placeholders = ", ".join("?" for _ in security_ids)
    query = f"""
        SELECT bars.security_id, bars.date,
               bars.price * COALESCE(factors.cum_factor, 1.0) AS price
        FROM (
            SELECT security_id, date, COALESCE(official_close, close) AS price
            FROM equity_daily_from_bars
            WHERE security_id IN ({placeholders})
              AND date BETWEEN ? AND ?
              AND COALESCE(official_close, close) > 0
        ) AS bars
        ASOF LEFT JOIN (
            SELECT security_id, effective_date, cum_factor
            FROM equity_adjustment_factors
            WHERE security_id IN ({placeholders})
              AND effective_date > ?
        ) AS factors
          ON factors.security_id = bars.security_id
         AND bars.date < factors.effective_date
        ORDER BY bars.security_id, bars.date
    """
    ids = [int(security_id) for security_id in security_ids]
    params = [*ids, start.isoformat(), end.isoformat(), *ids, start.isoformat()]
    frame = _query(query, params)
    if frame.empty:
        return {}
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.drop_duplicates(["security_id", "date"])
    previous_date = frame.groupby("security_id")["date"].shift(1)
    session_after = {
        stamp: pd.Timestamp(next_session(stamp.date()))
        for stamp in pd.DatetimeIndex(previous_date.dropna().unique())
    }
    consecutive = previous_date.map(session_after) == frame["date"]
    log_price = pd.Series(np.log(frame["price"].to_numpy(dtype=np.float64)), index=frame.index)
    frame["log_return"] = log_price - log_price.groupby(frame["security_id"]).shift(1)
    returns = frame.loc[consecutive]

    result: dict[int, pd.Series] = {}
    for security_id, group in returns.groupby("security_id"):
        series = pd.Series(
            group["log_return"].to_numpy(dtype=np.float64),
            index=pd.DatetimeIndex(group["date"]),
            name="log_return",
        ).sort_index()
        assert_point_in_time(series, end)
        result[int(cast(int, security_id))] = series
    return result


def split_adjustment_factor(ticker: str, quote_date: date) -> float:
    """Factor that moves an option-chain row quoted on `quote_date` onto
    the basis of `ticker`'s `instrument_prices` close: the product of
    `1 / ratio` over the ticker's rows in the lake's
    `yahoo_split_close_basis` with an ex-date after `quote_date` (`ratio`
    is old shares per new share, so a 4-for-1 split is 0.25 and
    contributes 4.0). Returns 1.0 when no such row exists; a missing or
    non-positive `ratio` contributes nothing.

    `dolthub_option_chain` strikes and premiums are quoted as traded and
    never adjusted, while `instrument_prices` closes are split-adjusted by
    Yahoo; dividing a chain row by this factor puts its strike and
    premiums on the same basis as the underlying's close. A row whose
    `in_close_basis` is false is a split Yahoo's close does not reflect,
    so it is skipped and the strike lands on the close's actual basis;
    rows where it is true or null count. Spin-offs Yahoo writes as
    fractional splits count too, because the close is adjusted for them."""
    query = """
        SELECT ratio
        FROM yahoo_split_close_basis
        WHERE ticker = ?
          AND ex_date > ?
          AND in_close_basis IS NOT FALSE
    """
    frame = _query(query, [ticker, quote_date.isoformat()])
    factor = 1.0
    for ratio in frame["ratio"]:
        if ratio and ratio > 0:
            factor /= float(ratio)
    return factor


def _apply_split_adjustment(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Rescales an option-chain frame's price-denominated columns
    (`strike`, `bid`, `ask`, `gamma`, `vega`, `theta`, `rho`) onto
    `instrument_prices`' continuously split-adjusted basis, per each
    row's own `date`. `delta` and `implied_volatility` are dimensionless
    and untouched. A no-op for any ticker/date with no later split."""
    frame = frame.copy()
    for quote_date in frame["date"].dt.date.unique():
        factor = split_adjustment_factor(ticker, quote_date)
        if factor == 1.0:
            continue
        mask = frame["date"].dt.date == quote_date
        frame.loc[mask, ["strike", "bid", "ask"]] /= factor
        frame.loc[mask, "gamma"] /= factor**2
        frame.loc[mask, ["vega", "theta", "rho"]] /= factor
    return frame


# Underlyings whose end-of-day chains `option_chain_history` holds from
# 2008, covering the years before `dolthub_option_chain` begins.
_OPTION_CHAIN_HISTORY_UNDERLYINGS = frozenset({"SPY", "QQQ", "IWM"})

# First `dolthub_option_chain` date per underlying, keyed by the lake
# client that answered; a `None` value records an underlying with no rows.
_FIRST_DOLTHUB_CHAIN_DATES: WeakKeyDictionary[Lake, dict[str, date | None]] = WeakKeyDictionary()


def _first_dolthub_chain_date(underlying: str) -> date | None:
    """The earliest snapshot `date` in `dolthub_option_chain` for
    `underlying`, or `None` when it has no rows. Read once per underlying
    and lake client and kept for the life of the process: the table only
    grows forward, so its first date never moves."""
    lake = _lake()
    cached = _FIRST_DOLTHUB_CHAIN_DATES.setdefault(lake, {})
    if underlying not in cached:
        query = """
            SELECT MIN(date) AS first
            FROM dolthub_option_chain
            WHERE underlying = ?
        """
        frame = _query(query, [underlying])
        first = None if frame.empty else frame.iloc[0]["first"]
        cached[underlying] = None if pd.isna(first) else pd.Timestamp(first).date()
    return cached[underlying]


def _option_chain_source(underlying: str, as_of_date: date) -> tuple[str, str]:
    """The chain table a read of `underlying` as of `as_of_date` queries,
    and that table's snapshot-date column. An underlying in
    `_OPTION_CHAIN_HISTORY_UNDERLYINGS` reads `option_chain_history`
    (`snapshot_date`) when `as_of_date` precedes its first
    `dolthub_option_chain` date or it has no dolthub rows; every other
    read uses `dolthub_option_chain` (`date`)."""
    if underlying in _OPTION_CHAIN_HISTORY_UNDERLYINGS:
        first = _first_dolthub_chain_date(underlying)
        if first is None or as_of_date < first:
            return "option_chain_history", "snapshot_date"
    return "dolthub_option_chain", "date"


# Calendar days before `as_of_date` within which
# `load_options_snapshot_from_lakehouse` looks for the latest snapshot by
# default: chain coverage is not daily-dense.
DEFAULT_OPTIONS_SNAPSHOT_LOOKBACK_DAYS = 10


def load_options_snapshot_from_lakehouse(
    ticker: str, as_of_date: date, lookback_days: int = DEFAULT_OPTIONS_SNAPSHOT_LOOKBACK_DAYS
) -> pd.DataFrame | None:
    """Real historical options-chain rows (strike, expiration, bid, ask,
    implied_volatility, delta, gamma, theta, vega, rho — no open interest,
    no volume, confirmed absent from the source) for `ticker`, from the
    most recent available snapshot date on or before `as_of_date` within
    `lookback_days` — coverage is real but not daily-dense (roughly
    every-other-trading-day for AAPL), so a same-day match isn't
    guaranteed even for a covered ticker.

    Rows come from the table `_option_chain_source` picks for `ticker`
    and `as_of_date`: `dolthub_option_chain`, or `option_chain_history`
    for SPY, QQQ and IWM before dolthub coverage begins. A history row's
    `snapshot_date` (a 16:00 ET end-of-day snapshot) is returned as
    `date`, and its `bid`/`ask` are the prices; its `mark` and
    `last_trade_price` are never read. Both tables yield the same
    columns and dtypes.

    Returns `None` if `ticker` has no coverage at all or the nearest
    snapshot is more than `lookback_days` stale — the caller reports a
    flat, zero-confidence signal in that case rather than treat a miss as
    fatal. `strike`/`bid`/`ask`/`gamma`/`vega`/`theta`/`rho` are rescaled
    onto `instrument_prices`' continuously split-adjusted basis via
    `_apply_split_adjustment` before returning, since the raw chain is
    archival and never itself adjusted for a later split. Self-asserts
    point-in-time on `date` before returning."""
    window_start = (as_of_date - pd.Timedelta(days=lookback_days)).isoformat()
    window_end = as_of_date.isoformat()
    table, date_column = _option_chain_source(ticker, as_of_date)
    query = f"""
        SELECT {date_column} AS date, expiration, option_type, strike, bid, ask,
               implied_volatility, delta, gamma, theta, vega, rho
        FROM {table}
        WHERE underlying = ?
          AND {date_column} BETWEEN ? AND ?
          AND {date_column} = (
              SELECT MAX({date_column})
              FROM {table}
              WHERE underlying = ?
                AND {date_column} BETWEEN ? AND ?
          )
    """
    frame = _query(query, [ticker, window_start, window_end, ticker, window_start, window_end])
    if frame.empty:
        return None
    frame["date"] = pd.to_datetime(frame["date"])
    frame["expiration"] = pd.to_datetime(frame["expiration"])
    frame = _apply_split_adjustment(frame, ticker)
    assert_point_in_time(frame, as_of_date, date_column="date")
    return frame


# CFTC contract-market codes for the tickers the loader covers, keyed by
# the `cftc_cot_disaggregated` (commodities) and `cftc_cot_tff` (FX and
# rate futures) tables.
_COT_DISAGGREGATED_CODES = {
    "GC=F": "088691",
    "CL=F": "067651",
    "SI=F": "084691",
    "NG=F": "023651",
    "HG=F": "085692",
}
_COT_TFF_CODES = {
    "EURUSD=X": "099741",
    "GBPUSD=X": "096742",
    "USDJPY=X": "097741",
    "AUDUSD=X": "232741",
    "ZT=F": "042601",
    "ZN=F": "043602",
    "ZB=F": "020601",
}
# Treasury ETFs have no COT category of their own; each reads the
# positioning of the Treasury future it tracks. Every other ticker the COT
# tables cover (commodities, FX and rate futures) is its own COT ticker.
DEFAULT_COT_TICKER_BY_ASSET: dict[str, str] = {"SHY": "ZT=F", "IEF": "ZN=F", "TLT": "ZB=F"}
# Calendar days of reports before `as_of_date` that
# `load_cot_positioning_from_lakehouse` returns by default: a little over
# a year of weekly reports.
DEFAULT_COT_LOOKBACK_DAYS = 400


def load_cot_positioning_from_lakehouse(
    ticker: str, as_of_date: date, lookback_days: int = DEFAULT_COT_LOOKBACK_DAYS
) -> pd.Series | None:
    """Net speculative positioning as a fraction of open interest, from
    CFTC Commitment of Traders reports — `managed_money_long -
    managed_money_short` (the disaggregated report's commodity category)
    for commodities, `leveraged_funds_long - leveraged_funds_short` (the
    TFF report's closest equivalent) for FX/rates, divided by
    `open_interest_all`. A given `ticker` only ever appears in one of the
    two tables (commodities in `cftc_cot_disaggregated`, FX/rate-futures
    in `cftc_cot_tff`), selected by its CFTC contract-market code.

    Point-in-time: each report is gated on its row in
    `cftc_cot_publication`, joined on `report_date`. A report whose
    `published_at` is set (always 15:30 New York time, before the close)
    is available from its `published_date`; one without a `published_at`
    is available only from the day after its `published_date`. A report
    with no publication row, or a null `published_date` (every report
    before 2000), is never available.

    Returns `None` if `ticker` has no COT coverage at all — callers
    should treat that as "no positioning read available," not an
    empty-but-valid series. Self-asserts point-in-time on the
    `report_date` index before returning."""
    window_start = (as_of_date - pd.Timedelta(days=lookback_days)).isoformat()
    availability_cutoff = as_of_date.isoformat()
    queries = [
        (
            "cftc_cot_disaggregated",
            _COT_DISAGGREGATED_CODES,
            "(managed_money_long - managed_money_short)",
        ),
        (
            "cftc_cot_tff",
            _COT_TFF_CODES,
            "(leveraged_funds_long - leveraged_funds_short)",
        ),
    ]
    for table, codes, net_position_expr in queries:
        code = codes.get(ticker)
        if code is None:
            continue
        query = f"""
            SELECT cot.report_date,
                   {net_position_expr} / cot.open_interest_all AS net_positioning_ratio
            FROM {table} AS cot
            JOIN cftc_cot_publication AS pub ON pub.report_date = cot.report_date
            WHERE cot.cftc_contract_market_code = ?
              AND cot.report_date BETWEEN ? AND ?
              AND cot.open_interest_all > 0
              AND (
                  (pub.published_at IS NOT NULL AND pub.published_date <= ?)
                  OR (pub.published_at IS NULL AND pub.published_date < ?)
              )
            ORDER BY cot.report_date
        """
        frame = _query(
            query,
            [code, window_start, availability_cutoff, availability_cutoff, availability_cutoff],
        )
        if not frame.empty:
            frame["report_date"] = pd.to_datetime(frame["report_date"])
            series = frame.set_index("report_date")["net_positioning_ratio"]
            assert_point_in_time(series, as_of_date)
            return series
    return None


# `equity_distributions.distribution_type` of an ordinary recurring cash
# dividend; specials, spin-offs and unclassified large payments carry
# other types.
_REGULAR_CASH = "regular_cash"


def load_dividends_from_lakehouse(ticker: str, as_of_date: date) -> pd.Series | None:
    """Regular cash dividends per share for `ticker`, from the
    `equity_distributions` table: rows whose `distribution_type` is
    `regular_cash`, as `amount_split_adjusted` indexed by ex-date
    (datetime64, ascending), with `ex_date <= as_of_date`.

    Every amount is on one split-consistent basis, so a stock split
    never reads as a dividend cut. Special cash distributions, spin-offs
    and other non-recurring payments are excluded. Gating on the ex-date
    is safe because a dividend is declared before its ex-date, so it is
    public by then. Returns `None` when no rows qualify. Self-asserts
    point-in-time on the ex-date index before returning."""
    query = """
        SELECT ex_date, amount_split_adjusted
        FROM equity_distributions
        WHERE ticker = ? AND distribution_type = ? AND ex_date <= ?
        ORDER BY ex_date
    """
    frame = _query(query, [ticker, _REGULAR_CASH, as_of_date.isoformat()])
    if frame.empty:
        return None
    frame["ex_date"] = pd.to_datetime(frame["ex_date"])
    series = frame.set_index("ex_date")["amount_split_adjusted"]
    assert_point_in_time(series, as_of_date)
    return series


def load_earnings_surprise_from_lakehouse(
    ticker: str, as_of_date: date
) -> tuple[float, float, pd.Timestamp] | None:
    """The most recent real EPS surprise for `ticker` known at the close
    of `as_of_date`: `(reported, estimate, announcement_date)` for the
    quarter with the latest announcement (`dolthub_earnings_calendar.
    date`) available by then. An announcement whose `when` is "Before
    market open" is available on its date; "After market close" or a
    NULL `when` makes it available the next NYSE session. The returned
    `announcement_date` is the calendar date itself. Joins the two
    tables on a 75-day window after `dolthub_eps_history.period_end_date`
    — a generous margin around the typical ~29-30 day real reporting
    lag, not a tight estimate.

    Returns `None` if `ticker` has no earnings history at all, or no
    quarter's announcement is available yet as of `as_of_date`.
    Self-asserts point-in-time on the availability date before
    returning."""
    query = """
        SELECT h.reported, h.estimate, c.date AS announcement_date, c."when" AS announced_when
        FROM (
            SELECT *
            FROM dolthub_eps_history
            WHERE act_symbol = ? AND reported IS NOT NULL AND estimate IS NOT NULL
        ) h
        JOIN (
            SELECT *
            FROM dolthub_earnings_calendar
            WHERE act_symbol = ?
        ) c
        ON c.act_symbol = h.act_symbol
           AND CAST(c.date AS DATE) BETWEEN CAST(h.period_end_date AS DATE)
               AND date_add(CAST(h.period_end_date AS DATE), INTERVAL 75 DAY)
        WHERE CAST(c.date AS DATE) <= ?
        ORDER BY c.date DESC
    """
    frame = _query(query, [ticker, ticker, as_of_date.isoformat()])
    if frame.empty:
        return None
    frame["announcement_date"] = pd.to_datetime(frame["announcement_date"])
    frame["available_date"] = [
        _earnings_available_date(announced.date(), when)
        for announced, when in zip(frame["announcement_date"], frame["announced_when"], strict=True)
    ]
    frame = frame[frame["available_date"] <= pd.Timestamp(as_of_date)]
    if frame.empty:
        return None
    assert_point_in_time(frame, as_of_date, date_column="available_date")
    row = frame.iloc[0]
    return float(row["reported"]), float(row["estimate"]), pd.Timestamp(row["announcement_date"])


# `dolthub_earnings_calendar.when` value for an announcement made before
# the open, so it is public for that day's session; any other value
# (after the close, or unknown) waits for the next session.
_BEFORE_MARKET_OPEN = "Before market open"


def _earnings_available_date(announced: date, when: str | None) -> pd.Timestamp:
    """The first day an announcement on `announced` is known at the
    close: that day for a `_BEFORE_MARKET_OPEN` announcement, otherwise
    the next NYSE session."""
    if when == _BEFORE_MARKET_OPEN:
        return pd.Timestamp(announced)
    return pd.Timestamp(next_session(announced))


def load_sic_code_from_lakehouse(ticker: str) -> str | None:
    """SIC (Standard Industrial Classification) code for `ticker` from
    SEC EDGAR, via `sec_sic_classification`; `None` if `ticker` has no SIC
    record (mostly non-equity ETF/future/FX tickers, which have no SEC
    filing to classify).

    Not point-in-time: the lake holds one current classification per
    ticker and no history of reclassifications, so a backtest date sees
    the code as of the lake's last scrape. A company reclassified after
    that date is grouped by its later industry. This is a known look-ahead
    limit of SIC-based pair screening, with no `as_of_date` gate possible
    until the lake carries classification history."""
    frame = _query("SELECT sic FROM sec_sic_classification WHERE ticker = ?", [ticker])
    if frame.empty or pd.isna(frame.iloc[0]["sic"]):
        return None
    return str(frame.iloc[0]["sic"])


def same_sic_candidate_pairs(tickers: list[str]) -> list[tuple[str, str]]:
    """Every pair within `tickers` that share a real SEC SIC code — a
    genuine industry-classification candidate list for pairs-cointegration
    screening, replacing a hand-picked "looks same-sector to me" list.
    Tickers with no SIC record (non-equities, or an equity outside SEC
    EDGAR's coverage) can't form a pair here, so this only ever
    under-generates candidates. The codes are current, not point-in-time
    (see `load_sic_code_from_lakehouse`)."""
    sic_by_ticker: dict[str, str] = {}
    for ticker in tickers:
        sic = load_sic_code_from_lakehouse(ticker)
        if sic is not None:
            sic_by_ticker[ticker] = sic

    pairs: list[tuple[str, str]] = []
    tickers_with_sic = sorted(sic_by_ticker)
    for i, ticker_a in enumerate(tickers_with_sic):
        for ticker_b in tickers_with_sic[i + 1 :]:
            if sic_by_ticker[ticker_a] == sic_by_ticker[ticker_b]:
                pairs.append((ticker_a, ticker_b))
    return pairs


# Lake symbol in `crypto_klines_1m` for each crypto ticker the loader
# serves: Binance USDT spot one-minute klines. USDT trades within a few
# basis points of USD, so the USDT price stands in for the USD price.
_CRYPTO_KLINE_SYMBOLS = {
    "BTC-USD": "BTCUSDT",
    "ETH-USD": "ETHUSDT",
}
# The NYSE regular session in America/New_York: the first and last
# one-minute kline start times of each session.
_SESSION_FIRST_MINUTE = "09:30:00"
_SESSION_LAST_MINUTE = "15:59:00"


def _equity_price_rows(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    """Daily `ticker`/`date`/`open`/`close`/`raw_close`/`volume` rows from
    `instrument_prices` for `tickers` between `start` and `end`
    inclusive: `close` is `adj_close`, `open` is the raw open scaled by
    `adj_close / close`, `raw_close` the unadjusted close.

    A row whose raw open lies outside its raw `[low, high]` range
    (boundaries included as valid), or whose raw low exceeds its raw
    high, has an inconsistent bar, so its `open` is NaN; the rest of the
    row is kept. A row with a null raw low or high keeps its open. Logs
    one WARNING naming each ticker with masked rows and its count when
    any row is masked."""
    placeholders = ", ".join("?" for _ in tickers)
    query = f"""
        SELECT ticker, date,
               CASE WHEN close > 0 THEN open * adj_close / close ELSE NULL END AS open,
               open AS raw_open,
               high AS raw_high,
               low AS raw_low,
               adj_close AS close,
               close AS raw_close,
               volume
        FROM instrument_prices
        WHERE ticker IN ({placeholders})
          AND date BETWEEN ? AND ?
        ORDER BY ticker, date
    """
    frame = _query(query, [*tickers, start.isoformat(), end.isoformat()])
    raw_open = frame["raw_open"].astype(float)
    raw_high = frame["raw_high"].astype(float)
    raw_low = frame["raw_low"].astype(float)
    inconsistent = (raw_open < raw_low) | (raw_open > raw_high) | (raw_low > raw_high)
    if inconsistent.any():
        frame.loc[inconsistent, "open"] = np.nan
        masked_by_ticker = frame.loc[inconsistent, "ticker"].value_counts().sort_index()
        logger.warning(
            "Masked the open of rows outside their raw low-high range: %s",
            ", ".join(f"{ticker}: {count}" for ticker, count in masked_by_ticker.items()),
        )
    return frame[["ticker", "date", "open", "close", "raw_close", "volume"]]


def _crypto_price_rows(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    """One `ticker`/`date`/`open`/`close`/`raw_close`/`volume` row per
    NYSE session between `start` and `end` inclusive for each crypto
    ticker in `tickers` (keys of `_CRYPTO_KLINE_SYMBOLS`), built from
    `crypto_klines_1m` minutes whose `open_time` (naive UTC, the minute's
    start) falls in the session in America/New_York: `open` is the open
    of the minute starting at `_SESSION_FIRST_MINUTE`, `close` and
    `raw_close` the close of the minute starting at
    `_SESSION_LAST_MINUTE`, `volume` the summed volume of the minutes
    from `_SESSION_FIRST_MINUTE` to `_SESSION_LAST_MINUTE`. A day without
    a `_SESSION_LAST_MINUTE` kline, or that is not an NYSE session, has
    no row. Only minutes inside the session on days in the range are
    read; the aggregation runs in the lake."""
    symbols = [_CRYPTO_KLINE_SYMBOLS[ticker] for ticker in tickers]
    placeholders = ", ".join("?" for _ in symbols)
    query = f"""
        SELECT symbol,
               CAST(local_time AS DATE) AS date,
               max(open) FILTER (WHERE CAST(local_time AS TIME) = CAST(? AS TIME)) AS open,
               max(close) FILTER (WHERE CAST(local_time AS TIME) = CAST(? AS TIME)) AS close,
               sum(volume) AS volume
        FROM (
            SELECT symbol, open, close, volume,
                   timezone(?, open_time AT TIME ZONE 'UTC') AS local_time
            FROM crypto_klines_1m
            WHERE symbol IN ({placeholders})
              AND open_time >= CAST(? AS TIMESTAMP)
              AND open_time < CAST(? AS TIMESTAMP)
        )
        WHERE CAST(local_time AS TIME) BETWEEN CAST(? AS TIME) AND CAST(? AS TIME)
        GROUP BY symbol, CAST(local_time AS DATE)
        HAVING max(close) FILTER (WHERE CAST(local_time AS TIME) = CAST(? AS TIME)) IS NOT NULL
        ORDER BY symbol, date
    """
    params = [
        _SESSION_FIRST_MINUTE,
        _SESSION_LAST_MINUTE,
        _DECISION_TIMEZONE,
        *symbols,
        start.isoformat(),
        (end + pd.Timedelta(days=1)).isoformat(),
        _SESSION_FIRST_MINUTE,
        _SESSION_LAST_MINUTE,
        _SESSION_LAST_MINUTE,
    ]
    frame = _query(query, params)
    if frame.empty:
        return frame
    ticker_by_symbol = {symbol: ticker for ticker, symbol in _CRYPTO_KLINE_SYMBOLS.items()}
    frame["ticker"] = frame["symbol"].map(ticker_by_symbol)
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame[[is_session(stamp.date()) for stamp in frame["date"]]]
    frame["raw_close"] = frame["close"]
    return frame[["ticker", "date", "open", "close", "raw_close", "volume"]]


def load_price_history_from_lakehouse(
    tickers: list[str], start: date, end: date
) -> pd.DataFrame | None:
    """Returns a `MultiIndex(ticker, field)` frame (`open`/`close`/
    `raw_close`/`volume`/`log_return`) for every ticker in `tickers` that the lake actually
    covers, or `None` if the lake covers none of them —
    `load_price_history_with_fallback` is what turns an incomplete result
    into a clear error naming the uncovered ticker(s). A failed request
    propagates.

    Crypto tickers (`_CRYPTO_KLINE_SYMBOLS`) are built from one-minute
    klines by `_crypto_price_rows`, one bar per NYSE session. Decisions
    are taken at the US close, so crypto is sampled at 16:00
    America/New_York (the close of the 15:59 minute) rather than at
    Yahoo's 23:59 UTC daily close, keeping it synchronous with every
    other asset's close.

    Every other ticker reads `instrument_prices` and uses `adj_close`
    (dividend/split-adjusted) rather than raw `close` — correct for a
    multi-year backtest where corporate actions actually occur — and
    scales the raw `open` by the same `adj_close / close` factor so the
    open the engine fills at and the close it marks at are on one price
    basis (an unadjusted open against an adjusted close would book the
    whole dividend-adjustment gap as a loss on every fill). A row whose
    raw open lies outside its raw low-high range, or whose low exceeds
    its high, gets a NaN `open` and keeps every other field (see
    `_equity_price_rows`, which logs one WARNING per call with the masked
    count per ticker); the engine does not fill on such a day.
    Self-asserts point-in-time on the combined frame's date index against
    `end` before returning.
    """
    crypto = [ticker for ticker in tickers if ticker in _CRYPTO_KLINE_SYMBOLS]
    others = [ticker for ticker in tickers if ticker not in _CRYPTO_KLINE_SYMBOLS]
    parts = []
    if others:
        parts.append(_equity_price_rows(others, start, end))
    if crypto:
        parts.append(_crypto_price_rows(crypto, start, end))
    parts = [part for part in parts if not part.empty]
    if not parts:
        return None
    frame = pd.concat(parts, ignore_index=True)

    frame["date"] = pd.to_datetime(frame["date"])
    covered_tickers = set(frame["ticker"].unique())
    missing = [ticker for ticker in tickers if ticker not in covered_tickers]
    if missing:
        logger.info("Lakehouse has no price coverage for %s in this range", missing)

    per_ticker: dict[str, pd.DataFrame] = {}
    for ticker, group in frame.groupby("ticker"):
        indexed = (
            group.set_index("date")[["open", "close", "raw_close", "volume"]].sort_index().copy()
        )
        # Some lakehouse rows have a real, non-zero
        # `close` but a corrupted `adj_close` of exactly 0.0 — a
        # zero/negative price here would silently produce
        # log(0) = -inf rather than NaN, which .dropna() downstream would
        # NOT catch. Mask non-positive prices to NaN before taking the
        # log, so a corrupted row degrades to "missing," the same way a
        # genuine data gap already does, not to a silent -inf/NaN
        # poisoning every consumer's arithmetic.
        indexed.loc[indexed["close"] <= 0.0, "close"] = np.nan
        indexed.loc[indexed["open"] <= 0.0, "open"] = np.nan
        # `raw_close` is the unadjusted print: option strikes, marks and
        # settlement live on that basis, never the dividend-adjusted one.
        indexed.loc[indexed["raw_close"] <= 0.0, "raw_close"] = np.nan
        # Spot FX and indices carry no volume; a zero means "no ADV", not
        # "no trading", so it becomes NaN and impact/participation skip it.
        indexed["volume"] = indexed["volume"].astype(float)
        indexed.loc[indexed["volume"] <= 0.0, "volume"] = np.nan
        indexed["log_return"] = np.log(indexed["close"] / indexed["close"].shift(1))
        per_ticker[str(ticker)] = indexed

    combined = pd.concat(per_ticker, axis=1)
    assert_point_in_time(combined, end)
    return combined


def load_raw_close_from_lakehouse(
    ticker: str, as_of_date: date, lookback_days: int
) -> float | None:
    """`ticker`'s unadjusted `instrument_prices` close (the `raw_close`
    basis of `load_price_history_from_lakehouse`, which option strikes are
    rescaled onto) on its last session on or before `as_of_date` within
    `lookback_days` calendar days. `None` when the lake has no positive
    close for `ticker` in that window. Self-asserts point-in-time on
    `date` before returning."""
    window_start = (as_of_date - pd.Timedelta(days=lookback_days)).isoformat()
    query = """
        SELECT date, close
        FROM instrument_prices
        WHERE ticker = ?
          AND date BETWEEN ? AND ?
          AND close > 0
        ORDER BY date DESC
        LIMIT 1
    """
    frame = _query(query, [ticker, window_start, as_of_date.isoformat()])
    if frame.empty:
        return None
    assert_point_in_time(frame, as_of_date, date_column="date")
    return float(frame["close"].iloc[0])


def load_fred_series_from_lakehouse(series_id: str, as_of_date: date) -> pd.Series | None:
    """A FRED macro series (`DGS3MO`/`DGS2`/`DGS10`/`VIXCLS`/`CPIAUCSL`/
    `BAMLH0A0HYM2`/`BAMLC0A0CM` — the same series `macro_loader.
    SUPPORTED_SERIES` names) as it stood on `as_of_date`, from the
    `fred_vintages` table (`series_id`, `date`, `realtime_start`, `value`;
    one row per published version of an observation). Each observation
    date carries the value of its latest version with `realtime_start <=
    as_of_date`; an observation with no version published by then is left
    out. The index is sorted and unique. Returns `None` if no rows remain.
    Self-asserts point-in-time on the date index before returning."""
    query = """
        SELECT date, value
        FROM (
            SELECT date, value, ROW_NUMBER() OVER (
                PARTITION BY date
                ORDER BY realtime_start DESC
            ) AS rn
            FROM fred_vintages
            WHERE series_id = ? AND realtime_start <= ? AND date <= ?
        )
        WHERE rn = 1
        ORDER BY date
    """
    cutoff = as_of_date.isoformat()
    frame = _query(query, [series_id, cutoff, cutoff])
    if frame.empty:
        return None
    frame["date"] = pd.to_datetime(frame["date"])
    series = frame.set_index("date")["value"]
    assert_point_in_time(series, as_of_date)
    return series


# `famafrench_returns` holds every Ken French dataset as long rows
# (`dataset`, `frequency`, `date`, `portfolio`, `value`); each model is one
# daily US factor dataset and the portfolios that make up its factors, in
# the column order the factor-pod analysts consume.
_KEN_FRENCH_DATASET_BY_MODEL = {
    "3factor": "F-F_Research_Data_Factors_daily",
    "5factor": "F-F_Research_Data_5_Factors_2x3_daily",
}
_KEN_FRENCH_FACTORS_BY_MODEL = {
    "3factor": ("Mkt-RF", "SMB", "HML", "RF"),
    "5factor": ("Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"),
}
_KEN_FRENCH_FREQUENCY = "daily"
# Ken French posts each month's daily factors after that month closes, so
# the end of the following month is the conservative public date.
_KEN_FRENCH_PUBLICATION_LAG_MONTHS = 1


def load_fama_french_factors_from_lakehouse(
    as_of_date: date, model: str = "3factor"
) -> pd.DataFrame | None:
    """Daily Fama-French factor returns for `model` ("3factor" or
    "5factor"), from the `famafrench_returns` table's matching daily US
    factor dataset. A day's values are returned only once public, which
    is the last day of the month after that day's month
    (`_KEN_FRENCH_PUBLICATION_LAG_MONTHS`) falling on or before
    `as_of_date`. The values are today's revised history, because the
    source keeps no daily vintages. The lake stores returns as plain
    percentage points (0.09 meaning 0.09%, Ken French's own
    convention); the result is on a 0-1 return scale with Ken French's
    `Mkt-RF`/`SMB`/`HML`/`RF`(/`RMW`/`CMA`) column names. Returns `None`
    if the table has no rows or `model` isn't one of the two supported
    names. Self-asserts point-in-time on the date index before
    returning."""
    dataset = _KEN_FRENCH_DATASET_BY_MODEL.get(model)
    if dataset is None:
        return None
    factors = _KEN_FRENCH_FACTORS_BY_MODEL[model]

    placeholders = ", ".join("?" for _ in factors)
    query = f"""
        SELECT date, portfolio, value
        FROM famafrench_returns
        WHERE dataset = ? AND frequency = ?
          AND portfolio IN ({placeholders})
          AND last_day(
                date_add(CAST(date AS DATE), INTERVAL {_KEN_FRENCH_PUBLICATION_LAG_MONTHS} MONTH)
              ) <= ?
        ORDER BY date
    """
    long = _query(query, [dataset, _KEN_FRENCH_FREQUENCY, *factors, as_of_date.isoformat()])
    if long.empty:
        return None
    long["date"] = pd.to_datetime(long["date"])
    wide = long.pivot_table(index="date", columns="portfolio", values="value", aggfunc="max")
    frame = wide[[factor for factor in factors if factor in wide.columns]]
    result = (frame.astype(float) / 100.0).sort_index()
    assert_point_in_time(result, as_of_date)
    return result


# The columns identifying one reported period of one concept; a later
# filing restating the same period replaces the earlier value.
_COMPANY_FACT_PERIOD_KEY = ["concept", "unit", "period_start", "period_end"]


def load_company_facts_from_lakehouse(ticker: str, as_of_date: date) -> pd.DataFrame | None:
    """Filed SEC EDGAR company facts for `ticker` known at the close of
    `as_of_date`, one row per reported period: columns `concept`, `unit`,
    `period_start`, `period_end`, `value`, `fiscal_year`,
    `fiscal_period`, `form`, `filed` and `available_date` (the four date
    columns datetime64), sorted by `filed` then `period_end`.

    Reads the `sec_company_facts_log` view, which keeps every scrape,
    because the deduplicated `sec_company_facts` view drops one of two
    facts that differ only in `period_start` until lake issue LK-13
    lands. Rows identical apart from their scrape (`scraped_at`, `dt`,
    `table`) count once.

    A filing reports several period windows for one concept (current
    quarter, year-to-date, prior-year comparatives) as separate rows, and
    later filings repeat or restate earlier periods. Gated on when the
    filing became public, never the period it describes:
    `available_date` is set from the filing's latest acceptance time in
    `sec_filings` (`_SEC_ACCEPTANCE_QUERY`) by `_sec_available_dates`,
    and only rows with `available_date <= as_of_date` are kept. Of
    those, each (`concept`, `unit`, `period_start`, `period_end`) keeps
    the row with the latest `available_date`, ties broken by the latest
    `filed`. Returns `None` if no rows remain. Self-asserts
    point-in-time on `filed` (the SQL gate) and on `available_date`
    before returning."""
    query = f"""
        SELECT x.concept, x.unit, x.period_start, x.period_end, x.value, x.fiscal_year,
               x.fiscal_period, x.form, x.filed_date AS filed, f.accepted_at
        FROM (
            SELECT DISTINCT * EXCLUDE (scraped_at, dt, "table")
            FROM sec_company_facts_log
            WHERE ticker = ? AND filed_date <= ?
        ) x
        LEFT JOIN ({_SEC_ACCEPTANCE_QUERY}) f ON f.accession_number = x.accession_number
        ORDER BY filed
    """
    frame = _query(query, [ticker, as_of_date.isoformat()])
    if frame.empty:
        return None
    frame["filed"] = pd.to_datetime(frame["filed"])
    assert_point_in_time(frame, as_of_date, date_column="filed")
    frame["period_start"] = pd.to_datetime(frame["period_start"])
    frame["period_end"] = pd.to_datetime(frame["period_end"])
    frame["available_date"] = _sec_available_dates(frame["filed"], frame["accepted_at"])
    frame = frame[frame["available_date"] <= pd.Timestamp(as_of_date)].drop(columns="accepted_at")
    if frame.empty:
        return None
    frame = (
        frame.sort_values(["available_date", "filed"], kind="stable")
        .drop_duplicates(subset=_COMPANY_FACT_PERIOD_KEY, keep="last")
        .sort_values(["filed", "period_end"], kind="stable")
        .reset_index(drop=True)
    )
    assert_point_in_time(frame, as_of_date, date_column="available_date")
    return frame


# One acceptance time per filing: `sec_filings` lists a filing once per
# issuer ticker, sometimes with different `accepted_at` values, and the
# latest one is the conservative availability.
_SEC_ACCEPTANCE_QUERY = """
    SELECT accession_number, max(accepted_at) AS accepted_at
    FROM sec_filings
    GROUP BY accession_number
"""

# Decisions are taken at the NYSE close (16:00 America/New_York); an SEC
# filing accepted before that time is public for that day's decision.
_DECISION_TIMEZONE = "America/New_York"
_DECISION_TIME = pd.Timedelta(hours=16)


def _sec_available_dates(filed: pd.Series, accepted_at: pd.Series) -> pd.Series:
    """The day each SEC filing is known at the decision time, from its
    `filed` date and EDGAR `accepted_at` timestamp (time-zone aware, any
    zone, or missing): the filed date when acceptance falls strictly
    before 16:00 America/New_York on that date, otherwise — including a
    missing `accepted_at` — the next NYSE session after the filed date.
    Returns a datetime64 series aligned with `filed`."""
    filed_days = pd.to_datetime(filed).dt.normalize()
    cutoff = (filed_days + _DECISION_TIME).dt.tz_localize(_DECISION_TIMEZONE)
    accepted = pd.to_datetime(accepted_at, utc=True).dt.tz_convert(_DECISION_TIMEZONE)
    before_close = accepted < cutoff
    following = filed_days.map(lambda day: pd.Timestamp(next_session(day.date())))
    return filed_days.where(before_close, following)


# FINRA publishes short interest about seven business days after the
# settlement date, after the close; eight NYSE sessions after settlement
# and then the next session is the conservative day it is known at the
# close.
_FINRA_DISSEMINATION_LAG_SESSIONS = 8
# FINRA short interest settles twice a month (mid-month and month-end).
FINRA_SETTLEMENTS_PER_YEAR = 24
# `finra_short_interest.revision_flag` value marking a row FINRA revised
# after first publication.
_FINRA_REVISED = "R"


def _finra_available_date(settlement: date) -> date:
    """The day a settlement's short interest is known at the close:
    `next_session(add_sessions(settlement, _FINRA_DISSEMINATION_LAG_SESSIONS))`."""
    return next_session(add_sessions(settlement, _FINRA_DISSEMINATION_LAG_SESSIONS))


def load_short_interest_from_lakehouse(ticker: str, as_of_date: date) -> pd.DataFrame | None:
    """FINRA short-interest history for `ticker`, from the
    `finra_short_interest` table (`symbol`, `settlement_date`,
    `revision_flag`, `current_short_position`, `days_to_cover`,
    `average_daily_volume`, ...), as `settlement_date`/
    `dissemination_date`/`short_interest_shares`/`days_to_cover`/
    `avg_daily_volume` rows sorted by settlement date.

    `dissemination_date` holds this loader's own availability estimate,
    `_finra_available_date(settlement_date)`, not the lake's
    `dissemination_date`, which counts weekdays rather than NYSE sessions
    (lake issue LK-47). A revised row (`revision_flag` 'R') carries a
    value published with the symbol's next settlement, so it takes that
    settlement's availability; a revised row with no later settlement is
    dropped, because the lake keeps no pre-revision value. Only rows
    available on or before `as_of_date` are returned; `None` if none are.
    Self-asserts point-in-time on `dissemination_date` before returning."""
    query = """
        SELECT settlement_date, revision_flag,
               current_short_position AS short_interest_shares,
               days_to_cover, average_daily_volume AS avg_daily_volume
        FROM finra_short_interest
        WHERE symbol = ?
        ORDER BY settlement_date
    """
    frame = _query(query, [ticker])
    if frame.empty:
        return None
    frame["settlement_date"] = pd.to_datetime(frame["settlement_date"])
    settlements = sorted({stamp.date() for stamp in frame["settlement_date"]})
    next_settlement: dict[date, date | None] = dict(
        zip(settlements, [*settlements[1:], None], strict=True)
    )
    revised = frame["revision_flag"].eq(_FINRA_REVISED).fillna(False).astype(bool)
    available: list[pd.Timestamp | None] = []
    for stamp, is_revised in zip(frame["settlement_date"], revised, strict=True):
        published_with = next_settlement[stamp.date()] if is_revised else stamp.date()
        available.append(
            None if published_with is None else pd.Timestamp(_finra_available_date(published_with))
        )
    frame["dissemination_date"] = pd.to_datetime(pd.Series(available, index=frame.index))
    frame = frame[frame["dissemination_date"] <= pd.Timestamp(as_of_date)]
    if frame.empty:
        return None
    frame = frame[
        [
            "settlement_date",
            "dissemination_date",
            "short_interest_shares",
            "days_to_cover",
            "avg_daily_volume",
        ]
    ].reset_index(drop=True)
    assert_point_in_time(frame, as_of_date, date_column="dissemination_date")
    return frame


def load_insider_transactions_from_lakehouse(ticker: str, as_of_date: date) -> pd.DataFrame | None:
    """SEC Form 4 insider transactions for `ticker` (`owner_cik`,
    `owner_name`, `is_officer`, `is_director`, `is_ten_percent_owner`,
    `transaction_date`, `transaction_code`, `shares`, `price_per_share`,
    `acquired_disposed`, `filed_date`, `available_date`), sorted by
    filed and transaction date.

    Reads the `sec_insider_transactions_log` view, which keeps every
    scrape, because the deduplicated `sec_insider_transactions` view
    collapses distinct lines of one filing into one row until lake issue
    LK-14 lands. Rows identical apart from their scrape (`scraped_at`,
    `dt`, `table`) count once, so two genuinely identical lines in one
    filing also count once.

    The trade date is not when a trade became public; the filing is.
    `available_date` (datetime64) is set from the filing's latest EDGAR
    acceptance time in `sec_filings` (`_SEC_ACCEPTANCE_QUERY`) by
    `_sec_available_dates` (16:00 America/New_York, the close, is the
    decision time), and only rows with `available_date <= as_of_date`
    are returned. Returns `None` if no rows remain. Self-asserts
    point-in-time on `filed_date` (the SQL gate) and on `available_date`
    before returning."""
    query = f"""
        SELECT t.owner_cik, t.owner_name, t.is_officer, t.is_director, t.is_ten_percent_owner,
               t.transaction_date, t.transaction_code, t.shares, t.price_per_share,
               t.acquired_disposed, t.filed_date, f.accepted_at
        FROM (
            SELECT DISTINCT * EXCLUDE (scraped_at, dt, "table")
            FROM sec_insider_transactions_log
            WHERE issuer_ticker = ? AND filed_date <= ?
        ) t
        LEFT JOIN ({_SEC_ACCEPTANCE_QUERY}) f ON f.accession_number = t.accession_number
        ORDER BY t.filed_date, t.transaction_date
    """
    frame = _query(query, [ticker, as_of_date.isoformat()])
    if frame.empty:
        return None
    frame["filed_date"] = pd.to_datetime(frame["filed_date"])
    frame["transaction_date"] = pd.to_datetime(frame["transaction_date"])
    assert_point_in_time(frame, as_of_date, date_column="filed_date")
    frame["available_date"] = _sec_available_dates(frame["filed_date"], frame["accepted_at"])
    frame = frame[frame["available_date"] <= pd.Timestamp(as_of_date)].drop(columns="accepted_at")
    if frame.empty:
        return None
    frame = frame.reset_index(drop=True)
    assert_point_in_time(frame, as_of_date, date_column="available_date")
    return frame


# Two chain rows quote the same strike when they differ by less than half a
# cent: a split rescale leaves a strike up to a third of a cent from its
# listed post-split value (625 / 3 = 208.333... against a listed 208.33),
# and listed strikes are at least 0.5 apart, so no two distinct strikes
# fall within this tolerance.
STRIKE_MATCH_TOLERANCE = 0.005


def load_option_quote_from_lakehouse(
    underlying: str,
    expiration: date,
    option_type: str,
    strike: float,
    as_of_date: date,
    lookback_days: int,
) -> pd.DataFrame | None:
    """The latest chain row quoting one contract on or before
    `as_of_date` within `lookback_days`, regardless of whether the latest
    snapshot still lists that expiry. Same source table, columns and
    prices as `load_options_snapshot_from_lakehouse` (the table
    `_option_chain_source` picks for `underlying` and `as_of_date`),
    including the `_apply_split_adjustment` rescaling. A row matches on
    `expiration` and `option_type` exactly and on a strike within
    `STRIKE_MATCH_TOLERANCE` of `strike`; the strike is compared in the
    adjusted basis rather than in SQL against the raw column, since the
    two differ for a contract quoted before a later split. Of several
    matching rows the one with the latest `date` is returned. `None`
    when never quoted in the window. Self-asserts point-in-time on
    `date` before returning."""
    window_start = (as_of_date - pd.Timedelta(days=lookback_days)).isoformat()
    table, date_column = _option_chain_source(underlying, as_of_date)
    query = f"""
        SELECT {date_column} AS date, expiration, option_type, strike, bid, ask,
               implied_volatility, delta, gamma, theta, vega, rho
        FROM {table}
        WHERE underlying = ?
          AND expiration = ?
          AND option_type = ?
          AND {date_column} BETWEEN ? AND ?
        ORDER BY {date_column} DESC
    """
    frame = _query(
        query,
        [underlying, expiration.isoformat(), option_type, window_start, as_of_date.isoformat()],
    )
    if frame.empty:
        return None
    frame["date"] = pd.to_datetime(frame["date"])
    frame["expiration"] = pd.to_datetime(frame["expiration"])
    frame = _apply_split_adjustment(frame, underlying)
    frame = frame[(frame["strike"] - strike).abs() < STRIKE_MATCH_TOLERANCE]
    if frame.empty:
        return None
    frame = frame.iloc[[0]]
    assert_point_in_time(frame, as_of_date, date_column="date")
    return frame


CoverageSource = Literal["options_chain", "cftc_cot", "earnings", "dividends", "sp500_membership"]


def _first_and_last(frame: pd.DataFrame) -> tuple[date, date] | None:
    """The `first`/`last` dates of a one-row `MIN`/`MAX` result, or `None`
    when the aggregate matched no rows (both values null)."""
    if frame.empty or pd.isna(frame.iloc[0]["first"]) or pd.isna(frame.iloc[0]["last"]):
        return None
    return pd.Timestamp(frame.iloc[0]["first"]).date(), pd.Timestamp(frame.iloc[0]["last"]).date()


def source_coverage(source: CoverageSource, identifier: str | None) -> tuple[date, date] | None:
    """The first and last date the lake holds for `source`, read from the
    tables the matching loader reads, with no point-in-time gate.
    `identifier` is a ticker for the per-ticker sources and `None` for a
    check over the whole table:

    - `options_chain`: snapshot `date`s in `dolthub_option_chain` for
      underlying `identifier`; for an underlying in
      `_OPTION_CHAIN_HISTORY_UNDERLYINGS`, also the `option_chain_history`
      `snapshot_date`s before its first dolthub date (all of them when it
      has no dolthub rows), the two tables' ranges joined as the option
      loaders read them.
    - `cftc_cot`: `cftc_cot_publication.published_date`s of the rows
      with positive open interest and a non-null `published_date` for the
      COT ticker of `identifier` (its `DEFAULT_COT_TICKER_BY_ASSET`
      future, else itself), from whichever of `cftc_cot_disaggregated` /
      `cftc_cot_tff` holds that ticker's contract-market code, joined on
      `report_date`; `None` when it has no code.
    - `earnings`: announcement `date`s in `dolthub_earnings_calendar`,
      over every symbol when `identifier` is `None`, else for that symbol.
    - `dividends`: ex-dates of regular cash distributions in
      `equity_distributions`, over every ticker when `identifier` is
      `None`, else for that ticker.
    - `sp500_membership`: the earliest `start_date` and the latest
      `end_date` in `sp500_membership`, an open membership span counting
      as today; `identifier` is not used.

    Returns `None` when the lake holds no matching rows."""
    if source == "options_chain":
        query = """
            SELECT MIN(date) AS first, MAX(date) AS last
            FROM dolthub_option_chain
            WHERE underlying = ?
        """
        dolthub = _first_and_last(_query(query, [identifier]))
        if identifier not in _OPTION_CHAIN_HISTORY_UNDERLYINGS:
            return dolthub
        history_query = """
            SELECT MIN(snapshot_date) AS first, MAX(snapshot_date) AS last
            FROM option_chain_history
            WHERE underlying = ?
        """
        if dolthub is None:
            return _first_and_last(_query(history_query, [identifier]))
        history = _first_and_last(
            _query(history_query + " AND snapshot_date < ?", [identifier, dolthub[0].isoformat()])
        )
        return dolthub if history is None else (history[0], dolthub[1])
    if source == "cftc_cot":
        cot_ticker = DEFAULT_COT_TICKER_BY_ASSET.get(identifier or "", identifier or "")
        for table, codes in (
            ("cftc_cot_disaggregated", _COT_DISAGGREGATED_CODES),
            ("cftc_cot_tff", _COT_TFF_CODES),
        ):
            code = codes.get(cot_ticker)
            if code is None:
                continue
            query = f"""
                SELECT MIN(pub.published_date) AS first, MAX(pub.published_date) AS last
                FROM {table} AS cot
                JOIN cftc_cot_publication AS pub ON pub.report_date = cot.report_date
                WHERE cot.cftc_contract_market_code = ?
                  AND cot.open_interest_all > 0
                  AND pub.published_date IS NOT NULL
            """
            return _first_and_last(_query(query, [code]))
        return None
    if source == "earnings":
        query = """
            SELECT MIN(date) AS first, MAX(date) AS last
            FROM dolthub_earnings_calendar
        """
        if identifier is None:
            return _first_and_last(_query(query, []))
        return _first_and_last(_query(query + " WHERE act_symbol = ?", [identifier]))
    if source == "dividends":
        query = """
            SELECT MIN(ex_date) AS first, MAX(ex_date) AS last
            FROM equity_distributions
            WHERE distribution_type = ?
        """
        if identifier is None:
            return _first_and_last(_query(query, [_REGULAR_CASH]))
        return _first_and_last(_query(query + " AND ticker = ?", [_REGULAR_CASH, identifier]))
    if source == "sp500_membership":
        query = """
            SELECT MIN(start_date) AS first,
                   MAX(COALESCE(end_date, CAST(? AS DATE))) AS last
            FROM sp500_membership
        """
        return _first_and_last(_query(query, [date.today().isoformat()]))
    raise ValueError(f"no coverage query for source {source!r}")
