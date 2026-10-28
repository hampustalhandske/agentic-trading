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
from typing import Any, cast

import numpy as np
import pandas as pd
from pydantic import BaseModel

from agentic_trading.config import _env_value
from agentic_trading.data.lakehouse_client import Lake
from agentic_trading.data.point_in_time import assert_point_in_time
from agentic_trading.data.trading_calendar import add_sessions, next_session
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
    """Real, point-in-time S&P 500 membership as of `as_of_date` — every
    ticker whose `sp500_membership` span covers this date, including
    names since removed (a name that left the index yesterday was still
    a genuine member today's-relative-yesterday, so it belongs in a
    point-in-time universe for that date; excluding it would be the
    lookahead-shaped mistake of using tomorrow's knowledge to prune
    today's universe). Returns `None` if the table has no matching rows —
    callers should fall back to a smaller static universe, not treat this
    as an empty-but-valid membership list.
    Calls `assert_point_in_time` on the result for consistency with every
    other loader in this module, though it's a no-op here since this
    returns tickers, not dates."""
    query = """
        SELECT DISTINCT ticker
        FROM sp500_membership
        WHERE start_date <= ?
          AND (end_date IS NULL OR end_date >= ?)
    """
    result = _query(query, [as_of_date.isoformat(), as_of_date.isoformat()])
    if result.empty:
        return None
    tickers = sorted(result["ticker"].tolist())
    assert_point_in_time(tickers, as_of_date)
    return tickers


def _split_adjustment_factor(ticker: str, quote_date: date) -> float:
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
        factor = _split_adjustment_factor(ticker, quote_date)
        if factor == 1.0:
            continue
        mask = frame["date"].dt.date == quote_date
        frame.loc[mask, ["strike", "bid", "ask"]] /= factor
        frame.loc[mask, "gamma"] /= factor**2
        frame.loc[mask, ["vega", "theta", "rho"]] /= factor
    return frame


def load_options_snapshot_from_lakehouse(
    ticker: str, as_of_date: date, lookback_days: int = 10
) -> pd.DataFrame | None:
    """Real historical options-chain rows (strike, expiration, bid, ask,
    implied_volatility, delta, gamma, theta, vega, rho — no open interest,
    no volume, confirmed absent from the source) for `ticker`, from the
    most recent available snapshot date on or before `as_of_date` within
    `lookback_days` — coverage is real but not daily-dense (roughly
    every-other-trading-day for AAPL), so a same-day match isn't
    guaranteed even for a covered ticker.

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
    query = """
        SELECT date, expiration, option_type, strike, bid, ask,
               implied_volatility, delta, gamma, theta, vega, rho
        FROM dolthub_option_chain
        WHERE underlying = ?
          AND date BETWEEN ? AND ?
          AND date = (
              SELECT MAX(date)
              FROM dolthub_option_chain
              WHERE underlying = ?
                AND date BETWEEN ? AND ?
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


def load_cot_positioning_from_lakehouse(
    ticker: str, as_of_date: date, lookback_days: int = 400
) -> pd.Series | None:
    """Net speculative positioning as a fraction of open interest, from
    CFTC Commitment of Traders reports — `managed_money_long -
    managed_money_short` (the disaggregated report's commodity category)
    for commodities, `leveraged_funds_long - leveraged_funds_short` (the
    TFF report's closest equivalent) for FX/rates, divided by
    `open_interest_all`. A given `ticker` only ever appears in one of the
    two tables (commodities in `cftc_cot_disaggregated`, FX/rate-futures
    in `cftc_cot_tff`), selected by its CFTC contract-market code.

    Point-in-time: COT reports have a real publication lag — the
    Tuesday-dated `report_date` isn't released until the following
    Friday. Gates on `report_date + 3 days` as the estimated
    availability date.

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
            SELECT report_date, {net_position_expr} / open_interest_all AS net_positioning_ratio
            FROM {table}
            WHERE cftc_contract_market_code = ?
              AND report_date BETWEEN ? AND ?
              AND date_add(CAST(report_date AS DATE), INTERVAL 3 DAY) <= ?
              AND open_interest_all > 0
            ORDER BY report_date
        """
        frame = _query(query, [code, window_start, availability_cutoff, availability_cutoff])
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
    """Real SIC (Standard Industrial Classification) code for `ticker`
    from SEC EDGAR, via `sec_sic_classification` — free, no GICS license
    restriction. Not time-varying in this lake (one row per ticker, not
    a point-in-time span), so no `as_of_date` gating; returns `None` if
    `ticker` has no SIC record (mostly non-equity ETF/future/FX tickers,
    which have no SEC filing to classify in the first place)."""
    frame = _query("SELECT sic FROM sec_sic_classification WHERE ticker = ?", [ticker])
    if frame.empty or pd.isna(frame.iloc[0]["sic"]):
        return None
    return str(frame.iloc[0]["sic"])


def same_sic_candidate_pairs(tickers: list[str]) -> list[tuple[str, str]]:
    """Every pair within `tickers` that share a real SEC SIC code — a
    genuine industry-classification candidate list for pairs-cointegration
    screening, replacing a hand-picked "looks same-sector to me" list.
    Tickers with no SIC record (non-equities, or an equity outside SEC
    EDGAR's coverage) simply can't form a pair here — this only ever
    under-generates candidates, never fabricates one for an
    unclassified ticker."""
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


def load_price_history_from_lakehouse(
    tickers: list[str], start: date, end: date
) -> pd.DataFrame | None:
    """Returns a `MultiIndex(ticker, field)` frame (`open`/`close`/
    `raw_close`/`volume`/`log_return`) for every ticker in `tickers` that the lake actually
    covers, or `None` if the lake covers none of them —
    `load_price_history_with_fallback` is what turns an incomplete result
    into a clear error naming the uncovered ticker(s). A failed request
    propagates.

    Uses `adj_close` (dividend/split-adjusted) rather than raw `close` —
    correct for a multi-year backtest where corporate actions actually
    occur — and scales the raw `open` by the same `adj_close / close`
    factor so the open the engine fills at and the close it marks at are
    on one price basis (an unadjusted open against an adjusted close would
    book the whole dividend-adjustment gap as a loss on every fill).
    Self-asserts point-in-time on the combined frame's date index against
    `end` before returning.
    """
    placeholders = ", ".join("?" for _ in tickers)
    query = f"""
        SELECT ticker, date,
               CASE WHEN close > 0 THEN open * adj_close / close ELSE NULL END AS open,
               adj_close AS close,
               close AS raw_close,
               volume
        FROM instrument_prices
        WHERE ticker IN ({placeholders})
          AND date BETWEEN ? AND ?
        ORDER BY ticker, date
    """
    frame = _query(query, [*tickers, start.isoformat(), end.isoformat()])
    if frame.empty:
        return None

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


def load_option_quote_from_lakehouse(
    underlying: str,
    expiration: date,
    option_type: str,
    strike: float,
    as_of_date: date,
    lookback_days: int,
) -> pd.DataFrame | None:
    """The latest chain row quoting one exact contract on or before
    `as_of_date` within `lookback_days`, regardless of whether the latest
    snapshot still lists that expiry. Same columns as
    `load_options_snapshot_from_lakehouse`, including the
    `_apply_split_adjustment` rescaling. `strike` is matched in that same
    adjusted basis rather than in SQL against the raw column, since the
    two can differ for a contract quoted before a later split — the
    strike filter is applied in pandas after adjustment instead. `None`
    when never quoted in the window. Self-asserts point-in-time on
    `date` before returning."""
    window_start = (as_of_date - pd.Timedelta(days=lookback_days)).isoformat()
    query = """
        SELECT date, expiration, option_type, strike, bid, ask,
               implied_volatility, delta, gamma, theta, vega, rho
        FROM dolthub_option_chain
        WHERE underlying = ?
          AND expiration = ?
          AND option_type = ?
          AND date BETWEEN ? AND ?
        ORDER BY date DESC
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
    frame = frame[(frame["strike"] - strike).abs() < 1e-6]
    if frame.empty:
        return None
    frame = frame.iloc[[0]]
    assert_point_in_time(frame, as_of_date, date_column="date")
    return frame
