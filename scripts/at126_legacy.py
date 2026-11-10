# at126_legacy.py

"""The per-call loaders as they stood before the window store (dev at
c6b479a), kept as the reference `at126_ab.py` compares the new paths
against: every function here sends its own query for each call, with
the SQL, post-query steps and return shapes of that commit. They call the
loader module's shared internals (`_table_sql`, `_query`, the guards and
availability helpers), which the window store does not change. The S&P
500 member reference does not log the unresolved-member warning; the bar
references run the split-move check as the loaders do, so the split-move
registries the paths leave can be compared. Delete
this module with `at126_ab.py` once the A/B has passed on the live lake.

Each `legacy_<loader>` takes the arguments of the public loader of the
same name (without `load_` and `_from_lakehouse`) and returns what that
loader returned at c6b479a, with these exceptions, which match the
loaders: in the security-bar references (`_legacy_adjusted_security_bars`)
a bar's raw price is this module's `SECURITY_RAW_PRICE_SQL` and its
`volume` is divided by the bar's `cum_factor`; the SEC references
(`legacy_company_facts`, `legacy_insider_transactions`) set
`available_date` with the loader module's shared `_sec_available_dates`
(the next NYSE session after the later of the filed date and the New
York date of the EDGAR acceptance), and `legacy_company_facts` keeps,
per period, the latest filed row known by the decision day (ties: the
latest `available_date`, then the latest acceptance); and inside a lake
session every reference reads a current table through the loader
module's `_table_sql`, whose dedupe takes, among one key's rows written at
its latest `scraped_at`, the first in the tie order the lake serves for
the table (`_tie_order_sql`), and refuses a current table the lake serves
no tie order for (`MissingTieOrderError`)."""

from datetime import date, timedelta

import numpy as np
import pandas as pd

from agentic_trading.data.instruments import futures_root
from agentic_trading.data.loaders import lakehouse_loader as ll
from agentic_trading.data.point_in_time import assert_point_in_time
from agentic_trading.data.trading_calendar import is_session, next_session


def legacy_sp500_member_securities_asof(as_of_date: date) -> list[ll.IndexMember] | None:
    ll._guard_data_end(as_of_date, "load_sp500_member_securities_asof")
    query = f"""
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
            FROM {ll._table_sql("sp500_price_coverage")}
            WHERE piece_start <= ?
              AND (piece_end IS NULL OR piece_end > ?)
        ) AS ranked
        WHERE security_rank = 1
        ORDER BY ticker, security_id
    """
    as_of = as_of_date.isoformat()
    result = ll._query(query, [as_of, as_of])
    if result.empty:
        return None
    unresolved = result["security_id"].isna()
    resolved = result.loc[~unresolved]
    members = [
        ll.IndexMember(security_id=int(security_id), ticker=str(ticker))
        for security_id, ticker in zip(resolved["security_id"], resolved["ticker"], strict=True)
    ]
    return members or None


# A bar's raw price: the exchange's official close where the lake keeps
# one, else the close of the session's last bar, the basis the grader
# (edgelab) settles on; the same expression as
# `lakehouse_loader.SECURITY_RAW_PRICE_SQL`, held here as its own copy.
SECURITY_RAW_PRICE_SQL = "COALESCE(official_close, close)"


def _legacy_adjusted_security_bars(
    security_ids: list[int], start: date, end: date, with_open_and_volume: bool
) -> pd.DataFrame:
    ll.adjustment_table_row_count("equity_adjustment_factors")
    placeholders = ", ".join("?" for _ in security_ids)
    bar_columns = ", open, volume" if with_open_and_volume else ""
    open_and_volume = (
        """
               CASE WHEN bars.open > 0
                    THEN bars.open * COALESCE(factors.cum_factor, 1.0) END AS open,
               CASE WHEN bars.volume > 0
                    THEN bars.volume / COALESCE(factors.cum_factor, 1.0) END AS volume,"""
        if with_open_and_volume
        else ""
    )
    query = f"""
        SELECT bars.security_id, bars.date,
               bars.raw_close * COALESCE(factors.cum_factor, 1.0) AS close,
               bars.raw_close,{open_and_volume}
               factors.effective_date AS factor_date
        FROM (
            SELECT security_id, date, {SECURITY_RAW_PRICE_SQL} AS raw_close{bar_columns}
            FROM {ll._table_sql("equity_daily_from_bars")}
            WHERE security_id IN ({placeholders})
              AND date BETWEEN ? AND ?
              AND {SECURITY_RAW_PRICE_SQL} > 0
        ) AS bars
        ASOF LEFT JOIN (
            SELECT security_id, effective_date, cum_factor
            FROM {ll._table_sql("equity_adjustment_factors")}
            WHERE security_id IN ({placeholders})
              AND effective_date > ?
        ) AS factors
          ON factors.security_id = bars.security_id
         AND bars.date < factors.effective_date
        ORDER BY bars.security_id, bars.date
    """
    ids = [int(security_id) for security_id in security_ids]
    params = [*ids, start.isoformat(), end.isoformat(), *ids, start.isoformat()]
    frame = ll._query(query, params)
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.drop_duplicates(["security_id", "date"]).reset_index(drop=True)
    previous_date = frame.groupby("security_id")["date"].shift(1)
    session_after = {
        stamp: pd.Timestamp(next_session(stamp.date()))
        for stamp in pd.DatetimeIndex(previous_date.dropna().unique())
    }
    consecutive = previous_date.map(session_after) == frame["date"]
    log_price = pd.Series(np.log(frame["close"].to_numpy(dtype=np.float64)), index=frame.index)
    log_return = log_price - log_price.groupby(frame["security_id"]).shift(1)
    frame["log_return"] = log_return.where(consecutive)
    ll._warn_unfactored_split_moves(ll._log_return_series(frame), frame)
    return frame


def legacy_security_log_returns(
    security_ids: list[int], start: date, end: date
) -> dict[int, pd.Series]:
    ll._guard_data_end(end, "load_security_log_returns")
    if not security_ids:
        return {}
    frame = _legacy_adjusted_security_bars(security_ids, start, end, with_open_and_volume=False)
    if frame.empty:
        return {}
    result = ll._log_return_series(frame)
    for series in result.values():
        assert_point_in_time(series, end)
    return result


def legacy_security_daily_bars(
    security_ids: list[int], start: date, end: date
) -> pd.DataFrame | None:
    ll._guard_data_end(end, "load_security_daily_bars")
    if not security_ids:
        return None
    frame = _legacy_adjusted_security_bars(security_ids, start, end, with_open_and_volume=True)
    if frame.empty:
        return None
    per_security = {
        int(security_id): group.set_index("date")[list(ll._DAILY_BAR_FIELDS)].astype(np.float64)
        for security_id, group in frame.groupby("security_id")
    }
    result = pd.concat(per_security, axis=1, names=["security_id", "field"], sort=True)
    result.index = pd.DatetimeIndex(result.index, name="date")
    assert_point_in_time(result, end)
    return result


def legacy_split_adjustment_factor(ticker: str, quote_date: date) -> float:
    ll.adjustment_table_row_count("yahoo_split_close_basis")
    query = f"""
        SELECT ratio
        FROM {ll._table_sql("yahoo_split_close_basis")}
        WHERE ticker = ?
          AND ex_date > ?
          AND in_close_basis IS NOT FALSE
    """
    frame = ll._query(query, [ticker, quote_date.isoformat()])
    factor = 1.0
    for ratio in frame["ratio"]:
        if ratio and ratio > 0:
            factor /= float(ratio)
    return factor


def _legacy_apply_split_adjustment(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    frame = frame.copy()
    for quote_date in frame["date"].dt.date.unique():
        factor = legacy_split_adjustment_factor(ticker, quote_date)
        if factor == 1.0:
            continue
        mask = frame["date"].dt.date == quote_date
        frame.loc[mask, ["strike", "bid", "ask"]] /= factor
        frame.loc[mask, "gamma"] /= factor**2
        frame.loc[mask, ["vega", "theta", "rho"]] /= factor
    return frame


def legacy_options_snapshot(
    ticker: str,
    as_of_date: date,
    lookback_days: int = ll.DEFAULT_OPTIONS_SNAPSHOT_LOOKBACK_DAYS,
) -> pd.DataFrame | None:
    ll._guard_data_end(as_of_date, "load_options_snapshot_from_lakehouse")
    window_start = (as_of_date - pd.Timedelta(days=lookback_days)).isoformat()
    window_end = as_of_date.isoformat()
    table, date_column = ll._option_chain_source(ticker, as_of_date)
    query = f"""
        SELECT {date_column} AS date, expiration, option_type, strike, bid, ask,
               implied_volatility, delta, gamma, theta, vega, rho
        FROM {ll._table_sql(table)}
        WHERE underlying = ?
          AND {date_column} BETWEEN ? AND ?
          AND {date_column} = (
              SELECT MAX({date_column})
              FROM {ll._table_sql(table)}
              WHERE underlying = ?
                AND {date_column} BETWEEN ? AND ?
          )
        ORDER BY expiration, option_type, strike, bid, ask, implied_volatility, delta, gamma,
                 theta, vega, rho
    """
    frame = ll._query(query, [ticker, window_start, window_end, ticker, window_start, window_end])
    if frame.empty:
        return None
    frame["date"] = pd.to_datetime(frame["date"])
    frame["expiration"] = pd.to_datetime(frame["expiration"])
    frame = _legacy_apply_split_adjustment(frame, ticker)
    assert_point_in_time(frame, as_of_date, date_column="date")
    return frame


def legacy_option_quote(
    underlying: str,
    expiration: date,
    option_type: str,
    strike: float,
    as_of_date: date,
    lookback_days: int,
) -> pd.DataFrame | None:
    ll._guard_data_end(as_of_date, "load_option_quote_from_lakehouse")
    window_start = (as_of_date - pd.Timedelta(days=lookback_days)).isoformat()
    table, date_column = ll._option_chain_source(underlying, as_of_date)
    query = f"""
        SELECT {date_column} AS date, expiration, option_type, strike, bid, ask,
               implied_volatility, delta, gamma, theta, vega, rho
        FROM {ll._table_sql(table)}
        WHERE underlying = ?
          AND expiration = ?
          AND option_type = ?
          AND {date_column} BETWEEN ? AND ?
        ORDER BY {date_column} DESC
    """
    frame = ll._query(
        query,
        [underlying, expiration.isoformat(), option_type, window_start, as_of_date.isoformat()],
    )
    if frame.empty:
        return None
    frame["date"] = pd.to_datetime(frame["date"])
    frame["expiration"] = pd.to_datetime(frame["expiration"])
    frame = _legacy_apply_split_adjustment(frame, underlying)
    frame = frame[(frame["strike"] - strike).abs() < ll.STRIKE_MATCH_TOLERANCE]
    if frame.empty:
        return None
    frame = frame.iloc[[0]]
    assert_point_in_time(frame, as_of_date, date_column="date")
    return frame


def legacy_cot_reports(
    ticker: str, as_of_date: date, lookback_days: int = ll.DEFAULT_COT_LOOKBACK_DAYS
) -> pd.DataFrame | None:
    ll._guard_data_end(as_of_date, "load_cot_reports_from_lakehouse")
    window_start = (as_of_date - pd.Timedelta(days=lookback_days)).isoformat()
    availability_cutoff = as_of_date.isoformat()
    queries = [
        (
            "cftc_cot_disaggregated",
            ll._COT_DISAGGREGATED_CODES,
            "(managed_money_long - managed_money_short)",
        ),
        ("cftc_cot_tff", ll._COT_TFF_CODES, "(leveraged_funds_long - leveraged_funds_short)"),
    ]
    for table, codes, net_position_expr in queries:
        code = codes.get(ticker)
        if code is None:
            continue
        ll._require_rows("cftc_cot_publication")
        query = f"""
            SELECT cot.report_date,
                   {net_position_expr} / cot.open_interest_all AS net_positioning_ratio,
                   pub.published_date
            FROM {ll._table_sql(table)} AS cot
            JOIN {ll._table_sql("cftc_cot_publication")} AS pub
              ON pub.report_date = cot.report_date
            WHERE cot.cftc_contract_market_code = ?
              AND cot.report_date BETWEEN ? AND ?
              AND cot.open_interest_all > 0
              AND (
                  (pub.published_at IS NOT NULL AND pub.published_date <= ?)
                  OR (pub.published_at IS NULL AND pub.published_date < ?)
              )
            ORDER BY cot.report_date
        """
        frame = ll._query(
            query,
            [code, window_start, availability_cutoff, availability_cutoff, availability_cutoff],
        )
        if not frame.empty:
            frame["report_date"] = pd.to_datetime(frame["report_date"])
            frame["published_date"] = pd.to_datetime(frame["published_date"])
            reports = frame.set_index("report_date")[["net_positioning_ratio", "published_date"]]
            assert_point_in_time(reports, as_of_date)
            return reports
    return None


def legacy_cot_positioning(
    ticker: str, as_of_date: date, lookback_days: int = ll.DEFAULT_COT_LOOKBACK_DAYS
) -> pd.Series | None:
    reports = legacy_cot_reports(ticker, as_of_date, lookback_days)
    return None if reports is None else reports["net_positioning_ratio"]


def legacy_dividends(ticker: str, as_of_date: date) -> pd.Series | None:
    ll._guard_data_end(as_of_date, "load_dividends_from_lakehouse")
    query = f"""
        SELECT ex_date, amount_split_adjusted
        FROM {ll._table_sql("equity_distributions")}
        WHERE ticker = ? AND distribution_type = ? AND ex_date <= ?
        ORDER BY ex_date
    """
    frame = ll._query(query, [ticker, ll._REGULAR_CASH, as_of_date.isoformat()])
    if frame.empty:
        return None
    frame["ex_date"] = pd.to_datetime(frame["ex_date"])
    series = frame.set_index("ex_date")["amount_split_adjusted"]
    assert_point_in_time(series, as_of_date)
    return series


def legacy_earnings_surprise(
    ticker: str, as_of_date: date
) -> tuple[float, float, pd.Timestamp] | None:
    ll._guard_data_end(as_of_date, "load_earnings_surprise_from_lakehouse")
    query = f"""
        SELECT h.reported, h.estimate, c.date AS announcement_date
        FROM (
            SELECT *
            FROM {ll._table_sql("dolthub_eps_history")}
            WHERE act_symbol = ? AND reported IS NOT NULL AND estimate IS NOT NULL
        ) h
        JOIN (
            SELECT *
            FROM {ll._table_sql("dolthub_earnings_calendar")}
            WHERE act_symbol = ?
        ) c
        ON c.act_symbol = h.act_symbol
           AND CAST(c.date AS DATE) BETWEEN CAST(h.period_end_date AS DATE)
               AND date_add(CAST(h.period_end_date AS DATE), INTERVAL 75 DAY)
        WHERE CAST(c.date AS DATE) < ?
        ORDER BY c.date DESC
    """
    frame = ll._query(query, [ticker, ticker, as_of_date.isoformat()])
    if frame.empty:
        return None
    frame["announcement_date"] = pd.to_datetime(frame["announcement_date"])
    frame["available_date"] = [
        pd.Timestamp(next_session(announced.date())) for announced in frame["announcement_date"]
    ]
    frame = frame[frame["available_date"] <= pd.Timestamp(as_of_date)]
    if frame.empty:
        return None
    assert_point_in_time(frame, as_of_date, date_column="available_date")
    row = frame.iloc[0]
    return float(row["reported"]), float(row["estimate"]), pd.Timestamp(row["announcement_date"])


def legacy_sic_code(ticker: str) -> str | None:
    frame = ll._query(
        f"SELECT sic FROM {ll._table_sql('sec_sic_classification')} WHERE ticker = ?", [ticker]
    )
    if frame.empty or pd.isna(frame.iloc[0]["sic"]):
        return None
    return str(frame.iloc[0]["sic"])


def _legacy_equity_price_rows(tickers: list[str], start: date, end: date) -> pd.DataFrame:
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
        FROM {ll._table_sql("instrument_prices")}
        WHERE ticker IN ({placeholders})
          AND date BETWEEN ? AND ?
        ORDER BY ticker, date
    """
    frame = ll._query(query, [*tickers, start.isoformat(), end.isoformat()])
    raw_open = frame["raw_open"].astype(float)
    raw_high = frame["raw_high"].astype(float)
    raw_low = frame["raw_low"].astype(float)
    inconsistent = (raw_open < raw_low) | (raw_open > raw_high) | (raw_low > raw_high)
    if inconsistent.any():
        frame.loc[inconsistent, "open"] = np.nan
    return frame[["ticker", "date", "open", "close", "raw_close", "volume"]]


def _legacy_crypto_price_rows(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    symbols = [ll._CRYPTO_KLINE_SYMBOLS[ticker] for ticker in tickers]
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
            FROM {ll._table_sql("crypto_klines_1m")}
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
        ll._SESSION_FIRST_MINUTE,
        ll._SESSION_LAST_MINUTE,
        ll._DECISION_TIMEZONE,
        *symbols,
        start.isoformat(),
        (end + pd.Timedelta(days=1)).isoformat(),
        ll._SESSION_FIRST_MINUTE,
        ll._SESSION_LAST_MINUTE,
        ll._SESSION_LAST_MINUTE,
    ]
    frame = ll._query(query, params)
    if frame.empty:
        return frame
    ticker_by_symbol = {symbol: ticker for ticker, symbol in ll._CRYPTO_KLINE_SYMBOLS.items()}
    frame["ticker"] = frame["symbol"].map(ticker_by_symbol)
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame[[is_session(stamp.date()) for stamp in frame["date"]]]
    frame["raw_close"] = frame["close"]
    return frame[["ticker", "date", "open", "close", "raw_close", "volume"]]


def legacy_price_history(tickers: list[str], start: date, end: date) -> pd.DataFrame | None:
    ll._guard_data_end(end, "load_price_history_from_lakehouse")
    crypto = [ticker for ticker in tickers if ticker in ll._CRYPTO_KLINE_SYMBOLS]
    others = [ticker for ticker in tickers if ticker not in ll._CRYPTO_KLINE_SYMBOLS]
    parts = []
    if others:
        parts.append(_legacy_equity_price_rows(others, start, end))
    if crypto:
        parts.append(_legacy_crypto_price_rows(crypto, start, end))
    parts = [part for part in parts if not part.empty]
    if not parts:
        return None
    frame = pd.concat(parts, ignore_index=True)
    frame["date"] = pd.to_datetime(frame["date"])
    per_ticker: dict[str, pd.DataFrame] = {}
    for ticker, group in frame.groupby("ticker"):
        indexed = (
            group.set_index("date")[["open", "close", "raw_close", "volume"]].sort_index().copy()
        )
        indexed.loc[indexed["close"] <= 0.0, "close"] = np.nan
        indexed.loc[indexed["open"] <= 0.0, "open"] = np.nan
        indexed.loc[indexed["raw_close"] <= 0.0, "raw_close"] = np.nan
        indexed["volume"] = indexed["volume"].astype(float)
        indexed.loc[indexed["volume"] <= 0.0, "volume"] = np.nan
        indexed["log_return"] = np.log(indexed["close"] / indexed["close"].shift(1))
        per_ticker[str(ticker)] = indexed
    combined = pd.concat(per_ticker, axis=1)
    assert_point_in_time(combined, end)
    return combined


def legacy_raw_close(ticker: str, as_of_date: date, lookback_days: int) -> float | None:
    ll._guard_data_end(as_of_date, "load_raw_close_from_lakehouse")
    window_start = (as_of_date - pd.Timedelta(days=lookback_days)).isoformat()
    query = f"""
        SELECT date, close
        FROM {ll._table_sql("instrument_prices")}
        WHERE ticker = ?
          AND date BETWEEN ? AND ?
          AND close > 0
        ORDER BY date DESC
        LIMIT 1
    """
    frame = ll._query(query, [ticker, window_start, as_of_date.isoformat()])
    if frame.empty:
        return None
    assert_point_in_time(frame, as_of_date, date_column="date")
    return float(frame["close"].iloc[0])


def legacy_fred_vintages(series_id: str, as_of_date: date) -> pd.DataFrame | None:
    ll._guard_data_end(as_of_date, "load_fred_vintages_from_lakehouse")
    query = f"""
        SELECT date, value, realtime_start, known_at
        FROM (
            SELECT date, value, realtime_start, known_at, ROW_NUMBER() OVER (
                PARTITION BY date
                ORDER BY realtime_start DESC
            ) AS rn
            FROM (
                SELECT date, value, realtime_start,
                    GREATEST(realtime_start, date) AS known_at
                FROM {ll._table_sql("fred_vintages")}
                WHERE series_id = ?
            )
            WHERE known_at < ?
        )
        WHERE rn = 1
        ORDER BY date
    """
    frame = ll._query(query, [series_id, as_of_date.isoformat()])
    if frame.empty:
        return None
    frame["date"] = pd.to_datetime(frame["date"])
    frame["realtime_start"] = pd.to_datetime(frame["realtime_start"])
    frame["known_at"] = pd.to_datetime(frame["known_at"])
    vintages = frame.set_index("date")[["value", "realtime_start", "known_at"]]
    assert_point_in_time(vintages, as_of_date)
    return vintages


def legacy_fred_series(series_id: str, as_of_date: date) -> pd.Series | None:
    vintages = legacy_fred_vintages(series_id, as_of_date)
    return None if vintages is None else vintages["value"]


def legacy_fama_french_factors(as_of_date: date, model: str = "3factor") -> pd.DataFrame | None:
    ll._guard_data_end(as_of_date, "load_fama_french_factors_from_lakehouse")
    dataset = ll._KEN_FRENCH_DATASET_BY_MODEL.get(model)
    if dataset is None:
        return None
    factors = ll._KEN_FRENCH_FACTORS_BY_MODEL[model]
    placeholders = ", ".join("?" for _ in factors)
    query = f"""
        SELECT date, portfolio, value
        FROM {ll._table_sql("famafrench_returns")}
        WHERE dataset = ? AND frequency = ?
          AND portfolio IN ({placeholders})
          AND last_day(
                date_add(CAST(date AS DATE), INTERVAL {ll._KEN_FRENCH_PUBLICATION_LAG_MONTHS} MONTH)
              ) <= ?
        ORDER BY date
    """
    long = ll._query(query, [dataset, ll._KEN_FRENCH_FREQUENCY, *factors, as_of_date.isoformat()])
    if long.empty:
        return None
    long["date"] = pd.to_datetime(long["date"])
    wide = long.pivot_table(index="date", columns="portfolio", values="value", aggfunc="max")
    frame = wide[[factor for factor in factors if factor in wide.columns]]
    result = (frame.astype(float) / 100.0).sort_index()
    assert_point_in_time(result, as_of_date)
    return result


def legacy_company_facts(ticker: str, as_of_date: date) -> pd.DataFrame | None:
    ll._guard_data_end(as_of_date, "load_company_facts_from_lakehouse")
    query = f"""
        SELECT x.concept, x.unit, x.period_start, x.period_end, x.value, x.fiscal_year,
               x.fiscal_period, x.form, x.filed_date AS filed, f.accepted_at
        FROM (
            SELECT DISTINCT * EXCLUDE (scraped_at, dt, "table")
            FROM {ll._table_sql("sec_company_facts_log")}
            WHERE ticker = ? AND filed_date <= ?
        ) x
        LEFT JOIN ({ll._sec_acceptance_query()}) f ON f.accession_number = x.accession_number
        ORDER BY filed, concept, unit, period_start, period_end, accepted_at, fiscal_year,
                 fiscal_period, form, value
    """
    frame = ll._query(query, [ticker, as_of_date.isoformat()])
    if frame.empty:
        return None
    frame["filed"] = pd.to_datetime(frame["filed"])
    assert_point_in_time(frame, as_of_date, date_column="filed")
    frame["period_start"] = pd.to_datetime(frame["period_start"])
    frame["period_end"] = pd.to_datetime(frame["period_end"])
    frame["available_date"] = ll._sec_available_dates(frame["filed"], frame["accepted_at"])
    frame = frame[frame["available_date"] <= pd.Timestamp(as_of_date)]
    if frame.empty:
        return None
    frame = (
        frame.assign(accepted_at=pd.to_datetime(frame["accepted_at"], utc=True))
        .sort_values(["filed", "available_date", "accepted_at"], kind="stable")
        .drop_duplicates(subset=ll._COMPANY_FACT_PERIOD_KEY, keep="last")
        .drop(columns="accepted_at")
        .sort_values(["filed", "period_end"], kind="stable")
        .reset_index(drop=True)
    )
    assert_point_in_time(frame, as_of_date, date_column="available_date")
    return frame


def legacy_short_interest(ticker: str, as_of_date: date) -> pd.DataFrame | None:
    ll._guard_data_end(as_of_date, "load_short_interest_from_lakehouse")
    query = f"""
        SELECT settlement_date, revision_flag,
               current_short_position AS short_interest_shares,
               days_to_cover, average_daily_volume AS avg_daily_volume
        FROM {ll._table_sql("finra_short_interest")}
        WHERE symbol = ?
        ORDER BY settlement_date
    """
    frame = ll._query(query, [ticker])
    if frame.empty:
        return None
    frame["settlement_date"] = pd.to_datetime(frame["settlement_date"])
    settlements = sorted({stamp.date() for stamp in frame["settlement_date"]})
    next_settlement: dict[date, date | None] = dict(
        zip(settlements, [*settlements[1:], None], strict=True)
    )
    revised = frame["revision_flag"].eq(ll._FINRA_REVISED).fillna(False).astype(bool)
    available: list[pd.Timestamp | None] = []
    for stamp, is_revised in zip(frame["settlement_date"], revised, strict=True):
        published_with = next_settlement[stamp.date()] if is_revised else stamp.date()
        available.append(
            None
            if published_with is None
            else pd.Timestamp(ll._finra_available_date(published_with))
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


def legacy_insider_transactions(ticker: str, as_of_date: date) -> pd.DataFrame | None:
    ll._guard_data_end(as_of_date, "load_insider_transactions_from_lakehouse")
    query = f"""
        SELECT t.owner_cik, t.owner_name, t.is_officer, t.is_director, t.is_ten_percent_owner,
               t.transaction_date, t.transaction_code, t.shares, t.price_per_share,
               t.acquired_disposed, t.filed_date, f.accepted_at
        FROM (
            SELECT DISTINCT * EXCLUDE (scraped_at, dt, "table")
            FROM {ll._table_sql("sec_insider_transactions_log")}
            WHERE issuer_ticker = ? AND filed_date <= ?
        ) t
        LEFT JOIN ({ll._sec_acceptance_query()}) f ON f.accession_number = t.accession_number
        ORDER BY t.filed_date, t.transaction_date, t.owner_cik, t.owner_name, t.transaction_code,
                 t.acquired_disposed, t.shares, t.price_per_share, t.is_officer, t.is_director,
                 t.is_ten_percent_owner, f.accepted_at
    """
    frame = ll._query(query, [ticker, as_of_date.isoformat()])
    if frame.empty:
        return None
    frame["filed_date"] = pd.to_datetime(frame["filed_date"])
    frame["transaction_date"] = pd.to_datetime(frame["transaction_date"])
    assert_point_in_time(frame, as_of_date, date_column="filed_date")
    frame["available_date"] = ll._sec_available_dates(frame["filed_date"], frame["accepted_at"])
    frame = frame[frame["available_date"] <= pd.Timestamp(as_of_date)].drop(columns="accepted_at")
    if frame.empty:
        return None
    frame = frame.reset_index(drop=True)
    assert_point_in_time(frame, as_of_date, date_column="available_date")
    return frame


def legacy_futures_price_history(tickers: list[str], start: date, end: date) -> pd.DataFrame | None:
    ll._guard_data_end(end, "load_futures_price_history_from_lakehouse")
    ticker_by_root: dict[str, str] = {}
    for ticker in tickers:
        root = futures_root(ticker)
        if root is None:
            raise ValueError(f"{ticker} is not a futures ticker")
        ticker_by_root[root] = ticker
    placeholders = ", ".join("?" for _ in ticker_by_root)
    query = f"""
        SELECT root, date, front_contract, unadjusted_close, ratio_adjusted_close,
               is_roll_day, roll_gap_known
        FROM (
            SELECT root, date, front_contract, unadjusted_close, ratio_adjusted_close,
                   is_roll_day, roll_gap_known,
                   max(CASE WHEN date < ? AND ratio_adjusted_close > 0 THEN date END)
                       OVER (PARTITION BY root) AS previous_kept
            FROM {ll._table_sql("futures_continuous_daily")}
            WHERE root IN ({placeholders})
              AND date <= ?
        ) AS daily
        WHERE date >= COALESCE(previous_kept, ?)
        ORDER BY root, date
    """
    frame = ll._query(
        query, [start.isoformat(), *ticker_by_root, end.isoformat(), start.isoformat()]
    )
    if frame.empty:
        return None
    frame["date"] = pd.to_datetime(frame["date"])

    per_ticker: dict[str, pd.DataFrame] = {}
    for lake_root, group in frame.groupby("root", sort=True):
        adjusted_all = group["ratio_adjusted_close"].astype(float)
        kept = (adjusted_all > 0.0).to_numpy(dtype=bool)
        unknown_gap = group["is_roll_day"].eq(True).to_numpy(dtype=bool) & group[
            "roll_gap_known"
        ].eq(False).fillna(False).to_numpy(dtype=bool)
        unknown_gaps_so_far = np.cumsum(unknown_gap)[kept]
        bars = group.loc[kept]
        adjusted = adjusted_all.to_numpy(dtype=np.float64)[kept]
        daily = np.zeros(len(adjusted))
        daily[1:] = adjusted[1:] / adjusted[:-1] - 1.0
        daily[1:][np.diff(unknown_gaps_so_far) > 0] = 0.0
        fronts = bars["front_contract"]
        previous_fronts = fronts.shift(1)
        roll = fronts.ne(previous_fronts) & fronts.notna() & previous_fronts.notna()

        in_range = (bars["date"] >= pd.Timestamp(start)).to_numpy(dtype=bool)
        if not in_range.any():
            continue
        bars = bars.loc[in_range]
        daily = daily[in_range]
        daily[0] = 0.0
        index = pd.DatetimeIndex(bars["date"])
        unadjusted = bars["unadjusted_close"].astype(float)
        unadjusted = unadjusted.where(unadjusted > 0.0)
        first_adjusted = float(adjusted[in_range][0])
        base = float(unadjusted.iloc[0]) if pd.notna(unadjusted.iloc[0]) else first_adjusted
        close = pd.Series(base * np.cumprod(1.0 + daily), index=index)
        per_ticker[ticker_by_root[str(lake_root)]] = pd.DataFrame(
            {
                "open": close,
                "close": close,
                "raw_close": unadjusted.to_numpy(),
                "volume": np.nan,
                "log_return": np.log(close / close.shift(1)),
                "roll": roll.loc[in_range].astype(float).to_numpy(),
            },
            index=index,
        )

    if not per_ticker:
        return None
    combined = pd.concat(per_ticker, axis=1)
    assert_point_in_time(combined, end)
    return combined


def legacy_policy_rates(currencies: list[str], start: date, end: date) -> pd.DataFrame | None:
    ll._guard_data_end(end, "load_policy_rates_from_lakehouse")
    wanted = list(dict.fromkeys(currencies))
    currency_by_area: dict[str, str] = {}
    for currency in wanted:
        area = ll._BIS_POLICY_RATE_AREAS.get(currency)
        if area is None:
            raise ValueError(f"no BIS policy-rate reference area for currency {currency}")
        currency_by_area[area] = currency
    placeholders = ", ".join("?" for _ in currency_by_area)
    query = f"""
        SELECT ref_area, date, value
        FROM {ll._table_sql("bis_observations")}
        WHERE dataflow = ?
          AND freq = ?
          AND ref_area IN ({placeholders})
          AND date BETWEEN ? AND ?
          AND value IS NOT NULL
        ORDER BY date, ref_area
    """
    window_start = start - timedelta(days=ll.POLICY_RATE_LOOKBACK_DAYS)
    frame = ll._query(
        query,
        [
            ll._BIS_POLICY_RATE_DATAFLOW,
            ll._BIS_DAILY,
            *currency_by_area,
            window_start.isoformat(),
            end.isoformat(),
        ],
    )
    if frame.empty:
        return None
    frame["date"] = pd.to_datetime(frame["date"])
    frame["currency"] = frame["ref_area"].map(currency_by_area)
    rates = frame.pivot(index="date", columns="currency", values="value").astype(float) / 100.0
    rates = rates.reindex(columns=wanted).sort_index()
    rates.index.name = "date"
    rates.columns.name = None
    assert_point_in_time(rates, end)
    return rates


# Each public loader's name and its frozen per-call reference.
LEGACY_LOADERS = {
    "load_futures_price_history_from_lakehouse": legacy_futures_price_history,
    "load_policy_rates_from_lakehouse": legacy_policy_rates,
    "load_sp500_member_securities_asof": legacy_sp500_member_securities_asof,
    "load_security_log_returns": legacy_security_log_returns,
    "load_security_daily_bars": legacy_security_daily_bars,
    "split_adjustment_factor": legacy_split_adjustment_factor,
    "load_options_snapshot_from_lakehouse": legacy_options_snapshot,
    "load_option_quote_from_lakehouse": legacy_option_quote,
    "load_cot_reports_from_lakehouse": legacy_cot_reports,
    "load_cot_positioning_from_lakehouse": legacy_cot_positioning,
    "load_dividends_from_lakehouse": legacy_dividends,
    "load_earnings_surprise_from_lakehouse": legacy_earnings_surprise,
    "load_sic_code_from_lakehouse": legacy_sic_code,
    "load_price_history_from_lakehouse": legacy_price_history,
    "load_raw_close_from_lakehouse": legacy_raw_close,
    "load_fred_vintages_from_lakehouse": legacy_fred_vintages,
    "load_fred_series_from_lakehouse": legacy_fred_series,
    "load_fama_french_factors_from_lakehouse": legacy_fama_french_factors,
    "load_company_facts_from_lakehouse": legacy_company_facts,
    "load_short_interest_from_lakehouse": legacy_short_interest,
    "load_insider_transactions_from_lakehouse": legacy_insider_transactions,
}
