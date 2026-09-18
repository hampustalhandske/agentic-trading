# lakehouse_loader.py

"""Every data read a run makes, from the external partitioned-parquet
lakehouse at `settings.data.lake_dir` (set by the `LAKEHOUSE_DIR`
environment variable): equities/ETFs/indices/commodities/FX/crypto
prices, point-in-time S&P 500 membership, option chains, dividends,
CFTC positioning, earnings, SIC codes, FRED series, Ken French factors,
FINRA short interest and SEC company facts, laid out as
`table=<name>/<partition>/*.parquet` and queried in-process with DuckDB
reading the parquet files directly.

This is the only market/alternative-data source a run reads:
`data/loader.py::load_price_history_with_fallback` raises for any ticker
the lake doesn't cover rather than falling back to a live API."""

from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from agentic_trading.config import settings
from agentic_trading.data.point_in_time import assert_point_in_time
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)


def _lake_dir() -> Path:
    return Path(settings.data.lake_dir)


def _instrument_prices_glob() -> str:
    return str(_lake_dir() / "table=instrument_prices" / "*" / "*.parquet")


def _sp500_membership_glob() -> str:
    return str(_lake_dir() / "table=sp500_membership" / "*" / "*.parquet")


def lakehouse_available() -> bool:
    return _lake_dir().exists() and any(_lake_dir().glob("table=instrument_prices/*/*.parquet"))


def load_sp500_constituents_asof(as_of_date: date) -> list[str] | None:
    """Real, point-in-time S&P 500 membership as of `as_of_date` — every
    ticker whose `sp500_membership` span covers this date, including
    names since removed (a name that left the index yesterday was still
    a genuine member today's-relative-yesterday, so it belongs in a
    point-in-time universe for that date; excluding it would be the
    lookahead-shaped mistake of using tomorrow's knowledge to prune
    today's universe). Returns `None` if the lake is unreachable or the
    table has no rows — callers should fall back to a smaller static
    universe, not treat this as an empty-but-valid membership list.
    Calls `assert_point_in_time` on the result for consistency with every
    other loader in this module, though it's a no-op here since this
    returns tickers, not dates."""
    if not _lake_dir().exists() or not any(_lake_dir().glob("table=sp500_membership/*/*.parquet")):
        return None

    query = f"""
        SELECT DISTINCT ticker
        FROM (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY ticker, start_date, end_date ORDER BY scraped_at DESC
            ) AS rn
            FROM read_parquet('{_sp500_membership_glob()}', hive_partitioning = true, union_by_name = true)
        )
        WHERE rn = 1
          AND start_date <= '{as_of_date.isoformat()}'
          AND (end_date IS NULL OR end_date >= '{as_of_date.isoformat()}')
    """
    try:
        connection = duckdb.connect(database=":memory:", read_only=False)
        result = connection.execute(query).df()
    except Exception:
        logger.warning("sp500_membership lakehouse query failed", exc_info=True)
        return None
    finally:
        connection.close()

    if result.empty:
        return None
    tickers = sorted(result["ticker"].tolist())
    assert_point_in_time(tickers, as_of_date)
    return tickers


def _option_chain_glob() -> str:
    return str(_lake_dir() / "table=dolthub_option_chain" / "*" / "*.parquet")


def load_options_snapshot_from_lakehouse(
    ticker: str, as_of_date: date, lookback_days: int = 10
) -> pd.DataFrame | None:
    """Real historical options-chain rows (strike, expiration, bid, ask,
    implied_volatility, delta, gamma, theta, vega, rho — no open interest,
    no volume, confirmed absent from the source) for `ticker`, from the
    most recent available snapshot date on or before `as_of_date` within
    `lookback_days` — coverage is real but not daily-dense (confirmed
    live: roughly every-other-trading-day for AAPL), so a same-day match
    isn't guaranteed even for a covered ticker.

    Returns `None` if the lake is unreachable, `ticker` has no coverage at
    all, or the nearest snapshot is more than `lookback_days` stale — the
    CVaR-pod analysts calling this fall back to their synthetic,
    theoretical approach in that case rather than treat a miss as fatal.
    Self-asserts point-in-time on `date` before returning.
    Every CVaR-pod analyst that touches options data is still a synthetic,
    model-implied read for any ticker outside this real coverage (11
    tickers as of this pull: AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA,
    JPM, XOM, SPY, COIN — confirm the current set directly against the
    lake if this list may have grown)."""
    if not _lake_dir().exists() or not any(
        _lake_dir().glob("table=dolthub_option_chain/*/*.parquet")
    ):
        return None

    window_start = (as_of_date - pd.Timedelta(days=lookback_days)).isoformat()
    query = f"""
        SELECT date, expiration, option_type, strike, bid, ask,
               implied_volatility, delta, gamma, theta, vega, rho
        FROM read_parquet('{_option_chain_glob()}', hive_partitioning = true, union_by_name = true)
        WHERE underlying = '{ticker}'
          AND date BETWEEN '{window_start}' AND '{as_of_date.isoformat()}'
          AND date = (
              SELECT MAX(date)
              FROM read_parquet('{_option_chain_glob()}', hive_partitioning = true, union_by_name = true)
              WHERE underlying = '{ticker}'
                AND date BETWEEN '{window_start}' AND '{as_of_date.isoformat()}'
          )
    """
    try:
        connection = duckdb.connect(database=":memory:", read_only=False)
        frame = connection.execute(query).df()
    except Exception:
        logger.warning("Options-chain lakehouse query failed for %s", ticker, exc_info=True)
        return None
    finally:
        connection.close()

    if frame.empty:
        return None
    frame["date"] = pd.to_datetime(frame["date"])
    frame["expiration"] = pd.to_datetime(frame["expiration"])
    assert_point_in_time(frame, as_of_date, date_column="date")
    return frame


def _cot_glob(table: str) -> str:
    return str(_lake_dir() / f"table={table}" / "*" / "*.parquet")


def load_cot_positioning_from_lakehouse(
    ticker: str, as_of_date: date, lookback_days: int = 400
) -> pd.Series | None:
    """Net speculative positioning as a fraction of open interest, from
    CFTC Commitment of Traders reports — `managed_money_long -
    managed_money_short` (the disaggregated report's commodity category)
    for commodities, `leveraged_funds_long - leveraged_funds_short` (the
    TFF report's closest equivalent) for FX/rates, divided by
    `open_interest_all`. A given `ticker` only ever appears in one of the
    two tables (commodities in `cot_disaggregated_positions`,
    FX/rate-futures in `cot_tff_positions`) — this tries both and uses
    whichever has rows.

    Point-in-time: COT reports have a real publication lag — the
    Tuesday-dated `report_date` isn't released until the following
    Friday. Gates on `report_date + 3 days` as the estimated
    availability date, the same estimated-lag convention this project's
    FINRA short-interest loader already uses for an identical
    reporting-lag shape.

    Returns `None` if the lake is unreachable or `ticker` has no COT
    coverage at all — callers should treat that as "no positioning read
    available," not an empty-but-valid series. Self-asserts point-in-time
    on the `report_date` index before returning."""
    if not _lake_dir().exists():
        return None

    window_start = (as_of_date - pd.Timedelta(days=lookback_days)).isoformat()
    availability_cutoff = as_of_date.isoformat()
    queries = [
        (
            "cot_disaggregated_positions",
            "(managed_money_long - managed_money_short)",
        ),
        (
            "cot_tff_positions",
            "(leveraged_funds_long - leveraged_funds_short)",
        ),
    ]
    for table, net_position_expr in queries:
        if not any(_lake_dir().glob(f"table={table}/*/*.parquet")):
            continue
        query = f"""
            SELECT report_date, {net_position_expr} / open_interest_all AS net_positioning_ratio
            FROM (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY ticker, report_date ORDER BY scraped_at DESC
                ) AS rn
                FROM read_parquet('{_cot_glob(table)}', hive_partitioning = true, union_by_name = true)
                WHERE ticker = '{ticker}'
                  AND report_date BETWEEN '{window_start}' AND '{availability_cutoff}'
                  AND date_add(CAST(report_date AS DATE), INTERVAL 3 DAY) <= '{availability_cutoff}'
            )
            WHERE rn = 1 AND open_interest_all > 0
            ORDER BY report_date
        """
        try:
            connection = duckdb.connect(database=":memory:", read_only=False)
            frame = connection.execute(query).df()
        except Exception:
            logger.warning("COT lakehouse query failed for %s/%s", table, ticker, exc_info=True)
            frame = pd.DataFrame()
        finally:
            connection.close()
        if not frame.empty:
            frame["report_date"] = pd.to_datetime(frame["report_date"])
            series = frame.set_index("report_date")["net_positioning_ratio"]
            assert_point_in_time(series, as_of_date)
            return series
    return None


def load_dividends_from_lakehouse(ticker: str, as_of_date: date) -> pd.Series | None:
    """Real dividend-per-share history (`dolthub_stock_dividends`) for
    `ticker`, indexed by ex-date, filtered `<= as_of_date` — the ex-date
    is itself the real, already-past-tense corporate-action date (no
    additional estimated-lag gating needed, unlike COT/FINRA). Returns
    `None` if the lake is unreachable or `ticker` has no dividend
    history at all. Self-asserts point-in-time on the `ex_date` index
    before returning."""
    if not _lake_dir().exists() or not any(
        _lake_dir().glob("table=dolthub_stock_dividends/*/*.parquet")
    ):
        return None

    query = f"""
        SELECT ex_date, amount
        FROM (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY act_symbol, ex_date ORDER BY scraped_at DESC
            ) AS rn
            FROM read_parquet('{_lake_dir() / "table=dolthub_stock_dividends" / "*" / "*.parquet"}', hive_partitioning = true, union_by_name = true)
            WHERE act_symbol = '{ticker}' AND ex_date <= '{as_of_date.isoformat()}'
        )
        WHERE rn = 1
        ORDER BY ex_date
    """
    try:
        connection = duckdb.connect(database=":memory:", read_only=False)
        frame = connection.execute(query).df()
    except Exception:
        logger.warning("Dividends lakehouse query failed for %s", ticker, exc_info=True)
        return None
    finally:
        connection.close()

    if frame.empty:
        return None
    frame["ex_date"] = pd.to_datetime(frame["ex_date"])
    series = frame.set_index("ex_date")["amount"]
    assert_point_in_time(series, as_of_date)
    return series


def load_earnings_surprise_from_lakehouse(
    ticker: str, as_of_date: date
) -> tuple[float, float, pd.Timestamp] | None:
    """The most recent real EPS surprise for `ticker` as of `as_of_date`:
    `(reported, estimate, announcement_date)` for whichever quarter's
    real announcement date (`dolthub_earnings_calendar.date`) is the
    latest one on or before `as_of_date` — the exact real availability
    gate, not an estimated lag, since both the fiscal period
    (`dolthub_eps_history.period_end_date`) and the real announcement
    date are both directly in the lake. Joins the two tables on a
    75-day window after `period_end_date` — a generous margin around the
    typical ~29-30 day real reporting lag, not a tight estimate.

    Returns `None` if the lake is unreachable, `ticker` has no earnings
    history at all, or no quarter's real announcement date has occurred
    yet as of `as_of_date`. Self-asserts point-in-time on
    `announcement_date` before returning."""
    if (
        not _lake_dir().exists()
        or not any(_lake_dir().glob("table=dolthub_eps_history/*/*.parquet"))
        or not any(_lake_dir().glob("table=dolthub_earnings_calendar/*/*.parquet"))
    ):
        return None

    eps_glob = str(_lake_dir() / "table=dolthub_eps_history" / "*" / "*.parquet")
    calendar_glob = str(_lake_dir() / "table=dolthub_earnings_calendar" / "*" / "*.parquet")
    query = f"""
        SELECT h.reported, h.estimate, c.date AS announcement_date
        FROM (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY act_symbol, period_end_date ORDER BY scraped_at DESC
            ) AS rn
            FROM read_parquet('{eps_glob}', hive_partitioning = true, union_by_name = true)
            WHERE act_symbol = '{ticker}' AND reported IS NOT NULL AND estimate IS NOT NULL
        ) h
        JOIN (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY act_symbol, date ORDER BY scraped_at DESC
            ) AS rn
            FROM read_parquet('{calendar_glob}', hive_partitioning = true, union_by_name = true)
            WHERE act_symbol = '{ticker}'
        ) c
        ON c.act_symbol = h.act_symbol
           AND h.rn = 1 AND c.rn = 1
           AND CAST(c.date AS DATE) BETWEEN CAST(h.period_end_date AS DATE)
               AND date_add(CAST(h.period_end_date AS DATE), INTERVAL 75 DAY)
        WHERE CAST(c.date AS DATE) <= '{as_of_date.isoformat()}'
        ORDER BY c.date DESC
        LIMIT 1
    """
    try:
        connection = duckdb.connect(database=":memory:", read_only=False)
        frame = connection.execute(query).df()
    except Exception:
        logger.warning("Earnings-surprise lakehouse query failed for %s", ticker, exc_info=True)
        return None
    finally:
        connection.close()

    if frame.empty:
        return None
    assert_point_in_time(frame, as_of_date, date_column="announcement_date")
    row = frame.iloc[0]
    return float(row["reported"]), float(row["estimate"]), pd.Timestamp(row["announcement_date"])


def load_sic_code_from_lakehouse(ticker: str) -> str | None:
    """Real SIC (Standard Industrial Classification) code for `ticker`
    from SEC EDGAR, via `sec_sic_classification` — free, no GICS license
    restriction. Not time-varying in this lake (one row per ticker, not
    a point-in-time span), so no `as_of_date` gating; returns `None` if
    the lake is unreachable or `ticker` has no SIC record (confirmed
    live: mostly non-equity ETF/future/FX tickers, which have no SEC
    filing to classify in the first place)."""
    if not _lake_dir().exists() or not any(
        _lake_dir().glob("table=sec_sic_classification/*/*.parquet")
    ):
        return None

    query = f"""
        SELECT sic
        FROM (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY ticker ORDER BY scraped_at DESC
            ) AS rn
            FROM read_parquet('{_lake_dir() / "table=sec_sic_classification" / "*" / "*.parquet"}', hive_partitioning = true, union_by_name = true)
            WHERE ticker = '{ticker}'
        )
        WHERE rn = 1
    """
    try:
        connection = duckdb.connect(database=":memory:", read_only=False)
        frame = connection.execute(query).df()
    except Exception:
        logger.warning("SIC-classification lakehouse query failed for %s", ticker, exc_info=True)
        return None
    finally:
        connection.close()

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
    covers, or `None` if the lake is unreachable or covers none of them
    — never raises; `load_price_history_with_fallback` is what turns an
    incomplete result into a clear error naming the uncovered ticker(s).

    Uses `adj_close` (dividend/split-adjusted) rather than raw `close` —
    correct for a multi-year backtest where corporate actions actually
    occur — and scales the raw `open` by the same `adj_close / close`
    factor so the open the engine fills at and the close it marks at are
    on one price basis (an unadjusted open against an adjusted close would
    book the whole dividend-adjustment gap as a loss on every fill).
    Deduplicates by latest `scraped_at` per (ticker, date), the same
    "log is append-only, latest write wins" convention the lake's own
    `db/analytics.py` views use. Self-asserts point-in-time on the
    combined frame's date index against `end` before returning.
    """
    if not lakehouse_available():
        return None

    tickers_sql = ", ".join(f"'{ticker}'" for ticker in tickers)
    query = f"""
        SELECT ticker, date,
               CASE WHEN close > 0 THEN open * adj_close / close ELSE NULL END AS open,
               adj_close AS close,
               close AS raw_close,
               volume
        FROM (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY ticker, date ORDER BY scraped_at DESC
            ) AS rn
            FROM read_parquet('{_instrument_prices_glob()}', hive_partitioning = true, union_by_name = true)
            WHERE ticker IN ({tickers_sql}) AND date BETWEEN '{start.isoformat()}' AND '{end.isoformat()}'
        )
        WHERE rn = 1
        ORDER BY ticker, date
    """
    try:
        connection = duckdb.connect(database=":memory:", read_only=False)
        frame = connection.execute(query).df()
    except Exception:
        logger.warning("Lakehouse price query failed for %s", tickers, exc_info=True)
        return None
    finally:
        connection.close()

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
        # Confirmed live: some lakehouse rows have a real, non-zero
        # `close` but a corrupted `adj_close` of exactly 0.0 (a
        # dividend-adjustment bug on that project's side, not this
        # loader's) — a zero/negative price here would silently produce
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


def _table_glob(table: str) -> str:
    return str(_lake_dir() / f"table={table}" / "*" / "*.parquet")


def load_fred_series_from_lakehouse(series_id: str, as_of_date: date) -> pd.Series | None:
    """A FRED macro series (`DGS3MO`/`DGS2`/`DGS10`/`VIXCLS`/`CPIAUCSL`/
    `BAMLH0A0HYM2`/`BAMLC0A0CM` — the same series `macro_loader.
    SUPPORTED_SERIES` names) from the `fred_series_observations` table
    (`series_id`, `date`, `realtime_start`, `value`), filtered
    `<= as_of_date`. Returns `None` if the lake or this table doesn't
    exist yet, or has no rows for `series_id`. Self-asserts point-in-time
    on the date index before returning."""
    if not _lake_dir().exists() or not any(
        _lake_dir().glob("table=fred_series_observations/*/*.parquet")
    ):
        return None

    query = f"""
        SELECT date, value
        FROM (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY series_id, date ORDER BY scraped_at DESC
            ) AS rn
            FROM read_parquet(
                '{_table_glob("fred_series_observations")}', hive_partitioning = true,
                union_by_name = true
            )
            WHERE series_id = '{series_id}' AND date <= '{as_of_date.isoformat()}'
        )
        WHERE rn = 1
        ORDER BY date
    """
    try:
        connection = duckdb.connect(database=":memory:", read_only=False)
        frame = connection.execute(query).df()
    except Exception:
        logger.warning("FRED lakehouse query failed for %s", series_id, exc_info=True)
        return None
    finally:
        connection.close()

    if frame.empty:
        return None
    frame["date"] = pd.to_datetime(frame["date"])
    series = frame.set_index("date")["value"]
    assert_point_in_time(series, as_of_date)
    return series


_KEN_FRENCH_TABLE_BY_MODEL = {
    "3factor": "fama_french_3factor_daily",
    "5factor": "fama_french_5factor_daily",
}
# The lakehouse stores factor returns as plain percentage points
# (e.g. 0.09 meaning 0.09%, matching Ken French's own published CSV
# convention) with lowercase snake_case columns — rename to Ken French's
# own `Mkt-RF`/`SMB`/`HML`/`RF`(/`RMW`/`CMA`) names and a 0-1 return
# scale, the shape the factor-pod analysts consume.
_KEN_FRENCH_COLUMN_RENAME = {
    "mkt_rf": "Mkt-RF",
    "smb": "SMB",
    "hml": "HML",
    "rmw": "RMW",
    "cma": "CMA",
    "rf": "RF",
}


def load_fama_french_factors_from_lakehouse(
    as_of_date: date, model: str = "3factor"
) -> pd.DataFrame | None:
    """Daily Fama-French factor returns for `model` ("3factor" or
    "5factor"), from the `fama_french_3factor_daily`/
    `fama_french_5factor_daily` tables, filtered `<= as_of_date`.
    Returns `None` if the lake or this table doesn't exist yet, has no
    rows, or `model` isn't one of the two supported names. Self-asserts
    point-in-time on the date index before returning."""
    table = _KEN_FRENCH_TABLE_BY_MODEL.get(model)
    if (
        table is None
        or not _lake_dir().exists()
        or not any(_lake_dir().glob(f"table={table}/*/*.parquet"))
    ):
        return None

    query = f"""
        SELECT *
        FROM (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY date ORDER BY scraped_at DESC
            ) AS rn
            FROM read_parquet('{_table_glob(table)}', hive_partitioning = true, union_by_name = true)
            WHERE date <= '{as_of_date.isoformat()}'
        )
        WHERE rn = 1
        ORDER BY date
    """
    try:
        connection = duckdb.connect(database=":memory:", read_only=False)
        frame = connection.execute(query).df()
    except Exception:
        logger.warning("Ken French lakehouse query failed for %s", model, exc_info=True)
        return None
    finally:
        connection.close()

    if frame.empty:
        return None
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.set_index("date")
    factor_columns = [column for column in _KEN_FRENCH_COLUMN_RENAME if column in frame.columns]
    frame = frame[factor_columns].rename(columns=_KEN_FRENCH_COLUMN_RENAME)
    result = (frame.astype(float) / 100.0).sort_index()
    assert_point_in_time(result, as_of_date)
    return result


def load_company_facts_from_lakehouse(ticker: str, as_of_date: date) -> pd.DataFrame | None:
    """Filed SEC EDGAR company facts for `ticker`, from the
    `sec_company_facts` table (`ticker`, `cik`, `concept`, `unit`,
    `period_start`, `period_end`, `value`, `fiscal_year`,
    `fiscal_period`, `form`, `accession_number`, `filed_date`) —
    `filed_date` renamed to `filed`, giving a `concept`/`value`/
    `fiscal_period`/`filed` shape. A single filing reports multiple period windows
    for the same concept (current quarter, year-to-date, prior-year
    comparatives) as separate rows sharing one `filed_date` — the same
    multi-row-per-filing shape the original SEC EDGAR XBRL API already
    produced, not new here. Gated on the real `filed_date` (never the
    fiscal period it describes — a lookahead bug otherwise)
    `<= as_of_date`. Returns `None` if the lake or this
    table doesn't exist yet, or has no rows for `ticker`. Self-asserts
    point-in-time on the returned `filed` column (the renamed
    `filed_date`) before returning."""
    if not _lake_dir().exists() or not any(_lake_dir().glob("table=sec_company_facts/*/*.parquet")):
        return None

    query = f"""
        SELECT concept, value, fiscal_period, filed_date AS filed
        FROM (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY ticker, concept, period_start, period_end, accession_number
                ORDER BY scraped_at DESC
            ) AS rn
            FROM read_parquet(
                '{_table_glob("sec_company_facts")}', hive_partitioning = true, union_by_name = true
            )
            WHERE ticker = '{ticker}' AND filed_date <= '{as_of_date.isoformat()}'
        )
        WHERE rn = 1
        ORDER BY filed
    """
    try:
        connection = duckdb.connect(database=":memory:", read_only=False)
        frame = connection.execute(query).df()
    except Exception:
        logger.warning("SEC EDGAR facts lakehouse query failed for %s", ticker, exc_info=True)
        return None
    finally:
        connection.close()

    if frame.empty:
        return None
    frame["filed"] = pd.to_datetime(frame["filed"])
    assert_point_in_time(frame, as_of_date, date_column="filed")
    return frame


def load_short_interest_from_lakehouse(ticker: str, as_of_date: date) -> pd.DataFrame | None:
    """FINRA short-interest history for `ticker`, from the
    `finra_short_interest` table (`symbol`, `settlement_date`,
    `dissemination_date`, `current_short_position`, `days_to_cover`,
    `average_daily_volume`, ...) — renamed to a
    `short_interest_shares`/`days_to_cover`/`avg_daily_volume`/
    `settlement_date`/`dissemination_date` shape. Gated on
    `dissemination_date <= as_of_date`. Returns `None` if the lake or
    this table doesn't exist yet, or has no rows for `ticker`.
    Self-asserts point-in-time on `dissemination_date` before
    returning."""
    if not _lake_dir().exists() or not any(
        _lake_dir().glob("table=finra_short_interest/*/*.parquet")
    ):
        return None

    query = f"""
        SELECT settlement_date, dissemination_date,
               current_short_position AS short_interest_shares,
               days_to_cover, average_daily_volume AS avg_daily_volume
        FROM (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY symbol, settlement_date ORDER BY scraped_at DESC
            ) AS rn
            FROM read_parquet(
                '{_table_glob("finra_short_interest")}', hive_partitioning = true,
                union_by_name = true
            )
            WHERE symbol = '{ticker}' AND dissemination_date <= '{as_of_date.isoformat()}'
        )
        WHERE rn = 1
        ORDER BY settlement_date
    """
    try:
        connection = duckdb.connect(database=":memory:", read_only=False)
        frame = connection.execute(query).df()
    except Exception:
        logger.warning("FINRA lakehouse query failed for %s", ticker, exc_info=True)
        return None
    finally:
        connection.close()

    if frame.empty:
        return None
    frame["settlement_date"] = pd.to_datetime(frame["settlement_date"])
    frame["dissemination_date"] = pd.to_datetime(frame["dissemination_date"])
    assert_point_in_time(frame, as_of_date, date_column="dissemination_date")
    return frame


def load_insider_transactions_from_lakehouse(ticker: str, as_of_date: date) -> pd.DataFrame | None:
    """SEC Form 4 insider transactions for `ticker` from the
    `sec_insider_transactions` table (`owner_cik`, `owner_name`,
    `is_officer`, `is_director`, `is_ten_percent_owner`, `transaction_date`,
    `transaction_code`, `shares`, `price_per_share`, `acquired_disposed`,
    `filed_date`), gated on `filed_date <= as_of_date` — the STOCK Act
    gives insiders two business days to file, so the filing date, not the
    trade date, is when the information became public. Latest
    `scraped_at` wins per (owner, accession, trade, security, code).
    Returns `None` if the lake or table is absent or has no rows.
    Self-asserts point-in-time on `filed_date` before returning."""
    if not _lake_dir().exists() or not any(
        _lake_dir().glob("table=sec_insider_transactions/*/*.parquet")
    ):
        return None

    query = f"""
        SELECT owner_cik, owner_name, is_officer, is_director, is_ten_percent_owner,
               transaction_date, transaction_code, shares, price_per_share,
               acquired_disposed, filed_date
        FROM (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY owner_cik, accession_number, transaction_date, security_title,
                             transaction_code
                ORDER BY scraped_at DESC
            ) AS rn
            FROM read_parquet(
                '{_table_glob("sec_insider_transactions")}', hive_partitioning = true,
                union_by_name = true
            )
            WHERE issuer_ticker = '{ticker}' AND filed_date <= '{as_of_date.isoformat()}'
        )
        WHERE rn = 1
        ORDER BY filed_date, transaction_date
    """
    try:
        connection = duckdb.connect(database=":memory:", read_only=False)
        frame = connection.execute(query).df()
    except Exception:
        logger.warning("Insider-transactions lakehouse query failed for %s", ticker, exc_info=True)
        return None
    finally:
        connection.close()

    if frame.empty:
        return None
    frame["filed_date"] = pd.to_datetime(frame["filed_date"])
    frame["transaction_date"] = pd.to_datetime(frame["transaction_date"])
    assert_point_in_time(frame, as_of_date, date_column="filed_date")
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
    `load_options_snapshot_from_lakehouse`; `None` when never quoted in
    the window. Self-asserts point-in-time on `date` before returning."""
    if not _lake_dir().exists() or not any(
        _lake_dir().glob("table=dolthub_option_chain/*/*.parquet")
    ):
        return None
    window_start = (as_of_date - pd.Timedelta(days=lookback_days)).isoformat()
    query = f"""
        SELECT date, expiration, option_type, strike, bid, ask,
               implied_volatility, delta, gamma, theta, vega, rho
        FROM read_parquet('{_option_chain_glob()}', hive_partitioning = true, union_by_name = true)
        WHERE underlying = '{underlying}'
          AND expiration = '{expiration.isoformat()}'
          AND option_type = '{option_type}'
          AND ABS(strike - {strike!r}) < 1e-6
          AND date BETWEEN '{window_start}' AND '{as_of_date.isoformat()}'
        ORDER BY date DESC
        LIMIT 1
    """
    try:
        connection = duckdb.connect(database=":memory:", read_only=False)
        frame = connection.execute(query).df()
    except Exception:
        logger.warning("Option quote lakehouse query failed for %s", underlying, exc_info=True)
        return None
    finally:
        connection.close()
    if frame.empty:
        return None
    frame["date"] = pd.to_datetime(frame["date"])
    frame["expiration"] = pd.to_datetime(frame["expiration"])
    assert_point_in_time(frame, as_of_date, date_column="date")
    return frame
