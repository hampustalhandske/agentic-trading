"""Unit tests for agentic_trading.data.loaders.lakehouse_loader.

The loaders run their real SQL against a `FakeLake` (see `tests/conftest.py`).
Each loader gates its own query on `<= as_of_date`; the "raises" tests make
the fake return a future-dated frame regardless of the SQL, to confirm
`assert_point_in_time` raises if a result ever contained a row the SQL gate
should have excluded.
"""

from datetime import date
from typing import Any

import pandas as pd
import pytest

from agentic_trading.data.loaders import lakehouse_loader


def _short_interest_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "settlement_date": ["2024-01-01"],
            "dissemination_date": ["2024-01-10"],
            "current_short_position": [1000],
            "days_to_cover": [1.5],
            "average_daily_volume": [500],
            "revision_flag": pd.Series([None], dtype="string"),
        }
    )


def test_load_short_interest_returns_valid_data(fake_lake: Any) -> None:
    """An unrevised settlement already available by as_of_date passes straight through."""
    fake_lake.add(
        "finra_short_interest",
        _short_interest_table(),
        date_columns=("settlement_date", "dissemination_date"),
    )

    result = lakehouse_loader.load_short_interest_from_lakehouse("AAPL", date(2024, 1, 15))

    assert result is not None
    assert result.loc[0, "short_interest_shares"] == 1000


def _company_facts_table(filed: str, accession: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAPL"],
            "concept": ["Revenues"],
            "value": [1_000_000.0],
            "fiscal_period": ["Q1"],
            "filed_date": [filed],
            "accession_number": [accession],
        }
    )


def _sec_filings_table(accession: str, accepted_at: str) -> pd.DataFrame:
    return pd.DataFrame(
        {"accession_number": [accession], "accepted_at": [pd.Timestamp(accepted_at, tz="UTC")]}
    )


def test_load_company_facts_returns_valid_data(fake_lake: Any) -> None:
    """Facts from a filing accepted well before as_of_date pass straight through."""
    fake_lake.add(
        "sec_company_facts",
        _company_facts_table("2024-01-10", "0000320193-24-000001"),
        date_columns=("filed_date",),
    )
    fake_lake.add("sec_filings", _sec_filings_table("0000320193-24-000001", "2024-01-10 14:00"))

    result = lakehouse_loader.load_company_facts_from_lakehouse("AAPL", date(2024, 1, 15))

    assert result is not None
    assert result.loc[0, "concept"] == "Revenues"


def test_load_company_facts_raises_on_future_filed_date(fake_lake: Any) -> None:
    """A filed date after as_of_date is a broken SQL gate — must raise, not pass through."""
    fake_lake.fixed_result = pd.DataFrame(
        {
            "concept": ["Revenues"],
            "value": [1_000_000.0],
            "fiscal_period": ["Q1"],
            "filed": ["2024-01-20"],
        }
    )

    with pytest.raises(ValueError, match="2024-01-20"):
        lakehouse_loader.load_company_facts_from_lakehouse("AAPL", date(2024, 1, 15))


def test_load_dividends_returns_valid_data(fake_lake: Any) -> None:
    """Valid, in-window ex-dates pass straight through."""
    fake_lake.add(
        "dolthub_stock_dividends",
        pd.DataFrame({"act_symbol": ["AAPL"], "ex_date": ["2024-01-10"], "amount": [0.25]}),
        date_columns=("ex_date",),
    )

    result = lakehouse_loader.load_dividends_from_lakehouse("AAPL", date(2024, 1, 15))

    assert result is not None
    assert result.iloc[0] == 0.25


def test_load_dividends_raises_on_future_ex_date(fake_lake: Any) -> None:
    """An ex-date after as_of_date in the index is a broken SQL gate — must raise."""
    fake_lake.fixed_result = pd.DataFrame({"ex_date": ["2024-01-20"], "amount": [0.25]})

    with pytest.raises(ValueError, match="2024-01-20"):
        lakehouse_loader.load_dividends_from_lakehouse("AAPL", date(2024, 1, 15))


def test_load_fred_series_returns_valid_data(fake_lake: Any) -> None:
    """Valid, in-window dates in the index pass straight through."""
    fake_lake.add(
        "fred_vintages",
        pd.DataFrame(
            {
                "series_id": ["DGS10"],
                "date": ["2024-01-10"],
                "realtime_start": ["2024-01-11"],
                "value": [5.25],
            }
        ),
        date_columns=("date", "realtime_start"),
    )

    result = lakehouse_loader.load_fred_series_from_lakehouse("DGS10", date(2024, 1, 15))

    assert result is not None
    assert result.iloc[0] == 5.25


def test_load_fred_series_raises_on_future_index_date(fake_lake: Any) -> None:
    """A date after as_of_date in the index is a broken SQL gate — must raise."""
    fake_lake.fixed_result = pd.DataFrame({"date": ["2024-01-20"], "value": [5.25]})

    with pytest.raises(ValueError, match="2024-01-20"):
        lakehouse_loader.load_fred_series_from_lakehouse("DGS10", date(2024, 1, 15))


def _option_chain_table(
    strike: float, bid: float, ask: float, implied_volatility: float, delta: float, **greeks: float
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "underlying": ["NVDA"],
            "date": ["2024-04-01"],
            "expiration": ["2024-04-26"],
            "option_type": ["call"],
            "strike": [strike],
            "bid": [bid],
            "ask": [ask],
            "implied_volatility": [implied_volatility],
            "delta": [delta],
            **{name: [value] for name, value in greeks.items()},
        }
    )


def _splits_table(ticker: str, ratios: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": pd.Series([ticker] * len(ratios), dtype="string"),
            "ex_date": ["2024-06-10"] * len(ratios),
            "ratio": pd.Series(ratios, dtype=float),
            "in_close_basis": pd.Series([True] * len(ratios), dtype="boolean"),
        }
    )


def test_options_snapshot_rescales_a_pre_split_strike_onto_todays_basis(
    fake_lake: Any,
) -> None:
    """A contract quoted before a later 10-for-1 split is archival and
    never itself adjusted; the loader must divide its strike/bid/ask by
    the split factor so it lines up with the always-adjusted spot from
    `instrument_prices` — this is the bug behind the NVDA 630 call that
    marked near zero against an $88 spot in the live 2024_2026 run
    (KNOWN_LIMITATIONS §0)."""
    fake_lake.add(
        "dolthub_option_chain",
        _option_chain_table(
            630.0, 274.35, 278.0, 0.5588, 0.9948, gamma=0.001, theta=-0.02, vega=0.03, rho=0.01
        ),
        date_columns=("date", "expiration"),
    )
    fake_lake.add(
        "yahoo_split_close_basis", _splits_table("NVDA", [0.1]), date_columns=("ex_date",)
    )

    result = lakehouse_loader.load_options_snapshot_from_lakehouse("NVDA", date(2024, 4, 1))

    assert result is not None
    row = result.iloc[0]
    assert row["strike"] == pytest.approx(63.0)
    assert row["bid"] == pytest.approx(27.435)
    assert row["ask"] == pytest.approx(27.8)
    assert row["gamma"] == pytest.approx(0.001 / 100.0)
    assert row["vega"] == pytest.approx(0.003)
    # Dimensionless fields are untouched by the rescale.
    assert row["delta"] == pytest.approx(0.9948)
    assert row["implied_volatility"] == pytest.approx(0.5588)


def test_options_snapshot_leaves_an_unsplit_ticker_unchanged(fake_lake: Any) -> None:
    """No split recorded after the quote date — the chain passes through
    exactly as quoted."""
    fake_lake.add(
        "dolthub_option_chain",
        _option_chain_table(
            180.0, 5.0, 5.2, 0.25, 0.5, gamma=0.01, theta=-0.05, vega=0.1, rho=0.02
        ).assign(underlying="AAPL"),
        date_columns=("date", "expiration"),
    )
    fake_lake.add("yahoo_split_close_basis", _splits_table("AAPL", []), date_columns=("ex_date",))

    result = lakehouse_loader.load_options_snapshot_from_lakehouse("AAPL", date(2024, 4, 1))

    assert result is not None
    assert result.iloc[0]["strike"] == pytest.approx(180.0)
    assert result.iloc[0]["bid"] == pytest.approx(5.0)


def test_cot_positioning_reads_the_cftc_table_by_contract_code(fake_lake: Any) -> None:
    """Gold maps to its CFTC contract code in `cftc_cot_disaggregated`; the
    ratio is managed-money net over open interest, gated on report_date + 3 days."""
    fake_lake.add(
        "cftc_cot_disaggregated",
        pd.DataFrame(
            {
                "cftc_contract_market_code": ["088691", "088691", "067651"],
                "report_date": ["2024-01-02", "2024-01-09", "2024-01-02"],
                "open_interest_all": [1000, 1000, 1000],
                "managed_money_long": [300, 400, 999],
                "managed_money_short": [100, 100, 0],
            }
        ),
        date_columns=("report_date",),
    )

    result = lakehouse_loader.load_cot_positioning_from_lakehouse("GC=F", date(2024, 1, 10))

    assert result is not None
    # The 2024-01-09 report is not yet public on 2024-01-10 (report_date + 3 days).
    assert list(result) == [pytest.approx(0.2)]
    assert lakehouse_loader.load_cot_positioning_from_lakehouse("AAPL", date(2024, 1, 10)) is None


def test_fama_french_pivots_the_long_table_onto_a_zero_one_scale(fake_lake: Any) -> None:
    """The daily 3-factor dataset becomes Mkt-RF/SMB/HML/RF columns in
    return units, ignoring other datasets and dates not yet public."""
    rows = [
        ("F-F_Research_Data_Factors_daily", "daily", "2024-01-02", "Mkt-RF", 0.5),
        ("F-F_Research_Data_Factors_daily", "daily", "2024-01-02", "SMB", 0.1),
        ("F-F_Research_Data_Factors_daily", "daily", "2024-01-02", "HML", -0.2),
        ("F-F_Research_Data_Factors_daily", "daily", "2024-01-02", "RF", 0.01),
        ("F-F_Research_Data_Factors_daily", "daily", "2024-02-01", "Mkt-RF", 9.0),
        ("F-F_Research_Data_5_Factors_2x3_daily", "daily", "2024-01-02", "RMW", 0.3),
    ]
    fake_lake.add(
        "famafrench_returns",
        pd.DataFrame(rows, columns=["dataset", "frequency", "date", "portfolio", "value"]),
        date_columns=("date",),
    )

    result = lakehouse_loader.load_fama_french_factors_from_lakehouse(date(2024, 2, 29))

    assert result is not None
    assert list(result.columns) == ["Mkt-RF", "SMB", "HML", "RF"]
    assert len(result) == 1
    assert result.iloc[0]["Mkt-RF"] == pytest.approx(0.005)
    assert result.iloc[0]["HML"] == pytest.approx(-0.002)
