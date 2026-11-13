# test_preflight_sources.py

"""Unit tests for the lake sources added to agentic_trading.backtest.preflight.

Covers the warm-up-aware coverage window, per-source scoping (equity-only,
coverage-scoped, unscoped), the `sec_sic` presence check, and the
`data_requirements` declared by the real analysts that read those sources."""

from datetime import date

import pandas as pd
import pytest

from agentic_trading.agents.analysts.alt_data_pod.network_centrality_analyst import (
    LeadLagNetworkCentralityAnalyst,
)
from agentic_trading.agents.analysts.cvar_pod.black_scholes_valuation_analyst import (
    BlackScholesValuationAnalyst,
)
from agentic_trading.agents.analysts.cvar_pod.greeks_delta_hedge_analyst import (
    GreeksDeltaHedgeAnalyst,
)
from agentic_trading.agents.analysts.cvar_pod.heston_vol_surface_analyst import (
    HestonVolSurfaceAnalyst,
)
from agentic_trading.agents.analysts.factor_pod.earnings_surprise_analyst import (
    EarningsSurpriseAnalyst,
)
from agentic_trading.agents.analysts.factor_pod.fama_macbeth_analyst import FamaMacbethAnalyst
from agentic_trading.agents.analysts.macro_rates_pod.cot_positioning_analyst import (
    COTPositioningAnalyst,
)
from agentic_trading.agents.analysts.risk_pod.dividend_payout_analyst import DividendPayoutAnalyst
from agentic_trading.agents.analysts.stat_arb_pod.kalman_hedge_ratio_analyst import (
    KalmanHedgeRatioAnalyst,
)
from agentic_trading.agents.analysts.stat_arb_pod.pairs_cointegration_analyst import (
    PairsCointegrationAnalyst,
)
from agentic_trading.agents.base import AnalystAgent, DataRequirement
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.backtest import preflight as preflight_module
from agentic_trading.backtest.preflight import check_data_availability

START = date(2020, 3, 1)
END = date(2020, 12, 31)
LAKE_FROM = date(2020, 1, 1)
LAKE_TO = date(2025, 1, 1)


def _analyst_for(source: str, asset_scoped: bool = True) -> AnalystAgent:
    class _Analyst(AnalystAgent):
        agent_id: str = f"{source}_analyst"
        data_requirements: tuple[DataRequirement, ...] = (
            DataRequirement(
                source=source,
                identifier="{ticker}" if asset_scoped else "all",
                asset_scoped=asset_scoped,
                description=f"{source} data",
            ),
        )

        def generate_signal(self, asset, market_data, as_of_date) -> AgentSignal:
            raise NotImplementedError

    return _Analyst()


def _patch_coverage(monkeypatch, by_ticker: dict[str | None, tuple[date, date] | None]) -> list:
    calls: list = []

    def _coverage(source: str, identifier: str | None):
        calls.append((source, identifier))
        return by_ticker.get(identifier)

    monkeypatch.setattr(preflight_module, "source_coverage", _coverage)
    return calls


def test_warmup_days_shift_the_required_start_for_new_and_existing_sources(monkeypatch) -> None:
    """Coverage from 2020-01-01 passes a 2020-03-01 run only when the warm-up fits inside it."""
    _patch_coverage(monkeypatch, {None: (LAKE_FROM, LAKE_TO)})
    fred = pd.Series([1.0, 2.0], index=pd.to_datetime([LAKE_FROM, LAKE_TO]))
    monkeypatch.setattr(preflight_module, "load_fred_series_from_lakehouse", lambda _i, _d: fred)
    membership = _analyst_for("sp500_membership", asset_scoped=False)
    fred_analyst = _analyst_for("fred", asset_scoped=False)

    for analyst in (membership, fred_analyst):
        assert check_data_availability([analyst], ["AAPL"], START, END).failing_analyst_ids == set()
        assert (
            check_data_availability(
                [analyst], ["AAPL"], START, END, warmup_days=0
            ).failing_analyst_ids
            == set()
        )
        failing = check_data_availability([analyst], ["AAPL"], START, END, warmup_days=90)
        assert failing.failing_analyst_ids == {analyst.agent_id}


def test_sp500_membership_is_one_unscoped_check(monkeypatch) -> None:
    """sp500_membership produces exactly one check with no asset, looked up with identifier None."""
    calls = _patch_coverage(monkeypatch, {None: (LAKE_FROM, LAKE_TO)})
    analyst = _analyst_for("sp500_membership", asset_scoped=False)

    report = check_data_availability([analyst], ["AAPL", "MSFT"], START, END)

    assert [check.asset for check in report.results[0].checks] == [None]
    assert calls == [("sp500_membership", None)]
    assert report.results[0].checks[0].available_from == LAKE_FROM
    assert report.results[0].checks[0].available_to == LAKE_TO


@pytest.mark.parametrize("source", ["options_chain", "cftc_cot"])
def test_coverage_scoped_sources_skip_tickers_without_data(monkeypatch, source) -> None:
    """A ticker with no coverage is not checked; partial coverage on a covered ticker fails."""
    _patch_coverage(
        monkeypatch,
        {
            "AAPL": (LAKE_FROM, LAKE_TO),
            "MSFT": (date(2020, 6, 1), LAKE_TO),
            "NVDA": (LAKE_FROM, date(2020, 6, 1)),
            "TSLA": None,
        },
    )
    analyst = _analyst_for(source)

    report = check_data_availability([analyst], ["AAPL", "MSFT", "NVDA", "TSLA"], START, END)

    checks = report.results[0].checks
    assert {check.asset: check.covers_run for check in checks} == {
        "AAPL": True,
        "MSFT": False,
        "NVDA": False,
    }
    summary = report.failure_summary(analyst.agent_id)
    assert f"{source} data for MSFT" in summary
    assert f"{source} data for NVDA" in summary
    assert "TSLA" not in summary

    only_uncovered = check_data_availability([analyst], ["TSLA"], START, END)
    assert only_uncovered.results[0].checks == []
    assert only_uncovered.passing_analyst_ids == {analyst.agent_id}


@pytest.mark.parametrize("source", ["earnings", "dividends"])
def test_event_sources_are_one_table_wide_check(monkeypatch, source) -> None:
    """earnings and dividends are checked once over the whole table, never per ticker,
    so a name with no events (a non-payer) or a late first event never fails a run."""
    calls = _patch_coverage(monkeypatch, {None: (LAKE_FROM, LAKE_TO), "AAPL": None})
    analyst = _analyst_for(source, asset_scoped=False)

    report = check_data_availability([analyst], ["AAPL", "AMZN", "GC=F"], START, END)

    assert [check.asset for check in report.results[0].checks] == [None]
    assert calls == [(source, None)]
    assert report.passing_analyst_ids == {analyst.agent_id}

    short = check_data_availability([analyst], ["AAPL"], START, date(2026, 1, 1))
    assert short.failing_analyst_ids == {analyst.agent_id}
    assert f"{source} data" in short.failure_summary(analyst.agent_id)


def test_sec_sic_checks_only_equities_with_a_code(monkeypatch) -> None:
    """A present SIC code is an undated covered check; a ticker without one, or a
    non-equity, produces no check (an unclassified name simply forms no pair)."""
    seen: list[str] = []

    def _sic(ticker: str):
        seen.append(ticker)
        return "3571" if ticker == "AAPL" else None

    monkeypatch.setattr(preflight_module, "load_sic_code_from_lakehouse", _sic)
    analyst = _analyst_for("sec_sic")

    report = check_data_availability([analyst], ["AAPL", "ZZZZ", "GC=F"], START, END)

    checks = {check.asset: check for check in report.results[0].checks}
    assert set(checks) == {"AAPL"}
    assert "GC=F" not in seen
    assert checks["AAPL"].covers_run is True
    assert checks["AAPL"].available_from is None and checks["AAPL"].available_to is None
    assert report.passing_analyst_ids == {analyst.agent_id}


def test_sec_sic_covered_regardless_of_warmup(monkeypatch) -> None:
    """An undated SIC code covers the run whatever the warm-up length."""
    monkeypatch.setattr(preflight_module, "load_sic_code_from_lakehouse", lambda _t: "3571")

    report = check_data_availability(
        [_analyst_for("sec_sic")], ["AAPL"], START, END, warmup_days=400
    )

    assert report.failing_analyst_ids == frozenset()


@pytest.mark.parametrize(
    ("analyst", "source", "asset_scoped"),
    [
        (BlackScholesValuationAnalyst(), "options_chain", True),
        (GreeksDeltaHedgeAnalyst(), "options_chain", True),
        (HestonVolSurfaceAnalyst(), "options_chain", True),
        (COTPositioningAnalyst(agent_id="cot_positioning_analyst_rates"), "cftc_cot", True),
        (EarningsSurpriseAnalyst(), "earnings", False),
        (DividendPayoutAnalyst(), "dividends", False),
        (LeadLagNetworkCentralityAnalyst(), "sp500_membership", False),
        (FamaMacbethAnalyst(), "sp500_membership", False),
        (PairsCointegrationAnalyst(partner_tickers_by_asset={"MSFT": ("ADBE",)}), "sec_sic", True),
        (KalmanHedgeRatioAnalyst(partner_ticker_by_asset={"MSFT": "ADBE"}), "sec_sic", True),
    ],
)
def test_analysts_declare_the_lake_sources_they_read(analyst, source, asset_scoped) -> None:
    """Each analyst that reads a new lake source declares a matching DataRequirement."""
    matching = [req for req in analyst.data_requirements if req.source == source]

    assert matching
    assert all(req.asset_scoped is asset_scoped for req in matching)
