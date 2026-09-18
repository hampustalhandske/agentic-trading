"""Unit tests for agentic_trading.backtest.preflight.

Covers the existence-check preflight: an analyst with no declared
`data_requirements` always passes trivially, an asset-scoped requirement
is checked once per run ticker, and coverage is judged against the
requirement's real [available_from, available_to] range, not a 1-2-date
sample."""

from datetime import date

import pandas as pd

from agentic_trading.agents.base import AnalystAgent, DataRequirement
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.backtest import preflight as preflight_module
from agentic_trading.backtest.preflight import check_data_availability

START = date(2024, 1, 1)
END = date(2024, 12, 31)


class _PricesOnlyAnalyst(AnalystAgent):
    agent_id: str = "prices_only_analyst"

    def generate_signal(self, asset, market_data, as_of_date) -> AgentSignal:
        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction="flat",
            confidence=0.0,
            rationale="unused in this test",
        )


class _FredAnalyst(AnalystAgent):
    agent_id: str = "fred_analyst"
    data_requirements: tuple[DataRequirement, ...] = (
        DataRequirement(
            source="fred", identifier="DGS3MO", asset_scoped=False, description="FRED DGS3MO"
        ),
    )

    def generate_signal(self, asset, market_data, as_of_date) -> AgentSignal:
        raise NotImplementedError


class _SecFactsAnalyst(AnalystAgent):
    agent_id: str = "sec_facts_analyst"
    data_requirements: tuple[DataRequirement, ...] = (
        DataRequirement(
            source="sec_edgar_facts",
            identifier="{ticker}",
            asset_scoped=True,
            description="SEC EDGAR company facts",
        ),
    )

    def generate_signal(self, asset, market_data, as_of_date) -> AgentSignal:
        raise NotImplementedError


def test_analyst_with_no_requirements_always_passes_with_zero_lakehouse_calls(monkeypatch) -> None:
    def _explode(*_a, **_k):
        raise AssertionError("should never be called for an analyst with no data_requirements")

    monkeypatch.setattr(preflight_module, "load_fred_series_from_lakehouse", _explode)

    report = check_data_availability([_PricesOnlyAnalyst()], ["AAPL"], START, END)

    assert report.passing_analyst_ids == frozenset({"prices_only_analyst"})
    assert report.failing_analyst_ids == frozenset()


def test_asset_independent_requirement_passes_when_lakehouse_covers_the_full_run(
    monkeypatch,
) -> None:
    full_range = pd.Series([1.0, 2.0], index=pd.to_datetime([date(2020, 1, 1), date(2025, 1, 1)]))
    monkeypatch.setattr(
        preflight_module, "load_fred_series_from_lakehouse", lambda _id, _d: full_range
    )

    report = check_data_availability([_FredAnalyst()], ["AAPL"], START, END)

    assert report.passing_analyst_ids == frozenset({"fred_analyst"})


def test_asset_independent_requirement_fails_when_coverage_is_partial(monkeypatch) -> None:
    partial_range = pd.Series(
        [1.0, 2.0], index=pd.to_datetime([date(2024, 6, 1), date(2024, 6, 30)])
    )
    monkeypatch.setattr(
        preflight_module, "load_fred_series_from_lakehouse", lambda _id, _d: partial_range
    )

    report = check_data_availability([_FredAnalyst()], ["AAPL"], START, END)

    assert report.failing_analyst_ids == frozenset({"fred_analyst"})
    assert "FRED DGS3MO" in report.failure_summary("fred_analyst")


def test_asset_independent_requirement_fails_when_lakehouse_has_no_coverage_at_all(
    monkeypatch,
) -> None:
    monkeypatch.setattr(preflight_module, "load_fred_series_from_lakehouse", lambda _id, _d: None)

    report = check_data_availability([_FredAnalyst()], ["AAPL"], START, END)

    assert report.failing_analyst_ids == frozenset({"fred_analyst"})
    assert "no coverage in the lakehouse" in report.failure_summary("fred_analyst")


def test_asset_scoped_requirement_is_checked_once_per_run_ticker(monkeypatch) -> None:
    calls: list[str] = []

    def _facts(ticker: str, _as_of_date):
        calls.append(ticker)
        if ticker == "AAPL":
            return pd.DataFrame(
                {"filed": pd.to_datetime([date(2020, 1, 1), date(2025, 1, 1)]), "value": [1.0, 2.0]}
            )
        return None

    monkeypatch.setattr(preflight_module, "load_company_facts_from_lakehouse", _facts)

    report = check_data_availability([_SecFactsAnalyst()], ["AAPL", "MSFT"], START, END)

    assert set(calls) == {"AAPL", "MSFT"}
    assert report.failing_analyst_ids == frozenset({"sec_facts_analyst"})
    assert "MSFT" in report.failure_summary("sec_facts_analyst")
