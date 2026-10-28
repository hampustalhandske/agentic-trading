# test_preflight_cot_scope.py

"""Unit tests for how the preflight scopes CFTC positioning checks.

Covers the Treasury-ETF proxy resolution in `source_coverage("cftc_cot", ...)`
(SHY/IEF/TLT read the Treasury future COT tracks) and the per-instance asset
scope an analyst declares through `AnalystAgent.covers_asset`, which limits
every asset_scoped requirement to the run tickers that instance covers."""

from datetime import date
from typing import Any

import pandas as pd

from agentic_trading.agents.analysts.macro_rates_pod import cot_positioning_analyst
from agentic_trading.agents.analysts.macro_rates_pod.cot_positioning_analyst import (
    COTPositioningAnalyst,
)
from agentic_trading.agents.base import AnalystAgent, DataRequirement
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.backtest import preflight as preflight_module
from agentic_trading.backtest.preflight import check_data_availability
from agentic_trading.data.loaders import lakehouse_loader

START = date(2020, 3, 1)
END = date(2020, 12, 31)
LAKE_FROM = date(2019, 1, 1)
LAKE_TO = date(2025, 1, 1)
_ZB_CODE = "020601"
_EURUSD_CODE = "099741"


def _seed_tff(lake: Any) -> None:
    """Two ZB=F reports, one EURUSD=X report, one ZB=F row with no open interest,
    and each report's release the following Friday."""
    lake.add(
        "cftc_cot_tff",
        pd.DataFrame(
            [
                ("2019-01-01", _ZB_CODE, 1_000),
                ("2024-12-31", _ZB_CODE, 1_000),
                ("2025-06-03", _ZB_CODE, 0),
                ("2015-01-06", _EURUSD_CODE, 1_000),
            ],
            columns=["report_date", "cftc_contract_market_code", "open_interest_all"],
        ),
        date_columns=("report_date",),
    )
    lake.add(
        "cftc_cot_publication",
        pd.DataFrame(
            {
                "report_date": ["2019-01-01", "2024-12-31", "2025-06-03", "2015-01-06"],
                "published_date": ["2019-01-04", "2025-01-03", "2025-06-06", "2015-01-09"],
                "published_at": pd.to_datetime(
                    [
                        "2019-01-04 20:30",
                        "2025-01-03 20:30",
                        "2025-06-06 20:30",
                        "2015-01-09 20:30",
                    ],
                    utc=True,
                ),
            }
        ),
        date_columns=("report_date", "published_date"),
    )


def test_treasury_etf_resolves_to_its_future(fake_lake: Any) -> None:
    """TLT's coverage is the span of ZB=F's published dates."""
    _seed_tff(fake_lake)

    assert lakehouse_loader.source_coverage("cftc_cot", "TLT") == (
        date(2019, 1, 4),
        date(2025, 1, 3),
    )
    assert lakehouse_loader.source_coverage("cftc_cot", "TLT") == lakehouse_loader.source_coverage(
        "cftc_cot", "ZB=F"
    )


def test_analyst_reads_the_loader_proxy_map() -> None:
    """The COT analyst's default remapping is the loader's proxy map."""
    assert (
        cot_positioning_analyst.DEFAULT_COT_TICKER_BY_ASSET
        is lakehouse_loader.DEFAULT_COT_TICKER_BY_ASSET
    )
    rates = COTPositioningAnalyst(agent_id="cot_positioning_analyst_rates")
    assert rates.cot_ticker_by_asset == {"SHY": "ZT=F", "IEF": "ZN=F", "TLT": "ZB=F"}


def test_rates_instance_checks_treasury_etfs_through_their_future(fake_lake: Any) -> None:
    """A run holding TLT but not ZB=F still checks the rates instance's COT coverage."""
    _seed_tff(fake_lake)
    rates = COTPositioningAnalyst(agent_id="cot_positioning_analyst_rates")

    report = check_data_availability([rates], ["TLT", "AAPL"], START, END)

    checks = report.results[0].checks
    assert [check.asset for check in checks] == ["TLT"]
    assert checks[0].available_from == date(2019, 1, 4)
    assert report.passing_analyst_ids == {"cot_positioning_analyst_rates"}

    short = check_data_availability([rates], ["TLT"], START, date(2026, 1, 1))
    assert short.failing_analyst_ids == {"cot_positioning_analyst_rates"}
    assert "for TLT" in short.failure_summary("cot_positioning_analyst_rates")


def test_covered_assets_limit_which_run_tickers_are_checked(monkeypatch) -> None:
    """An instance limited to commodities is not checked (or failed) on FX tickers."""
    calls: list[tuple[str, str | None]] = []

    def _coverage(source: str, identifier: str | None):
        calls.append((source, identifier))
        if identifier == "GC=F":
            return (LAKE_FROM, LAKE_TO)
        return (date(2020, 6, 1), LAKE_TO)

    monkeypatch.setattr(preflight_module, "source_coverage", _coverage)
    commodities = COTPositioningAnalyst(
        agent_id="cot_positioning_analyst_commodities", covered_assets=("GC=F",)
    )
    unrestricted = COTPositioningAnalyst(agent_id="cot_positioning_analyst")

    report = check_data_availability([commodities, unrestricted], ["GC=F", "EURUSD=X"], START, END)

    by_id = {result.analyst_id: result for result in report.results}
    assert [check.asset for check in by_id["cot_positioning_analyst_commodities"].checks] == [
        "GC=F"
    ]
    assert [check.asset for check in by_id["cot_positioning_analyst"].checks] == [
        "GC=F",
        "EURUSD=X",
    ]
    assert report.passing_analyst_ids == {"cot_positioning_analyst_commodities"}
    assert report.failing_analyst_ids == {"cot_positioning_analyst"}
    assert calls.count(("cftc_cot", "EURUSD=X")) == 1


class _AaplOnlyAnalyst(AnalystAgent):
    """Covers only AAPL; reads one per-ticker and one table-wide source."""

    agent_id: str = "aapl_only_analyst"
    data_requirements: tuple[DataRequirement, ...] = (
        DataRequirement(
            source="options_chain",
            identifier="{ticker}",
            asset_scoped=True,
            description="Option chain snapshots",
        ),
        DataRequirement(
            source="sp500_membership",
            identifier="all",
            asset_scoped=False,
            description="S&P 500 membership history",
        ),
    )

    def covers_asset(self, asset: str) -> bool:
        return asset == "AAPL"

    def generate_signal(self, asset, market_data, as_of_date) -> AgentSignal:
        raise NotImplementedError


def test_covers_asset_scopes_asset_requirements_only(monkeypatch) -> None:
    """A `covers_asset` override drops other tickers from asset_scoped checks but
    leaves a table-wide requirement's single check in place."""
    calls: list[tuple[str, str | None]] = []

    def _coverage(source: str, identifier: str | None):
        calls.append((source, identifier))
        return (LAKE_FROM, LAKE_TO)

    monkeypatch.setattr(preflight_module, "source_coverage", _coverage)

    report = check_data_availability([_AaplOnlyAnalyst()], ["AAPL", "MSFT"], START, END)

    assert [check.asset for check in report.results[0].checks] == ["AAPL", None]
    assert calls == [("options_chain", "AAPL"), ("sp500_membership", None)]


class _PlainAnalyst(AnalystAgent):
    agent_id: str = "plain_analyst"

    def generate_signal(self, asset, market_data, as_of_date) -> AgentSignal:
        raise NotImplementedError


def test_default_analyst_covers_every_asset() -> None:
    """Without an override, `covers_asset` is True for any ticker, and an
    unrestricted COT instance covers every asset too."""
    for analyst in (_PlainAnalyst(), COTPositioningAnalyst()):
        assert all(analyst.covers_asset(asset) for asset in ("AAPL", "GC=F", "EURUSD=X"))
