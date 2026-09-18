# preflight.py

"""Real-data preflight — runs once per backtest/experiment, after the
analyst roster and ticker list are both final, and before the first
simulated day. For every analyst's declared `DataRequirement`s
(`agents/base.py`), checks whether the lakehouse actually covers the
run's tickers/date range and reports the result.

This project never calls a live external API during a run (prices,
FRED, Ken French, SEC EDGAR, and FINRA are all lakehouse-only
— see `data/loader.py` and `data/loaders/lakehouse_loader.py`), so a
requirement this module reports as unavailable means the lakehouse
doesn't have that data yet, not a fetch that might still succeed
mid-run. Every coverage check below queries the lakehouse as of the
real wall-clock date (`date.today()`), not the run's own `end_date` —
there is no look-ahead concern here, since this check happens once,
before the simulated run starts, and wants the lakehouse's true real
coverage range, not a point-in-time-gated slice of it."""

from datetime import date

from pydantic import BaseModel

from agentic_trading.agents.base import AnalystAgent, DataRequirement, is_likely_equity
from agentic_trading.data.loaders.lakehouse_loader import (
    load_company_facts_from_lakehouse,
    load_fama_french_factors_from_lakehouse,
    load_fred_series_from_lakehouse,
    load_insider_transactions_from_lakehouse,
    load_short_interest_from_lakehouse,
)
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)


class RequirementCoverage(BaseModel):
    """One requirement's real coverage range in the lakehouse, checked
    against one run ticker (`asset`, `None` for an `asset_scoped=False`
    requirement)."""

    requirement: DataRequirement
    asset: str | None
    available_from: date | None
    available_to: date | None
    covers_run: bool


class AnalystPreflightResult(BaseModel):
    analyst_id: str
    passed: bool
    checks: list[RequirementCoverage]

    @property
    def failure_summary(self) -> str:
        failing = [check for check in self.checks if not check.covers_run]
        if not failing:
            return f"{self.analyst_id}: ok"
        parts = []
        for check in failing:
            label = check.requirement.description
            if check.asset is not None:
                label = f"{label} for {check.asset}"
            if check.available_from is None or check.available_to is None:
                parts.append(f"{label} — no coverage in the lakehouse")
            else:
                parts.append(
                    f"{label} — lakehouse covers {check.available_from.isoformat()} to "
                    f"{check.available_to.isoformat()}, run needs it too"
                )
        return f"{self.analyst_id} cannot run fully: " + "; ".join(parts)


class PreflightReport(BaseModel):
    checked_at: date
    results: list[AnalystPreflightResult]

    @property
    def failing_analyst_ids(self) -> frozenset[str]:
        return frozenset(result.analyst_id for result in self.results if not result.passed)

    @property
    def passing_analyst_ids(self) -> frozenset[str]:
        return frozenset(result.analyst_id for result in self.results if result.passed)

    def failure_summary(self, analyst_id: str) -> str:
        for result in self.results:
            if result.analyst_id == analyst_id:
                return result.failure_summary
        return f"{analyst_id}: not checked"


def _coverage_for_requirement(
    requirement: DataRequirement, asset: str | None, start_date: date, end_date: date
) -> RequirementCoverage:
    today = date.today()
    available_from: date | None = None
    available_to: date | None = None

    if requirement.source == "fred":
        series = load_fred_series_from_lakehouse(requirement.identifier, today)
        if series is not None and not series.empty:
            available_from = series.index.min().date()
            available_to = series.index.max().date()
    elif requirement.source == "ken_french_factors":
        factors = load_fama_french_factors_from_lakehouse(today, model=requirement.identifier)
        if factors is not None and not factors.empty:
            available_from = factors.index.min().date()
            available_to = factors.index.max().date()
    elif requirement.source == "sec_edgar_facts" and asset is not None:
        facts = load_company_facts_from_lakehouse(asset, today)
        if facts is not None and not facts.empty:
            available_from = facts["filed"].min().date()
            available_to = facts["filed"].max().date()
    elif requirement.source == "finra_short_interest" and asset is not None:
        history = load_short_interest_from_lakehouse(asset, today)
        if history is not None and not history.empty:
            available_from = history["dissemination_date"].min().date()
            available_to = history["dissemination_date"].max().date()
    elif requirement.source == "sec_insider_transactions" and asset is not None:
        filings = load_insider_transactions_from_lakehouse(asset, today)
        if filings is not None and not filings.empty:
            available_from = filings["filed_date"].min().date()
            available_to = filings["filed_date"].max().date()

    covers_run = (
        available_from is not None
        and available_to is not None
        and available_from <= start_date
        and available_to >= end_date
    )
    return RequirementCoverage(
        requirement=requirement,
        asset=asset,
        available_from=available_from,
        available_to=available_to,
        covers_run=covers_run,
    )


# Every asset_scoped source currently in this taxonomy happens to be
# equity-only (SEC filings, FINRA short interest) — an unscoped pod
# (Event/Credit) evaluates every asset in the run, equities and
# commodities/FX/crypto alike, so this filter is what keeps preflight
# from checking (and an analyst from firing on) an asset class these
# sources were never designed for. If a genuinely asset-class-agnostic
# asset_scoped source is ever added, it does not belong in this set.
_EQUITY_ONLY_SOURCES = frozenset(
    {"sec_edgar_facts", "finra_short_interest", "sec_insider_transactions"}
)


def check_data_availability(
    analysts: list[AnalystAgent], assets: list[str], start_date: date, end_date: date
) -> PreflightReport:
    """Checks every analyst's declared `data_requirements` against
    `assets`/`[start_date, end_date]`. An analyst with no requirements
    (the majority — anything reading only `market_data` or an
    already-graceful lakehouse table) passes trivially, with zero
    lakehouse calls. An equity-only asset_scoped requirement is only
    checked against the run's equity-like tickers (`is_likely_equity`)
    — a non-equity ticker isn't a gap to report, it's simply out of
    scope for that requirement."""
    results: list[AnalystPreflightResult] = []
    for analyst in analysts:
        checks: list[RequirementCoverage] = []
        for requirement in analyst.data_requirements:
            if requirement.asset_scoped:
                scoped_assets = (
                    [asset for asset in assets if is_likely_equity(asset)]
                    if requirement.source in _EQUITY_ONLY_SOURCES
                    else assets
                )
                checks.extend(
                    _coverage_for_requirement(requirement, asset, start_date, end_date)
                    for asset in scoped_assets
                )
            else:
                checks.append(_coverage_for_requirement(requirement, None, start_date, end_date))
        passed = all(check.covers_run for check in checks)
        results.append(
            AnalystPreflightResult(analyst_id=analyst.agent_id, passed=passed, checks=checks)
        )
    return PreflightReport(checked_at=date.today(), results=results)
