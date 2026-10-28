# preflight.py

"""Real-data preflight — runs once per backtest/experiment, after the
analyst roster and ticker list are both final, and before the first
simulated day. For every analyst's declared `DataRequirement`s
(`agents/base.py`), checks whether the lakehouse actually covers the
run's tickers and its date range, warm-up included, and reports the
result.

This project never calls a live external API during a run (prices,
FRED, Ken French, SEC EDGAR, FINRA, option chains, CFTC positioning,
earnings, dividends and S&P 500 membership are all lakehouse-only
— see `data/loader.py` and `data/loaders/lakehouse_loader.py`), so a
requirement this module reports as unavailable means the lakehouse
doesn't have that data yet, not a fetch that might still succeed
mid-run. Every coverage check below queries the lakehouse as of the
real wall-clock date (`date.today()`), not the run's own `end_date` —
there is no look-ahead concern here, since this check happens once,
before the simulated run starts, and wants the lakehouse's true real
coverage range, not a point-in-time-gated slice of it."""

from datetime import date, timedelta

from pydantic import BaseModel

from agentic_trading.agents.base import AnalystAgent, DataRequirement, is_likely_equity
from agentic_trading.data.loaders.lakehouse_loader import (
    load_company_facts_from_lakehouse,
    load_fama_french_factors_from_lakehouse,
    load_fred_series_from_lakehouse,
    load_insider_transactions_from_lakehouse,
    load_short_interest_from_lakehouse,
    load_sic_code_from_lakehouse,
    source_coverage,
)
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)


class RequirementCoverage(BaseModel):
    """One requirement's real coverage range in the lakehouse, checked
    against one run ticker (`asset`, `None` for an `asset_scoped=False`
    requirement). `required_from` is the first date this requirement needs
    covered (the start date less its `lookback_days`, or less the run's
    warm-up when it has none) when that precedes the run's start date, and
    `run_end` the run's end date; both are `None` when not recorded."""

    requirement: DataRequirement
    asset: str | None
    available_from: date | None
    available_to: date | None
    covers_run: bool
    required_from: date | None = None
    run_end: date | None = None


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
                needs = (
                    f"run needs {check.required_from.isoformat()} to {check.run_end.isoformat()}"
                    if check.required_from is not None and check.run_end is not None
                    else "run needs it too"
                )
                parts.append(
                    f"{label} — lakehouse covers {check.available_from.isoformat()} to "
                    f"{check.available_to.isoformat()}, {needs}"
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


def _available_range(requirement: DataRequirement, asset: str | None) -> tuple[date, date] | None:
    """The real `[first, last]` date range the lake holds for one dated
    requirement on `asset` (`None` for an unscoped requirement), or
    `None` when it holds nothing."""
    today = date.today()
    source = requirement.source
    if source == "fred":
        series = load_fred_series_from_lakehouse(requirement.identifier, today)
        if series is not None and not series.empty:
            return series.index.min().date(), series.index.max().date()
    elif source == "ken_french_factors":
        factors = load_fama_french_factors_from_lakehouse(today, model=requirement.identifier)
        if factors is not None and not factors.empty:
            return factors.index.min().date(), factors.index.max().date()
    elif source == "sec_edgar_facts" and asset is not None:
        facts = load_company_facts_from_lakehouse(asset, today)
        if facts is not None and not facts.empty:
            return facts["filed"].min().date(), facts["filed"].max().date()
    elif source == "finra_short_interest" and asset is not None:
        history = load_short_interest_from_lakehouse(asset, today)
        if history is not None and not history.empty:
            return (
                history["dissemination_date"].min().date(),
                history["dissemination_date"].max().date(),
            )
    elif source == "sec_insider_transactions" and asset is not None:
        filings = load_insider_transactions_from_lakehouse(asset, today)
        if filings is not None and not filings.empty:
            return filings["filed_date"].min().date(), filings["filed_date"].max().date()
    elif source == "sp500_membership":
        return source_coverage("sp500_membership", None)
    elif source == "options_chain" and asset is not None:
        return source_coverage("options_chain", asset)
    elif source == "cftc_cot" and asset is not None:
        return source_coverage("cftc_cot", asset)
    elif source == "earnings":
        return source_coverage("earnings", None)
    elif source == "dividends":
        return source_coverage("dividends", None)
    return None


def _coverage_for_requirement(
    requirement: DataRequirement,
    asset: str | None,
    required_from: date,
    start_date: date,
    end_date: date,
) -> RequirementCoverage | None:
    """One requirement's coverage check on `asset`, judged against the
    window `[required_from, end_date]` (`required_from` is on or before
    `start_date`). An undated source (`sec_sic`)
    covers the run, with both `available_*` dates `None`, when the lake
    holds a record for `asset`. Returns `None`, meaning "not checked",
    for an undated or coverage-scoped source on an asset the lake holds
    nothing for."""
    if requirement.source in _UNDATED_SOURCES:
        if asset is None or load_sic_code_from_lakehouse(asset) is None:
            return None
        return RequirementCoverage(
            requirement=requirement,
            asset=asset,
            available_from=None,
            available_to=None,
            covers_run=True,
        )

    available = _available_range(requirement, asset)
    if available is None and requirement.source in _COVERAGE_SCOPED_SOURCES:
        return None
    available_from, available_to = available if available is not None else (None, None)
    covers_run = (
        available_from is not None
        and available_to is not None
        and available_from <= required_from
        and available_to >= end_date
    )
    return RequirementCoverage(
        requirement=requirement,
        asset=asset,
        available_from=available_from,
        available_to=available_to,
        covers_run=covers_run,
        required_from=required_from if required_from < start_date else None,
        run_end=end_date,
    )


# Asset-scoped sources that only exist for equities (SEC filings and
# classifications, FINRA short interest). An unscoped
# pod (Event/Credit) evaluates every asset in the run, equities and
# commodities/FX/crypto alike, so this filter is what keeps preflight
# from checking (and an analyst from firing on) an asset class these
# sources were never designed for.
_EQUITY_ONLY_SOURCES = frozenset(
    {
        "sec_edgar_facts",
        "finra_short_interest",
        "sec_insider_transactions",
        "sec_sic",
    }
)
# Asset-scoped sources the lake holds only for some tickers of any asset
# class (option chains, CFTC positioning). An analyst reading one reports
# a ticker without rows as unavailable, so preflight checks only the
# tickers the lake holds rows for.
_COVERAGE_SCOPED_SOURCES = frozenset({"options_chain", "cftc_cot"})
# Sources with one current record per ticker and no dates: present means
# covered, whatever the run window, and a ticker without a record is not
# checked (an unclassified name simply forms no pair).
_UNDATED_SOURCES = frozenset({"sec_sic"})


def check_data_availability(
    analysts: list[AnalystAgent],
    assets: list[str],
    start_date: date,
    end_date: date,
    warmup_days: int = 0,
) -> PreflightReport:
    """Checks every analyst's declared `data_requirements` against
    `assets` and the window from each requirement's required start to
    `end_date`: a dated requirement covers the run when the lake's first
    date is on or before `start_date - lookback_days` (its
    `DataRequirement.lookback_days`, or the run's `warmup_days` when that
    is `None`) and its last date on or after `end_date`. An analyst with no requirements
    (anything reading only `market_data`) passes trivially, with zero
    lakehouse calls. An equity-only asset_scoped requirement is only
    checked against the run's equity-like tickers (`is_likely_equity`)
    — a non-equity ticker isn't a gap to report, it's simply out of
    scope for that requirement. A coverage-scoped requirement
    (`options_chain`, `cftc_cot`) is only checked against tickers the
    lake holds rows for, and `sec_sic` only against tickers with a code.
    Every asset_scoped requirement is checked only against tickers the
    analyst instance covers (`AnalystAgent.covers_asset`)."""
    results: list[AnalystPreflightResult] = []
    for analyst in analysts:
        checks: list[RequirementCoverage] = []
        for requirement in analyst.data_requirements:
            lookback_days = (
                warmup_days if requirement.lookback_days is None else requirement.lookback_days
            )
            required_from = start_date - timedelta(days=lookback_days)
            if requirement.asset_scoped:
                scoped_assets: list[str | None] = [
                    asset
                    for asset in assets
                    if analyst.covers_asset(asset)
                    and (requirement.source not in _EQUITY_ONLY_SOURCES or is_likely_equity(asset))
                ]
            else:
                scoped_assets = [None]
            for asset in scoped_assets:
                check = _coverage_for_requirement(
                    requirement, asset, required_from, start_date, end_date
                )
                if check is not None:
                    checks.append(check)
        passed = all(check.covers_run for check in checks)
        results.append(
            AnalystPreflightResult(analyst_id=analyst.agent_id, passed=passed, checks=checks)
        )
    return PreflightReport(checked_at=date.today(), results=results)
