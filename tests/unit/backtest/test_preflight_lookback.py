# test_preflight_lookback.py

"""Unit tests for per-requirement lookbacks in agentic_trading.backtest.preflight.

A requirement's `lookback_days` sets how far before `start_date` the lake must
cover it, in place of the run's warm-up; `None` falls back to the warm-up. The
failure summary names each failing requirement's own required start."""

from datetime import date, timedelta

import pandas as pd
import pytest

from agentic_trading.agents.analysts.alt_data_pod.insider_cluster_buying_analyst import (
    InsiderClusterBuyingAnalyst,
)
from agentic_trading.agents.base import AnalystAgent, DataRequirement
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.backtest import preflight as preflight_module
from agentic_trading.backtest.preflight import check_data_availability
from agentic_trading.config import temporary_overrides

START = date(2010, 1, 4)
END = date(2017, 12, 29)
LAKE_TO = date(2025, 1, 1)
WARMUP_DAYS = 1100


def _analyst(*requirements: DataRequirement) -> AnalystAgent:
    class _Analyst(AnalystAgent):
        agent_id: str = "lookback_analyst"
        data_requirements: tuple[DataRequirement, ...] = requirements

        def generate_signal(self, asset, market_data, as_of_date) -> AgentSignal:
            raise NotImplementedError

    return _Analyst()


def _fred(series_id: str, lookback_days: int | None) -> DataRequirement:
    return DataRequirement(
        source="fred",
        identifier=series_id,
        asset_scoped=False,
        description=f"FRED {series_id}",
        lookback_days=lookback_days,
    )


def _patch_fred(monkeypatch, first_by_series: dict[str, date]) -> None:
    """Each FRED series covers from its `first_by_series` date to `LAKE_TO`."""

    def _load(series_id: str, _as_of: date) -> pd.Series:
        return pd.Series([1.0, 2.0], index=pd.to_datetime([first_by_series[series_id], LAKE_TO]))

    monkeypatch.setattr(preflight_module, "load_fred_series_from_lakehouse", _load)


@pytest.mark.parametrize(
    ("lookback_days", "lake_from", "covered"),
    [
        (30, START - timedelta(days=30), True),
        (30, START - timedelta(days=29), False),
        (0, START, True),
        (0, START + timedelta(days=1), False),
        (2000, START - timedelta(days=1500), False),
    ],
)
def test_lookback_overrides_the_warmup(monkeypatch, lookback_days, lake_from, covered) -> None:
    """With `lookback_days` set, the required start is `start_date - lookback_days`
    whatever the warm-up, shorter or longer than it."""
    _patch_fred(monkeypatch, {"DGS3MO": lake_from})
    analyst = _analyst(_fred("DGS3MO", lookback_days))

    report = check_data_availability([analyst], ["AAPL"], START, END, warmup_days=WARMUP_DAYS)

    assert report.results[0].checks[0].covers_run is covered


@pytest.mark.parametrize(
    ("lake_from", "covered"),
    [
        (START - timedelta(days=WARMUP_DAYS), True),
        (START - timedelta(days=WARMUP_DAYS - 1), False),
    ],
)
def test_no_lookback_falls_back_to_the_warmup(monkeypatch, lake_from, covered) -> None:
    """With `lookback_days` None, the required start is `start_date - warmup_days`."""
    _patch_fred(monkeypatch, {"VIXCLS": lake_from})
    analyst = _analyst(_fred("VIXCLS", None))

    report = check_data_availability([analyst], ["AAPL"], START, END, warmup_days=WARMUP_DAYS)

    assert report.results[0].checks[0].covers_run is covered


def test_summary_names_each_requirements_own_required_start(monkeypatch) -> None:
    """Failing requirements with different lookbacks each report their own start;
    one whose required start is the run's start keeps the plain wording."""
    lake_from = START + timedelta(days=10)
    _patch_fred(monkeypatch, {"DGS3MO": lake_from, "DGS10": lake_from, "VIXCLS": lake_from})
    analyst = _analyst(_fred("DGS3MO", 90), _fred("DGS10", None), _fred("VIXCLS", 0))

    report = check_data_availability([analyst], ["AAPL"], START, END, warmup_days=WARMUP_DAYS)

    summary = report.failure_summary(analyst.agent_id)
    covers = f"lakehouse covers {lake_from.isoformat()} to {LAKE_TO.isoformat()}"
    lookback_start = (START - timedelta(days=90)).isoformat()
    warmup_start = (START - timedelta(days=WARMUP_DAYS)).isoformat()
    assert f"FRED DGS3MO — {covers}, run needs {lookback_start} to {END.isoformat()}" in summary
    assert f"FRED DGS10 — {covers}, run needs {warmup_start} to {END.isoformat()}" in summary
    assert f"FRED VIXCLS — {covers}, run needs it too" in summary
    by_series = {check.requirement.identifier: check for check in report.results[0].checks}
    assert by_series["DGS3MO"].required_from == START - timedelta(days=90)
    assert by_series["DGS10"].required_from == START - timedelta(days=WARMUP_DAYS)
    assert by_series["VIXCLS"].required_from is None


def test_insider_analyst_passes_a_2010_run_on_2009_filings(monkeypatch) -> None:
    """Form 4 filings from 2009 cover a 2010 run under the full warm-up, because the
    analyst's lookback is its insider window rather than the warm-up."""
    filings = pd.DataFrame({"filed_date": pd.to_datetime([date(2009, 6, 1), LAKE_TO])})
    monkeypatch.setattr(
        preflight_module, "load_insider_transactions_from_lakehouse", lambda _t, _d: filings
    )

    report = check_data_availability(
        [InsiderClusterBuyingAnalyst()], ["AAPL"], START, END, warmup_days=WARMUP_DAYS
    )

    assert report.passing_analyst_ids == {"insider_cluster_buying_analyst"}


def test_settings_derived_lookback_follows_overrides_at_construction() -> None:
    """An analyst built under an override declares the overridden window."""
    with temporary_overrides({"analysts.insider_window_days": 45}):
        overridden = InsiderClusterBuyingAnalyst()
    default = InsiderClusterBuyingAnalyst()

    assert overridden.data_requirements[0].lookback_days == 45
    assert default.data_requirements[0].lookback_days == 30
