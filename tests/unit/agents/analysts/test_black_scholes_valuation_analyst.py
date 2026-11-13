# test_black_scholes_valuation_analyst.py

"""`BlackScholesValuationAnalyst` prices both legs at the annual DGS3MO rate as known on the
decision day, needs a positive finite volatility on each leg, and dates its signal by the newest
of the chain snapshot and the rate it priced off. The chain reduction, the GARCH leg and the FRED
loader are stubbed on the analyst's own module, so no test reads the lake."""

from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import pytest

from agentic_trading.agents.analysts.cvar_pod import black_scholes_valuation_analyst as module
from agentic_trading.agents.analysts.cvar_pod.black_scholes_valuation_analyst import (
    BlackScholesValuationAnalyst,
)
from agentic_trading.data.loaders.macro_loader import DGS3MO
from agentic_trading.stage_a.decisions import row_source

AS_OF = pd.Timestamp("2024-06-17")
SNAPSHOT = date(2024, 6, 14)


def _market(periods: int = 300) -> pd.DataFrame:
    """A single-asset frame closing at 100 on the `periods` business days ending AS_OF, with a
    mean daily log return of about 1%, far from any annual T-bill rate."""
    returns = np.random.default_rng(0).normal(0.01, 0.01, periods)
    return pd.DataFrame(
        {"close": np.full(periods, 100.0), "log_return": returns},
        index=pd.bdate_range(end=AS_OF, periods=periods),
    )


def _options(implied_volatility: float, snapshot: date = SNAPSHOT) -> pd.DataFrame:
    """A `real_atm_options_by_expiration` frame: one 30-day expiration at `implied_volatility`."""
    return pd.DataFrame(
        {
            "implied_volatility": [implied_volatility],
            "days_to_expiration": [30],
            "snapshot_date": [pd.Timestamp(snapshot)],
        }
    )


def _rates(values: list[float | None], known_at: list[str]) -> pd.DataFrame:
    """A `load_fred_vintages_from_lakehouse` frame of DGS3MO in percent, one row per value, each
    observed and published the business day before it is known."""
    known = pd.to_datetime(known_at)
    observed = known - pd.offsets.BDay(1)
    return pd.DataFrame(
        {"value": values, "realtime_start": known, "known_at": known},
        index=pd.DatetimeIndex(observed, name="date"),
    )


def _serve(
    monkeypatch: pytest.MonkeyPatch,
    *,
    daily_volatility: float = 0.01,
    implied_volatility: float = 0.20,
    rates: pd.DataFrame | None = None,
    snapshot: date = SNAPSHOT,
) -> list[tuple[str, date]]:
    """Stubs the chain, the GARCH leg and the FRED read; returns the FRED reads made."""
    reads: list[tuple[str, date]] = []

    def _load(series_id: str, as_of_date: date) -> pd.DataFrame | None:
        reads.append((series_id, as_of_date))
        return rates

    monkeypatch.setattr(
        module, "real_atm_options_by_expiration", lambda *_: _options(implied_volatility, snapshot)
    )
    monkeypatch.setattr(module, "conditional_daily_volatility", lambda *_a, **_k: daily_volatility)
    monkeypatch.setattr(module, "load_fred_vintages_from_lakehouse", _load, raising=False)
    return reads


def _recording_pricers(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Wraps both pricers to record the rate each call receives."""
    seen: list[float] = []
    call, put = module.black_scholes_call, module.black_scholes_put

    def _wrap(pricer: Any) -> Any:
        def _priced(spot: float, strike: float, rate: float, vol: float, tau: float) -> float:
            seen.append(rate)
            return float(pricer(spot, strike, rate, vol, tau))

        return _priced

    monkeypatch.setattr(module, "black_scholes_call", _wrap(call))
    monkeypatch.setattr(module, "black_scholes_put", _wrap(put))
    return seen


_KNOWN_RATES = _rates([5.40, 5.25, 5.30], ["2024-06-12", "2024-06-13", "2024-06-14"])


def test_prices_at_the_annual_rate_known_on_the_day(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both legs price at the latest DGS3MO value as known on the decision day, in decimals."""
    reads = _serve(monkeypatch, rates=_KNOWN_RATES)
    seen = _recording_pricers(monkeypatch)

    signal = BlackScholesValuationAnalyst().generate_signal("AAPL", _market(), AS_OF)

    assert reads == [(DGS3MO, AS_OF.date())]
    assert seen and all(rate == pytest.approx(0.0530) for rate in seen)
    assert signal.status == "active"


def test_the_latest_known_rate_skips_a_missing_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """A newest observation with no value (a FRED holiday row) leaves the previous value."""
    rates = _rates([5.25, None], ["2024-06-13", "2024-06-14"])
    _serve(monkeypatch, rates=rates)
    seen = _recording_pricers(monkeypatch)

    BlackScholesValuationAnalyst().generate_signal("AAPL", _market(), AS_OF)

    assert seen and all(rate == pytest.approx(0.0525) for rate in seen)


@pytest.mark.parametrize("rates", [None, _rates([None], ["2024-06-14"])])
def test_no_known_rate_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, rates: pd.DataFrame | None
) -> None:
    _serve(monkeypatch, rates=rates)

    signal = BlackScholesValuationAnalyst().generate_signal("AAPL", _market(), AS_OF)

    assert (signal.status, signal.direction, signal.confidence) == ("unavailable", "flat", 0.0)


def test_source_as_of_is_the_newer_of_snapshot_and_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    """The rate known after the snapshot dates the signal; a snapshot newer than the rate does."""
    later_rate = _rates([5.30, 5.31], ["2024-06-13", "2024-06-15"])
    _serve(monkeypatch, rates=later_rate)
    assert BlackScholesValuationAnalyst().generate_signal(
        "AAPL", _market(), AS_OF
    ).source_as_of == date(2024, 6, 15)

    _serve(monkeypatch, rates=_rates([5.30], ["2024-06-11"]))
    assert (
        BlackScholesValuationAnalyst().generate_signal("AAPL", _market(), AS_OF).source_as_of
        == SNAPSHOT
    )


def test_declares_fred_dgs3mo() -> None:
    requirements = BlackScholesValuationAnalyst().data_requirements
    fred = [r for r in requirements if r.source == "fred"]
    assert [(r.identifier, r.asset_scoped) for r in fred] == [(DGS3MO, False)]
    assert any(r.source == "options_chain" for r in requirements)
    assert row_source(BlackScholesValuationAnalyst()) == "fred"


@pytest.mark.parametrize("daily_volatility", [0.0, -0.01, float("nan"), float("inf")])
def test_a_degenerate_fitted_volatility_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, daily_volatility: float
) -> None:
    _serve(monkeypatch, daily_volatility=daily_volatility, rates=_KNOWN_RATES)

    signal = BlackScholesValuationAnalyst().generate_signal("AAPL", _market(), AS_OF)

    assert (signal.status, signal.direction, signal.confidence) == ("unavailable", "flat", 0.0)


@pytest.mark.parametrize("implied_volatility", [0.0, -0.2, float("nan"), float("inf")])
def test_a_degenerate_chain_volatility_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, implied_volatility: float
) -> None:
    _serve(monkeypatch, implied_volatility=implied_volatility, rates=_KNOWN_RATES)

    signal = BlackScholesValuationAnalyst().generate_signal("AAPL", _market(), AS_OF)

    assert (signal.status, signal.direction, signal.confidence) == ("unavailable", "flat", 0.0)
    assert signal.source_as_of == SNAPSHOT
