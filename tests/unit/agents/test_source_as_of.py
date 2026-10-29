# test_source_as_of.py

"""AT-118 item 1: each round-1 sourced analyst sets `AgentSignal.source_as_of` on every active
signal to the source's own publication date of the newest data the signal rests on, never the
later date that data becomes usable. A signal with no data to rest on leaves it `None`.

The FRED and CFTC analysts read a `fake_lake`, because their loaders return values without a
publication date to stub; the others stub their loader on the analyst's own module."""

from collections.abc import Callable
from datetime import date
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest

from agentic_trading.agents.analysts import (
    CIRRateRegimeAnalyst,
    CreditSpreadSignalAnalyst,
    MacroVarSpilloverAnalyst,
)
from agentic_trading.agents.analysts.alt_data_pod import network_centrality_analyst
from agentic_trading.agents.analysts.factor_pod import (
    earnings_surprise_analyst,
    fama_french_factor_analyst,
    fama_macbeth_analyst,
)
from agentic_trading.agents.analysts.macro_rates_pod.cot_positioning_analyst import (
    COTPositioningAnalyst,
)
from agentic_trading.data.loaders.lakehouse_loader import IndexMember

_VINTAGE_DATES = ("date", "realtime_start")
_ZB, _ZN = "020601", "043602"


def _returning(value: Any) -> Callable[..., Any]:
    """A stand-in loader that ignores its arguments and returns `value`."""
    return lambda *_args, **_kwargs: value


def _vintages(series_id: str, published_end: str, seed: int, periods: int = 260) -> pd.DataFrame:
    """`periods` business-day observations of `series_id` oscillating around 3.0, each first
    published the next business day, the last on `published_end`."""
    days = pd.bdate_range(end=published_end, periods=periods + 1)
    wave = 0.3 * np.sin(np.linspace(0.0, 12.0 * np.pi, periods))
    noise = np.random.default_rng(seed).normal(0.0, 0.02, periods)
    return pd.DataFrame(
        {
            "series_id": series_id,
            "date": days[:-1],
            "realtime_start": days[1:],
            "value": 3.0 + wave + noise,
        }
    )


def _market(end: str, periods: int = 300, seed: int = 0) -> pd.DataFrame:
    """A single-asset frame of `periods` business-day closes and log returns ending `end`."""
    returns = np.random.default_rng(seed).normal(0.0, 0.01, periods)
    return pd.DataFrame(
        {"close": 100.0 * np.exp(np.cumsum(returns)), "log_return": returns},
        index=pd.bdate_range(end=end, periods=periods),
    )


def _cot(lake: Any, reports: pd.DatetimeIndex, timed_newest: bool, code: str = _ZB) -> None:
    """Weekly TFF reports for `code` on `reports`, each released the Friday after at 15:30 ET;
    the newest carries no release time unless `timed_newest`."""
    published = reports + pd.Timedelta(days=3)
    stamps: list[str | None] = [f"{day.date()} 19:30" for day in published]
    if not timed_newest:
        stamps[-1] = None
    long = np.random.default_rng(6).integers(100, 400, reports.size)
    tff = pd.DataFrame(
        {
            "cftc_contract_market_code": code,
            "report_date": reports,
            "open_interest_all": 1000,
            "leveraged_funds_long": long,
            "leveraged_funds_short": 100,
        }
    )
    publication = pd.DataFrame(
        {
            "report_date": reports,
            "published_date": published,
            "published_at": pd.to_datetime(stamps, utc=True),
            "rule_date": published,
            "basis": "test",
        }
    )
    lake.add("cftc_cot_tff", tff, date_columns=("report_date",))
    lake.add(
        "cftc_cot_publication",
        publication,
        date_columns=("report_date", "published_date", "rule_date"),
    )


def _factors(start: str, end: str, seed: int) -> pd.DataFrame:
    """Daily Ken French 3-factor returns on the business days from `start` to `end`."""
    days = pd.bdate_range(start, end)
    values = np.random.default_rng(seed).normal(0.0, 0.01, (days.size, 3))
    frame = pd.DataFrame(values, index=days, columns=["Mkt-RF", "SMB", "HML"])
    frame["RF"] = 0.0001
    return frame


def _universe(end: str, periods: int, seed: int) -> pd.DataFrame:
    """Daily log returns of tickers T000 to T034 on the `periods` business days ending `end`."""
    values = np.random.default_rng(seed).normal(0.0, 0.01, (periods, 35))
    columns = [f"T{n:03d}" for n in range(35)]
    return pd.DataFrame(values, index=pd.bdate_range(end=end, periods=periods), columns=columns)


def _members(monkeypatch: pytest.MonkeyPatch, module: ModuleType, returns: pd.DataFrame) -> None:
    """Serves `returns`' columns as the S&P 500 members on `module`, security ids by position,
    each with its column as its security-keyed log returns."""
    members = [IndexMember(security_id=n, ticker=t) for n, t in enumerate(returns.columns)]
    by_security = {member.security_id: returns[member.ticker] for member in members}
    monkeypatch.setattr(module, "load_sp500_member_securities_asof", _returning(members))
    monkeypatch.setattr(module, "load_security_log_returns", _returning(by_security))


def test_cir_rate_regime(fake_lake: Any) -> None:
    """The realtime_start of the newest DGS3MO vintage used dates the signal; no series, None."""
    as_of = pd.Timestamp("2010-06-15")
    fake_lake.add("fred_vintages", _vintages("DGS3MO", "2010-06-15", seed=0), _VINTAGE_DATES)
    signal = CIRRateRegimeAnalyst().generate_signal("TLT", pd.DataFrame(), as_of)

    assert signal.status == "active"
    assert signal.source_as_of == date(2010, 6, 14)

    fake_lake.add("fred_vintages", _vintages("DGS10", "2010-06-15", seed=0), _VINTAGE_DATES)
    missing = CIRRateRegimeAnalyst().generate_signal("TLT", pd.DataFrame(), as_of)
    assert missing.status == "unavailable"
    assert missing.source_as_of is None


def test_macro_var_spillover(fake_lake: Any) -> None:
    """The latest realtime_start across VIXCLS, DGS10 and DGS3MO dates the signal."""
    as_of = pd.Timestamp("2010-06-15")
    market = _market("2010-06-15")
    series = {
        name: _vintages(name, "2010-06-11", seed=seed)
        for seed, name in enumerate(("VIXCLS", "DGS10", "DGS3MO"), start=1)
    }
    series["DGS10"].loc[series["DGS10"].index[-1], "realtime_start"] = pd.Timestamp("2010-06-14")
    every_series = pd.concat(list(series.values()), ignore_index=True)
    fake_lake.add("fred_vintages", every_series, _VINTAGE_DATES)
    signal = MacroVarSpilloverAnalyst().generate_signal("SPY", market, as_of)

    assert signal.status == "active"
    assert signal.source_as_of == date(2010, 6, 14)

    without_dgs10 = pd.concat([series["VIXCLS"], series["DGS3MO"]], ignore_index=True)
    fake_lake.add("fred_vintages", without_dgs10, _VINTAGE_DATES)
    missing = MacroVarSpilloverAnalyst().generate_signal("SPY", market, as_of)
    assert missing.status == "unavailable"
    assert missing.source_as_of is None


def test_credit_spread_signal(fake_lake: Any) -> None:
    """The realtime_start of the newest BAMLH0A0HYM2 vintage used dates the signal."""
    as_of = pd.Timestamp("2024-06-17")
    spread = _vintages("BAMLH0A0HYM2", "2024-06-17", seed=4, periods=150)
    fake_lake.add("fred_vintages", spread, _VINTAGE_DATES)
    signal = CreditSpreadSignalAnalyst().generate_signal("HYG", pd.DataFrame(), as_of)

    assert signal.status == "active"
    assert signal.source_as_of == date(2024, 6, 14)

    fake_lake.add("fred_vintages", _vintages("DGS10", "2024-06-17", seed=4), _VINTAGE_DATES)
    missing = CreditSpreadSignalAnalyst().generate_signal("HYG", pd.DataFrame(), as_of)
    assert missing.status == "unavailable"
    assert missing.source_as_of is None


def test_cot_positioning(fake_lake: Any) -> None:
    """The CFTC release date of the newest report used dates the signal, not its usable date."""
    analyst = COTPositioningAnalyst(agent_id="cot_positioning_analyst_rates")
    reports = pd.date_range(end="2024-06-11", periods=30, freq="W-TUE")
    cases = [
        (True, "2024-06-14", date(2024, 6, 14)),
        (False, "2024-06-14", date(2024, 6, 7)),
        (False, "2024-06-17", date(2024, 6, 14)),
    ]
    for timed_newest, day, expected in cases:
        _cot(fake_lake, reports, timed_newest)
        signal = analyst.generate_signal("TLT", pd.DataFrame(), pd.Timestamp(day))
        assert signal.status == "active", (timed_newest, day)
        assert signal.source_as_of == expected, (timed_newest, day)

    _cot(fake_lake, reports, timed_newest=True, code=_ZN)
    missing = analyst.generate_signal("TLT", pd.DataFrame(), pd.Timestamp("2024-06-14"))
    assert missing.status == "unavailable"
    assert missing.source_as_of is None


def test_earnings_surprise(monkeypatch: pytest.MonkeyPatch) -> None:
    """The newest announcement date dates the signal, inside and past the drift window."""
    module = earnings_surprise_analyst
    as_of = pd.Timestamp("2024-02-05")
    for announced, direction in (("2024-01-25", "long"), ("2023-10-26", "flat")):
        result = (1.10, 1.00, pd.Timestamp(announced))
        monkeypatch.setattr(module, "load_earnings_surprise_from_lakehouse", _returning(result))
        signal = module.EarningsSurpriseAnalyst().generate_signal("AAPL", pd.DataFrame(), as_of)
        assert (signal.status, signal.direction) == ("active", direction), announced
        assert signal.source_as_of == date.fromisoformat(announced)

    monkeypatch.setattr(module, "load_earnings_surprise_from_lakehouse", _returning(None))
    missing = module.EarningsSurpriseAnalyst().generate_signal("AAPL", pd.DataFrame(), as_of)
    assert missing.status == "unavailable"
    assert missing.source_as_of is None


def test_fama_french_factor(monkeypatch: pytest.MonkeyPatch) -> None:
    """The newest factor row's public date, the last day of the next month, dates the signal."""
    module = fama_french_factor_analyst
    as_of, market = pd.Timestamp("2022-06-15"), _market("2022-06-15")
    factors = _factors("2021-05-03", "2022-04-29", seed=7)
    monkeypatch.setattr(module, "load_fama_french_factors_from_lakehouse", _returning(factors))
    signal = module.FamaFrenchFactorAnalyst().generate_signal("AAPL", market, as_of)

    assert signal.status == "active"
    assert signal.source_as_of == date(2022, 5, 31)

    monkeypatch.setattr(module, "load_fama_french_factors_from_lakehouse", _returning(None))
    missing = module.FamaFrenchFactorAnalyst().generate_signal("AAPL", market, as_of)
    assert missing.status == "unavailable"
    assert missing.source_as_of is None


def test_fama_macbeth(monkeypatch: pytest.MonkeyPatch) -> None:
    """The newer of the factors' public date and the latest price bar used dates the signal."""
    module = fama_macbeth_analyst
    as_of = pd.Timestamp("2022-06-15")
    factors = _factors("2021-05-03", "2022-04-29", seed=8)
    monkeypatch.setattr(module, "load_fama_french_factors_from_lakehouse", _returning(factors))

    _members(monkeypatch, module, _universe("2022-05-27", periods=250, seed=9))
    signal = module.FamaMacbethAnalyst().generate_signal("T000", pd.DataFrame(), as_of)
    assert signal.status == "active"
    assert signal.source_as_of == date(2022, 5, 31)

    _members(monkeypatch, module, _universe("2022-06-15", periods=250, seed=9))
    later = module.FamaMacbethAnalyst().generate_signal("T000", pd.DataFrame(), as_of)
    assert later.status == "active"
    assert later.source_as_of is not None
    assert date(2022, 5, 31) <= later.source_as_of <= date(2022, 6, 15)

    monkeypatch.setattr(module, "load_fama_french_factors_from_lakehouse", _returning(None))
    missing = module.FamaMacbethAnalyst().generate_signal("T000", pd.DataFrame(), as_of)
    assert missing.status == "unavailable"
    assert missing.source_as_of is None


def test_lead_lag_network_centrality(monkeypatch: pytest.MonkeyPatch) -> None:
    """The latest price bar used dates the signal, as membership is announced in advance."""
    module = network_centrality_analyst
    as_of, market = pd.Timestamp("2022-07-18"), pd.DataFrame()
    _members(monkeypatch, module, _universe("2022-07-15", periods=150, seed=10))
    signal = module.LeadLagNetworkCentralityAnalyst().generate_signal("T000", market, as_of)

    assert signal.status == "active"
    assert signal.source_as_of == date(2022, 7, 15)

    monkeypatch.setattr(module, "load_sp500_member_securities_asof", _returning(None))
    monkeypatch.setattr(module, "load_fixed_universe_returns", _returning({}))
    missing = module.LeadLagNetworkCentralityAnalyst().generate_signal("T000", market, as_of)
    assert missing.status == "unavailable"
    assert missing.source_as_of is None
