# test_source_as_of_round2.py

"""The options, dividend and stat-arb analysts that declare a data source date every active
signal with `AgentSignal.source_as_of`: the publication date of the newest data the signal rests
on. A chain snapshot is an end-of-day quote published at its own date's close, so it carries its
snapshot date, whatever `settings.options.publication_lag_days` the agent waits before reading it;
a dividend carries its ex-date, the day `load_dividends_from_lakehouse` admits it. An abstention
on data read keeps that date, an unavailable signal with nothing read carries `None`, and
`SignalStore.write_day` refuses an active row from any of these analysts without one.

Chain snapshots are served by stubbing `load_options_snapshot_from_lakehouse` on `_options_data`,
the module that imports it, so the shared ATM reduction runs for real; the dividend loader is
stubbed on the analyst's own module. The SIC lookup behind the stat-arb pairs is an undated input
(`decisions.UNDATED_SOURCES`), so their rows are price-only (see
`tests/unit/stage_a/test_undated_sources.py`); only their partnerless unavailable signal is
specified here."""

from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from agentic_trading.agents.analysts.cvar_pod import (
    _options_data,
    black_scholes_valuation_analyst,
    greeks_delta_hedge_analyst,
    heston_vol_surface_analyst,
)
from agentic_trading.agents.analysts.risk_pod import dividend_payout_analyst
from agentic_trading.agents.analysts.stat_arb_pod.kalman_hedge_ratio_analyst import (
    KalmanHedgeRatioAnalyst,
)
from agentic_trading.agents.analysts.stat_arb_pod.pairs_cointegration_analyst import (
    PairsCointegrationAnalyst,
)
from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.config import temporary_overrides
from agentic_trading.data.loaders.lakehouse_loader import DEFAULT_OPTIONS_SNAPSHOT_LOOKBACK_DAYS
from agentic_trading.stage_a.decisions import LookAheadError
from agentic_trading.stage_a.spec import SpecRecord, build_spec, spec_hash
from agentic_trading.stage_a.store import SignalStore

# A Monday, and the Friday snapshot before it: the snapshot date, the usable date and
# AS_OF minus the lag all differ.
AS_OF = pd.Timestamp("2024-06-17")
SNAPSHOT = date(2024, 6, 14)
EX_DATES = ("2024-02-09", "2024-05-10")
_IDS = {"AAPL": "14593"}
_WRITTEN = ("signals.parquet", "decisions.parquet", "progress.json")
_ANALYSTS: tuple[type[AnalystAgent], ...] = (
    black_scholes_valuation_analyst.BlackScholesValuationAnalyst,
    greeks_delta_hedge_analyst.GreeksDeltaHedgeAnalyst,
    heston_vol_surface_analyst.HestonVolSurfaceAnalyst,
    dividend_payout_analyst.DividendPayoutAnalyst,
)


def _returning(value: Any) -> Callable[..., Any]:
    """A stand-in that ignores its arguments and returns `value`."""
    return lambda *_args, **_kwargs: value


def _market(periods: int = 300) -> pd.DataFrame:
    """A single-asset frame closing at 100 on the `periods` business days ending AS_OF: 1% daily
    log-return noise, then a 2% rise on each of the last five days."""
    returns = np.random.default_rng(0).normal(0.0, 0.01, periods)
    returns[-5:] = 0.02
    return pd.DataFrame(
        {"close": np.full(periods, 100.0), "log_return": returns},
        index=pd.bdate_range(end=AS_OF, periods=periods),
    )


def _chain(day: date, ivs: tuple[float, ...], gamma: float = 0.02) -> pd.DataFrame:
    """A snapshot dated `day` with a call and a put struck at 100 per IV in `ivs`, the n-th
    expiring 30 * n days after `day`; every row carries `gamma`."""
    stamp = pd.Timestamp(day)
    return pd.DataFrame(
        [
            {
                "date": stamp,
                "expiration": stamp + pd.Timedelta(days=30 * n),
                "option_type": kind,
                "strike": 100.0,
                "implied_volatility": iv,
                "delta": 0.5 if kind == "call" else -0.5,
                "gamma": gamma,
            }
            for n, iv in enumerate(ivs, start=1)
            for kind in ("call", "put")
        ]
    )


def _serve(monkeypatch: pytest.MonkeyPatch, chain: pd.DataFrame | None) -> None:
    """Serves `chain` to the CVaR pod's snapshot reads as the loader would: only for a requested
    day on or after its date and within the lookback. `None` serves no snapshot."""

    def _load(
        ticker: str, as_of_date: date, lookback_days: int = DEFAULT_OPTIONS_SNAPSHOT_LOOKBACK_DAYS
    ) -> pd.DataFrame | None:
        if chain is None:
            return None
        age = (as_of_date - chain["date"].iloc[0].date()).days
        return chain if 0 <= age <= lookback_days else None

    monkeypatch.setattr(_options_data, "load_options_snapshot_from_lakehouse", _load)


def _black_scholes(monkeypatch: pytest.MonkeyPatch, chain: pd.DataFrame | None) -> AgentSignal:
    """BlackScholesValuationAnalyst on AAPL at AS_OF off `chain`, GARCH leg fixed at 1% a day."""
    module = black_scholes_valuation_analyst
    monkeypatch.setattr(module, "conditional_daily_volatility", _returning(0.01))
    _serve(monkeypatch, chain)
    return module.BlackScholesValuationAnalyst().generate_signal("AAPL", _market(), AS_OF)


def _heston(monkeypatch: pytest.MonkeyPatch, chain: pd.DataFrame | None) -> AgentSignal:
    """HestonVolSurfaceAnalyst on AAPL at AS_OF off `chain`, its model volatility fixed at 20%."""
    module = heston_vol_surface_analyst
    monkeypatch.setattr(module, "annualized_garch_volatility", _returning(0.20))
    _serve(monkeypatch, chain)
    return module.HestonVolSurfaceAnalyst().generate_signal("AAPL", _market(), AS_OF)


def _greeks(monkeypatch: pytest.MonkeyPatch, chain: pd.DataFrame | None) -> AgentSignal:
    """GreeksDeltaHedgeAnalyst on AAPL at AS_OF off `chain`, after the four sessions before it
    read snapshots with ATM gammas 0.010 to 0.0115, filling a five-point minimum history."""
    analyst = greeks_delta_hedge_analyst.GreeksDeltaHedgeAnalyst()
    market = _market()
    days, gammas = pd.bdate_range("2024-06-10", periods=4), (0.010, 0.0105, 0.011, 0.0115)
    with temporary_overrides({"analysts.gamma_min_history": 5}):
        for day, gamma in zip(days, gammas, strict=True):
            _serve(monkeypatch, _chain(day.date() - timedelta(days=3), (0.2, 0.2), gamma))
            analyst.generate_signal("AAPL", market, day)
        _serve(monkeypatch, chain)
        return analyst.generate_signal("AAPL", market, AS_OF)


def _dividends(monkeypatch: pytest.MonkeyPatch, amounts: tuple[float, float] | None) -> AgentSignal:
    """DividendPayoutAnalyst on AAPL at AS_OF over regular payouts `amounts` with ex-dates
    EX_DATES. `None` serves no dividends."""
    module = dividend_payout_analyst
    series = None if amounts is None else pd.Series(amounts, index=pd.to_datetime(EX_DATES))
    monkeypatch.setattr(module, "load_dividends_from_lakehouse", _returning(series))
    return module.DividendPayoutAnalyst().generate_signal("AAPL", pd.DataFrame(), AS_OF)


def _store(store_dir: Path, analyst: AnalystAgent) -> tuple[SignalStore, Path]:
    """An opened store for `analyst` over AAPL through 2024, and its trial directory."""
    spec = build_spec(analyst, list(_IDS), date(2024, 1, 2), date(2024, 12, 31))
    record = SpecRecord(
        spec=spec,
        spec_hash=spec_hash(spec),
        lake_time=datetime(2026, 11, 5, tzinfo=UTC),
        lake_version="f89aef2f428a",
        quantcore_version="0.4.0",
    )
    store = SignalStore(store_dir, record)
    store.open()
    return store, store_dir / record.spec_hash


def _flat(signal: AgentSignal) -> tuple[str, str, float]:
    return signal.status, signal.direction, signal.confidence


def test_black_scholes_valuation(monkeypatch: pytest.MonkeyPatch) -> None:
    """The snapshot used dates the signal, whatever the publication lag; no snapshot read, None."""
    for lag in (1, 2):
        with temporary_overrides({"options.publication_lag_days": lag}):
            signal = _black_scholes(monkeypatch, _chain(SNAPSHOT, (0.60, 0.60)))
        assert (signal.status, signal.direction) == ("active", "short"), lag
        assert signal.source_as_of == SNAPSHOT, lag

    missing = _black_scholes(monkeypatch, None)
    assert (missing.status, missing.source_as_of) == ("unavailable", None)


def test_greeks_delta_hedge(monkeypatch: pytest.MonkeyPatch) -> None:
    """The snapshot used dates a fade and an abstention inside the z band alike."""
    for lag in (1, 2):
        with temporary_overrides({"options.publication_lag_days": lag}):
            signal = _greeks(monkeypatch, _chain(SNAPSHOT, (0.2, 0.2), gamma=0.05))
        assert (signal.status, signal.direction) == ("active", "short"), lag
        assert signal.source_as_of == SNAPSHOT, lag

    abstain = _greeks(monkeypatch, _chain(SNAPSHOT, (0.2, 0.2), gamma=0.011))
    assert _flat(abstain) == ("active", "flat", 0.0)
    assert abstain.source_as_of == SNAPSHOT

    missing = _greeks(monkeypatch, None)
    assert (missing.status, missing.source_as_of) == ("unavailable", None)


def test_heston_vol_surface(monkeypatch: pytest.MonkeyPatch) -> None:
    """The snapshot used dates a rich-premium long and a no-premium abstention."""
    for lag in (1, 2):
        with temporary_overrides({"options.publication_lag_days": lag}):
            signal = _heston(monkeypatch, _chain(SNAPSHOT, (0.28, 0.32)))
        assert (signal.status, signal.direction) == ("active", "long"), lag
        assert signal.source_as_of == SNAPSHOT, lag

    abstain = _heston(monkeypatch, _chain(SNAPSHOT, (0.20, 0.21)))
    assert _flat(abstain) == ("active", "flat", 0.0)
    assert abstain.source_as_of == SNAPSHOT

    missing = _heston(monkeypatch, None)
    assert (missing.status, missing.source_as_of) == ("unavailable", None)


def test_dividend_payout(monkeypatch: pytest.MonkeyPatch) -> None:
    """The newer ex-date of the two payouts compared dates a raise and an unchanged payout."""
    raised = _dividends(monkeypatch, (0.24, 0.25))
    assert (raised.status, raised.direction) == ("active", "long")
    assert raised.source_as_of == date(2024, 5, 10)

    unchanged = _dividends(monkeypatch, (0.25, 0.25))
    assert _flat(unchanged) == ("active", "flat", 0.0)
    assert unchanged.source_as_of == date(2024, 5, 10)

    missing = _dividends(monkeypatch, None)
    assert (missing.status, missing.source_as_of) == ("unavailable", None)


def test_stat_arb_without_a_partner() -> None:
    """With no partner configured nothing is read, so the unavailable signal carries no date."""
    for analyst in (PairsCointegrationAnalyst(), KalmanHedgeRatioAnalyst()):
        signal = analyst.generate_signal("AAPL", _market(), AS_OF)
        assert (signal.status, signal.source_as_of) == ("unavailable", None), analyst.agent_id


def test_store_refuses_an_unsourced_active_row(tmp_path: Path) -> None:
    """An analyst's active row without a source_as_of, flat included, raises and writes nothing."""
    day = AS_OF.date()
    for analyst_class in _ANALYSTS:
        for direction, confidence in (("long", 0.5), ("flat", 0.0)):
            analyst = analyst_class()
            store, trial = _store(tmp_path / f"{analyst.agent_id}-{direction}", analyst)
            unsourced = AgentSignal(
                agent_id=analyst.agent_id,
                asset="AAPL",
                as_of_date=day,
                direction=direction,
                confidence=confidence,
                rationale="",
            )
            with pytest.raises(LookAheadError):
                store.write_day(day, [unsourced], _IDS, "equity")
            assert store.last_completed() is None, analyst.agent_id
            assert [name for name in _WRITTEN if (trial / name).exists()] == [], analyst.agent_id


def test_store_accepts_the_analysts_own_active_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Each dated analyst's own active signal is written with its source and source_as_of, and
    refused once that date is stripped."""
    day = AS_OF.date()
    cases: list[tuple[AnalystAgent, str, AgentSignal]] = [
        (
            black_scholes_valuation_analyst.BlackScholesValuationAnalyst(),
            "options_chain",
            _black_scholes(monkeypatch, _chain(SNAPSHOT, (0.60, 0.60))),
        ),
        (
            greeks_delta_hedge_analyst.GreeksDeltaHedgeAnalyst(),
            "options_chain",
            _greeks(monkeypatch, _chain(SNAPSHOT, (0.2, 0.2), gamma=0.05)),
        ),
        (
            heston_vol_surface_analyst.HestonVolSurfaceAnalyst(),
            "options_chain",
            _heston(monkeypatch, _chain(SNAPSHOT, (0.28, 0.32))),
        ),
        (
            dividend_payout_analyst.DividendPayoutAnalyst(),
            "dividends",
            _dividends(monkeypatch, (0.24, 0.25)),
        ),
    ]
    for analyst, source, signal in cases:
        assert signal.status == "active", analyst.agent_id
        assert signal.source_as_of is not None, analyst.agent_id
        store, trial = _store(tmp_path / analyst.agent_id, analyst)
        stripped = signal.model_copy(update={"source_as_of": None})
        with pytest.raises(LookAheadError):
            store.write_day(day, [stripped], _IDS, "equity")

        store.write_day(day, [signal], _IDS, "equity")
        rows = pq.read_table(trial / "decisions.parquet").to_pylist()
        assert [(row["status"], row["source"], row["source_as_of"]) for row in rows] == [
            ("active", source, signal.source_as_of)
        ]
        assert store.last_completed() == day
