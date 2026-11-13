"""Unit tests for VixTailHedgeAnalyst — a standing short-SPY insurance
sleeve whose confidence floors at a base level and scales up with VIX
stress."""

from datetime import timedelta

import numpy as np
import pandas as pd

from agentic_trading.agents.analysts.tail_hedge_pod.vix_tail_hedge_analyst import (
    VIX_TICKER,
    VixTailHedgeAnalyst,
)
from agentic_trading.config import settings
from agentic_trading.data.trading_calendar import sessions_between

AS_OF = pd.Timestamp("2024-06-20")
MARKET_DATA = pd.DataFrame({"close": [500.0], "log_return": [0.0]}, index=[AS_OF])


def _vix_frame(closes: np.ndarray) -> pd.DataFrame:
    dates = pd.date_range(end=AS_OF, periods=closes.size, freq="B")
    columns = pd.MultiIndex.from_product([[VIX_TICKER], ["open", "close", "log_return"]])
    frame = pd.DataFrame(index=dates, columns=columns, dtype=np.float64)
    frame[(VIX_TICKER, "open")] = closes
    frame[(VIX_TICKER, "close")] = closes
    frame[(VIX_TICKER, "log_return")] = 0.0
    return frame


def _patch_vix(monkeypatch, frame: pd.DataFrame | None) -> None:
    monkeypatch.setattr(
        "agentic_trading.agents.analysts.tail_hedge_pod.vix_tail_hedge_analyst."
        "load_price_history_from_lakehouse",
        lambda _tickers, _start, _end: frame,
    )


def test_calm_vix_keeps_base_protection_on(monkeypatch) -> None:
    closes = np.full(60, 15.0)
    closes[-1] = 14.0
    _patch_vix(monkeypatch, _vix_frame(closes))

    signal = VixTailHedgeAnalyst().generate_signal("SPY", MARKET_DATA, AS_OF)

    assert signal.status == "active"
    assert signal.direction == "short"
    assert signal.confidence == settings.analysts.tail_hedge_base_confidence


def test_elevated_vix_scales_protection_above_base(monkeypatch) -> None:
    closes = np.linspace(14.0, 16.0, 60)
    closes[-1] = 40.0
    _patch_vix(monkeypatch, _vix_frame(closes))

    signal = VixTailHedgeAnalyst().generate_signal("SPY", MARKET_DATA, AS_OF)

    assert signal.direction == "short"
    assert signal.confidence > settings.analysts.tail_hedge_base_confidence
    assert signal.confidence <= 1.0


def test_flat_on_insufficient_vix_history(monkeypatch) -> None:
    _patch_vix(monkeypatch, _vix_frame(np.full(5, 15.0)))

    signal = VixTailHedgeAnalyst().generate_signal("SPY", MARKET_DATA, AS_OF)

    assert signal.status == "unavailable"
    assert signal.direction == "flat"
    assert signal.confidence == 0.0


def test_flat_on_no_lakehouse_coverage(monkeypatch) -> None:
    _patch_vix(monkeypatch, None)

    signal = VixTailHedgeAnalyst().generate_signal("SPY", MARKET_DATA, AS_OF)

    assert signal.status == "unavailable"
    assert signal.direction == "flat"
    assert signal.confidence == 0.0


def test_lookback_counts_nyse_sessions(monkeypatch) -> None:
    """The VIX baseline is the last `vix_lookback_days` NYSE sessions up to the as-of date: the
    load starts on the session that many sessions back, and older bars stay out of the z-score."""
    lookback = settings.analysts.vix_lookback_days
    sessions = sessions_between(AS_OF.date() - timedelta(days=3 * lookback), AS_OF.date())
    expected_start = sessions[-lookback]
    calls: list[tuple] = []
    dates = pd.DatetimeIndex(sessions)
    closes = np.full(dates.size, 15.0)
    closes[: dates.size - lookback] = 100.0  # bars older than the lookback: must not count
    closes[-1] = 16.0
    columns = pd.MultiIndex.from_product([[VIX_TICKER], ["open", "close", "log_return"]])
    frame = pd.DataFrame(index=dates, columns=columns, dtype=np.float64)
    for field in ("open", "close"):
        frame[(VIX_TICKER, field)] = closes
    frame[(VIX_TICKER, "log_return")] = 0.0

    def loader(tickers, start, end):
        calls.append((tickers, start, end))
        return frame

    monkeypatch.setattr(
        "agentic_trading.agents.analysts.tail_hedge_pod.vix_tail_hedge_analyst."
        "load_price_history_from_lakehouse",
        loader,
    )

    signal = VixTailHedgeAnalyst().generate_signal("SPY", MARKET_DATA, AS_OF)

    assert calls == [([VIX_TICKER], expected_start, AS_OF.date())]
    window = closes[-lookback:]
    z_score = (window[-1] - window.mean()) / window.std(ddof=1)
    assert signal.rationale == (
        f"^VIX at {window[-1]:.1f}, z-score={z_score:.2f} vs its own trailing mean"
    )
