"""Unit tests for VixTailHedgeAnalyst — a standing short-SPY insurance
sleeve whose confidence floors at a base level and scales up with VIX
stress."""

import numpy as np
import pandas as pd

from agentic_trading.agents.analysts.tail_hedge_pod.vix_tail_hedge_analyst import (
    VIX_TICKER,
    VixTailHedgeAnalyst,
)
from agentic_trading.config import settings

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
