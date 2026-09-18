"""Unit tests for CreditSpreadSignalAnalyst — a z-score read on FRED's
ICE BofA HY option-adjusted spread (read from the lakehouse), market-wide
(not per-issuer)."""

import numpy as np
import pandas as pd

from agentic_trading.agents.analysts.credit_pod.credit_spread_signal_analyst import (
    CreditSpreadSignalAnalyst,
)

AS_OF = pd.Timestamp("2024-06-20")
MARKET_DATA = pd.DataFrame({"close": [100.0], "log_return": [0.0]}, index=[AS_OF])


def _patch_spread(monkeypatch, spread_series: pd.Series | None) -> None:
    monkeypatch.setattr(
        "agentic_trading.agents.analysts.credit_pod.credit_spread_signal_analyst."
        "load_fred_series_from_lakehouse",
        lambda _series_id, _as_of_date: spread_series,
    )


def test_widening_spread_is_bearish(monkeypatch) -> None:
    """A spread that just spiked well above its own trailing mean (a
    real "risk-off" credit signal) should read short/bearish for
    whatever credit-sensitive asset it's attached to."""
    dates = pd.date_range("2024-01-01", periods=120, freq="D")
    values = np.full(120, 4.0)
    values[-1] = 8.0  # today's spread spikes far above the trailing mean
    _patch_spread(monkeypatch, pd.Series(values, index=dates))

    signal = CreditSpreadSignalAnalyst().generate_signal("HYG", MARKET_DATA, AS_OF)

    assert signal.direction == "short"
    assert signal.confidence > 0.0


def test_tightening_spread_is_bullish(monkeypatch) -> None:
    """A spread well below its own trailing mean (improving credit
    conditions) should read long/bullish."""
    dates = pd.date_range("2024-01-01", periods=120, freq="D")
    values = np.full(120, 4.0)
    values[-1] = 1.0
    _patch_spread(monkeypatch, pd.Series(values, index=dates))

    signal = CreditSpreadSignalAnalyst().generate_signal("HYG", MARKET_DATA, AS_OF)

    assert signal.direction == "long"
    assert signal.confidence > 0.0


def test_flat_on_insufficient_history(monkeypatch) -> None:
    _patch_spread(monkeypatch, pd.Series([4.0, 4.1, 4.2]))

    signal = CreditSpreadSignalAnalyst().generate_signal("HYG", MARKET_DATA, AS_OF)

    assert signal.direction == "flat"
    assert signal.confidence == 0.0


def test_flat_on_no_lakehouse_coverage(monkeypatch) -> None:
    """Missing lakehouse coverage is reported as an explicit unavailable
    status, not a silent flat."""
    _patch_spread(monkeypatch, None)

    signal = CreditSpreadSignalAnalyst().generate_signal("HYG", MARKET_DATA, AS_OF)

    assert signal.status == "unavailable"
    assert signal.direction == "flat"
    assert signal.confidence == 0.0
