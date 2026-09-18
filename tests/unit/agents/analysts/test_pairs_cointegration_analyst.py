"""Unit tests for `PairsCointegrationAnalyst`'s spread construction and
the pod's shared entry/hold/exit rule (`spread_direction`)."""

import numpy as np
import pandas as pd
import pytest

from agentic_trading.agents.analysts.stat_arb_pod import pairs_cointegration_analyst as mod
from agentic_trading.config import settings

ENTRY = settings.analysts.z_entry_threshold
EXIT = settings.analysts.z_exit_threshold


def _market_data(n: int = 120) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2023-01-01", periods=n)
    a = 100 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
    b = 100 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
    frame = pd.concat(
        {
            "A": pd.DataFrame({"close": a}, index=dates),
            "B": pd.DataFrame({"close": b}, index=dates),
        },
        axis=1,
    )
    return frame, dates


def test_no_partner_configured_is_flat() -> None:
    frame, dates = _market_data()

    signal = mod.PairsCointegrationAnalyst().generate_signal("A", frame, dates[-1])

    assert signal.direction == "flat"
    assert "no partner" in signal.rationale


def test_spread_uses_configured_hedge_ratio(monkeypatch) -> None:
    frame, dates = _market_data()
    captured: dict[str, np.ndarray] = {}

    def _spy(spread, window):
        captured["spread"] = spread
        return np.zeros_like(spread)

    monkeypatch.setattr(mod, "spread_zscore", _spy)
    analyst = mod.PairsCointegrationAnalyst(
        partner_tickers_by_asset={"A": ("B",)}, hedge_ratio_by_pair={("A", "B"): 1.7}
    )

    analyst.generate_signal("A", frame, dates[-1])

    returns_a = np.diff(np.log(frame["A"]["close"].to_numpy()))
    returns_b = np.diff(np.log(frame["B"]["close"].to_numpy()))
    expected = np.cumsum(returns_a) - 1.7 * np.cumsum(returns_b)
    np.testing.assert_allclose(captured["spread"], expected)


def test_missing_hedge_ratio_falls_back_to_unit(monkeypatch) -> None:
    frame, dates = _market_data()
    captured: dict[str, np.ndarray] = {}

    def _spy(spread, window):
        captured["spread"] = spread
        return np.zeros_like(spread)

    monkeypatch.setattr(mod, "spread_zscore", _spy)
    analyst = mod.PairsCointegrationAnalyst(partner_tickers_by_asset={"A": ("B",)})

    analyst.generate_signal("A", frame, dates[-1])

    returns_a = np.diff(np.log(frame["A"]["close"].to_numpy()))
    returns_b = np.diff(np.log(frame["B"]["close"].to_numpy()))
    np.testing.assert_allclose(captured["spread"], np.cumsum(returns_a) - np.cumsum(returns_b))


@pytest.mark.parametrize(
    ("z", "expected_direction", "expected_confidence"),
    [
        (ENTRY + 1.0, "short", 0.5),
        (-(ENTRY + 2.0), "long", 1.0),
        (ENTRY - 0.1, "flat", 0.0),
    ],
)
def test_entry_direction_and_confidence(
    monkeypatch, z: float, expected_direction: str, expected_confidence: float
) -> None:
    frame, dates = _market_data()
    monkeypatch.setattr(mod, "spread_zscore", lambda spread, window: np.full(spread.size, z))
    analyst = mod.PairsCointegrationAnalyst(partner_tickers_by_asset={"A": ("B",)})

    signal = analyst.generate_signal("A", frame, dates[-1])

    assert signal.direction == expected_direction
    assert signal.confidence == pytest.approx(expected_confidence)


def test_open_position_is_held_until_spread_reverts_inside_exit_band(monkeypatch) -> None:
    frame, dates = _market_data()
    path = iter([ENTRY + 0.5, (ENTRY + EXIT) / 2, EXIT - 0.1, (ENTRY + EXIT) / 2])
    monkeypatch.setattr(
        mod, "spread_zscore", lambda spread, window: np.full(spread.size, next(path))
    )
    analyst = mod.PairsCointegrationAnalyst(partner_tickers_by_asset={"A": ("B",)})

    directions = [analyst.generate_signal("A", frame, dates[-1]).direction for _ in range(4)]

    # Enter short, hold through the band, exit inside it, and stay flat
    # afterwards since the band alone never re-enters.
    assert directions == ["short", "short", "flat", "flat"]


def test_spread_direction_rule() -> None:
    assert mod.spread_direction(ENTRY + 0.1, None) == "short"
    assert mod.spread_direction(-(ENTRY + 0.1), None) == "long"
    assert mod.spread_direction((ENTRY + EXIT) / 2, None) == "flat"
    assert mod.spread_direction((ENTRY + EXIT) / 2, "long") == "long"
    assert mod.spread_direction(-(ENTRY + EXIT) / 2, "short") == "short"
    assert mod.spread_direction(EXIT / 2, "long") == "flat"
    # A spread that has flipped past the opposite entry reverses the position.
    assert mod.spread_direction(-(ENTRY + 0.1), "short") == "long"
