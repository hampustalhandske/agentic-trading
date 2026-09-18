"""Unit tests for `_pairs_screening.rescreen_pairs`'s orchestration: the
pre-filters, running Engle-Granger in both directions and keeping the
better one, and the verdict's orientation. The statistics themselves
(ADF, Engle-Granger, OU half-life) are quantcore's and are monkeypatched
here with hand-built results."""

import numpy as np
import pandas as pd
import pytest

from agentic_trading.agents.analysts.stat_arb_pod import _pairs_screening as mod
from agentic_trading.config import settings

FORMATION_DAYS = 100


def _market_data(correlation: float = 0.9, n: int = 300) -> tuple[pd.DataFrame, pd.Timestamp]:
    rng = np.random.default_rng(0)
    common = rng.normal(0.0, 0.01, n)
    idio_a = rng.normal(0.0, 0.01, n)
    idio_b = rng.normal(0.0, 0.01, n)
    returns_a = correlation * common + np.sqrt(1 - correlation**2) * idio_a
    returns_b = correlation * common + np.sqrt(1 - correlation**2) * idio_b
    dates = pd.bdate_range("2023-01-01", periods=n)
    frame = pd.concat(
        {
            "A": pd.DataFrame({"close": 100 * np.exp(np.cumsum(returns_a))}, index=dates),
            "B": pd.DataFrame({"close": 100 * np.exp(np.cumsum(returns_b))}, index=dates),
        },
        axis=1,
    )
    return frame, dates[-1]


def _patch_direction_aware_stats(monkeypatch, p_ab: float, p_ba: float) -> None:
    """Engle-Granger results keyed on argument order via the first value
    of the dependent series, which differs between the two legs."""
    monkeypatch.setattr(mod, "adf_test", lambda series: (-1.0, 0.5))
    first_values: dict[str, float] = {}

    def _engle_granger(y, x):
        first_values.setdefault("first_call_y", float(y[0]))
        if float(y[0]) == first_values["first_call_y"]:
            return 2.0, -4.0, p_ab
        return 0.5, -4.0, p_ba

    monkeypatch.setattr(mod, "engle_granger_test", _engle_granger)
    monkeypatch.setattr(mod, "ou_half_life", lambda spread: 15.0)


def test_verdict_takes_direction_with_lower_pvalue_and_its_beta(monkeypatch) -> None:
    frame, as_of = _market_data()
    _patch_direction_aware_stats(monkeypatch, p_ab=0.04, p_ba=0.01)

    verdicts = mod.rescreen_pairs(frame, as_of, (("A", "B"),), FORMATION_DAYS)

    assert set(verdicts) == {("A", "B")}
    verdict = verdicts[("A", "B")]
    assert verdict.dependent == "B"
    assert verdict.beta == 0.5
    assert verdict.p_value == 0.01
    assert verdict.half_life == 15.0


def test_verdict_keeps_first_direction_when_it_is_better(monkeypatch) -> None:
    frame, as_of = _market_data()
    _patch_direction_aware_stats(monkeypatch, p_ab=0.01, p_ba=0.04)

    verdict = mod.rescreen_pairs(frame, as_of, (("A", "B"),), FORMATION_DAYS)[("A", "B")]

    assert verdict.dependent == "A"
    assert verdict.beta == 2.0
    assert verdict.p_value == 0.01


def test_stationary_leg_is_rejected(monkeypatch) -> None:
    frame, as_of = _market_data()
    _patch_direction_aware_stats(monkeypatch, p_ab=0.01, p_ba=0.04)
    monkeypatch.setattr(mod, "adf_test", lambda series: (-5.0, 0.001))

    assert mod.rescreen_pairs(frame, as_of, (("A", "B"),), FORMATION_DAYS) == {}


def test_low_return_correlation_is_rejected(monkeypatch) -> None:
    frame, as_of = _market_data(correlation=0.0)
    _patch_direction_aware_stats(monkeypatch, p_ab=0.01, p_ba=0.04)
    monkeypatch.setattr(settings.analysts, "pair_min_correlation", 0.5)

    assert mod.rescreen_pairs(frame, as_of, (("A", "B"),), FORMATION_DAYS) == {}


def test_too_little_overlapping_history_is_rejected(monkeypatch) -> None:
    frame, as_of = _market_data(n=40)
    _patch_direction_aware_stats(monkeypatch, p_ab=0.01, p_ba=0.04)

    assert mod.rescreen_pairs(frame, as_of, (("A", "B"),), FORMATION_DAYS) == {}


def test_half_life_failure_is_rejected_not_raised(monkeypatch) -> None:
    frame, as_of = _market_data()
    _patch_direction_aware_stats(monkeypatch, p_ab=0.01, p_ba=0.04)

    def _diverging(spread):
        raise ValueError("does not exhibit mean reversion")

    monkeypatch.setattr(mod, "ou_half_life", _diverging)

    assert mod.rescreen_pairs(frame, as_of, (("A", "B"),), FORMATION_DAYS) == {}


def test_missing_ticker_is_rejected(monkeypatch) -> None:
    frame, as_of = _market_data()
    _patch_direction_aware_stats(monkeypatch, p_ab=0.01, p_ba=0.04)

    assert mod.rescreen_pairs(frame, as_of, (("A", "ZZZ"),), FORMATION_DAYS) == {}


@pytest.mark.parametrize("formation_days", [None, FORMATION_DAYS])
def test_formation_window_defaults_to_settings(monkeypatch, formation_days) -> None:
    frame, as_of = _market_data(n=1200)
    _patch_direction_aware_stats(monkeypatch, p_ab=0.01, p_ba=0.04)
    captured: dict[str, int] = {}
    real_screen = mod._screen_pair

    def _spy(market_data, ticker_a, ticker_b, as_of_date, window):
        captured["window"] = window
        return real_screen(market_data, ticker_a, ticker_b, as_of_date, window)

    monkeypatch.setattr(mod, "_screen_pair", _spy)

    mod.rescreen_pairs(frame, as_of, (("A", "B"),), formation_days)

    expected = formation_days if formation_days else settings.analysts.pair_formation_days
    assert captured["window"] == expected
