# test_newey_west_threshold.py

"""NeweyWestSignificanceAnalyst's `t_stat_threshold` defaults to `None`, meaning
`settings.analysts.t_stat_threshold` read when the signal is computed; an explicit value wins."""

import numpy as np
import pandas as pd
import pytest

from agentic_trading.agents.analysts.factor_pod import newey_west_significance_analyst as mod
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.config import settings, temporary_overrides

_DATES = pd.date_range("2024-01-01", periods=150, freq="D")
# The HAC t-stat the patched regression reports: slope 1.0 over a standard error of 0.4.
_T_STAT = 2.5
_LOW, _HIGH = 2.0, 3.0


@pytest.fixture(autouse=True)
def _fixed_t_stat(monkeypatch: pytest.MonkeyPatch) -> None:
    """The regression reports slope 1.0 with a HAC variance of 0.16, so |t| is `_T_STAT`."""
    monkeypatch.setattr(
        mod, "ols", lambda _y, _x: (np.array([0.0, 1.0]), np.zeros(_x.shape[0]), 1.0)
    )
    variance = (1.0 / _T_STAT) ** 2
    monkeypatch.setattr(
        mod, "newey_west_cov", lambda *_args, **_kwargs: np.array([[0.0, 0.0], [0.0, variance]])
    )


def _signal(analyst: mod.NeweyWestSignificanceAnalyst) -> AgentSignal:
    returns = np.random.default_rng(3).normal(0.0005, 0.01, size=len(_DATES))
    market_data = pd.DataFrame(
        {"close": np.arange(100.0, 100.0 + len(_DATES)), "log_return": returns}, index=_DATES
    )
    return analyst.generate_signal("AAPL", market_data, _DATES[-1])


def _bets(signal: AgentSignal) -> bool:
    return signal.direction != "flat" and signal.confidence > 0.0


def test_default_threshold_is_read_at_use_time() -> None:
    """An analyst built before the override follows the setting in force when it signals."""
    analyst = mod.NeweyWestSignificanceAnalyst()
    with temporary_overrides({"analysts.t_stat_threshold": _LOW}):
        low = _signal(analyst)
    with temporary_overrides({"analysts.t_stat_threshold": _HIGH}):
        high = _signal(analyst)

    assert analyst.t_stat_threshold is None
    assert _bets(low) and low.confidence == pytest.approx(_T_STAT / (2 * _LOW))
    assert not _bets(high)
    assert (high.status, high.direction, high.confidence) == ("active", "flat", 0.0)


def test_explicit_threshold_wins_over_the_setting() -> None:
    """A threshold given at construction is used whatever the setting says."""
    with temporary_overrides({"analysts.t_stat_threshold": _HIGH}):
        explicit_low = _signal(mod.NeweyWestSignificanceAnalyst(t_stat_threshold=_LOW))
    with temporary_overrides({"analysts.t_stat_threshold": _LOW}):
        explicit_high = _signal(mod.NeweyWestSignificanceAnalyst(t_stat_threshold=_HIGH))

    assert _LOW < _T_STAT < _HIGH
    assert settings.analysts.t_stat_threshold == 2.0
    assert _bets(explicit_low)
    assert not _bets(explicit_high)
