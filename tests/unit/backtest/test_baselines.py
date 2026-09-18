"""Unit tests for agentic_trading.backtest.baselines.BaselineCommander."""

import numpy as np
import pandas as pd
import pytest

from agentic_trading.backtest.baselines import BaselineCommander
from agentic_trading.scoring.scoreboard import PerformanceScoreboard
from agentic_trading.scoring.track_record import TrackRecord


def _market_data(assets: list[str], daily_returns: dict[str, list[float]]) -> pd.DataFrame:
    """A minimal multi-asset frame with a `log_return` column per asset,
    shaped the way `data/loader.py` produces it (MultiIndex columns)."""
    n_days = len(next(iter(daily_returns.values())))
    dates = pd.date_range("2024-01-02", periods=n_days, freq="B")
    frames = {}
    for asset in assets:
        returns = daily_returns[asset]
        close = 100.0 * np.cumprod([1.0 + r for r in returns])
        frames[asset] = pd.DataFrame(
            {"close": close, "log_return": returns, "open": close}, index=dates
        )
    return pd.concat(frames, axis=1)


def _scoreboard() -> PerformanceScoreboard:
    return PerformanceScoreboard(TrackRecord())


def test_equal_weight_returns_the_same_fixed_weight_every_day() -> None:
    commander = BaselineCommander("equal_weight")
    market_data = _market_data(["AAA", "BBB"], {"AAA": [0.01, 0.02], "BBB": [-0.01, 0.03]})

    first_day = commander.decide(market_data, pd.Timestamp("2024-01-02"), _scoreboard())
    second_day = commander.decide(market_data, pd.Timestamp("2024-01-03"), _scoreboard())

    assert {d.asset: d.target_weight for d in first_day} == {"AAA": 0.5, "BBB": 0.5}
    assert {d.asset: d.target_weight for d in second_day} == {"AAA": 0.5, "BBB": 0.5}


def test_buy_and_hold_issues_decisions_only_on_the_first_call() -> None:
    """Unlike equal_weight, buy_and_hold must never reissue a decision
    after the initial buy — the engine only rebalances on days it receives
    a decision, so a real "buy once and hold" has to mean an empty
    decision list on every later day, not a repeated fixed weight."""
    commander = BaselineCommander("buy_and_hold")
    market_data = _market_data(
        ["AAA", "BBB"], {"AAA": [0.01, 0.02, 0.03], "BBB": [-0.01, 0.03, 0.0]}
    )

    first_day = commander.decide(market_data, pd.Timestamp("2024-01-02"), _scoreboard())
    second_day = commander.decide(market_data, pd.Timestamp("2024-01-03"), _scoreboard())
    third_day = commander.decide(market_data, pd.Timestamp("2024-01-04"), _scoreboard())

    assert {d.asset: d.target_weight for d in first_day} == {"AAA": 0.5, "BBB": 0.5}
    assert second_day == []
    assert third_day == []


def test_momentum_weights_only_assets_with_positive_trailing_return() -> None:
    commander = BaselineCommander("momentum")
    # AAA trends up over the trailing 20-day window, BBB trends down.
    market_data = _market_data(
        ["AAA", "BBB"],
        {"AAA": [0.01] * 25, "BBB": [-0.01] * 25},
    )

    decisions = commander.decide(market_data, market_data.index[-1], _scoreboard())

    weights = {d.asset: d.target_weight for d in decisions}
    assert weights["AAA"] == pytest.approx(0.5)
    assert weights["BBB"] == pytest.approx(0.0)


def test_volatility_target_scales_down_a_high_volatility_asset() -> None:
    """A daily stdev well above the ~0.94%/day threshold implied by a 15%
    annualized vol target must pull the weight below equal-weight — this
    is the case the un-annualized threshold (comparing daily vol against a
    bare 0.15) could never trigger, since daily vol essentially never
    exceeds 15%."""
    commander = BaselineCommander("volatility_target")
    rng = np.random.default_rng(seed=0)
    high_vol_returns = list(rng.normal(loc=0.0, scale=0.05, size=25))  # ~5%/day, high vol
    low_vol_returns = list(rng.normal(loc=0.0, scale=0.005, size=25))  # ~0.5%/day, low vol
    market_data = _market_data(["HIGH", "LOW"], {"HIGH": high_vol_returns, "LOW": low_vol_returns})

    decisions = commander.decide(market_data, market_data.index[-1], _scoreboard())

    weights = {d.asset: d.target_weight for d in decisions}
    equal_weight = 0.5
    assert weights["HIGH"] < equal_weight
    assert weights["LOW"] == pytest.approx(equal_weight, rel=1e-2)


def test_unsupported_strategy_raises() -> None:
    with pytest.raises(ValueError, match="unsupported baseline strategy"):
        BaselineCommander("not_a_real_strategy")
