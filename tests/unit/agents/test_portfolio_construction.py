import math
from datetime import date

import numpy as np
import pandas as pd

from agentic_trading.agents.portfolio_construction import PortfolioConstructor
from agentic_trading.agents.schemas import PortfolioDecision


def _decision(asset: str, target_weight: float) -> PortfolioDecision:
    return PortfolioDecision(
        asset=asset,
        as_of_date=date(2024, 6, 1),
        target_weight=target_weight,
        contributing_signals=[],
    )


def _returns_history(assets: list[str], periods: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {asset: rng.normal(0.0005, 0.01, size=periods) for asset in assets}
    return pd.DataFrame(data)


def test_disabled_constructor_returns_decisions_unchanged() -> None:
    constructor = PortfolioConstructor(enabled=False)
    decisions = [_decision("AAPL", 0.5), _decision("MSFT", -0.3)]

    result = constructor.construct(
        decisions=decisions,
        returns_history=_returns_history(["AAPL", "MSFT"]),
        market_weights={"AAPL": 0.5, "MSFT": 0.5},
        as_of_date=pd.Timestamp("2024-06-01"),
    )

    assert result == decisions


def test_enabled_constructor_produces_sane_weights_for_small_universe() -> None:
    assets = ["AAPL", "MSFT", "GOOGL"]
    constructor = PortfolioConstructor(enabled=True, method="risk_parity", max_gross_leverage=2.0)
    decisions = [_decision("AAPL", 0.6), _decision("MSFT", -0.2), _decision("GOOGL", 0.1)]

    result = constructor.construct(
        decisions=decisions,
        returns_history=_returns_history(assets),
        market_weights={"AAPL": 0.4, "MSFT": 0.35, "GOOGL": 0.25},
        as_of_date=pd.Timestamp("2024-06-01"),
    )

    assert len(result) == len(decisions)
    gross_leverage = sum(abs(decision.target_weight) for decision in result)
    assert gross_leverage <= 2.0 + 1e-6
    for decision in result:
        assert math.isfinite(decision.target_weight)
        assert not math.isnan(decision.target_weight)


def test_enabled_constructor_skips_gracefully_with_too_few_observations() -> None:
    """Early backtest days can hand PortfolioConstructor a returns_history
    with too few rows (or none at all — the first simulated day's
    log_return is NaN and gets dropped) to estimate a covariance matrix.
    It must degrade to a no-op for that day rather than raising."""
    assets = ["AAPL", "MSFT"]
    constructor = PortfolioConstructor(enabled=True)
    decisions = [_decision("AAPL", 0.4), _decision("MSFT", -0.2)]

    for periods in (0, 1, 3):
        result = constructor.construct(
            decisions=decisions,
            returns_history=_returns_history(assets, periods=periods),
            market_weights={"AAPL": 0.5, "MSFT": 0.5},
            as_of_date=pd.Timestamp("2024-06-01"),
        )
        assert result == decisions


def test_enabled_constructor_skips_when_observations_equal_asset_count() -> None:
    """quantcore's ledoit_wolf_shrinkage requires strictly more observations
    (T) than assets (k) — a basket of 5 assets (the project's
    DEFAULT_BASKET) with exactly 5 days of returns history is a real case
    hit in an actual run: T=5, k=5 fails T > k even though 5 observations
    clears the flat MIN_OBSERVATIONS_FOR_COVARIANCE floor. Must degrade to
    a no-op, not raise."""
    assets = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM"]
    constructor = PortfolioConstructor(enabled=True)
    decisions = [_decision(asset, 0.1) for asset in assets]

    result = constructor.construct(
        decisions=decisions,
        returns_history=_returns_history(assets, periods=5),
        market_weights={asset: 0.2 for asset in assets},
        as_of_date=pd.Timestamp("2024-06-01"),
    )

    assert result == decisions


def test_enabled_constructor_supports_min_variance_and_mean_variance() -> None:
    assets = ["AAPL", "MSFT"]
    decisions = [_decision("AAPL", 0.4), _decision("MSFT", 0.2)]
    returns_history = _returns_history(assets)
    market_weights = {"AAPL": 0.5, "MSFT": 0.5}

    for method in ("min_variance", "mean_variance", "risk_parity"):
        constructor = PortfolioConstructor(enabled=True, method=method)
        result = constructor.construct(
            decisions=decisions,
            returns_history=returns_history,
            market_weights=market_weights,
            as_of_date=pd.Timestamp("2024-06-01"),
        )
        assert len(result) == len(decisions)
        assert all(math.isfinite(decision.target_weight) for decision in result)
