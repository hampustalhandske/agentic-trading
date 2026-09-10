from __future__ import annotations

import numpy as np
import pandas as pd

from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.schemas import PortfolioDecision
from agentic_trading.scoring.scoreboard import PerformanceScoreboard


class BaselineCommander:
    """Simple deterministic portfolio baselines for out-of-sample comparison."""

    def __init__(self, strategy: str) -> None:
        valid = {"equal_weight", "buy_and_hold", "momentum", "volatility_target"}
        if strategy not in valid:
            raise ValueError(f"unsupported baseline strategy: {strategy}")
        self.strategy = strategy

    def close(self) -> None:
        pass

    def decide(
        self,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
        scoreboard: PerformanceScoreboard,
    ) -> list[PortfolioDecision]:
        assets = (
            list(dict.fromkeys(market_data.columns.get_level_values(0)))
            if isinstance(market_data.columns, pd.MultiIndex)
            else [market_data.attrs.get("asset", "AAPL")]
        )
        target_weights = self._target_weights(market_data, assets)
        return [
            PortfolioDecision(
                asset=asset,
                as_of_date=as_of_date.date(),
                target_weight=float(target_weights[asset]),
                contributing_signals=[],
            )
            for asset in assets
        ]

    def _target_weights(self, market_data: pd.DataFrame, assets: list[str]) -> dict[str, float]:
        equal_weight = 1.0 / len(assets) if assets else 0.0
        if self.strategy in {"equal_weight", "buy_and_hold"}:
            return dict.fromkeys(assets, equal_weight)

        weights: dict[str, float] = {}
        for asset in assets:
            returns = self._returns(market_data, asset)
            if returns.size < 2:
                weights[asset] = 0.0
                continue
            if self.strategy == "momentum":
                weights[asset] = equal_weight if float(np.mean(returns[-20:])) > 0 else 0.0
            else:
                volatility = float(np.std(returns[-20:], ddof=1))
                weights[asset] = min(equal_weight, equal_weight * 0.15 / max(volatility, 1e-8))
        return weights

    @staticmethod
    def _returns(market_data: pd.DataFrame, asset: str) -> np.ndarray:
        if isinstance(market_data.columns, pd.MultiIndex):
            values = market_data[asset]["log_return"].dropna()
        else:
            values = market_data["log_return"].dropna()
        return values.to_numpy(dtype=np.float64)


class SingleAnalystCommander:
    """Run one analyst as an isolated, unscaled portfolio strategy."""

    def __init__(self, analyst: AnalystAgent) -> None:
        self.analyst = analyst

    def close(self) -> None:
        pass

    def decide(self, market_data, as_of_date, scoreboard):
        assets = (
            list(dict.fromkeys(market_data.columns.get_level_values(0)))
            if isinstance(market_data.columns, pd.MultiIndex)
            else [market_data.attrs.get("asset", "AAPL")]
        )
        decisions = []
        for asset in assets:
            try:
                signal = self.analyst.generate_signal(asset, market_data, as_of_date)
                direction_sign = {"long": 1.0, "short": -1.0, "flat": 0.0}[signal.direction]
                target_weight = direction_sign * signal.confidence
            except Exception:
                target_weight = 0.0
            decisions.append(
                PortfolioDecision(
                    asset=asset,
                    as_of_date=as_of_date.date(),
                    target_weight=max(min(target_weight, 1.0), -1.0),
                    contributing_signals=[],
                )
            )
        return decisions
