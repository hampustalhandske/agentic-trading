from __future__ import annotations

import math

import numpy as np
import pandas as pd

from agentic_trading.agents._market_math import single_asset_from_attrs
from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.commander import CommanderRiskState
from agentic_trading.agents.schemas import PortfolioDecision
from agentic_trading.scoring.scoreboard import PerformanceScoreboard

# volatility_target compares a *daily* trailing-stdev against this
# threshold, so the threshold must itself be daily-scaled — annualizing a
# 15%-per-year vol target via sqrt(252) gives the equivalent daily figure
# rather than comparing a daily stdev against a raw annual number.
_ANNUAL_VOL_TARGET = 0.15
_TRADING_DAYS_PER_YEAR = 252
_DAILY_VOL_TARGET = _ANNUAL_VOL_TARGET / math.sqrt(_TRADING_DAYS_PER_YEAR)


class BaselineCommander:
    """Simple deterministic portfolio baselines for out-of-sample comparison."""

    def __init__(self, strategy: str) -> None:
        valid = {"equal_weight", "buy_and_hold", "momentum", "volatility_target"}
        if strategy not in valid:
            raise ValueError(f"unsupported baseline strategy: {strategy}")
        self.strategy = strategy
        # buy_and_hold only: True once the single initial buy has been
        # issued, so every later `decide()` call returns no decisions at
        # all and the engine never rebalances — the actual distinction
        # from `equal_weight`, which *is* a daily-rebalanced constant-weight
        # index. Returning the same fixed weight every day (the previous
        # behavior) made the engine rebalance both strategies identically.
        self._bought_and_held = False

    def close(self) -> None:
        pass

    def risk_state(self) -> CommanderRiskState:
        """Baselines carry no pod-level risk state to checkpoint."""
        return CommanderRiskState()

    def restore_risk_state(self, state: CommanderRiskState) -> None:
        pass

    def decide(
        self,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
        scoreboard: PerformanceScoreboard,
        instrument_returns: dict[str, float] | None = None,
    ) -> list[PortfolioDecision]:
        if self.strategy == "buy_and_hold" and self._bought_and_held:
            return []
        assets = (
            list(dict.fromkeys(market_data.columns.get_level_values(0)))
            if isinstance(market_data.columns, pd.MultiIndex)
            else [single_asset_from_attrs(market_data)]
        )
        target_weights = self._target_weights(market_data, assets)
        if self.strategy == "buy_and_hold":
            self._bought_and_held = True
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
                weights[asset] = min(
                    equal_weight, equal_weight * _DAILY_VOL_TARGET / max(volatility, 1e-8)
                )
        return weights

    @staticmethod
    def _returns(market_data: pd.DataFrame, asset: str) -> np.ndarray:
        if isinstance(market_data.columns, pd.MultiIndex):
            values = market_data[asset]["log_return"].dropna()
        else:
            values = market_data["log_return"].dropna()
        return np.asarray(values.to_numpy(dtype=np.float64), dtype=np.float64)


class SingleAnalystCommander:
    """Run one analyst as an isolated, unscaled portfolio strategy."""

    def __init__(self, analyst: AnalystAgent) -> None:
        self.analyst = analyst

    def close(self) -> None:
        pass

    def risk_state(self) -> CommanderRiskState:
        """An isolated single-analyst run carries no pod-level risk state."""
        return CommanderRiskState()

    def restore_risk_state(self, state: CommanderRiskState) -> None:
        pass

    def decide(self, market_data, as_of_date, scoreboard, instrument_returns=None):
        assets = (
            list(dict.fromkeys(market_data.columns.get_level_values(0)))
            if isinstance(market_data.columns, pd.MultiIndex)
            else [single_asset_from_attrs(market_data)]
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
