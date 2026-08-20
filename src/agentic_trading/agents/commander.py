# commander.py

import pandas as pd

from agentic_trading.agents.base import ManagerAgent
from agentic_trading.agents.schemas import PortfolioDecision
from agentic_trading.scoring.scoreboard import PerformanceScoreboard
from agentic_trading.scoring.weighting import WeightingStrategy


class Commander:
    """Layer 3, final allocation authority. Holds the managers it draws
    signals from and the strategy it uses to weight them by trailing
    track record; both are public attributes, kept inspectable like every
    other component in this project."""

    def __init__(
        self,
        managers: list["ManagerAgent"],
        weighting_strategy: "WeightingStrategy",
    ) -> None:
        self.managers = managers
        self.weighting_strategy = weighting_strategy

    def decide(
        self,
        market_data: "pd.DataFrame",
        as_of_date: "pd.Timestamp",
        scoreboard: "PerformanceScoreboard",
    ) -> list["PortfolioDecision"]:
        """Produces one PortfolioDecision per asset covered by this
        Commander's managers, for `as_of_date`.

        Intended flow: call each manager's evaluate() to get its scaled
        ManagerSignals, apply self.weighting_strategy (evaluated against
        `scoreboard`) per manager to weight their relative influence, then
        combine each asset's weighted signals into one target_weight,
        carrying the full contributing ManagerSignal list forward onto the
        resulting PortfolioDecision for traceability.
        """
        if market_data.empty or not self.managers:
            return []

        if isinstance(market_data.columns, pd.MultiIndex):
            assets = list(dict.fromkeys(market_data.columns.get_level_values(0)))
        else:
            assets = [market_data.attrs.get("asset", "AAPL")]

        # See ManagerAgent.evaluate()'s comment: scoring/ is keyed by plain
        # `date`, so normalize the pd.Timestamp before it reaches that layer.
        manager_weights = self.weighting_strategy.get_weights(
            scoreboard, [manager.agent_id for manager in self.managers], as_of_date.date()
        )
        decisions: list[PortfolioDecision] = []
        for asset in assets:
            contributed_signals = []
            for manager in self.managers:
                analyst_signals = manager.collect_signals(asset, market_data, as_of_date)
                contributed_signals.extend(
                    manager.evaluate(analyst_signals, market_data, as_of_date, scoreboard)
                )

            target_weight = 0.0
            for signal in contributed_signals:
                direction_sign = {"long": 1.0, "short": -1.0, "flat": 0.0}[signal.direction]
                target_weight += direction_sign * signal.confidence * manager_weights.get(
                    signal.manager_id, 1.0
                )

            decisions.append(
                PortfolioDecision(
                    asset=asset,
                    as_of_date=as_of_date.date(),
                    target_weight=max(min(target_weight, 1.0), -1.0),
                    contributing_signals=contributed_signals,
                )
            )

        return decisions
