# roster.py

"""Default analyst/manager roster shared by the CLI and GUI entry points.

Splits the five analysts across two managers so the Commander sees two
genuinely different risk overlays rather than duplicate coverage of the same
signals under two names: RiskManager (VaR-based dampening) owns the
directional/sentiment analysts, CVaRManager (tail-risk dampening) owns the
valuation analysts. More analysts and a richer ownership/communication
structure will be layered on later — this is the starting split.
"""

from __future__ import annotations

from agentic_trading.agents.analysts import (
    BlackScholesValuationAnalyst,
    MonteCarloValuationAnalyst,
    NewsSentimentAnalyst,
    RiskMaximizingAnalyst,
    RiskMinimizingAnalyst,
)
from agentic_trading.agents.base import AnalystAgent, ManagerAgent
from agentic_trading.agents.managers import CVaRManager, RiskManager
from agentic_trading.scoring.weighting import Hedge

RISK_MANAGER_ANALYST_IDS: frozenset[str] = frozenset(
    {
        "risk_minimizing_analyst",
        "risk_maximizing_analyst",
        "news_sentiment_analyst",
    }
)
CVAR_MANAGER_ANALYST_IDS: frozenset[str] = frozenset(
    {
        "monte_carlo_valuation_analyst",
        "black_scholes_valuation_analyst",
    }
)


def build_default_analysts() -> list[AnalystAgent]:
    """The full Layer-1 roster, in a stable, deterministic order."""
    return [
        RiskMinimizingAnalyst(),
        RiskMaximizingAnalyst(),
        NewsSentimentAnalyst(),
        MonteCarloValuationAnalyst(),
        BlackScholesValuationAnalyst(),
    ]


def build_default_managers(analysts: list[AnalystAgent] | None = None) -> list[ManagerAgent]:
    """RiskManager and CVaRManager, each owning a disjoint subset of
    `analysts` (defaults to build_default_analysts()) per the split above."""
    analysts = analysts if analysts is not None else build_default_analysts()
    risk_owned = [a for a in analysts if a.agent_id in RISK_MANAGER_ANALYST_IDS]
    cvar_owned = [a for a in analysts if a.agent_id in CVAR_MANAGER_ANALYST_IDS]
    return [
        RiskManager(managed_analysts=risk_owned, weighting_strategy=Hedge()),
        CVaRManager(managed_analysts=cvar_owned, weighting_strategy=Hedge()),
    ]
