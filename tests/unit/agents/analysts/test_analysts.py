from agentic_trading.agents.analysts import (
    BlackScholesValuationAnalyst,
    MonteCarloValuationAnalyst,
    NewsSentimentAnalyst,
    RiskMaximizingAnalyst,
    RiskMinimizingAnalyst,
)
from agentic_trading.agents.base import AnalystAgent


def test_analysts_imports() -> None:
    assert NewsSentimentAnalyst.__name__ == "NewsSentimentAnalyst"
    assert RiskMaximizingAnalyst.__name__ == "RiskMaximizingAnalyst"
    assert RiskMinimizingAnalyst.__name__ == "RiskMinimizingAnalyst"
    assert BlackScholesValuationAnalyst.__name__ == "BlackScholesValuationAnalyst"
    assert MonteCarloValuationAnalyst.__name__ == "MonteCarloValuationAnalyst"


def test_analysts_behave_as_analysts() -> None:
    for analyst_cls in (
        NewsSentimentAnalyst,
        RiskMaximizingAnalyst,
        RiskMinimizingAnalyst,
        BlackScholesValuationAnalyst,
        MonteCarloValuationAnalyst,
    ):
        assert issubclass(analyst_cls, AnalystAgent)