from agentic_trading.agents.base import ManagerAgent
from agentic_trading.agents.managers import (
    AltDataManager,
    ConditionalValueAtRiskManager,
    CVaRManager,
    EventCreditManager,
    FactorManager,
    MacroRatesManager,
    RiskManager,
    StatArbManager,
)
from agentic_trading.agents.portfolio_construction import PortfolioConstructor
from agentic_trading.scoring.weighting import Hedge

ALL_MANAGER_CLASSES = (
    RiskManager,
    ConditionalValueAtRiskManager,
    CVaRManager,
    StatArbManager,
    MacroRatesManager,
    FactorManager,
    EventCreditManager,
    AltDataManager,
)


def test_managers_imports() -> None:
    assert RiskManager.__name__ == "RiskManager"
    assert ConditionalValueAtRiskManager.__name__ == "ConditionalValueAtRiskManager"
    assert CVaRManager.__name__ == "ConditionalValueAtRiskManager"
    assert StatArbManager.__name__ == "StatArbManager"
    assert MacroRatesManager.__name__ == "MacroRatesManager"
    assert FactorManager.__name__ == "FactorManager"
    assert EventCreditManager.__name__ == "EventCreditManager"
    assert AltDataManager.__name__ == "AltDataManager"


def test_managers_can_be_constructed_with_defaults() -> None:
    for manager_cls in ALL_MANAGER_CLASSES:
        instance = manager_cls(
            managed_analysts=[],
            weighting_strategy=Hedge(),
            portfolio_constructor=PortfolioConstructor(),
        )
        assert isinstance(instance.agent_id, str) and instance.agent_id


def test_managers_behave_as_managers() -> None:
    for manager_cls in ALL_MANAGER_CLASSES:
        assert issubclass(manager_cls, ManagerAgent)
