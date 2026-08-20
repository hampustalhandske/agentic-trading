from agentic_trading.agents.base import ManagerAgent
from agentic_trading.agents.managers import ConditionalValueAtRiskManager, CVaRManager, RiskManager


def test_managers_imports() -> None:
    assert RiskManager.__name__ == "RiskManager"
    assert ConditionalValueAtRiskManager.__name__ == "ConditionalValueAtRiskManager"
    assert CVaRManager.__name__ == "ConditionalValueAtRiskManager"


def test_managers_behave_as_managers() -> None:
    for manager_cls in (RiskManager, ConditionalValueAtRiskManager, CVaRManager):
        assert issubclass(manager_cls, ManagerAgent)