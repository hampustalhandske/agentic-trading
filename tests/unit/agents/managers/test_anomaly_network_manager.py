import pandas as pd

from agentic_trading.agents.managers.anomaly_network_manager import AnomalyNetworkManager
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.scoring.weighting import Hedge

AS_OF = pd.Timestamp("2024-06-20")


def _signal(rationale: str, direction: str = "long", confidence: float = 0.8) -> AgentSignal:
    return AgentSignal(
        agent_id="some_anomaly_network_analyst",
        asset="AAPL",
        as_of_date=AS_OF.date(),
        direction=direction,
        confidence=confidence,
        rationale=rationale,
    )


def _manager() -> AnomalyNetworkManager:
    return AnomalyNetworkManager(managed_analysts=[], weighting_strategy=Hedge())


def test_scale_factor_dampens_low_sample_count() -> None:
    deep_sample = _signal("[sample_count=50]")
    thin_sample = _signal("[sample_count=2]")

    deep_factor = _manager().scale_factor(deep_sample, pd.DataFrame(), AS_OF)
    thin_factor = _manager().scale_factor(thin_sample, pd.DataFrame(), AS_OF)

    assert thin_factor < deep_factor
    assert deep_factor == 1.0


def test_scale_factor_never_flips_direction_only_dampens_confidence() -> None:
    manager = _manager()
    thin_sample = _signal("[sample_count=1]", direction="short")

    factor = manager.scale_factor(thin_sample, pd.DataFrame(), AS_OF)
    scaled_confidence = thin_sample.confidence * factor

    assert 0.0 <= factor <= 1.0
    assert scaled_confidence <= thin_sample.confidence
    assert thin_sample.direction == "short"


def test_scale_factor_defaults_to_no_dampening_without_tag() -> None:
    untagged = _signal("no tag here")

    factor = _manager().scale_factor(untagged, pd.DataFrame(), AS_OF)

    assert factor == 1.0
