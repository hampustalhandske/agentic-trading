import pandas as pd

from agentic_trading.agents.managers.alt_data_manager import AltDataManager
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.scoring.weighting import Hedge

AS_OF = pd.Timestamp("2024-06-20")


def _signal(rationale: str, direction: str = "long", confidence: float = 0.8) -> AgentSignal:
    return AgentSignal(
        agent_id="some_alt_data_analyst",
        asset="AAPL",
        as_of_date=AS_OF.date(),
        direction=direction,
        confidence=confidence,
        rationale=rationale,
    )


def _manager() -> AltDataManager:
    return AltDataManager(managed_analysts=[], weighting_strategy=Hedge())


def test_scale_factor_dampens_stale_signals() -> None:
    fresh = _signal("[source_as_of=2024-06-20] [sample_count=10]")
    stale = _signal("[source_as_of=2024-05-01] [sample_count=10]")

    fresh_factor = _manager().scale_factor(fresh, pd.DataFrame(), AS_OF)
    stale_factor = _manager().scale_factor(stale, pd.DataFrame(), AS_OF)

    assert stale_factor < fresh_factor


def test_scale_factor_dampens_sparse_signals() -> None:
    dense = _signal("[source_as_of=2024-06-20] [sample_count=10]")
    sparse = _signal("[source_as_of=2024-06-20] [sample_count=1]")

    dense_factor = _manager().scale_factor(dense, pd.DataFrame(), AS_OF)
    sparse_factor = _manager().scale_factor(sparse, pd.DataFrame(), AS_OF)

    assert sparse_factor < dense_factor


def test_scale_factor_never_flips_direction_only_dampens_confidence() -> None:
    manager = _manager()
    stale_sparse = _signal("[source_as_of=2024-01-01] [sample_count=1]", direction="short")

    factor = manager.scale_factor(stale_sparse, pd.DataFrame(), AS_OF)
    scaled_confidence = stale_sparse.confidence * factor

    assert 0.0 <= factor <= 1.0
    assert scaled_confidence <= stale_sparse.confidence
    # scale_factor never touches direction — it only feeds a confidence multiplier.
    assert stale_sparse.direction == "short"


def test_scale_factor_defaults_to_no_dampening_without_tags() -> None:
    untagged = _signal("no tags here")

    factor = _manager().scale_factor(untagged, pd.DataFrame(), AS_OF)

    assert factor == 1.0
