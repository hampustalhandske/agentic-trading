"""Unit tests for agentic_trading.agents._market_math's shared reshaping
helpers used by every pod manager's construct_portfolio()."""

import pandas as pd
import pytest

from agentic_trading.agents._market_math import clamp, single_asset_from_attrs, wide_log_returns
from agentic_trading.agents.base import AnalystAgent, ManagerAgent
from agentic_trading.agents.portfolio_construction import PortfolioConstructor
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.scoring.weighting import Hedge


def _multi_index_market_data() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    return pd.concat(
        {
            "AAPL": pd.DataFrame({"log_return": [0.01, 0.02, -0.01]}, index=index),
            "MSFT": pd.DataFrame({"log_return": [0.00, 0.01, 0.02]}, index=index),
        },
        axis=1,
    )


def test_wide_log_returns_returns_empty_frame_for_no_assets() -> None:
    """A scoped pod with zero assets in scope on a given day (e.g. a
    partial universe that includes none of its asset class) must not
    crash — see the bug this guards against in agents/base.py's
    construct_portfolio()."""
    result = wide_log_returns([], _multi_index_market_data())

    assert result.empty


def test_wide_log_returns_reshapes_multi_index_columns() -> None:
    result = wide_log_returns(["AAPL", "MSFT"], _multi_index_market_data())

    assert list(result.columns) == ["AAPL", "MSFT"]
    assert len(result) == 3


def test_single_asset_from_attrs_reads_the_set_attr() -> None:
    market_data = pd.DataFrame({"log_return": [0.01]})
    market_data.attrs["asset"] = "TLT"

    assert single_asset_from_attrs(market_data) == "TLT"


def test_single_asset_from_attrs_raises_when_unset() -> None:
    """Must fail loudly rather than silently defaulting to some other
    ticker — a silent default here would attribute this asset's real
    return to the wrong pod-return history entry."""
    market_data = pd.DataFrame({"log_return": [0.01]})

    with pytest.raises(ValueError, match="attrs"):
        single_asset_from_attrs(market_data)


def test_clamp_bounds_a_finite_value() -> None:
    assert clamp(1.7) == 1.0
    assert clamp(-0.3) == 0.0
    assert clamp(0.4) == 0.4
    assert clamp(0.9, 0.0, 0.5) == 0.5


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_clamp_raises_on_a_non_finite_value(value: float) -> None:
    """A non-finite value has no place in [lower, upper]; clamping it
    would hand a NaN confidence the upper bound."""
    with pytest.raises(ValueError, match="finite"):
        clamp(value)


class _NanConfidenceAnalyst(AnalystAgent):
    """An analyst whose confidence comes out of `clamp` on an undefined
    statistic."""

    agent_id: str = "nan_confidence_analyst"

    def generate_signal(
        self, asset: str, market_data: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> AgentSignal:
        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction="long",
            confidence=clamp(float("nan")),
            rationale="confidence from an undefined statistic",
        )


class _UnitManager(ManagerAgent):
    agent_id: str = "unit_manager"

    def scale_factor(
        self, signal: AgentSignal, market_data: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> float:
        return 1.0


def test_an_analyst_clamping_nan_reports_an_error_signal() -> None:
    manager = _UnitManager(
        managed_analysts=[_NanConfidenceAnalyst()],
        weighting_strategy=Hedge(),
        portfolio_constructor=PortfolioConstructor(),
    )
    as_of = pd.Timestamp("2024-01-02")
    market_data = pd.DataFrame({"close": [100.0]}, index=pd.to_datetime(["2024-01-02"]))

    (signal,) = manager.collect_signals("AAPL", market_data, as_of)

    assert signal.status == "error"
    assert signal.direction == "flat"
    assert signal.confidence == 0.0
