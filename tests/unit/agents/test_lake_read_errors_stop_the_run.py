# test_lake_read_errors_stop_the_run.py

"""A failed lake read stops the run instead of becoming an analyst's error signal: every error
the data layer raises for a lake read that failed is a `LakeReadError`, and each place that turns
an analyst's or a manager's exception into a flat signal (`ManagerAgent.collect_signals` and
`evaluate`, the stage (a) runner's `_signal`, `SingleAnalystCommander.decide`, the network
centrality analyst's universe load) re-raises it. A model failure (a `ValueError`, a plain
`RuntimeError`) still becomes an error signal."""

from datetime import UTC, date, datetime

import pandas as pd
import pytest
from pydantic import ConfigDict

from agentic_trading.agents.analysts.alt_data_pod.network_centrality_analyst import (
    LeadLagNetworkCentralityAnalyst,
)
from agentic_trading.agents.base import AnalystAgent, ManagerAgent
from agentic_trading.agents.portfolio_construction import PortfolioConstructor
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.backtest.baselines import SingleAnalystCommander
from agentic_trading.data.lake_errors import LakeReadError
from agentic_trading.data.lakehouse_client import (
    LakeConnectionError,
    LakehouseError,
    LakeQuietPeriodError,
    LakeStreamCutError,
)
from agentic_trading.data.loaders import lakehouse_loader as ll
from agentic_trading.scoring.scoreboard import PerformanceScoreboard
from agentic_trading.scoring.track_record import TrackRecord
from agentic_trading.scoring.weighting import Hedge
from agentic_trading.stage_a import runner as stage_a_runner

AS_OF = pd.Timestamp("2024-01-02")
MARKET_DATA = pd.DataFrame({"close": [100.0]}, index=pd.to_datetime(["2024-01-02"]))

LAKE_ERRORS: list[Exception] = [
    LakehouseError("HTTP 400: Binder Error", status=400),
    LakehouseError("HTTP 503: busy", status=503),
    LakeStreamCutError("the response was cut off"),
    LakeConnectionError("POST /v1/sql: ConnectError: Connection refused"),
    LakeQuietPeriodError("lake call refused", datetime(2026, 1, 1, tzinfo=UTC)),
    ll.LakeTimeError("the lake's served code changed"),
    ll.MissingTieOrderError("no tie order"),
    ll.UnpinnedTableReadError("names a table other than through <table>_at(T)"),
    ll.UnpinnedLakeReadError("no lake_session open"),
    ll.HoldoutReadError("asked for data known after data_end"),
    ll.EmptyTableReadError("sp500_membership read empty twice"),
    ll.EmptyAdjustmentTableError("equity_adjustment_factors is empty"),
    ll.AdjustmentFactorCollisionError("two factor rows collide"),
    ll.LakeHealthError("the lake's health refuses a run"),
]
_IDS = [type(error).__name__ for error in LAKE_ERRORS]


class _RaisingAnalyst(AnalystAgent):
    agent_id: str = "raising_analyst"
    error: Exception

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def generate_signal(
        self, asset: str, market_data: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> AgentSignal:
        raise self.error


class _Manager(ManagerAgent):
    agent_id: str = "test_manager"
    error: Exception | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def scale_factor(
        self, signal: AgentSignal, market_data: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> float:
        if self.error is not None:
            raise self.error
        return 1.0


def _manager(analyst_error: Exception, scale_error: Exception | None = None) -> _Manager:
    return _Manager(
        managed_analysts=[_RaisingAnalyst(error=analyst_error)],
        weighting_strategy=Hedge(),
        portfolio_constructor=PortfolioConstructor(),
        error=scale_error,
    )


def _signal() -> AgentSignal:
    return AgentSignal(
        agent_id="raising_analyst",
        asset="AAPL",
        as_of_date=AS_OF.date(),
        direction="long",
        confidence=0.8,
        rationale="test",
    )


@pytest.mark.parametrize("error", LAKE_ERRORS, ids=_IDS)
def test_every_lake_error_is_a_lake_read_error(error: Exception) -> None:
    assert isinstance(error, LakeReadError)
    assert isinstance(error, RuntimeError)


@pytest.mark.parametrize("error", LAKE_ERRORS, ids=_IDS)
def test_a_lake_error_in_generate_signal_propagates_from_collect_signals(
    error: Exception,
) -> None:
    with pytest.raises(type(error)):
        _manager(error).collect_signals("AAPL", MARKET_DATA, AS_OF)


@pytest.mark.parametrize(
    "error", [ValueError("singular matrix"), RuntimeError("no convergence"), ZeroDivisionError()]
)
def test_a_model_failure_in_generate_signal_is_still_an_error_signal(error: Exception) -> None:
    (signal,) = _manager(error).collect_signals("AAPL", MARKET_DATA, AS_OF)
    assert signal.status == "error"
    assert signal.confidence == 0.0


def test_a_lake_error_in_scale_factor_propagates_from_evaluate() -> None:
    manager = _manager(ValueError("unused"), scale_error=LakehouseError("HTTP 400: x"))
    with pytest.raises(LakehouseError):
        manager.evaluate([_signal()], MARKET_DATA, AS_OF, PerformanceScoreboard(TrackRecord()))


def test_a_model_failure_in_scale_factor_is_still_a_zero_factor() -> None:
    manager = _manager(ValueError("unused"), scale_error=ValueError("bad vol"))
    (scaled,) = manager.evaluate(
        [_signal()], MARKET_DATA, AS_OF, PerformanceScoreboard(TrackRecord())
    )
    assert scaled.status == "error"
    assert scaled.scale_applied == 0.0


def test_a_lake_error_propagates_from_the_stage_a_signal() -> None:
    analyst = _RaisingAnalyst(error=LakeStreamCutError("cut off"))
    with pytest.raises(LakeStreamCutError):
        stage_a_runner._signal(analyst, "AAPL", MARKET_DATA, AS_OF)
    control = _RaisingAnalyst(error=ValueError("model"))
    assert stage_a_runner._signal(control, "AAPL", MARKET_DATA, AS_OF).status == "error"


def test_a_lake_error_propagates_from_a_single_analyst_baseline() -> None:
    commander = SingleAnalystCommander(_RaisingAnalyst(error=LakehouseError("HTTP 423: frozen")))
    market = pd.DataFrame({("AAPL", "close"): [100.0]}, index=pd.to_datetime([AS_OF]))
    with pytest.raises(LakehouseError):
        commander.decide(market, AS_OF, PerformanceScoreboard(TrackRecord()))
    control = SingleAnalystCommander(_RaisingAnalyst(error=ValueError("model")))
    (decision,) = control.decide(market, AS_OF, PerformanceScoreboard(TrackRecord()))
    assert decision.target_weight == 0.0


def test_a_lake_error_loading_the_broad_universe_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The network centrality analyst's universe load turns a model failure into an error
    signal, but a failed lake read propagates."""

    def failing(self: LeadLagNetworkCentralityAnalyst, as_of_date: pd.Timestamp) -> None:
        raise ll.HoldoutReadError("asked for data known after data_end")

    monkeypatch.setattr(LeadLagNetworkCentralityAnalyst, "_load_broad_lakehouse_universe", failing)
    with pytest.raises(ll.HoldoutReadError):
        LeadLagNetworkCentralityAnalyst().generate_signal("AAPL", MARKET_DATA, AS_OF)

    def broken(self: LeadLagNetworkCentralityAnalyst, as_of_date: pd.Timestamp) -> None:
        raise ValueError("bad cross-section")

    monkeypatch.setattr(LeadLagNetworkCentralityAnalyst, "_load_broad_lakehouse_universe", broken)
    signal = LeadLagNetworkCentralityAnalyst().generate_signal("AAPL", MARKET_DATA, AS_OF)
    assert signal.status == "error"
    assert signal.as_of_date == date(2024, 1, 2)
