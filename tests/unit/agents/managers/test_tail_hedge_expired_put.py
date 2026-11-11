"""TailHedgeManager emits no row for a put on or after the session it settles: a held put
whose expiry is on or before the decision day is neither closed nor held, only replaced."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.managers.tail_hedge_manager import TailHedgeManager
from agentic_trading.agents.portfolio_construction import PortfolioConstructor
from agentic_trading.agents.schemas import AgentSignal, OptionContract, TailHedgeState
from agentic_trading.data.loaders import option_marks
from agentic_trading.data.loaders.option_marks import OptionMark
from agentic_trading.scoring.scoreboard import PerformanceScoreboard
from agentic_trading.scoring.track_record import TrackRecord
from agentic_trading.scoring.weighting import EqualWeight

pytestmark = pytest.mark.unpinned_lake_reads

SPOT = 500.0
_TENOR_DAYS = 60
# Expires on Saturday 2024-01-20; Friday is the last session it trades, Monday the first
# after its expiry.
_EXPIRED = OptionContract(
    underlying="SPY", expiry=date(2024, 1, 20), strike=475.0, option_type="put"
)
_MONDAY = pd.Timestamp("2024-01-22")


class _ShortAnalyst(AnalystAgent):
    agent_id: str = "vix_tail_hedge_analyst"

    def generate_signal(
        self, asset: str, market_data: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> AgentSignal:
        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction="short",
            confidence=0.5,
            rationale="VIX elevated",
        )


def _market_data(end: pd.Timestamp, periods: int = 3) -> pd.DataFrame:
    dates = pd.bdate_range(end=end, periods=periods)
    return pd.concat(
        {
            "SPY": pd.DataFrame(
                {"close": np.full(periods, SPOT), "log_return": np.zeros(periods)}, index=dates
            )
        },
        axis=1,
    )


def _mark(contract: OptionContract, as_of_date: date, spot: float) -> OptionMark:
    return OptionMark(
        contract=contract,
        as_of_date=as_of_date,
        snapshot_date=as_of_date,
        mid=8.0,
        bid=7.8,
        ask=8.2,
        implied_volatility=0.2,
        delta=-0.3,
        gamma=0.01,
        vega=0.5,
        theta=-0.1,
        is_model_mark=False,
        is_expired=False,
        tradeable=True,
    )


@pytest.fixture(autouse=True)
def chain(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stub chain listing one expiry `_TENOR_DAYS` after the decision day, the strike the
    manager asks for, and a tradeable mark."""

    def select_expiry(underlying, as_of_date, target_tenor_days, after=None) -> date:
        return (after or as_of_date) + timedelta(days=_TENOR_DAYS)

    monkeypatch.setattr(option_marks, "select_expiry", select_expiry)
    monkeypatch.setattr(option_marks, "nearest_strike", lambda _u, _e, _t, tgt, _d: round(tgt))
    monkeypatch.setattr(option_marks, "load_option_mark", _mark)


def _manager_holding(contract: OptionContract) -> TailHedgeManager:
    manager = TailHedgeManager(
        managed_analysts=[_ShortAnalyst()],
        weighting_strategy=EqualWeight(),
        portfolio_constructor=PortfolioConstructor(),
        assets=("SPY",),
    )
    manager.restore_manager_state(
        manager.manager_state().model_copy(
            update={"tail_hedge": TailHedgeState(held_contract_id=contract.contract_id)}
        )
    )
    return manager


def test_a_put_past_its_expiry_gets_no_row_when_the_sleeve_rolls() -> None:
    """On the first session after a Saturday expiry the sleeve buys a new put and emits no
    row, not even a close, for the expired one."""
    manager = _manager_holding(_EXPIRED)

    decisions = manager.construct_portfolio(
        ["SPY"], _market_data(_MONDAY), _MONDAY, PerformanceScoreboard(TrackRecord())
    )

    option_rows = [decision for decision in decisions if decision.instrument is not None]
    assert all(decision.asset != _EXPIRED.contract_id for decision in option_rows)
    (new_put,) = option_rows
    assert new_put.instrument is not None
    assert new_put.instrument.expiry == _MONDAY.date() + timedelta(days=_TENOR_DAYS)
    assert new_put.target_weight > 0.0
    assert manager.manager_state().tail_hedge is not None
    assert manager.manager_state().tail_hedge.held_contract_id == new_put.asset


def test_a_put_before_its_expiry_is_still_closed_when_the_sleeve_rolls() -> None:
    """On the Friday before a Saturday expiry the put still trades: rolling closes it."""
    friday = pd.Timestamp("2024-01-19")
    manager = _manager_holding(_EXPIRED)

    decisions = manager.construct_portfolio(
        ["SPY"], _market_data(friday), friday, PerformanceScoreboard(TrackRecord())
    )

    by_asset = {decision.asset: decision for decision in decisions}
    assert by_asset[_EXPIRED.contract_id].target_weight == 0.0
    assert len([decision for decision in decisions if decision.instrument is not None]) == 2
