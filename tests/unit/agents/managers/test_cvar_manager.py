"""Unit tests for ConditionalValueAtRiskManager's listed-option
expression: a sized long/short underlying weight becomes a
delta-equivalent long call/put, a confident variance-risk-premium read
adds a delta-hedged short straddle, and an untradeable chain keeps the
equity weight."""

from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
import pytest

from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.managers import cvar_manager
from agentic_trading.agents.managers.cvar_manager import (
    VRP_ANALYST_ID,
    ConditionalValueAtRiskManager,
)
from agentic_trading.agents.portfolio_construction import PortfolioConstructor
from agentic_trading.agents.schemas import AgentSignal, OptionContract, PortfolioDecision
from agentic_trading.config import settings
from agentic_trading.data.loaders import option_marks
from agentic_trading.data.loaders.option_marks import OptionMark
from agentic_trading.scoring.scoreboard import PerformanceScoreboard
from agentic_trading.scoring.track_record import TrackRecord
from agentic_trading.scoring.weighting import WeightingStrategy

AS_OF = pd.Timestamp("2024-06-20")
SPOT = 100.0
MID = {"call": 5.0, "put": 4.0}
DELTA = {"call": 0.55, "put": -0.45}


def _market_data(periods: int = 3) -> pd.DataFrame:
    dates = pd.bdate_range(end=AS_OF, periods=periods)
    return pd.concat(
        {
            "AAPL": pd.DataFrame(
                {"close": np.full(periods, SPOT), "log_return": np.zeros(periods)}, index=dates
            )
        },
        axis=1,
    )


class _FixedAnalyst(AnalystAgent):
    direction: str = "long"
    confidence: float = 0.8

    def generate_signal(
        self, asset: str, market_data: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> AgentSignal:
        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction=self.direction,  # type: ignore[arg-type]
            confidence=self.confidence,
            rationale="fixed",
        )


class _UnitWeights(WeightingStrategy):
    def get_weights(self, scoreboard, source_ids, as_of_date):
        return dict.fromkeys(source_ids, 1.0)


def _manager(*analysts: AnalystAgent) -> ConditionalValueAtRiskManager:
    return ConditionalValueAtRiskManager(
        managed_analysts=list(analysts),
        weighting_strategy=_UnitWeights(),
        portfolio_constructor=PortfolioConstructor(),
        assets=("AAPL",),
    )


def _scoreboard() -> PerformanceScoreboard:
    return PerformanceScoreboard(TrackRecord())


def _mark(
    contract: OptionContract, as_of_date: date, tradeable: bool = True, delta: float | None = None
) -> OptionMark:
    mid = MID[contract.option_type]
    return OptionMark(
        contract=contract,
        as_of_date=as_of_date,
        snapshot_date=as_of_date,
        mid=mid,
        bid=mid - 0.1 if tradeable else 0.0,
        ask=mid + 0.1,
        implied_volatility=0.25,
        delta=DELTA[contract.option_type] if delta is None else delta,
        gamma=0.02,
        vega=0.1,
        theta=-0.05,
        is_model_mark=False,
        is_expired=False,
        tradeable=tradeable,
    )


@pytest.fixture(autouse=True)
def _no_vol_dampening(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cvar_manager, "vol_regime_scale_factor", lambda *_: 1.0)


@pytest.fixture
def chain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        option_marks,
        "select_expiry",
        lambda underlying, as_of_date, tenor, after=None: as_of_date + timedelta(days=tenor),
    )
    monkeypatch.setattr(
        option_marks,
        "nearest_strike",
        lambda underlying, expiry, option_type, target, as_of_date: round(target),
    )
    monkeypatch.setattr(
        option_marks,
        "load_option_mark",
        lambda contract, as_of_date, spot: _mark(contract, as_of_date),
    )


def _split(decisions: list[PortfolioDecision]) -> tuple[PortfolioDecision, list[PortfolioDecision]]:
    equity = next(decision for decision in decisions if decision.asset == "AAPL")
    options = [decision for decision in decisions if decision.instrument is not None]
    return equity, options


def _underlying_weight(*analysts: AnalystAgent) -> float:
    """The weight the base constructor sizes for `analysts`' views on a
    fresh manager, the reference every option leg below is sized against."""
    manager = _manager(*analysts)
    decisions = super(ConditionalValueAtRiskManager, manager).construct_portfolio(
        ["AAPL"], _market_data(), AS_OF, _scoreboard()
    )
    return decisions[0].target_weight


def test_long_weight_becomes_a_delta_equivalent_call(chain) -> None:
    weight = _underlying_weight(_FixedAnalyst(agent_id="a"))
    assert weight > 0.0

    decisions = _manager(_FixedAnalyst(agent_id="a")).construct_portfolio(
        ["AAPL"], _market_data(), AS_OF, _scoreboard()
    )

    equity, options = _split(decisions)
    assert equity.target_weight == 0.0
    assert len(options) == 1
    call = options[0]
    assert call.instrument is not None
    assert call.instrument.option_type == "call"
    assert call.instrument.strike == SPOT
    assert call.instrument.expiry == AS_OF.date() + timedelta(
        days=settings.options.target_tenor_days
    )
    expected = min(weight * MID["call"] / (DELTA["call"] * SPOT), weight)
    assert call.target_weight == pytest.approx(expected)
    assert call.contributing_signals == equity.contributing_signals


def test_short_weight_becomes_a_delta_equivalent_put(chain) -> None:
    weight = _underlying_weight(_FixedAnalyst(agent_id="a", direction="short"))
    assert weight < 0.0

    decisions = _manager(_FixedAnalyst(agent_id="a", direction="short")).construct_portfolio(
        ["AAPL"], _market_data(), AS_OF, _scoreboard()
    )

    equity, options = _split(decisions)
    assert equity.target_weight == 0.0
    assert [option.instrument.option_type for option in options if option.instrument] == ["put"]
    expected = min(abs(weight) * MID["put"] / (abs(DELTA["put"]) * SPOT), abs(weight))
    assert options[0].target_weight == pytest.approx(expected)


def test_premium_never_exceeds_the_stock_weight(chain, monkeypatch) -> None:
    monkeypatch.setattr(
        option_marks,
        "load_option_mark",
        lambda contract, as_of_date, spot: _mark(contract, as_of_date, delta=0.02),
    )
    weight = _underlying_weight(_FixedAnalyst(agent_id="a"))

    decisions = _manager(_FixedAnalyst(agent_id="a")).construct_portfolio(
        ["AAPL"], _market_data(), AS_OF, _scoreboard()
    )

    # A 0.02 delta is below the floor: the view stays in stock.
    equity, options = _split(decisions)
    assert options == []
    assert equity.target_weight == pytest.approx(weight)


def test_keeps_the_equity_weight_when_the_contract_is_untradeable(chain, monkeypatch) -> None:
    monkeypatch.setattr(
        option_marks,
        "load_option_mark",
        lambda contract, as_of_date, spot: _mark(contract, as_of_date, tradeable=False),
    )
    weight = _underlying_weight(_FixedAnalyst(agent_id="a"))

    decisions = _manager(_FixedAnalyst(agent_id="a")).construct_portfolio(
        ["AAPL"], _market_data(), AS_OF, _scoreboard()
    )

    equity, options = _split(decisions)
    assert options == []
    assert equity.target_weight == pytest.approx(weight)


def test_keeps_the_equity_weight_without_a_listed_expiry(monkeypatch) -> None:
    monkeypatch.setattr(option_marks, "select_expiry", lambda *_, **__: None)
    weight = _underlying_weight(_FixedAnalyst(agent_id="a"))

    decisions = _manager(_FixedAnalyst(agent_id="a")).construct_portfolio(
        ["AAPL"], _market_data(), AS_OF, _scoreboard()
    )

    equity, options = _split(decisions)
    assert options == []
    assert equity.target_weight == pytest.approx(weight)


def test_confident_vrp_signal_adds_a_delta_hedged_short_straddle(chain) -> None:
    """A bullish view plus a rich VRP read: the long ATM call carrying the
    view and the short ATM straddle share the call contract, so the call
    row is the net of both, the put row is the straddle's put leg, and
    the equity row is exactly the hedge that neutralizes the straddle's
    delta."""
    vrp_confidence = max(settings.options.short_straddle_min_confidence, 0.6)
    analysts = (
        _FixedAnalyst(agent_id="a"),
        _FixedAnalyst(agent_id=VRP_ANALYST_ID, confidence=vrp_confidence),
    )
    weight = _underlying_weight(*analysts)
    assert weight > 0.0

    decisions = _manager(*analysts).construct_portfolio(
        ["AAPL"], _market_data(), AS_OF, _scoreboard()
    )

    equity, options = _split(decisions)
    by_type = {option.instrument.option_type: option for option in options if option.instrument}
    assert set(by_type) == {"call", "put"}
    assert len({(o.instrument.expiry, o.instrument.strike) for o in options if o.instrument}) == 1

    directional = min(weight * MID["call"] / (DELTA["call"] * SPOT), weight)
    # Straddle underlying notional is the VRP confidence times the pod's
    # own weight; premium follows from the two mids over spot, split by
    # mid so both legs hold equal contracts.
    total_premium = vrp_confidence * weight * (MID["call"] + MID["put"]) / SPOT
    straddle = {
        kind: total_premium * MID[kind] / (MID["call"] + MID["put"]) for kind in ("call", "put")
    }
    assert straddle["call"] / MID["call"] == pytest.approx(straddle["put"] / MID["put"])
    assert by_type["call"].target_weight == pytest.approx(directional - straddle["call"])
    assert by_type["put"].target_weight == pytest.approx(-straddle["put"])

    short_straddle_delta = sum(
        -straddle[kind] * DELTA[kind] * SPOT / MID[kind] for kind in ("call", "put")
    )
    assert equity.target_weight + short_straddle_delta == pytest.approx(0.0)
    assert equity.target_weight != 0.0


def test_vrp_signal_below_the_floor_sells_no_straddle(chain) -> None:
    manager = _manager(
        _FixedAnalyst(agent_id="a"),
        _FixedAnalyst(
            agent_id=VRP_ANALYST_ID,
            confidence=settings.options.short_straddle_min_confidence / 2.0,
        ),
    )

    decisions = manager.construct_portfolio(["AAPL"], _market_data(), AS_OF, _scoreboard())

    _, options = _split(decisions)
    assert all(option.target_weight > 0.0 for option in options)


def test_flipped_view_closes_the_held_leg_and_opens_the_other_side(chain) -> None:
    manager = _manager(_FixedAnalyst(agent_id="a"))
    first = manager.construct_portfolio(["AAPL"], _market_data(), AS_OF, _scoreboard())
    call_id = next(decision.asset for decision in first if decision.instrument is not None)

    manager.managed_analysts = [_FixedAnalyst(agent_id="a", direction="short")]
    # A reversal only switches side once it has persisted for the
    # configured number of consecutive days; until then the call is held.
    for _ in range(settings.options.expression_confirmation_days - 1):
        interim = manager.construct_portfolio(["AAPL"], _market_data(), AS_OF, _scoreboard())
        assert next(d for d in interim if d.asset == call_id).target_weight > 0.0
    second = manager.construct_portfolio(["AAPL"], _market_data(), AS_OF, _scoreboard())

    by_asset = {decision.asset: decision for decision in second}
    assert by_asset[call_id].target_weight == 0.0
    assert by_asset[call_id].instrument is not None
    opened = [
        decision
        for decision in second
        if decision.instrument is not None and decision.target_weight > 0.0
    ]
    assert len(opened) == 1 and opened[0].instrument is not None
    assert opened[0].instrument.option_type == "put"


def _lake_chain_expiring_on(fake_lake: Any, underlying: str, day: date, strike: float) -> None:
    """Seeds `fake_lake` with one quoted snapshot dated `day - 1` whose only
    expiry is `day`, and no splits."""
    fake_lake.add(
        "dolthub_option_chain",
        pd.DataFrame(
            [
                {
                    "underlying": underlying,
                    "date": (day - timedelta(days=1)).isoformat(),
                    "expiration": day.isoformat(),
                    "option_type": option_type,
                    "strike": strike,
                    "bid": 1.0,
                    "ask": 1.2,
                    "implied_volatility": 0.25,
                    "delta": 0.5 if option_type == "call" else -0.5,
                    "gamma": 0.02,
                    "theta": -0.05,
                    "vega": 0.1,
                    "rho": 0.01,
                }
                for option_type in ("call", "put")
            ]
        ),
        date_columns=("date", "expiration"),
    )
    fake_lake.add_empty(
        "yahoo_split_close_basis",
        {"ticker": "VARCHAR", "ex_date": "DATE", "ratio": "DOUBLE", "in_close_basis": "BOOLEAN"},
    )


def test_lagged_chain_listing_only_todays_expiry_selects_no_contract(fake_lake: Any) -> None:
    """The chain read at `day - 1` lists only `day`'s expiry; selection never returns it."""
    day = AS_OF.date()
    _lake_chain_expiring_on(fake_lake, "AAPL", day, SPOT)
    option_marks.clear_cache()
    try:
        assert option_marks.list_expiries("AAPL", day - timedelta(days=1)) == [day]
        assert cvar_manager._select_contract("AAPL", "call", SPOT, SPOT, day) is None
    finally:
        option_marks.clear_cache()
