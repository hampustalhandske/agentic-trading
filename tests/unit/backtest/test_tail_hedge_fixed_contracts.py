# test_tail_hedge_fixed_contracts.py

"""A held tail-hedge put keeps the contract count it opened with until it is rolled, closed or
expires: `PortfolioDecision.hold_contracts` marks the row of a put held from an earlier day, the
book treats such a row as a hold, and every open, roll or reopen is a new commitment against
`settings.options.tail_hedge_annual_premium_cap`."""

from collections.abc import Callable, Iterator
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel, Field

from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.commander import Commander
from agentic_trading.agents.managers.tail_hedge_manager import TailHedgeManager
from agentic_trading.agents.portfolio_construction import PortfolioConstructor
from agentic_trading.agents.schemas import AgentSignal, OptionContract, PortfolioDecision
from agentic_trading.backtest.engine import BacktestEngine
from agentic_trading.backtest.portfolio import Portfolio
from agentic_trading.config import settings, temporary_overrides
from agentic_trading.data.loaders import option_marks
from agentic_trading.data.loaders.option_marks import OptionMark
from agentic_trading.scoring.scoreboard import PerformanceScoreboard
from agentic_trading.scoring.track_record import TrackRecord
from agentic_trading.scoring.weighting import EqualWeight

SPOT = 500.0
NAV = 10_000_000.0
# Every put the stub chain offers expires this many calendar days after its
# decision day: two weeks of holding before it is inside `roll_min_dte`.
_EXPIRY_DAYS = 21
# Monday 2024-01-08 to Friday 2024-01-26: fifteen sessions with one roll.
_INDEX = pd.bdate_range("2024-01-08", periods=15)
_DAYS = [day.date() for day in _INDEX]
# The analyst's confidence per session, so the held put's premium target moves every day.
_CONFIDENCES = (0.3, 0.5, 0.2, 0.45, 0.25, 0.4, 0.3, 0.5, 0.2, 0.35, 0.45, 0.4, 0.3, 0.5, 0.25)
# A session on which every quote is untradeable while the first put is held.
_UNTRADEABLE_DAY = _DAYS[4]

_HELD = OptionContract(underlying="SPY", expiry=date(2024, 2, 16), strike=475.0, option_type="put")
_OPENED = OptionContract(
    underlying="SPY", expiry=date(2024, 3, 15), strike=475.0, option_type="put"
)
_ROLLED_OUT = OptionContract(
    underlying="SPY", expiry=date(2024, 1, 19), strike=475.0, option_type="put"
)
# (contract, premium weight, hold_contracts) of each option row `_FixedRowsPod` emits.
_POD_ROWS = ((_HELD, 0.004, True), (_OPENED, 0.006, False), (_ROLLED_OUT, 0.0, True))


class _InterruptError(Exception):
    pass


class _Chain(BaseModel):
    """Stub chain days with no expiry (`outages`) and with untradeable quotes."""

    outages: set[date] = Field(default_factory=set)
    untradeable: set[date] = Field(default_factory=set)


class _ShortAnalyst(AnalystAgent):
    """Short at the day's confidence in `confidences`; unavailable on a day without one."""

    agent_id: str = "vix_tail_hedge_analyst"
    confidences: dict[date, float | None] = Field(default_factory=dict)

    def generate_signal(
        self, asset: str, market_data: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> AgentSignal:
        confidence = self.confidences.get(as_of_date.date())
        if confidence is None:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="VIX unavailable",
                status="unavailable",
            )
        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction="short",
            confidence=confidence,
            rationale="VIX elevated",
        )


class _FixedRowsPod(TailHedgeManager):
    """A tail-hedge pod whose option rows are `_POD_ROWS` every day."""

    def construct_portfolio(
        self,
        assets: list[str],
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
        scoreboard: PerformanceScoreboard,
    ) -> list[PortfolioDecision]:
        day = as_of_date.date()
        equity = PortfolioDecision(
            asset="SPY", as_of_date=day, target_weight=0.0, contributing_signals=[]
        )
        return [equity] + [_option_row(contract, day, w, hold) for contract, w, hold in _POD_ROWS]


def _option_row(
    contract: OptionContract, day: date, weight: float, hold_contracts: bool
) -> PortfolioDecision:
    return PortfolioDecision(
        asset=contract.contract_id,
        as_of_date=day,
        target_weight=weight,
        contributing_signals=[],
        instrument=contract,
        hold_contracts=hold_contracts,
    )


def _mid_at(days_to_expiry: int) -> float:
    """Time value decaying with days to expiry; the put stays out of the money at `SPOT`."""
    return 2.0 + 0.5 * days_to_expiry


def _dte(contract_id: str, day: date) -> int:
    return (OptionContract.parse(contract_id).expiry - day).days


def _quote(
    contract: OptionContract,
    as_of_date: date,
    mid: float,
    tradeable: bool = True,
    expired: bool = False,
) -> OptionMark:
    """A quote 0.10 either side of `mid`, or a zero-width expired mark at `mid`."""
    half_spread = 0.0 if expired else 0.1
    return OptionMark(
        contract=contract,
        as_of_date=as_of_date,
        snapshot_date=None if expired else as_of_date,
        mid=mid,
        bid=mid - half_spread,
        ask=mid + half_spread,
        implied_volatility=0.2,
        delta=-0.3,
        gamma=0.01,
        vega=0.5,
        theta=-0.1,
        is_model_mark=False,
        is_expired=expired,
        tradeable=tradeable and not expired,
    )


def _uncapped(confidence: float, days_to_expiry: int) -> float:
    """The tail-hedge premium weight of a put at `days_to_expiry` before the premium cap."""
    budget = settings.commander.tail_hedge_budget_fraction * max(
        confidence, settings.analysts.tail_hedge_base_confidence
    )
    notional = settings.options.tail_hedge_notional_fraction * _mid_at(days_to_expiry) / SPOT
    return min(budget, notional)


def _roll_index() -> int:
    """The session on which the backtest's first put is inside `roll_min_dte`."""
    expiry = _DAYS[0] + timedelta(days=_EXPIRY_DAYS)
    return next(
        index
        for index, day in enumerate(_DAYS)
        if (expiry - day).days < settings.options.roll_min_dte
    )


@pytest.fixture(autouse=True)
def chain(monkeypatch: pytest.MonkeyPatch) -> _Chain:
    """Stub SPY chain and marks for both the manager and the engine; no split adjustment."""
    stub = _Chain()

    def select_expiry(underlying, as_of_date, target_tenor_days, after=None) -> date | None:
        day = after or as_of_date
        return None if day in stub.outages else day + timedelta(days=_EXPIRY_DAYS)

    def load_option_mark(contract: OptionContract, as_of_date: date, spot: float) -> OptionMark:
        if as_of_date >= contract.expiry:
            return _quote(contract, as_of_date, max(contract.strike - spot, 0.0), expired=True)
        mid = _mid_at((contract.expiry - as_of_date).days)
        return _quote(contract, as_of_date, mid, tradeable=as_of_date not in stub.untradeable)

    monkeypatch.setattr(option_marks, "select_expiry", select_expiry)
    monkeypatch.setattr(option_marks, "nearest_strike", lambda _u, _e, _t, tgt, _d: round(tgt))
    monkeypatch.setattr(option_marks, "load_option_mark", load_option_mark)
    monkeypatch.setattr("agentic_trading.backtest.engine.load_option_mark", load_option_mark)
    monkeypatch.setattr(
        "agentic_trading.backtest.engine.lakehouse_loader.split_adjustment_factor",
        lambda *_args: 1.0,
    )
    return stub


@pytest.fixture
def scenario(chain: _Chain) -> Iterator[float]:
    """The backtest's cap (the first put plus half the roll) and one untradeable session."""
    chain.untradeable.add(_UNTRADEABLE_DAY)
    first = _uncapped(_CONFIDENCES[0], _EXPIRY_DAYS)
    rolled = _uncapped(_CONFIDENCES[_roll_index()], _EXPIRY_DAYS)
    value = first + 0.5 * rolled
    with temporary_overrides({"options.tail_hedge_annual_premium_cap": value}):
        yield value


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


def _frame() -> pd.DataFrame:
    flat = np.full(len(_INDEX), SPOT)
    return pd.DataFrame(
        {
            ("SPY", "open"): flat,
            ("SPY", "close"): flat,
            ("SPY", "log_return"): np.zeros(len(_INDEX)),
        },
        index=_INDEX,
    )


def _manager(confidences: dict[date, float | None]) -> TailHedgeManager:
    return TailHedgeManager(
        managed_analysts=[_ShortAnalyst(confidences=confidences)],
        weighting_strategy=EqualWeight(),
        portfolio_constructor=PortfolioConstructor(),
        assets=("SPY",),
    )


def _engine(checkpoint_path: Path | None = None, resume: bool = False) -> BacktestEngine:
    commander = Commander(
        managers=[_manager(dict(zip(_DAYS, _CONFIDENCES, strict=True)))],
        weighting_strategy=EqualWeight(),
    )
    return BacktestEngine(
        commander=commander,
        portfolio=Portfolio(cash=NAV, positions={}, commission_rate=0.0, slippage_bps=0.0),
        track_record=TrackRecord(),
        data_loader=lambda _assets, _start, _end: _frame(),
        start_date=_DAYS[0],
        end_date=_DAYS[-1],
        assets=["SPY"],
        checkpoint_path=checkpoint_path,
        resume=resume,
    )


def _stop_after(day: int) -> Callable[..., None]:
    def stop(day_index: int, *_args: object) -> None:
        if day_index == day:
            raise _InterruptError

    return stop


def test_portfolio_decision_hold_contracts_flag() -> None:
    """`hold_contracts` defaults to False and survives a checkpoint's JSON round trip."""
    day = _DAYS[0]
    equity = PortfolioDecision(
        asset="SPY", as_of_date=day, target_weight=0.0, contributing_signals=[]
    )
    option = PortfolioDecision(
        asset=_HELD.contract_id,
        as_of_date=day,
        target_weight=0.01,
        contributing_signals=[],
        instrument=_HELD,
    )
    held = _option_row(_HELD, day, 0.01, hold_contracts=True)

    assert equity.hold_contracts is False
    assert option.hold_contracts is False
    assert held.hold_contracts is True
    assert PortfolioDecision.model_validate(held.model_dump(mode="json")) == held
    assert PortfolioDecision.model_validate(option.model_dump(mode="json")).hold_contracts is False


def test_book_holds_a_held_contract_whatever_the_target() -> None:
    """A flagged row on a held put trades nothing; weight 0 closes; other rows size as today."""
    day = _DAYS[0]
    contract_id = _OPENED.contract_id
    multiplier = float(settings.options.contract_multiplier)
    portfolio = Portfolio(cash=NAV, positions={}, commission_rate=0.0, slippage_bps=0.0)

    def quote(mid: float, tradeable: bool = True) -> OptionMark:
        return _quote(_OPENED, day, mid, tradeable)

    opening = _option_row(_OPENED, day, 0.005, hold_contracts=True)
    assert portfolio.option_target_differs_from_held(opening, quote(12.5, False), NAV, spot=SPOT)
    opened = portfolio.apply_option_decision(opening, quote(12.5), NAV, spot=SPOT)
    assert (opened.action, opened.contracts) == ("buy", 40.0)
    cash = portfolio.cash

    for weight, mid, value in [
        (0.005, 12.5, NAV),
        (0.01, 6.0, NAV),
        (0.002, 12.5, NAV),
        (0.005, 3.0, 2.0 * NAV),
        (0.005, 12.5, 0.5 * NAV),
        (-0.005, 12.5, NAV),
    ]:
        held = _option_row(_OPENED, day, weight, hold_contracts=True)
        for tradeable in (True, False):
            case = (weight, mid, value, tradeable)
            mark = quote(mid, tradeable)
            assert not portfolio.option_target_differs_from_held(held, mark, value, spot=SPOT), case
            record = portfolio.apply_option_decision(held, mark, value, spot=SPOT)
            assert (record.action, record.contracts, record.target_shares) == (
                "hold",
                0.0,
                40.0,
            ), case
            assert portfolio.option_positions[contract_id].contracts == 40.0, case
            assert portfolio.cash == cash, case

    resized = _option_row(_OPENED, day, 0.01, hold_contracts=False)
    assert portfolio.option_target_differs_from_held(resized, quote(12.5), NAV, spot=SPOT)
    topped_up = portfolio.apply_option_decision(resized, quote(12.5), NAV, spot=SPOT)
    assert (topped_up.action, topped_up.contracts) == ("buy", 40.0)
    assert portfolio.option_positions[contract_id].contracts == 80.0
    cash = portfolio.cash

    closing = _option_row(_OPENED, day, 0.0, hold_contracts=True)
    assert portfolio.option_target_differs_from_held(closing, quote(12.5, False), NAV, spot=SPOT)
    closed = portfolio.apply_option_decision(closing, quote(12.5), NAV, spot=SPOT)
    assert (closed.action, closed.contracts) == ("sell", -80.0)
    assert contract_id not in portfolio.option_positions
    assert portfolio.cash == pytest.approx(
        cash + 80 * 12.4 * multiplier - 80 * settings.options.commission_per_contract
    )


def test_manager_flags_held_rows_and_commits_every_open(chain: _Chain) -> None:
    """Opens, rolls and reopens are unflagged and count against the cap; held rows are flagged."""
    start = date(2024, 1, 8)
    days = [start + timedelta(days=offset) for offset in range(90)]
    cycle = (0.3, 0.5, 0.2, 0.45, 0.25)
    confidences: dict[date, float | None] = {
        day: None if offset in (3, 4) else cycle[offset % len(cycle)]
        for offset, day in enumerate(days)
    }
    chain.outages.update(days[15:23])
    cap = 0.025
    manager = _manager(confidences)
    scoreboard = PerformanceScoreboard(TrackRecord())
    held: str | None = None
    opens: list[tuple[date, float]] = []
    committed: list[float] = []
    positive: list[PortfolioDecision] = []
    rolls = reopens = capped_holds = 0

    with temporary_overrides({"options.tail_hedge_annual_premium_cap": cap}):
        for day in days:
            timestamp = pd.Timestamp(day)
            rows = [
                row
                for row in manager.construct_portfolio(
                    ["SPY"], _market_data(timestamp), timestamp, scoreboard
                )
                if row.instrument is not None
            ]
            if held is not None and OptionContract.parse(held).expiry <= day:
                held = None
            positive = [row for row in rows if row.target_weight > 0.0]
            closed = {row.asset for row in rows if row.target_weight == 0.0}
            assert len(positive) <= 1, day
            for row in positive:
                assert row.hold_contracts == (row.asset == held), day
                if not row.hold_contracts:
                    if held is not None:
                        assert held in closed, day
                        rolls += 1
                    elif opens:
                        reopens += 1
                    opens.append((day, row.target_weight))
                elif (
                    _dte(row.asset, day) < settings.options.roll_min_dte
                    and day not in chain.outages
                ):
                    capped_holds += 1
            committed.append(sum(weight for opened, weight in opens if (day - opened).days < 365))
            assert manager.premium_committed(day) == pytest.approx(committed[-1], abs=1e-15), day
            assert committed[-1] <= cap + 1e-12, day
            if positive:
                held = positive[0].asset
            elif held in closed:
                held = None

    assert rolls >= 1
    assert reopens >= 1
    assert capped_holds >= 1
    reached = [
        day for day, total in zip(days, committed, strict=True) if total == pytest.approx(cap)
    ]
    assert reached
    assert all(opened <= reached[0] for opened, _ in opens)
    assert positive == []


def test_commander_carries_hold_contracts_to_the_combined_row() -> None:
    """Each tail-hedge option row reaches the Commander's output with its weight and flag."""
    pod = _FixedRowsPod(
        managed_analysts=[_ShortAnalyst()],
        weighting_strategy=EqualWeight(),
        portfolio_constructor=PortfolioConstructor(),
        assets=("SPY",),
    )
    commander = Commander(managers=[pod], weighting_strategy=EqualWeight())

    decisions = commander.decide(
        _frame().iloc[:2],
        _INDEX[1],
        PerformanceScoreboard(TrackRecord()),
        instrument_returns={},
    )

    combined = {
        decision.asset: (decision.target_weight, decision.hold_contracts)
        for decision in decisions
        if decision.instrument is not None
    }
    assert combined == {contract.contract_id: (w, hold) for contract, w, hold in _POD_ROWS}


def test_backtest_buys_each_put_once_within_the_cap(scenario: float) -> None:
    """Each put is bought once, holds its count until closed, and every open fits the cap."""
    cap = scenario
    result = _engine().run()
    nav = dict(result.portfolio_value_history)
    multiplier = float(settings.options.contract_multiplier)
    rows: dict[str, list[tuple[date, PortfolioDecision]]] = {}
    for day in _DAYS:
        for decision in result.decisions_by_day[day]:
            if decision.instrument is not None and decision.target_weight > 0.0:
                rows.setdefault(decision.asset, []).append((day, decision))
    opens = [
        (held[0][0], contract_id, held[0][1].target_weight) for contract_id, held in rows.items()
    ]

    roll = _roll_index()
    assert [day for day, _, _ in opens] == [_DAYS[0], _DAYS[roll]]
    first_weight = _uncapped(_CONFIDENCES[0], _EXPIRY_DAYS)
    assert opens[0][2] == pytest.approx(first_weight)
    assert opens[1][2] == pytest.approx(cap - first_weight)
    assert opens[1][2] < _uncapped(_CONFIDENCES[roll], _EXPIRY_DAYS)

    first_id = opens[0][1]
    first_count = next(
        trade.contracts
        for trade in result.trade_log
        if trade.asset == first_id and trade.action == "buy"
    )
    assert any(
        abs(
            round(
                _uncapped(_CONFIDENCES[index], _dte(first_id, day))
                * nav[day]
                / (_mid_at(_dte(first_id, day)) * multiplier)
            )
            - first_count
        )
        >= settings.options.rebalance_band * first_count
        for index, day in enumerate(_DAYS[1:roll], start=1)
    )

    spent = slack = 0.0
    for day, contract_id, _ in opens:
        flags = [row.hold_contracts for _, row in rows[contract_id]]
        assert flags == [False] + [True] * (len(flags) - 1), contract_id
        trades = [trade for trade in result.trade_log if trade.asset == contract_id]
        fills = [trade for trade in trades if trade.action != "hold"]
        assert [trade.action for trade in fills] in (["buy"], ["buy", "sell"], ["buy", "expire"])
        buy = fills[0]
        assert buy.as_of_date == day
        assert all(trade.target_shares == 0.0 for trade in fills[1:])
        holds = [trade for trade in trades if trade.action == "hold"]
        assert len(holds) == len(flags) - 1, contract_id
        assert {trade.target_shares for trade in holds} == {buy.contracts}, contract_id

        assert buy.reference_price is not None
        premium = buy.contracts * buy.reference_price * multiplier
        rounding = 0.5 * buy.reference_price * multiplier
        headroom = cap - sum(
            weight for opened, _, weight in opens if opened < day and (day - opened).days < 365
        )
        assert premium <= headroom * nav[day] + rounding, contract_id
        spent += premium / nav[day]
        slack += rounding / nav[day]
    assert spent <= cap + slack
    assert result.unfilled_orders == []


def test_resume_reproduces_the_fixed_contract_run(scenario: float, tmp_path: Path) -> None:
    """Interrupted after any session and resumed, the run equals the uninterrupted one."""
    expected = _engine().run()
    assert any(trade.action == "buy" for trade in expected.trade_log if trade.instrument)

    for k in range(len(_DAYS)):
        checkpoint_path = tmp_path / f"checkpoint_{k}.db"
        with pytest.raises(_InterruptError):
            _engine(checkpoint_path).run(on_day_complete=_stop_after(k))

        result = _engine(checkpoint_path, resume=True).run()

        assert result.trade_log == expected.trade_log, f"interrupted after day {k}"
        assert result.decisions_by_day == expected.decisions_by_day, k
        assert result.portfolio_value_history == expected.portfolio_value_history, k
        assert result.pod_costs == expected.pod_costs, k
