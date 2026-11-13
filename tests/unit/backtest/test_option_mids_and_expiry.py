"""Option instrument returns and expiry settlement in `BacktestEngine.run`.

Every held option contract is marked at the close of every simulated day, and the day's
instrument return the Commander's pod-return tracker sees is that mark over the previous
day's, whether or not a decision on the contract executed in between; a held contract with no
mark for a day raises `MissingOptionMidError`. A contract is settled at intrinsic on the first
session on or after its expiry before the Commander decides that session, a decision on it
emitted the session before (with a one-day execution lag) is not filled on the settlement
session, and a Commander that emits a row for a settled contract raises
`ExpiredOptionDecisionError`.
"""

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from agentic_trading.agents.schemas import OptionContract, PortfolioDecision
from agentic_trading.backtest import engine as engine_module
from agentic_trading.backtest.engine import BacktestEngine
from agentic_trading.backtest.portfolio import Portfolio
from agentic_trading.config import settings, temporary_overrides
from agentic_trading.data.loaders.option_marks import OptionMark
from agentic_trading.scoring.track_record import TrackRecord

NAV = 10_000.0
_TIME_VALUE = 1.0
_HALF_SPREAD = 0.1


@pytest.fixture(autouse=True)
def _lift_option_notional_cap():
    """One contract on a $10,000 book exceeds the firm's option-notional cap; lift it so
    fills follow the premium targets."""
    with temporary_overrides({"options.max_notional_fraction": 1_000.0}):
        yield


@pytest.fixture(autouse=True)
def _no_split_adjustment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "agentic_trading.backtest.engine.lakehouse_loader.split_adjustment_factor",
        lambda *_args: 1.0,
    )


def _mark(contract: OptionContract, as_of_date: date, spot: float) -> OptionMark:
    """Intrinsic plus a flat time value before expiry (0.10 either side of mid), and an
    untradeable zero-width intrinsic mark on or after expiry."""
    if contract.option_type == "call":
        intrinsic = max(spot - contract.strike, 0.0)
    else:
        intrinsic = max(contract.strike - spot, 0.0)
    expired = as_of_date >= contract.expiry
    mid = intrinsic if expired else intrinsic + _TIME_VALUE
    half_spread = 0.0 if expired else _HALF_SPREAD
    return OptionMark(
        contract=contract,
        as_of_date=as_of_date,
        snapshot_date=None if expired else as_of_date,
        mid=mid,
        bid=mid - half_spread,
        ask=mid + half_spread,
        implied_volatility=0.0 if expired else 0.25,
        delta=0.0 if expired else 0.5,
        gamma=0.0,
        vega=0.0,
        theta=0.0,
        is_model_mark=False,
        is_expired=expired,
        tradeable=not expired,
    )


def _frame(closes: dict[str, float]) -> pd.DataFrame:
    """A single-asset price frame with open equal to close on every listed day."""
    index = pd.to_datetime(list(closes))
    close = pd.Series(list(closes.values()), index=index, dtype=float)
    return pd.DataFrame(
        {"open": close, "close": close, "log_return": np.log(close / close.shift(1))}
    )


def _row(
    contract: OptionContract, day: date, weight: float, hold_contracts: bool = False
) -> PortfolioDecision:
    return PortfolioDecision(
        asset=contract.contract_id,
        as_of_date=day,
        target_weight=weight,
        contributing_signals=[],
        instrument=contract,
        hold_contracts=hold_contracts,
    )


class _RecordingCommander:
    """Emits the rows given per day and records, at each decide, the instrument returns it
    was handed, the option positions held and the portfolio's cash."""

    def __init__(self, portfolio: Portfolio, rows: dict[date, list[PortfolioDecision]]) -> None:
        self._portfolio = portfolio
        self._rows = rows
        self.instrument_returns: dict[date, dict[str, float]] = {}
        self.held_at_decide: dict[date, set[str]] = {}
        self.cash_at_decide: dict[date, float] = {}

    def decide(
        self,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
        scoreboard: object,
        instrument_returns: dict[str, float] | None = None,
    ) -> list[PortfolioDecision]:
        day = as_of_date.date()
        self.instrument_returns[day] = dict(instrument_returns or {})
        self.held_at_decide[day] = set(self._portfolio.option_positions)
        self.cash_at_decide[day] = self._portfolio.cash
        return list(self._rows.get(day, []))


def _engine(
    closes: dict[str, float],
    rows: dict[date, list[PortfolioDecision]],
    asset: str,
    execution_lag_days: int = 0,
) -> tuple[BacktestEngine, _RecordingCommander, Portfolio]:
    portfolio = Portfolio(cash=NAV, positions={}, commission_rate=0.0, slippage_bps=0.0)
    commander = _RecordingCommander(portfolio, rows)
    days = [date.fromisoformat(day) for day in closes]
    engine = BacktestEngine(
        commander=commander,
        portfolio=portfolio,
        track_record=TrackRecord(),
        data_loader=lambda _assets, _start, _end: _frame(closes),
        start_date=days[0],
        end_date=days[-1],
        assets=[asset],
        execution_lag_days=execution_lag_days,
    )
    return engine, commander, portfolio


# A call bought on the first day and never decided again before it expires on the fourth.
_CALL = OptionContract(underlying="AAPL", expiry=date(2024, 1, 5), strike=100.0, option_type="call")
# Spot 102, 101, 104, 103: call mids 3.00, 2.00, 5.00, then 3.00 intrinsic at expiry.
_CALL_CLOSES = {"2024-01-02": 102.0, "2024-01-03": 101.0, "2024-01-04": 104.0, "2024-01-05": 103.0}
_CALL_ROWS = {date(2024, 1, 2): [_row(_CALL, date(2024, 1, 2), 0.05)]}


def test_a_held_contract_without_a_decision_is_measured_against_the_previous_days_mid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each day's instrument return of a held call is its close mid over the previous
    session's, though no decision on the call executes after the first day; on its expiry
    day the return runs to the intrinsic it settles at."""
    monkeypatch.setattr("agentic_trading.backtest.engine.load_option_mark", _mark)
    engine, commander, portfolio = _engine(_CALL_CLOSES, _CALL_ROWS, "AAPL")

    engine.run()

    contract_id = _CALL.contract_id
    assert commander.instrument_returns[date(2024, 1, 2)] == {}
    assert commander.instrument_returns[date(2024, 1, 3)][contract_id] == pytest.approx(
        math.log(2.0 / 3.0)
    )
    assert commander.instrument_returns[date(2024, 1, 4)][contract_id] == pytest.approx(
        math.log(5.0 / 2.0)
    )
    assert commander.instrument_returns[date(2024, 1, 5)][contract_id] == pytest.approx(
        math.log(3.0 / 5.0)
    )
    assert portfolio.option_positions == {}


def test_a_held_contract_without_a_mark_for_a_day_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A held call with no mark on its third day stops the run rather than measuring a later
    return against the second day's mid."""

    def mark_with_a_gap(contract: OptionContract, as_of_date: date, spot: float):
        if as_of_date == date(2024, 1, 4):
            return None
        return _mark(contract, as_of_date, spot)

    monkeypatch.setattr("agentic_trading.backtest.engine.load_option_mark", mark_with_a_gap)
    engine, commander, _ = _engine(_CALL_CLOSES, _CALL_ROWS, "AAPL")

    with pytest.raises(engine_module.MissingOptionMidError, match=_CALL.contract_id):
        engine.run()
    assert date(2024, 1, 4) not in commander.instrument_returns
    assert [day for day, _ in engine.decisions_by_day.items()] == [
        date(2024, 1, 2),
        date(2024, 1, 3),
    ]


# A put expiring on Saturday 2024-01-20: opened Thursday, held Friday, settled Monday.
_PUT = OptionContract(underlying="SPY", expiry=date(2024, 1, 20), strike=100.0, option_type="put")
_PUT_CLOSES = {"2024-01-18": 100.0, "2024-01-19": 99.0, "2024-01-22": 97.0, "2024-01-23": 98.0}
_THURSDAY, _FRIDAY, _MONDAY, _TUESDAY = (date.fromisoformat(day) for day in _PUT_CLOSES)


def test_a_saturday_expiry_settles_before_the_next_sessions_decide(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With a one-day execution lag, Thursday's open fills on Friday and Friday's hold row
    would fill on Monday. The put is settled at Monday's intrinsic (3.00) before the
    Commander decides Monday, and Friday's row is not filled: the put's only record after
    its buy is the "expire" record, and no order on it is left unfilled."""
    monkeypatch.setattr("agentic_trading.backtest.engine.load_option_mark", _mark)
    rows = {
        _THURSDAY: [_row(_PUT, _THURSDAY, 0.05)],
        _FRIDAY: [_row(_PUT, _FRIDAY, 0.05, hold_contracts=True)],
    }
    engine, commander, portfolio = _engine(_PUT_CLOSES, rows, "SPY", execution_lag_days=1)

    result = engine.run()

    contract_id = _PUT.contract_id
    multiplier = float(settings.options.contract_multiplier)
    bought, *after = [trade for trade in result.trade_log if trade.asset == contract_id]
    assert bought.action == "buy"
    assert bought.contracts > 0.0
    assert [(trade.as_of_date, trade.action, trade.price) for trade in after] == [
        (_MONDAY, "expire", 3.0)
    ]
    assert commander.held_at_decide[_MONDAY] == set()
    assert commander.cash_at_decide[_MONDAY] == pytest.approx(
        NAV + bought.premium - bought.transaction_cost + bought.contracts * 3.0 * multiplier
    )
    assert result.unfilled_orders == []
    assert portfolio.option_positions == {}


def test_a_row_emitted_for_a_settled_contract_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Commander that emits a row on Monday for the put that expired on Saturday stops the
    run on Monday."""
    monkeypatch.setattr("agentic_trading.backtest.engine.load_option_mark", _mark)
    rows = {
        _THURSDAY: [_row(_PUT, _THURSDAY, 0.05)],
        _MONDAY: [_row(_PUT, _MONDAY, 0.0001, hold_contracts=True)],
    }
    engine, _, _ = _engine(_PUT_CLOSES, rows, "SPY")

    with pytest.raises(engine_module.ExpiredOptionDecisionError, match=_PUT.contract_id):
        engine.run()
