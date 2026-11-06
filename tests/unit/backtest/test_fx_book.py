# test_fx_book.py

"""Spot FX in a USD account: `target_weight` is the USD value of the
base-currency position. Long EURUSD earns S_t / S_{t-1} - 1 per USD of
notional and long USDJPY earns 1 - S_{t-1} / S_t; the position is held
without cash, and its carry (base minus quote policy rate, actual/365 over
calendar days) accrues on the USD notional with the position's sign."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from agentic_trading.agents.schemas import BacktestResult, PortfolioDecision
from agentic_trading.backtest.engine import BacktestEngine
from agentic_trading.backtest.portfolio import MarkedPosition, Portfolio
from agentic_trading.config import settings
from agentic_trading.scoring.track_record import TrackRecord

_CAPITAL = 1_000_000.0
_DAY = date(2024, 1, 2)
_EURUSD = [1.10, 1.12, 1.09, 1.11, 1.11]
_USDJPY = [150.0, 152.0, 148.0, 149.0, 151.0]


def _book(cash: float = _CAPITAL) -> Portfolio:
    return Portfolio(cash=cash, positions={}, commission_rate=0.0, slippage_bps=0.0)


def _decision(asset: str, weight: float, day: date = _DAY) -> PortfolioDecision:
    return PortfolioDecision(
        asset=asset, as_of_date=day, target_weight=weight, contributing_signals=[]
    )


def test_long_eurusd_earns_the_spot_return_on_its_usd_notional() -> None:
    book = _book()
    book.apply_decision(_decision("EURUSD=X", 0.5), 1.10, _CAPITAL)

    assert book.cash == _CAPITAL
    assert book.marked_positions["EURUSD=X"].quantity == pytest.approx(500_000.0 / 1.10)
    assert book.settle_marked_positions({"EURUSD=X": 1.12}) == pytest.approx(
        500_000.0 * (1.12 / 1.10 - 1.0)
    )


def test_long_usdjpy_earns_one_minus_the_inverse_return_on_its_usd_notional() -> None:
    """Long USDJPY is long USD against JPY: its quantity is the USD notional, and each
    mark pays notional x (1 - S_{t-1} / S_t) in USD."""
    book = _book()
    book.apply_decision(_decision("USDJPY=X", 0.5), 150.0, _CAPITAL)

    assert book.cash == _CAPITAL
    assert book.marked_positions["USDJPY=X"] == MarkedPosition(quantity=500_000.0, mark=150.0)
    first = book.settle_marked_positions({"USDJPY=X": 152.0})
    second = book.settle_marked_positions({"USDJPY=X": 148.0})

    assert first == pytest.approx(500_000.0 * (1.0 - 150.0 / 152.0))
    assert second == pytest.approx(500_000.0 * (1.0 - 152.0 / 148.0))
    assert book.cash == pytest.approx(_CAPITAL + first + second)


def test_short_usdjpy_earns_the_negated_pnl() -> None:
    book = _book()
    book.apply_decision(_decision("USDJPY=X", -0.5), 150.0, _CAPITAL)

    assert book.settle_marked_positions({"USDJPY=X": 152.0}) == pytest.approx(
        -500_000.0 * (1.0 - 150.0 / 152.0)
    )


def test_usdjpy_previous_weight_is_the_usd_notional_over_nav() -> None:
    book = _book()
    book.apply_decision(_decision("USDJPY=X", 0.5), 150.0, _CAPITAL)
    trade = book.apply_decision(_decision("USDJPY=X", 0.25), 160.0, _CAPITAL)

    assert trade.previous_weight == pytest.approx(0.5)
    assert trade.delta_shares == pytest.approx(-250_000.0)
    assert trade.notional == pytest.approx(250_000.0)


class _FixedCommander:
    """Targets the same weights every day."""

    def __init__(self, weights: dict[str, float]) -> None:
        self.weights = weights

    def decide(
        self,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
        scoreboard: object,
        instrument_returns: dict[str, float] | None = None,
    ) -> list[PortfolioDecision]:
        return [
            _decision(asset, weight, as_of_date.date()) for asset, weight in self.weights.items()
        ]


def _fx_frame(
    closes: dict[str, list[float]], opens: dict[str, list[float]] | None = None
) -> pd.DataFrame:
    """A `(ticker, field)` frame on business days from 2024-01-02; each open is its close
    unless `opens` gives it."""
    index = pd.bdate_range("2024-01-02", periods=len(next(iter(closes.values()))))
    columns: dict[tuple[str, str], pd.Series] = {}
    for ticker, values in closes.items():
        close = pd.Series(values, index=index)
        columns[(ticker, "open")] = pd.Series((opens or {}).get(ticker, values), index=index)
        columns[(ticker, "close")] = close
        columns[(ticker, "log_return")] = np.log(close / close.shift(1))
    return pd.DataFrame(columns)


def _run(
    frame: pd.DataFrame,
    weights: dict[str, float],
    policy_rates: pd.DataFrame | None = None,
    *,
    execution_lag_days: int = 0,
    risk_free_rate_annual: float | None = None,
) -> BacktestResult:
    days = frame.index
    return BacktestEngine(
        commander=_FixedCommander(weights),  # type: ignore[arg-type]
        portfolio=_book(),
        track_record=TrackRecord(),
        data_loader=lambda _assets, _start, _end: frame,
        start_date=days[0].date(),
        end_date=days[-1].date(),
        assets=list(weights),
        execution_lag_days=execution_lag_days,
        risk_free_rate_series=(
            None
            if risk_free_rate_annual is None
            else pd.Series(risk_free_rate_annual / settings.firm.trading_days_per_year, index=days)
        ),
        policy_rates=policy_rates,
    ).run()


def _usd_pnl(asset: str, quantity: float, start: float, end: float) -> float:
    """USD profit of `quantity` units of `asset` moving from `start` to `end`: EURUSD pays
    quantity x (end - start) in USD, USDJPY pays it in JPY, worth 1 / end USD per yen."""
    move = quantity * (end - start)
    return move / end if asset == "USDJPY=X" else move


def _units(asset: str, usd_notional: float, price: float) -> float:
    """Units of `asset` worth `usd_notional` USD at `price`: euros for EURUSD, dollars for
    USDJPY."""
    return usd_notional if asset == "USDJPY=X" else usd_notional / price


def test_engine_lag_one_fx_fills_at_the_open_and_marks_at_the_close() -> None:
    """With a one-day lag, day t's NAV marks the held units at day t's close, then day t-1's
    target fills at day t's open: held units move from that close back to the open, and the
    new units, half of day t's NAV each, are held from the open on."""
    closes = {"EURUSD=X": _EURUSD, "USDJPY=X": _USDJPY}
    opens = {
        "EURUSD=X": [1.09, 1.115, 1.10, 1.105, 1.12],
        "USDJPY=X": [149.0, 151.0, 149.5, 148.5, 150.0],
    }
    weights = {"EURUSD=X": 0.5, "USDJPY=X": 0.5}

    result = _run(_fx_frame(closes, opens), weights, execution_lag_days=1)

    cash = _CAPITAL
    held: dict[str, tuple[float, float]] = {}
    expected: list[float] = []
    for day in range(len(_EURUSD)):
        for asset, (quantity, mark) in held.items():
            cash += _usd_pnl(asset, quantity, mark, closes[asset][day])
            held[asset] = (quantity, closes[asset][day])
        nav = cash
        expected.append(nav)
        if day == 0:
            continue
        for asset, weight in weights.items():
            open_price = opens[asset][day]
            quantity, mark = held.get(asset, (0.0, open_price))
            cash += _usd_pnl(asset, quantity, mark, open_price)
            held[asset] = (_units(asset, weight * nav, open_price), open_price)
    values = [value for _, value in result.portfolio_value_history]
    assert values == pytest.approx(expected, rel=1e-12)
    assert values[1] == _CAPITAL
    assert values[2] != pytest.approx(_CAPITAL)


def test_engine_futures_and_fx_cash_earns_the_account_rate_and_notional_nothing() -> None:
    """At a 4% risk-free rate, with a future and a pair rebalanced to half of NAV each and no
    policy rates, each day's NAV is the previous NAV plus the day's mark-to-market, all in
    cash, grown by 4% x calendar days / 365: the notionals earn no interest and pay none."""
    rate = 0.04
    cl = [50.0, 51.0, 49.5, 49.5, 52.0]
    frame = _fx_frame({"CL=F": cl, "EURUSD=X": _EURUSD})

    result = _run(frame, {"CL=F": 0.5, "EURUSD=X": 0.5}, risk_free_rate_annual=rate)

    days = [stamp.date() for stamp in frame.index]
    expected = [_CAPITAL]
    interest = 0.0
    for day in range(1, len(days)):
        previous = expected[-1]
        cash = previous * (
            1.0
            + 0.5 * (cl[day] / cl[day - 1] - 1.0)
            + 0.5 * (_EURUSD[day] / _EURUSD[day - 1] - 1.0)
        )
        accrued = cash * rate * (days[day] - days[day - 1]).days / 365
        interest += accrued
        expected.append(cash + accrued)
    values = [value for _, value in result.portfolio_value_history]
    assert values == pytest.approx(expected, rel=1e-12)
    assert result.total_financing_cost == pytest.approx(-interest, rel=1e-12)


def test_engine_books_the_exact_usd_pnl_of_eurusd_and_usdjpy() -> None:
    """Rebalanced to half of NAV in each pair daily, NAV grows by half of each pair's USD
    return per USD of notional."""
    result = _run(
        _fx_frame({"EURUSD=X": _EURUSD, "USDJPY=X": _USDJPY}),
        {"EURUSD=X": 0.5, "USDJPY=X": 0.5},
    )

    expected = [_CAPITAL]
    for day in range(1, len(_EURUSD)):
        eur = _EURUSD[day] / _EURUSD[day - 1] - 1.0
        jpy = 1.0 - _USDJPY[day - 1] / _USDJPY[day]
        expected.append(expected[-1] * (1.0 + 0.5 * eur + 0.5 * jpy))
    values = [value for _, value in result.portfolio_value_history]
    assert values == pytest.approx(expected, rel=1e-12)
    assert result.total_financing_cost == 0.0


def test_carry_credits_a_long_and_debits_a_short_actual_365() -> None:
    """Long base earns (base rate - quote rate) x USD notional x days / 365; a short pays it."""
    for weight, sign in ((0.5, 1.0), (-0.5, -1.0)):
        book = _book()
        book.apply_decision(_decision("EURUSD=X", weight), 1.10, _CAPITAL)

        net = book.apply_carry({"EURUSD=X": 0.03}, 3)

        carry = sign * 500_000.0 * 0.03 * 3 / 365
        assert net == pytest.approx(-carry)
        assert book.cash == pytest.approx(_CAPITAL + carry)


def test_usdjpy_carry_is_the_usd_minus_jpy_rate_on_the_usd_notional() -> None:
    book = _book()
    book.apply_decision(_decision("USDJPY=X", 0.5), 150.0, _CAPITAL)
    differential = 0.05 - (-0.001)

    net = book.apply_carry({"USDJPY=X": differential}, 4)

    assert net == pytest.approx(-500_000.0 * differential * 4 / 365)


def test_a_negative_differential_debits_a_long() -> None:
    book = _book()
    book.apply_decision(_decision("EURUSD=X", 0.5), 1.10, _CAPITAL)

    net = book.apply_carry({"EURUSD=X": -0.02}, 1)

    assert net == pytest.approx(500_000.0 * 0.02 / 365)


def test_zero_rates_accrue_no_carry() -> None:
    book = _book()
    book.apply_decision(_decision("EURUSD=X", 0.5), 1.10, _CAPITAL)

    assert book.apply_carry({"EURUSD=X": 0.0}, 3) == 0.0
    assert book.apply_carry({}, 3) == 0.0
    assert book.apply_financing({"EURUSD=X": 1.10}, 0.0, 3) == 0.0
    assert book.cash == _CAPITAL


def test_engine_accrues_carry_at_the_previous_days_rate_over_calendar_days() -> None:
    """At a constant spot, each day's NAV gains half the previous NAV times the EUR minus
    USD rate in force on the previous valuation day times the calendar days between them
    over 365. A rate is in force from its observation date until the next one, and a
    currency missing on a date keeps its previous observation."""
    frame = _fx_frame({"EURUSD=X": [1.10] * 5})
    policy_rates = pd.DataFrame(
        {"EUR": [0.045, np.nan, 0.07], "USD": [0.025, 0.03, 0.03]},
        index=pd.to_datetime(["2023-12-29", "2024-01-03", "2024-01-04"]),
    )
    result = _run(frame, {"EURUSD=X": 0.5}, policy_rates)

    carry = [0.045 - 0.025, 0.045 - 0.03, 0.07 - 0.03, 0.04, 0.04]
    days = [stamp.date() for stamp in frame.index]
    expected = [_CAPITAL]
    for day in range(1, len(days)):
        gap = (days[day] - days[day - 1]).days
        expected.append(expected[-1] * (1.0 + 0.5 * carry[day - 1] * gap / 365))
    values = [value for _, value in result.portfolio_value_history]
    assert values == pytest.approx(expected, rel=1e-12)
    assert result.total_financing_cost == pytest.approx(-(values[-1] - _CAPITAL))


def test_engine_accrues_no_carry_without_policy_rates() -> None:
    result = _run(_fx_frame({"EURUSD=X": [1.10] * 5}), {"EURUSD=X": 0.5})

    assert [value for _, value in result.portfolio_value_history] == [_CAPITAL] * 5
    assert result.total_financing_cost == 0.0
