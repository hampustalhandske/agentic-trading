# test_cash_interest.py

"""Idle cash earns interest at the risk-free rate on an actual/365 basis;
`Portfolio.apply_financing` returns the net charge (charges minus credit),
`BacktestEngine` compounds that credit into NAV, and excess returns subtract
the same accrual over the same calendar gap."""

import numpy as np
import pandas as pd
import pytest

from agentic_trading.agents.schemas import PortfolioDecision
from agentic_trading.backtest.engine import BacktestEngine
from agentic_trading.backtest.portfolio import Portfolio
from agentic_trading.config import settings
from agentic_trading.scoring.track_record import TrackRecord

_RATE = 0.05
_CAPITAL = 100_000.0
_DAYS = pd.bdate_range("2024-01-02", periods=250)


def _book(cash: float, positions: dict[str, float] | None = None) -> Portfolio:
    return Portfolio(cash=cash, positions=positions or {}, commission_rate=0.0, slippage_bps=0.0)


def test_apply_financing_credits_positive_cash_actual_365() -> None:
    """Positive cash earns `cash * r * days / 365` and the return is the negative credit."""
    portfolio = _book(200_000.0)
    net = portfolio.apply_financing({}, _RATE, 10)
    credit = 200_000.0 * _RATE * 10 / 365
    assert net == pytest.approx(-credit, abs=1e-9)
    assert portfolio.cash == pytest.approx(200_000.0 + credit, abs=1e-9)


def test_apply_financing_negative_cash_charged_without_credit() -> None:
    """Borrowed cash pays `r + margin_spread` on the debt and earns nothing."""
    portfolio = _book(-50_000.0)
    net = portfolio.apply_financing({}, _RATE, 7)
    expected = 50_000.0 * (_RATE + settings.firm.margin_spread_annual) * 7 / 365
    assert net == pytest.approx(expected, abs=1e-9)
    assert net > 0.0
    assert portfolio.cash == pytest.approx(-50_000.0 - expected, abs=1e-9)


def test_apply_financing_short_book_nets_borrow_against_credit() -> None:
    """Short proceeds sitting in cash earn interest while the short pays stock borrow."""
    portfolio = _book(105_000.0, {"XYZ": -100.0})
    days = 12
    net = portfolio.apply_financing({"XYZ": 50.0}, _RATE, days)
    borrow = 5_000.0 * settings.firm.short_borrow_rate_annual * days / 365
    credit = 105_000.0 * _RATE * days / 365
    assert net == pytest.approx(borrow - credit, abs=1e-9)
    assert portfolio.cash == pytest.approx(105_000.0 - net, abs=1e-9)


def test_apply_financing_zero_rate_and_non_positive_days_earn_nothing() -> None:
    """A zero rate gives no credit and `days_held <= 0` returns 0.0 with cash unchanged."""
    portfolio = _book(100_000.0)
    assert portfolio.apply_financing({}, 0.0, 30) == pytest.approx(0.0, abs=1e-12)
    assert portfolio.cash == pytest.approx(100_000.0, abs=1e-9)
    for days in (0, -3):
        assert portfolio.apply_financing({}, _RATE, days) == 0.0
        assert portfolio.cash == pytest.approx(100_000.0, abs=1e-9)


class _IdleCommander:
    """Never trades."""

    def decide(
        self,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
        scoreboard: object,
        instrument_returns: dict[str, float] | None = None,
    ) -> list[PortfolioDecision]:
        return []


def _frame() -> pd.DataFrame:
    close = pd.Series(100.0, index=_DAYS)
    return pd.DataFrame(
        {
            ("AAPL", "open"): close,
            ("AAPL", "close"): close,
            ("AAPL", "log_return"): np.log(close / close.shift(1)),
        }
    )


def test_engine_all_cash_book_compounds_interest_into_nav() -> None:
    """An idle cash book compounds actual/365 interest per calendar gap into NAV."""
    daily_rate = _RATE / settings.firm.trading_days_per_year
    result = BacktestEngine(
        commander=_IdleCommander(),  # type: ignore[arg-type]
        portfolio=_book(_CAPITAL),
        track_record=TrackRecord(),
        data_loader=lambda _assets, _start, _end: _frame(),
        start_date=_DAYS[0].date(),
        end_date=_DAYS[-1].date(),
        assets=["AAPL"],
        risk_free_rate_series=pd.Series(daily_rate, index=_DAYS),
    ).run()

    gaps = np.diff(_DAYS.values).astype("timedelta64[D]").astype(int)
    expected = _CAPITAL * float(np.prod(1.0 + _RATE * gaps / 365))
    values = np.array([value for _, value in result.portfolio_value_history])

    assert values[-1] == pytest.approx(expected, rel=1e-9)
    assert result.total_financing_cost < 0.0
    assert result.total_financing_cost == pytest.approx(-(values[-1] - _CAPITAL), rel=1e-9)

    # The benchmark accrues the same rate over the same calendar gap, so a
    # cash book's excess return is zero every day and its ratios report 0.0.
    excess = values[1:] / values[:-1] - 1.0 - _RATE * gaps / 365
    assert np.max(np.abs(excess)) < 1e-12
    assert result.final_sharpe_ratio == 0.0
    assert result.final_sortino_ratio == 0.0
