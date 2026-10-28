# test_cash_interest_rate_timing.py

"""Interest over the gap from valuation date t-1 to t accrues at the
risk-free rate in force on t-1, both on cash and in the benchmark that
excess returns subtract."""

import itertools

import numpy as np
import pandas as pd
import pytest

from agentic_trading.agents.schemas import PortfolioDecision
from agentic_trading.backtest.engine import BacktestEngine
from agentic_trading.backtest.portfolio import Portfolio
from agentic_trading.config import settings
from agentic_trading.scoring.track_record import TrackRecord

_OLD_RATE = 0.02
_NEW_RATE = 0.06
# A Monday, so the first gap after the step is the Friday-to-Monday one.
_STEP_DATE = pd.Timestamp("2024-01-08")
_CAPITAL = 100_000.0
_DAYS = pd.bdate_range("2024-01-02", "2024-01-19")


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


def _stepped_daily_rate() -> pd.Series:
    """Calendar-indexed daily decimal rate, as `load_risk_free_rate_series`
    returns it: `_OLD_RATE` before `_STEP_DATE` and `_NEW_RATE` from it on."""
    calendar = pd.date_range(_DAYS[0], _DAYS[-1], freq="D")
    annual = np.where(calendar < _STEP_DATE, _OLD_RATE, _NEW_RATE)
    return pd.Series(annual / settings.firm.trading_days_per_year, index=calendar)


def test_first_gap_after_a_rate_step_accrues_at_the_previous_rate() -> None:
    """The Friday 2024-01-05 to Monday 2024-01-08 gap accrues three days at
    the old rate even though the new rate is in force on Monday; the next
    gap accrues at the new rate. The benchmark uses the same rate, so the
    cash book's excess return is zero every day."""
    result = BacktestEngine(
        commander=_IdleCommander(),  # type: ignore[arg-type]
        portfolio=Portfolio(cash=_CAPITAL, positions={}, commission_rate=0.0, slippage_bps=0.0),
        track_record=TrackRecord(),
        data_loader=lambda _assets, _start, _end: _frame(),
        start_date=_DAYS[0].date(),
        end_date=_DAYS[-1].date(),
        assets=["AAPL"],
        risk_free_rate_series=_stepped_daily_rate(),
    ).run()

    history = dict(result.portfolio_value_history)
    friday, monday, tuesday = (
        pd.Timestamp(day).date() for day in ("2024-01-05", "2024-01-08", "2024-01-09")
    )

    assert history[monday] / history[friday] - 1.0 == pytest.approx(_OLD_RATE * 3 / 365, abs=1e-15)
    assert history[tuesday] / history[monday] - 1.0 == pytest.approx(_NEW_RATE * 1 / 365, abs=1e-15)

    dates = [day for day, _ in result.portfolio_value_history]
    values = np.array([value for _, value in result.portfolio_value_history])
    gaps = np.array([(current - previous).days for previous, current in itertools.pairwise(dates)])
    rates = np.array(
        [_OLD_RATE if pd.Timestamp(day) < _STEP_DATE else _NEW_RATE for day in dates[:-1]]
    )
    expected_final = _CAPITAL * float(np.prod(1.0 + rates * gaps / 365))

    assert values[-1] == pytest.approx(expected_final, rel=1e-12)
    assert result.total_financing_cost == pytest.approx(-(values[-1] - _CAPITAL), rel=1e-9)
    excess = values[1:] / values[:-1] - 1.0 - rates * gaps / 365
    assert np.max(np.abs(excess)) < 1e-12
    assert result.final_sharpe_ratio == 0.0
    assert result.final_sortino_ratio == 0.0
