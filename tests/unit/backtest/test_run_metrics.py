# test_run_metrics.py

"""The engine's whole-run results as pure functions of a run's histories (`run_metrics`,
`risk_free_accrual`, `benchmark_returns_on`, `source_returns_of`, `source_performance_of`,
`execution_cost_totals`, `pod_performance_of`): what `BacktestEngine.run` returns and what an
offline reader of its checkpoint computes."""

import itertools
import math
from datetime import date

import numpy as np
import pandas as pd
import pytest
from quantcore.performance.metrics import (
    information_ratio,
    maximum_drawdown,
    sharpe_ratio,
    sortino_ratio,
)

from agentic_trading.agents.schemas import TradeRecord
from agentic_trading.backtest.engine import (
    ZERO_EXCESS_RETURN_TOLERANCE,
    _pod_performance,
    benchmark_returns_on,
    execution_cost_totals,
    pod_performance_of,
    risk_free_accrual,
    risk_free_rate_annual,
    run_metrics,
    source_performance_of,
    source_returns_of,
)
from agentic_trading.config import settings

_DATES = [date(2024, 1, 5), date(2024, 1, 8), date(2024, 1, 9), date(2024, 1, 10)]
_VALUES = [100_000.0, 101_000.0, 100_500.0, 102_000.0]


def _rates(annual: float) -> pd.Series:
    calendar = pd.date_range("2024-01-01", "2024-01-31", freq="D")
    return pd.Series(annual / settings.firm.trading_days_per_year, index=calendar)


def _returns(values: list[float]) -> np.ndarray:
    return np.array([(b - a) / a for a, b in itertools.pairwise(values)])


def test_the_risk_free_accrual_runs_at_the_rate_in_force_when_each_gap_starts() -> None:
    rates = _rates(0.05)
    rates.loc[pd.Timestamp(date(2024, 1, 8))] = 0.10 / settings.firm.trading_days_per_year

    accrual = risk_free_accrual(rates, _DATES)

    assert accrual.tolist() == pytest.approx([0.05 * 3 / 365, 0.10 * 1 / 365, 0.05 * 1 / 365])


def test_a_negative_or_missing_rate_accrues_nothing() -> None:
    assert risk_free_rate_annual(_rates(-0.01), _DATES[0]) == 0.0
    assert risk_free_rate_annual(None, _DATES[0]) == 0.0
    assert risk_free_rate_annual(_rates(0.05), date(2025, 1, 1)) == 0.0
    assert risk_free_accrual(None, _DATES).tolist() == [0.0, 0.0, 0.0]
    assert risk_free_accrual(_rates(0.05), _DATES[:1]).size == 0


def test_benchmark_returns_are_read_on_the_return_dates_or_not_at_all() -> None:
    benchmark = pd.Series([0.01, -0.02, 0.03], index=pd.to_datetime(_DATES[1:]))

    assert benchmark_returns_on(benchmark, _DATES[1:]).tolist() == [0.01, -0.02, 0.03]
    assert benchmark_returns_on(benchmark, _DATES) is None
    assert benchmark_returns_on(None, _DATES[1:]) is None


def test_sharpe_and_sortino_are_on_excess_returns_and_the_information_ratio_on_raw() -> None:
    rates = _rates(0.04)
    benchmark = pd.Series([0.004, -0.001, 0.002], index=pd.to_datetime(_DATES[1:]))
    returns = _returns(_VALUES)
    excess = returns - risk_free_accrual(rates, _DATES)

    metrics = run_metrics(
        list(zip(_DATES, _VALUES, strict=True)),
        risk_free_rate_series=rates,
        benchmark_returns=benchmark,
    )

    assert metrics.final_sharpe_ratio == sharpe_ratio(excess)
    assert metrics.final_sortino_ratio == sortino_ratio(excess)
    assert metrics.final_information_ratio == information_ratio(returns, benchmark.to_numpy())
    assert metrics.final_max_drawdown == maximum_drawdown(returns)
    assert metrics.final_hit_rate == pytest.approx(2 / 3)


def test_a_book_earning_exactly_the_risk_free_rate_has_no_sharpe_or_sortino() -> None:
    rates = _rates(0.04)
    values = [100_000.0]
    for previous, current in itertools.pairwise(_DATES):
        values.append(values[-1] * (1.0 + 0.04 * (current - previous).days / 365.0))

    metrics = run_metrics(
        list(zip(_DATES, values, strict=True)), risk_free_rate_series=rates, benchmark_returns=None
    )

    excess = _returns(values) - risk_free_accrual(rates, _DATES)
    assert np.all(np.abs(excess) <= ZERO_EXCESS_RETURN_TOLERANCE)
    assert metrics.final_sharpe_ratio is None
    assert metrics.final_sortino_ratio is None
    assert metrics.final_information_ratio is None


def test_fewer_than_two_returns_leave_every_ratio_undefined() -> None:
    metrics = run_metrics(
        list(zip(_DATES[:2], _VALUES[:2], strict=True)),
        risk_free_rate_series=None,
        benchmark_returns=None,
    )

    assert (
        metrics.final_sharpe_ratio,
        metrics.final_sortino_ratio,
        metrics.final_calmar_ratio,
        metrics.final_profit_factor,
        metrics.final_information_ratio,
    ) == (None, None, None, None, None)
    assert run_metrics([], risk_free_rate_series=None, benchmark_returns=None).final_hit_rate == 0.0


def test_source_returns_keep_row_order_and_first_appearance() -> None:
    rows = [("b", 0.1, 0.2), ("a", 0.3, 0.4), ("b", -0.1, -0.2)]

    assert source_returns_of(rows) == {"b": ([0.1, -0.1], [0.2, -0.2]), "a": ([0.3], [0.4])}
    assert list(source_returns_of(rows)) == ["b", "a"]


def test_source_performance_reports_rates_sums_sharpe_and_drawdown() -> None:
    returns = [0.01, -0.02, 0.005]
    raw = [0.02, -0.01, 0.0]

    performance = source_performance_of({"analyst": (returns, raw)})["analyst"]

    mean = float(np.mean(returns))
    assert performance["cumulative_return"] == pytest.approx(mean * 252.0)
    assert performance["total_attributed_return"] == pytest.approx(sum(returns))
    assert performance["sharpe_ratio"] == pytest.approx(
        mean / float(np.std(returns, ddof=1)) * math.sqrt(252)
    )
    assert performance["max_drawdown"] == pytest.approx(-0.02)
    assert performance["observations"] == 3.0
    assert performance["raw_total_attributed_return"] == pytest.approx(0.01)
    assert performance["raw_max_drawdown"] == pytest.approx(-0.01)


def test_a_source_with_one_return_has_a_zero_sharpe() -> None:
    performance = source_performance_of({"analyst": ([0.01], [0.01])})["analyst"]

    assert performance["sharpe_ratio"] == 0.0
    assert performance["max_drawdown"] == 0.0


def _trade(**costs: float) -> TradeRecord:
    return TradeRecord(
        as_of_date=_DATES[0],
        asset="AAPL",
        action="buy",
        target_weight=0.1,
        previous_weight=0.0,
        target_shares=10.0,
        delta_shares=10.0,
        price=100.0,
        notional=1_000.0,
        **costs,
    )


def test_execution_cost_totals_sum_each_kind_and_all_of_them() -> None:
    trades = [
        _trade(transaction_cost=1.0, slippage_cost=0.5, impact_cost=0.25, spread_cost=0.0),
        _trade(transaction_cost=2.0, slippage_cost=0.0, impact_cost=0.0, spread_cost=0.75),
    ]

    totals = execution_cost_totals(trades)

    assert totals.total_transaction_cost == 3.0
    assert totals.total_slippage_cost == 0.5
    assert totals.total_impact_cost == 0.25
    assert totals.total_spread_cost == 0.75
    assert totals.total_execution_cost == 4.5
    assert execution_cost_totals([]).total_execution_cost == 0.0


def test_pod_performance_is_each_pods_own_over_its_dated_returns() -> None:
    history = {
        "pod_b": [(_DATES[0], 0.01), (_DATES[1], -0.005), (_DATES[2], 0.002)],
        "pod_a": [(_DATES[0], 0.0)],
    }

    performance = pod_performance_of(history)

    assert list(performance) == ["pod_b", "pod_a"]
    assert performance["pod_b"] == _pod_performance([0.01, -0.005, 0.002])
    assert performance["pod_a"].observations == 1
