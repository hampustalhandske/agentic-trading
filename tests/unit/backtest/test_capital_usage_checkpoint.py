"""Capital usage persists through the checkpoint store and a resumed run."""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from agentic_trading.agents.schemas import CapitalUsage, PortfolioDecision
from agentic_trading.backtest import checkpoint_store
from agentic_trading.backtest.engine import BacktestEngine
from agentic_trading.backtest.portfolio import Portfolio
from agentic_trading.scoring.track_record import TrackRecord

_INDEX = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])


def test_store_round_trips_capital_usage(tmp_path: Path) -> None:
    connection = checkpoint_store.connect(tmp_path / "checkpoint.db")
    for index, day in enumerate([date(2024, 1, 2), date(2024, 1, 3)]):
        checkpoint_store.append_day(
            connection,
            as_of_date=day,
            portfolio_value=100_000.0,
            new_track_record=[],
            decisions=[],
            new_trades=[],
            next_day_index=index + 1,
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 3),
            assets=["AAPL"],
            portfolio={"cash": 100_000.0},
            signal_health={},
            pending_decisions=[],
            capital_usage=CapitalUsage(
                as_of_date=day, idle_pod_share=None if index == 0 else 0.25, cash_share=0.5
            ),
        )

    payload = checkpoint_store.read_all(connection)

    assert payload is not None
    assert [CapitalUsage.model_validate(entry) for entry in payload["capital_usage"]] == [
        CapitalUsage(as_of_date=date(2024, 1, 2), idle_pod_share=None, cash_share=0.5),
        CapitalUsage(as_of_date=date(2024, 1, 3), idle_pod_share=0.25, cash_share=0.5),
    ]


class _Commander:
    def decide(self, market_data, as_of_date, scoreboard, instrument_returns=None):
        return [
            PortfolioDecision(
                asset="AAPL",
                as_of_date=pd.Timestamp(as_of_date).date(),
                target_weight=0.5,
                contributing_signals=[],
            )
        ]

    def idle_pod_share(self) -> float:
        return 0.3


def _frame() -> pd.DataFrame:
    close = pd.Series([100.0, 101.0, 102.0, 103.0], index=_INDEX)
    return pd.DataFrame(
        {
            ("AAPL", "open"): close,
            ("AAPL", "close"): close,
            ("AAPL", "log_return"): np.log(close / close.shift(1)),
        }
    )


def _engine(end: pd.Timestamp, db_path: Path, *, resume: bool) -> BacktestEngine:
    return BacktestEngine(
        commander=_Commander(),  # type: ignore[arg-type]
        portfolio=Portfolio(cash=100_000.0, positions={}, commission_rate=0.0, slippage_bps=0.0),
        track_record=TrackRecord(),
        data_loader=lambda _assets, _start, _end: _frame(),
        start_date=_INDEX[0].date(),
        end_date=end.date(),
        assets=["AAPL"],
        execution_lag_days=1,
        checkpoint_path=db_path,
        resume=resume,
    )


def test_resumed_run_has_complete_capital_usage(tmp_path: Path) -> None:
    db_path = tmp_path / "checkpoint.db"
    fresh = _engine(_INDEX[-1], db_path, resume=False).run()

    result = _engine(_INDEX[-1], db_path, resume=True).run()

    assert result.capital_usage == fresh.capital_usage

    assert [entry.as_of_date for entry in result.capital_usage] == [day.date() for day in _INDEX]
    assert all(entry.idle_pod_share == 0.3 for entry in result.capital_usage)


def test_store_round_trips_pod_returns(tmp_path: Path) -> None:
    connection = checkpoint_store.connect(tmp_path / "checkpoint.db")
    days = [date(2024, 1, 2), date(2024, 1, 3)]
    for index, day in enumerate(days):
        checkpoint_store.append_day(
            connection,
            as_of_date=day,
            portfolio_value=100_000.0,
            new_track_record=[],
            decisions=[],
            new_trades=[],
            next_day_index=index + 1,
            start_date=days[0],
            end_date=days[-1],
            assets=["AAPL"],
            portfolio={"cash": 100_000.0},
            signal_health={},
            pending_decisions=[],
            new_pod_returns=[("pod_a", day, 0.01 * (index + 1)), ("pod_b", day, -0.02)],
        )

    payload = checkpoint_store.read_all(connection)

    assert payload is not None
    assert payload["pod_return_history"] == {
        "pod_a": [("2024-01-02", 0.01), ("2024-01-03", 0.02)],
        "pod_b": [("2024-01-02", -0.02), ("2024-01-03", -0.02)],
    }
