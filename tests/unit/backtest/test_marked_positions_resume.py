# test_marked_positions_resume.py

"""A book holding a future and an FX pair, rolling and accruing carry, resumes
from its checkpoint after any day to a `BacktestResult` byte-identical to the
uninterrupted run's: the marked positions and their marks are checkpointed
and restored."""

from collections.abc import Callable
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agentic_trading.agents.schemas import PortfolioDecision
from agentic_trading.backtest.engine import BacktestEngine
from agentic_trading.backtest.portfolio import Portfolio
from agentic_trading.scoring.track_record import TrackRecord

_INDEX = pd.bdate_range("2024-01-02", periods=8)
_DAYS = [stamp.date() for stamp in _INDEX]


class _InterruptError(Exception):
    pass


class _AlternatingCommander:
    """Long both instruments on even days and short them on odd days."""

    def decide(
        self,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
        scoreboard: object,
        instrument_returns: dict[str, float] | None = None,
    ) -> list[PortfolioDecision]:
        sign = 1.0 if as_of_date.day % 2 == 0 else -1.0
        return [
            PortfolioDecision(
                asset=asset,
                as_of_date=as_of_date.date(),
                target_weight=sign * weight,
                contributing_signals=[],
            )
            for asset, weight in (("CL=F", 0.4), ("USDJPY=X", 0.3))
        ]


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    future = pd.Series(70.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, len(_INDEX)))), _INDEX)
    spot = pd.Series(150.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, len(_INDEX)))), _INDEX)
    return pd.DataFrame(
        {
            ("CL=F", "open"): future,
            ("CL=F", "close"): future,
            ("CL=F", "log_return"): np.log(future / future.shift(1)),
            ("CL=F", "roll"): pd.Series([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0], _INDEX),
            ("USDJPY=X", "open"): spot.shift(1).fillna(spot.iloc[0]),
            ("USDJPY=X", "close"): spot,
            ("USDJPY=X", "log_return"): np.log(spot / spot.shift(1)),
        }
    )


def _engine(checkpoint_path: Path | None = None, resume: bool = False) -> BacktestEngine:
    return BacktestEngine(
        commander=_AlternatingCommander(),  # type: ignore[arg-type]
        portfolio=Portfolio(cash=1_000_000.0, positions={}, commission_rate=5e-4, slippage_bps=2.0),
        track_record=TrackRecord(),
        data_loader=lambda _assets, _start, _end: _frame(),
        start_date=_DAYS[0],
        end_date=_DAYS[-1],
        assets=["CL=F", "USDJPY=X"],
        execution_lag_days=1,
        checkpoint_path=checkpoint_path,
        resume=resume,
        policy_rates=pd.DataFrame(
            {"USD": [0.05, 0.0525], "JPY": [-0.001, 0.001]},
            index=pd.to_datetime(["2023-12-29", "2024-01-05"]),
        ),
    )


def _stop_after(day: int) -> Callable[..., None]:
    def stop(day_index: int, *_args: object) -> None:
        if day_index == day:
            raise _InterruptError

    return stop


def test_resume_with_marked_positions_is_byte_identical(tmp_path: Path) -> None:
    expected = _engine().run()
    expected_json = expected.model_dump_json()
    assert any(trade.action == "roll" for trade in expected.trade_log)
    assert expected.total_financing_cost != 0.0

    for k in range(len(_DAYS) - 1):
        checkpoint_path = tmp_path / f"checkpoint_{k}.db"
        with pytest.raises(_InterruptError):
            _engine(checkpoint_path).run(on_day_complete=_stop_after(k))

        resumed = _engine(checkpoint_path, resume=True).run()

        assert resumed.model_dump_json() == expected_json, f"interrupted after day {k}"


def test_portfolio_dump_carries_the_marked_positions() -> None:
    portfolio = Portfolio(cash=1_000_000.0, positions={})
    portfolio.apply_decision(
        PortfolioDecision(
            asset="CL=F",
            as_of_date=date(2024, 1, 2),
            target_weight=0.5,
            contributing_signals=[],
        ),
        50.0,
        1_000_000.0,
    )

    restored = Portfolio.model_validate(portfolio.model_dump(mode="json"))

    assert restored.marked_positions == portfolio.marked_positions
