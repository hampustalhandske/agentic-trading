# test_checkpoint_lake_time.py

"""AT-106: a checkpointed `BacktestEngine` run stores its lake time T; a resume under a session
with a different T raises `LakeTimeError`, and one under the same T continues."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from agentic_trading.agents.schemas import PortfolioDecision
from agentic_trading.backtest.engine import BacktestEngine
from agentic_trading.backtest.portfolio import Portfolio
from agentic_trading.data.loaders import lakehouse_loader as ll
from agentic_trading.scoring.track_record import TrackRecord

_INDEX = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
_T1 = datetime(2024, 6, 1, tzinfo=UTC)
_T2 = datetime(2024, 6, 2, tzinfo=UTC)


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


def _engine(db_path: Path, *, resume: bool) -> BacktestEngine:
    close = pd.Series([100.0, 101.0, 102.0, 103.0], index=_INDEX)
    frame = pd.DataFrame(
        {
            ("AAPL", "open"): close,
            ("AAPL", "close"): close,
            ("AAPL", "log_return"): np.log(close / close.shift(1)),
        }
    )
    return BacktestEngine(
        commander=_Commander(),  # type: ignore[arg-type]
        portfolio=Portfolio(cash=100_000.0, positions={}, commission_rate=0.0, slippage_bps=0.0),
        track_record=TrackRecord(),
        data_loader=lambda _assets, _start, _end: frame,
        start_date=_INDEX[0].date(),
        end_date=_INDEX[-1].date(),
        assets=["AAPL"],
        execution_lag_days=1,
        checkpoint_path=db_path,
        resume=resume,
    )


def test_resume_requires_the_checkpointed_lake_time(fake_lake: Any, tmp_path: Path) -> None:
    """Resuming under another T raises LakeTimeError; the stored T reproduces the run."""
    db_path = tmp_path / "checkpoint.db"
    with ll.lake_session(_T1):
        fresh = _engine(db_path, resume=False).run()

    with pytest.raises(ll.LakeTimeError), ll.lake_session(_T2):
        _engine(db_path, resume=True).run()
    with ll.lake_session(_T1):
        resumed = _engine(db_path, resume=True).run()

    assert resumed.portfolio_value_history == fresh.portfolio_value_history
