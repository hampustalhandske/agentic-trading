# test_resume_byte_identity.py

"""A backtest interrupted after any day and resumed from its checkpoint gives a `BacktestResult`
whose JSON is byte-identical to the uninterrupted run's, with an analyst and a manager whose
signals rest on pydantic private state (a `RefitCache` and day counters) that the checkpoint
carries. A resume under different strategy settings or a different roster raises `ValueError`; a
resume that differs only in runtime-only settings reproduces the uninterrupted run."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pydantic import PrivateAttr

from agentic_trading.agents._market_math import RefitCache
from agentic_trading.agents.base import AnalystAgent, ManagerAgent
from agentic_trading.agents.commander import Commander
from agentic_trading.agents.portfolio_construction import PortfolioConstructor
from agentic_trading.agents.schemas import AgentSignal, PortfolioDecision
from agentic_trading.backtest.engine import BacktestEngine
from agentic_trading.backtest.portfolio import Portfolio
from agentic_trading.config import temporary_overrides
from agentic_trading.scoring.scoreboard import PerformanceScoreboard
from agentic_trading.scoring.track_record import TrackRecord
from agentic_trading.scoring.weighting import EqualWeight

_WARMUP = 10
# Ten warm-up bars, then Monday 2024-01-08 to Monday 2024-01-15: six sessions.
_INDEX = pd.bdate_range("2023-12-25", periods=_WARMUP + 6)
_SESSIONS = _INDEX[_WARMUP:]
_DAYS = [day.date() for day in _SESSIONS]
# Calendar days a `_StatefulAnalyst` fit is reused for: it refits on the 8th, 11th and 15th.
_REFIT_INTERVAL_DAYS = 3


class _InterruptError(Exception):
    pass


class _StatefulAnalyst(AnalystAgent):
    """Long on odd and short on even days it has signalled on, at a confidence read from its
    signal count and from a fit (the refit day's day of year) refit every `_REFIT_INTERVAL_DAYS`
    calendar days."""

    agent_id: str = "stateful_analyst"
    base_confidence: float = 0.2
    _refit_cache: RefitCache = PrivateAttr(
        default_factory=lambda: RefitCache(refit_interval_days=_REFIT_INTERVAL_DAYS)
    )
    _days_seen: int = PrivateAttr(default=0)

    def generate_signal(
        self, asset: str, market_data: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> AgentSignal:
        self._days_seen += 1
        fit = self._refit_cache.get_or_refit(asset, as_of_date, lambda: as_of_date.dayofyear)
        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction="long" if self._days_seen % 2 else "short",
            confidence=self.base_confidence + 0.1 * (fit % 4) + 0.05 * (self._days_seen % 3),
            rationale="stub",
        )


class _ViewConstructor(PortfolioConstructor):
    """Sizes each asset at half its pod view's net signal."""

    def construct(self, views, returns_history, as_of_date, previous_weights=None):
        return {asset: 0.5 * view.net_signal for asset, view in views.items()}


class _CountingManager(ManagerAgent):
    """Scales every signal by the share of the run's sessions it has built a portfolio on."""

    agent_id: str = "counting_manager"
    _days_seen: int = PrivateAttr(default=0)

    def construct_portfolio(
        self,
        assets: list[str],
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
        scoreboard: PerformanceScoreboard,
    ) -> list[PortfolioDecision]:
        self._days_seen += 1
        return super().construct_portfolio(assets, market_data, as_of_date, scoreboard)

    def scale_factor(self, signal, market_data, as_of_date) -> float:
        return min(self._days_seen, len(_DAYS)) / len(_DAYS)


def _frame() -> pd.DataFrame:
    """AAPL on a seeded random walk: open at the previous close, zero return on the first bar."""
    rng = np.random.default_rng(5)
    close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, len(_INDEX)))), _INDEX)
    return pd.DataFrame(
        {
            ("AAPL", "open"): close.shift(1).fillna(close.iloc[0]),
            ("AAPL", "close"): close,
            ("AAPL", "log_return"): np.log(close / close.shift(1)).fillna(0.0),
        }
    )


def _commander(*analysts: AnalystAgent, assets: tuple[str, ...] | None = ("AAPL",)) -> Commander:
    """One `_CountingManager` pod over `analysts` (one `_StatefulAnalyst` when none are given)."""
    manager = _CountingManager(
        managed_analysts=list(analysts) or [_StatefulAnalyst()],
        weighting_strategy=EqualWeight(),
        portfolio_constructor=_ViewConstructor(),
        assets=assets,
    )
    return Commander(managers=[manager], weighting_strategy=EqualWeight())


def _engine(
    commander: Commander, checkpoint_path: Path | None = None, resume: bool = False
) -> BacktestEngine:
    return BacktestEngine(
        commander=commander,
        portfolio=Portfolio(cash=1_000_000.0, positions={}, commission_rate=5e-4, slippage_bps=2.0),
        track_record=TrackRecord(),
        data_loader=lambda _assets, _start, _end: _frame(),
        start_date=_DAYS[0],
        end_date=_DAYS[-1],
        assets=["AAPL"],
        execution_lag_days=1,
        checkpoint_path=checkpoint_path,
        resume=resume,
    )


def _stop_after(day: int) -> Callable[..., None]:
    def stop(day_index: int, *_args: object) -> None:
        if day_index == day:
            raise _InterruptError

    return stop


def _interrupted(checkpoint_path: Path, day: int = 2) -> Path:
    """`checkpoint_path`, holding the default run interrupted after `day`."""
    with pytest.raises(_InterruptError):
        _engine(_commander(), checkpoint_path).run(on_day_complete=_stop_after(day))
    return checkpoint_path


def _signal_path(analyst: _StatefulAnalyst, sessions: pd.DatetimeIndex) -> list[tuple[str, float]]:
    """The analyst's (direction, confidence) on each of `sessions`, in order."""
    frame = _frame()
    signals = [analyst.generate_signal("AAPL", frame, day) for day in sessions]
    return [(signal.direction, signal.confidence) for signal in signals]


def test_resume_after_any_day_is_byte_identical(tmp_path: Path) -> None:
    """Interrupted after any day k and resumed, the run's `BacktestResult` JSON equals the
    uninterrupted run's exactly, though the analyst's next signal differs without its state."""
    expected = _engine(_commander()).run()
    expected_json = expected.model_dump_json()
    assert expected.trade_log
    assert [day for day, _ in expected.portfolio_value_history] == _DAYS
    uninterrupted = _signal_path(_StatefulAnalyst(), _SESSIONS)

    for k in range(len(_DAYS)):
        if k + 1 < len(_DAYS):
            assert _signal_path(_StatefulAnalyst(), _SESSIONS[k + 1 :])[0] != uninterrupted[k + 1]
        checkpoint_path = _interrupted(tmp_path / f"checkpoint_{k}.db", k)

        result = _engine(_commander(), checkpoint_path, resume=True).run()

        assert result.model_dump_json() == expected_json, f"interrupted after day {k}"


@pytest.mark.parametrize(
    ("overrides", "changes_strategy"),
    [
        ({"firm.commission_rate": 0.001}, True),
        ({"firm.execution_lag_days": 0}, True),
        ({"analysts.tail_hedge_base_confidence": 0.3}, True),
        ({"stage_a.min_free_memory_pct": 35.0, "stage_a.min_free_disk_gb": 1.0}, False),
    ],
    ids=["commission_rate", "execution_lag_days", "analysts_setting", "runtime_only"],
)
def test_resume_checks_the_settings_fingerprint(
    tmp_path: Path, overrides: dict[str, Any], changes_strategy: bool
) -> None:
    """A resume under changed strategy settings raises `ValueError`; one under changed
    runtime-only settings reproduces the uninterrupted run byte for byte."""
    expected_json = _engine(_commander()).run().model_dump_json()
    checkpoint_path = _interrupted(tmp_path / "checkpoint.db")

    with temporary_overrides(overrides):
        resumed = _engine(_commander(), checkpoint_path, resume=True)
        if changes_strategy:
            with pytest.raises(ValueError):
                resumed.run()
        else:
            assert resumed.run().model_dump_json() == expected_json


@pytest.mark.parametrize(
    "build",
    [
        lambda: _commander(_StatefulAnalyst(base_confidence=0.3)),
        lambda: _commander(_StatefulAnalyst(agent_id="renamed_analyst")),
        lambda: _commander(_StatefulAnalyst(), _StatefulAnalyst(agent_id="second_analyst")),
        lambda: _commander(_StatefulAnalyst(), assets=None),
    ],
    ids=["analyst_parameter", "analyst_id", "extra_analyst", "manager_parameter"],
)
def test_resume_checks_the_roster_fingerprint(
    tmp_path: Path, build: Callable[[], Commander]
) -> None:
    """A resume whose roster differs from the checkpointed one in an analyst's or a manager's
    agent_id or constructor parameters, or in its analysts, raises `ValueError`."""
    checkpoint_path = _interrupted(tmp_path / "checkpoint.db")
    resumed = _engine(build(), checkpoint_path, resume=True)

    with pytest.raises(ValueError):
        resumed.run()
