# test_runner_history_horizon.py

"""AT-140: under a history horizon far shorter than its window, a firm record writes
analyst_availability.json, unfilled_orders.json and agentic_meta.json from its engine checkpoint
with the same bytes as the files built from the checkpoint's full read
(`checkpoint_store.read_all`), at a pause and at the end of the resumed record, and a paused and
resumed record ends with the files of an unbroken one. The window crosses a year end, and its
unfilled orders and unavailable and failed signals fall on days long outside the horizon."""

import json
from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pydantic import PrivateAttr

from agentic_trading.agents.base import AnalystAgent, ManagerAgent
from agentic_trading.agents.commander import Commander
from agentic_trading.agents.portfolio_construction import PortfolioConstructor
from agentic_trading.agents.schemas import AgentSignal, PortfolioDecision, UnfilledOrder
from agentic_trading.backtest import checkpoint_store
from agentic_trading.backtest.engine_inputs import EngineInputs
from agentic_trading.config import temporary_overrides
from agentic_trading.data.trading_calendar import sessions_between
from agentic_trading.firm_record.availability import analyst_availability
from agentic_trading.firm_record.runner import FirmRecordResult, run_firm_record
from agentic_trading.firm_record.submission import (
    AgenticMeta,
    write_analyst_availability,
    write_unfilled_orders,
)
from agentic_trading.scoring.weighting import EqualWeight

_T = datetime(2026, 11, 6, tzinfo=UTC)
# The smallest horizon the measured-IC window (at least 20 calendar days) fits inside.
_HORIZON_DAYS = 20
_WARMUP = pd.DatetimeIndex(sessions_between(date(2015, 10, 1), date(2015, 10, 30)))
_START, _END = date(2015, 11, 2), date(2016, 2, 26)
_SESSIONS = pd.DatetimeIndex(sessions_between(_START, _END))
_UNIVERSE = ["SPY", "QQQ"]
# Sessions whose SPY open is masked, so the fill due that morning has no price.
_NAN_OPEN_DAYS = _SESSIONS[[3, 30, 70]]
# A session QQQ has no bar on: its decision carries the signals of the session before.
_GAP_DAY = _SESSIONS[12]
_PAUSE_AFTER = 45
_ROSTER = ["price_stub_analyst", "patchy_stub_analyst", "flaky_stub_analyst"]
_REPORTS = ("analyst_availability.json", "unfilled_orders.json", "agentic_meta.json")


class _PriceAnalyst(AnalystAgent):
    """Long after an up close and short after a down one."""

    agent_id: str = "price_stub_analyst"

    def generate_signal(
        self, asset: str, market_data: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> AgentSignal:
        returns = market_data[(asset, "log_return")].dropna()
        up = returns.empty or float(returns.iloc[-1]) >= 0.0
        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction="long" if up else "short",
            confidence=0.4,
            rationale="stub",
        )


class _PatchyAnalyst(AnalystAgent):
    """Unavailable on every third signal, long otherwise."""

    agent_id: str = "patchy_stub_analyst"
    _calls: int = PrivateAttr(default=0)

    def generate_signal(
        self, asset: str, market_data: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> AgentSignal:
        self._calls += 1
        if self._calls % 3 == 0:
            return self.unavailable_signal(asset, as_of_date, "no data")
        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction="long",
            confidence=0.3,
            rationale="stub",
        )


class _FlakyAnalyst(AnalystAgent):
    """Fails on every fifth signal, long otherwise."""

    agent_id: str = "flaky_stub_analyst"
    _calls: int = PrivateAttr(default=0)

    def generate_signal(
        self, asset: str, market_data: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> AgentSignal:
        self._calls += 1
        if self._calls % 5 == 0:
            raise RuntimeError("model failed")
        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction="long",
            confidence=0.2,
            rationale="stub",
        )


class _ViewConstructor(PortfolioConstructor):
    def construct(self, views, returns_history, as_of_date, previous_weights=None):
        return {asset: 0.3 * view.net_signal for asset, view in views.items()}


class _PassThroughManager(ManagerAgent):
    def scale_factor(self, signal, market_data, as_of_date) -> float:
        return 1.0


def _commander() -> Commander:
    manager = _PassThroughManager(
        agent_id="broad_manager",
        managed_analysts=[_PriceAnalyst(), _PatchyAnalyst(), _FlakyAnalyst()],
        weighting_strategy=EqualWeight(),
        portfolio_constructor=_ViewConstructor(),
    )
    return Commander(managers=[manager], weighting_strategy=EqualWeight())


def _frame() -> pd.DataFrame:
    """Warm-up then the window on random walks; QQQ lacks `_GAP_DAY`, SPY's open is masked."""
    index = _WARMUP.append(_SESSIONS)
    rng = np.random.default_rng(11)
    columns: dict[tuple[str, str], pd.Series] = {}
    for asset in _UNIVERSE:
        close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, len(index)))), index)
        if asset == "QQQ":
            close[_GAP_DAY] = np.nan
        present = close.dropna()
        columns[(asset, "open")] = close.copy()
        columns[(asset, "close")] = close
        columns[(asset, "log_return")] = np.log(present / present.shift(1)).reindex(index)
    frame = pd.DataFrame(columns)
    frame.loc[_NAN_OPEN_DAYS, ("SPY", "open")] = np.nan
    return frame


def _run(store_dir: Path, **kwargs: Any) -> FirmRecordResult:
    inputs = EngineInputs(
        assets=_UNIVERSE, market_data=_frame(), market_data_source="lakehouse data"
    )
    return run_firm_record(
        _commander(), _UNIVERSE, _START, _END, store_dir, inputs,
        lake_time=_T, lake_version="lake", quantcore_version="0.3.1", **kwargs,
    )  # fmt: skip


def _reports(directory: Path) -> dict[str, bytes]:
    return {name: (directory / name).read_bytes() for name in _REPORTS}


def _full_read_reports(
    store_dir: Path, result: FirmRecordResult, out_dir: Path
) -> dict[str, bytes]:
    """The reports built from `read_all` of the record's checkpoint, starting from its own meta."""
    connection = checkpoint_store.connect(
        store_dir / f"{result.spec_hash}.checkpoint.sqlite", resume=True
    )
    try:
        payload = checkpoint_store.read_all(connection)
    finally:
        connection.close()
    assert payload is not None
    meta = AgenticMeta.model_validate_json(
        (store_dir / result.spec_hash / "agentic_meta.json").read_text()
    )
    orders = [UnfilledOrder.model_validate(order) for order in payload["unfilled_orders"]]
    decisions_by_day = {
        date.fromisoformat(day): [PortfolioDecision.model_validate(entry) for entry in decisions]
        for day, decisions in payload["decisions_by_day"].items()
    }
    out_dir.mkdir(parents=True)
    meta = write_unfilled_orders(out_dir, orders, meta)
    write_analyst_availability(out_dir, analyst_availability(decisions_by_day, _ROSTER), meta)
    return _reports(out_dir)


@pytest.fixture(autouse=True)
def _short_horizon() -> Iterator[None]:
    """A 20-day firm history horizon, with the measured-IC window shortened to fit inside it."""
    with temporary_overrides(
        {"firm.history_horizon_days": _HORIZON_DAYS, "portfolio.ic_lookback_days": _HORIZON_DAYS}
    ):
        yield


def test_checkpoint_reports_match_the_full_read(tmp_path: Path) -> None:
    """Paused, resumed and unbroken records write the reports a full checkpoint read builds."""
    paused = _run(tmp_path / "paused", max_days=_PAUSE_AFTER)
    assert paused.last_completed == _SESSIONS[_PAUSE_AFTER - 1].date()
    assert _reports(tmp_path / "paused" / paused.spec_hash) == _full_read_reports(
        tmp_path / "paused", paused, tmp_path / "full_read_paused"
    )

    resumed = _run(tmp_path / "paused")
    unbroken = _run(tmp_path / "unbroken")
    expected = _full_read_reports(tmp_path / "unbroken", unbroken, tmp_path / "full_read")

    assert resumed.complete and unbroken.complete
    assert (_END - _START).days > 4 * _HORIZON_DAYS
    assert _reports(tmp_path / "unbroken" / unbroken.spec_hash) == expected
    assert _reports(tmp_path / "paused" / resumed.spec_hash) == expected
    availability = json.loads(expected["analyst_availability.json"])
    assert list(availability["patchy_stub_analyst"]) == ["2015", "2016", "total"]
    assert availability["price_stub_analyst"]["2015"]["active"] > 0
    assert availability["patchy_stub_analyst"]["2015"]["unavailable"] > 0
    assert availability["flaky_stub_analyst"]["2015"]["error"] > 0
    spy_unfilled = {
        order["as_of_date"]
        for order in json.loads(expected["unfilled_orders.json"])["orders"]
        if order["asset"] == "SPY"
    }
    assert {day.date().isoformat() for day in _NAN_OPEN_DAYS} <= spy_unfilled
