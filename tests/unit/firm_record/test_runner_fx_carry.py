# test_runner_fx_carry.py

"""AT-23: a firm record over a spot FX pair accrues the pair's carry from the BIS policy rates
in its engine inputs (base minus quote rate, actual/365 over calendar days, on the USD
notional), and a record whose universe holds a pair without policy rates is refused before
anything is written."""

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from agentic_trading.agents.base import AnalystAgent, ManagerAgent
from agentic_trading.agents.commander import Commander
from agentic_trading.agents.portfolio_construction import PortfolioConstructor
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.backtest.engine_inputs import EngineInputs
from agentic_trading.backtest.portfolio import Portfolio
from agentic_trading.data.trading_calendar import sessions_between
from agentic_trading.firm_record import runner as runner_module
from agentic_trading.firm_record.runner import FirmRecordResult, run_firm_record
from agentic_trading.scoring.weighting import EqualWeight

_T = datetime(2026, 11, 6, tzinfo=UTC)
_PAIR = "USDJPY=X"
_UNIVERSE = [_PAIR]
# Thursday to the next Friday: the first carry gap after the first fill runs Friday to
# Monday, three calendar days.
_START, _END = date(2016, 1, 7), date(2016, 1, 15)
_WARMUP = pd.DatetimeIndex(sessions_between(date(2015, 12, 1), date(2016, 1, 6)))
_SESSIONS = pd.DatetimeIndex(sessions_between(_START, _END))
_USD_RATE, _JPY_RATE = 0.05, 0.001


class _LongAnalyst(AnalystAgent):
    agent_id: str = "long_stub_analyst"

    def generate_signal(
        self, asset: str, market_data: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> AgentSignal:
        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction="long",
            confidence=0.5,
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
        agent_id="fx_manager",
        managed_analysts=[_LongAnalyst()],
        weighting_strategy=EqualWeight(),
        portfolio_constructor=_ViewConstructor(),
        assets=None,
    )
    return Commander(managers=[manager], weighting_strategy=EqualWeight())


def _frame() -> pd.DataFrame:
    index = _WARMUP.append(_SESSIONS)
    rng = np.random.default_rng(7)
    close = pd.Series(120.0 * np.exp(np.cumsum(rng.normal(0.0, 0.004, len(index)))), index)
    return pd.DataFrame(
        {
            (_PAIR, "open"): close,
            (_PAIR, "close"): close,
            (_PAIR, "log_return"): np.log(close / close.shift(1)),
        }
    )


def _rates(usd: float, jpy: float) -> pd.DataFrame:
    """One policy-rate observation per currency, in force from before the warm-up."""
    return pd.DataFrame(
        {"USD": [usd], "JPY": [jpy]}, index=pd.DatetimeIndex(["2015-11-30"], name="date")
    )


def _inputs(policy_rates: pd.DataFrame) -> EngineInputs:
    return EngineInputs(
        assets=_UNIVERSE,
        market_data=_frame(),
        market_data_source="lakehouse data",
        policy_rates=policy_rates,
    )


def _run(store_dir: Path, inputs: EngineInputs, **kwargs: Any) -> FirmRecordResult:
    return run_firm_record(
        _commander(), _UNIVERSE, _START, _END, store_dir, inputs,
        lake_time=_T, lake_version="lake", quantcore_version="0.3.1", **kwargs,
    )  # fmt: skip


def _checkpoint(store_dir: Path, result: FirmRecordResult) -> dict[str, Any]:
    payload = runner_module._checkpoint_payload(store_dir / f"{result.spec_hash}.checkpoint.sqlite")
    assert payload is not None
    return payload


def _values(payload: dict[str, Any]) -> dict[date, float]:
    return {date.fromisoformat(day): value for day, value in payload["portfolio_value_history"]}


def _portfolio(payload: dict[str, Any]) -> Portfolio:
    raw = payload["portfolio"]
    return (
        Portfolio.model_validate_json(raw)
        if isinstance(raw, str)
        else Portfolio.model_validate(raw)
    )


def test_a_record_over_a_pair_accrues_its_carry(tmp_path: Path) -> None:
    """The pair is first held from the second session, a Friday; on the Monday after it the
    book is worth its USD notional x (USD rate - JPY rate) x 3 / 365 more than the same record
    at zero policy rates, which matches it to the cent up to the Friday."""
    friday, monday = _SESSIONS[1].date(), _SESSIONS[2].date()
    assert (friday.weekday(), (monday - friday).days) == (4, 3)

    carried_dir = tmp_path / "carried"
    paused = _run(carried_dir, _inputs(_rates(_USD_RATE, _JPY_RATE)), max_days=2)
    held = _portfolio(_checkpoint(carried_dir, paused)).marked_positions[_PAIR].quantity
    carried = _run(carried_dir, _inputs(_rates(_USD_RATE, _JPY_RATE)))
    flat = _run(tmp_path / "flat", _inputs(_rates(0.0, 0.0)))

    assert paused.last_completed == friday and held > 0.0
    assert carried.complete and flat.complete
    carried_values = _values(_checkpoint(carried_dir, carried))
    flat_values = _values(_checkpoint(tmp_path / "flat", flat))
    assert carried_values[friday] == pytest.approx(flat_values[friday], abs=1e-6)
    # USDJPY's base is USD, so one unit held is one USD of notional.
    expected = held * (_USD_RATE - _JPY_RATE) * 3 / 365.0
    assert carried_values[monday] - flat_values[monday] == pytest.approx(expected, rel=1e-9)


def test_a_pair_without_policy_rates_is_refused_before_anything_is_written(
    tmp_path: Path,
) -> None:
    """Engine inputs holding a spot FX pair cannot be built without policy rates, and
    run_firm_record refuses inputs that carry none, naming the pair, before writing."""
    with pytest.raises(ValueError, match=r"USDJPY=X"):
        EngineInputs(assets=_UNIVERSE, market_data=_frame(), market_data_source="lakehouse data")
    unrated = EngineInputs.model_construct(
        assets=_UNIVERSE,
        market_data=_frame(),
        market_data_source="lakehouse data",
        risk_free_rate_series=None,
        policy_rates=None,
        benchmark_returns=None,
        starting_cash=1_000_000.0,
        commission_rate=0.0,
        slippage_bps=0.0,
        execution_lag_days=1,
    )

    with pytest.raises(ValueError, match=r"spot FX pair\(s\) \['USDJPY=X'\]"):
        _run(tmp_path, unrated)
    assert list(tmp_path.iterdir()) == []


def test_inputs_for_another_universe_are_refused(tmp_path: Path) -> None:
    """Engine inputs whose assets are not the record's universe are refused before writing."""
    other = EngineInputs(assets=["SPY"], market_data=_frame(), market_data_source="lakehouse data")

    with pytest.raises(ValueError, match="not the universe"):
        _run(tmp_path, other)
    assert list(tmp_path.iterdir()) == []


def test_the_rates_do_not_enter_the_spec(tmp_path: Path) -> None:
    """Policy rates are data read at the record's lake time: two records differing only in
    their rates share one spec hash."""
    carried = _run(tmp_path / "a", _inputs(_rates(_USD_RATE, _JPY_RATE)), max_days=1)
    flat = _run(tmp_path / "b", _inputs(_rates(0.0, 0.0)), max_days=1)

    assert carried.spec_hash == flat.spec_hash
