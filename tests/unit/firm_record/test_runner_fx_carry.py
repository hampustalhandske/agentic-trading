# test_runner_fx_carry.py

"""AT-23: a firm record over a spot FX pair accrues the pair's carry from the BIS policy rates
in its engine inputs (base minus quote rate, actual/365 over calendar days, on the USD
notional), and a record whose universe holds a pair without policy rates is refused before
anything is written."""

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal, NamedTuple

import numpy as np
import pandas as pd
import pytest

from agentic_trading.agents.base import AnalystAgent, ManagerAgent
from agentic_trading.agents.commander import Commander
from agentic_trading.agents.portfolio_construction import PortfolioConstructor
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.backtest.engine_inputs import EngineInputs
from agentic_trading.backtest.portfolio import Portfolio
from agentic_trading.data.instruments import currency_pair
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


class _CarryCase(NamedTuple):
    """A record over one `pair` held in `direction`, its first close near `level`, at the
    policy rates `base_rate` and `quote_rate`; `usd_base` when one unit held is one USD."""

    pair: str
    direction: Literal["long", "short"]
    level: float
    base_rate: float
    quote_rate: float
    usd_base: bool


_CASES = [
    _CarryCase("USDJPY=X", "long", 120.0, _USD_RATE, _JPY_RATE, usd_base=True),
    _CarryCase("USDJPY=X", "short", 120.0, _USD_RATE, _JPY_RATE, usd_base=True),
    _CarryCase("EURUSD=X", "long", 1.08, 0.0005, _USD_RATE, usd_base=False),
]


class _StubAnalyst(AnalystAgent):
    agent_id: str = "fx_stub_analyst"
    direction: Literal["long", "short"] = "long"

    def generate_signal(
        self, asset: str, market_data: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> AgentSignal:
        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction=self.direction,
            confidence=0.5,
            rationale="stub",
        )


class _ViewConstructor(PortfolioConstructor):
    def construct(self, views, returns_history, as_of_date, previous_weights=None):
        return {asset: 0.3 * view.net_signal for asset, view in views.items()}


class _PassThroughManager(ManagerAgent):
    def scale_factor(self, signal, market_data, as_of_date) -> float:
        return 1.0


def _commander(direction: Literal["long", "short"] = "long") -> Commander:
    manager = _PassThroughManager(
        agent_id="fx_manager",
        managed_analysts=[_StubAnalyst(direction=direction)],
        weighting_strategy=EqualWeight(),
        portfolio_constructor=_ViewConstructor(),
        assets=None,
    )
    return Commander(managers=[manager], weighting_strategy=EqualWeight())


def _frame(pair: str = _PAIR, level: float = 120.0) -> pd.DataFrame:
    index = _WARMUP.append(_SESSIONS)
    rng = np.random.default_rng(7)
    close = pd.Series(level * np.exp(np.cumsum(rng.normal(0.0, 0.004, len(index)))), index)
    return pd.DataFrame(
        {
            (pair, "open"): close,
            (pair, "close"): close,
            (pair, "log_return"): np.log(close / close.shift(1)),
        }
    )


def _rates(base_rate: float, quote_rate: float, pair: str = _PAIR) -> pd.DataFrame:
    """One policy-rate observation for each currency of `pair`, in force from before the
    warm-up."""
    pair_currencies = currency_pair(pair)
    assert pair_currencies is not None
    return pd.DataFrame(
        {pair_currencies.base: [base_rate], pair_currencies.quote: [quote_rate]},
        index=pd.DatetimeIndex(["2015-11-30"], name="date"),
    )


def _inputs(policy_rates: pd.DataFrame, pair: str = _PAIR, level: float = 120.0) -> EngineInputs:
    return EngineInputs(
        assets=[pair],
        market_data=_frame(pair, level),
        market_data_source="lakehouse data",
        policy_rates=policy_rates,
    )


def _run(
    store_dir: Path,
    inputs: EngineInputs,
    direction: Literal["long", "short"] = "long",
    **kwargs: Any,
) -> FirmRecordResult:
    return run_firm_record(
        _commander(direction), list(inputs.assets), _START, _END, store_dir, inputs,
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


@pytest.mark.parametrize("case", _CASES, ids=["long-usdjpy", "short-usdjpy", "long-eurusd"])
def test_a_record_over_a_pair_accrues_its_carry(tmp_path: Path, case: _CarryCase) -> None:
    """The pair is first held from the second session, a Friday. On the Monday after it the
    book differs from the same record at zero policy rates by the position's signed USD
    notional x (base rate - quote rate) x 3 / 365: credited on a long USDJPY (USD above JPY),
    debited on a short one, and debited on a long EURUSD (EUR below USD) over a notional of
    its quantity x Monday's close, the mark the position is settled to before carry accrues.
    The two records match up to the Friday."""
    friday, monday = _SESSIONS[1].date(), _SESSIONS[2].date()
    assert (friday.weekday(), (monday - friday).days) == (4, 3)
    rated = _rates(case.base_rate, case.quote_rate, case.pair)

    carried_dir = tmp_path / "carried"
    paused = _run(carried_dir, _inputs(rated, case.pair, case.level), case.direction, max_days=2)
    held = _portfolio(_checkpoint(carried_dir, paused)).marked_positions[case.pair].quantity
    carried = _run(carried_dir, _inputs(rated, case.pair, case.level), case.direction)
    flat = _run(
        tmp_path / "flat",
        _inputs(_rates(0.0, 0.0, case.pair), case.pair, case.level),
        case.direction,
    )

    assert paused.last_completed == friday
    assert held > 0.0 if case.direction == "long" else held < 0.0
    assert carried.complete and flat.complete
    carried_values = _values(_checkpoint(carried_dir, carried))
    flat_values = _values(_checkpoint(tmp_path / "flat", flat))
    assert carried_values[friday] == pytest.approx(flat_values[friday], abs=1e-6)
    # One unit of a USD-base pair is one USD; one unit of a USD-quoted pair is worth its price.
    monday_close = float(
        _frame(case.pair, case.level).loc[pd.Timestamp(monday), (case.pair, "close")]
    )
    notional = held if case.usd_base else held * monday_close
    expected = notional * (case.base_rate - case.quote_rate) * 3 / 365.0
    # A position earns carry when its sign matches the rate differential's, and pays it otherwise.
    assert (expected < 0.0) == ((case.direction == "short") != (case.base_rate < case.quote_rate))
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
    )

    with pytest.raises(ValueError, match=r"spot FX pair\(s\) \['USDJPY=X'\]"):
        _run(tmp_path, unrated)
    assert list(tmp_path.iterdir()) == []


def test_inputs_for_another_universe_are_refused(tmp_path: Path) -> None:
    """Engine inputs whose assets are not the record's universe are refused before writing."""
    other = EngineInputs(assets=["SPY"], market_data=_frame(), market_data_source="lakehouse data")

    with pytest.raises(ValueError, match="not the universe"):
        run_firm_record(
            _commander(), _UNIVERSE, _START, _END, tmp_path, other,
            lake_time=_T, lake_version="lake", quantcore_version="0.3.1",
        )  # fmt: skip
    assert list(tmp_path.iterdir()) == []


def test_the_rates_do_not_enter_the_spec(tmp_path: Path) -> None:
    """Policy rates are data read at the record's lake time: two records differing only in
    their rates share one spec hash."""
    carried = _run(tmp_path / "a", _inputs(_rates(_USD_RATE, _JPY_RATE)), max_days=1)
    flat = _run(tmp_path / "b", _inputs(_rates(0.0, 0.0)), max_days=1)

    assert carried.spec_hash == flat.spec_hash
