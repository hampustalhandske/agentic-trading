# test_execution_costs.py

"""Execution costs beyond commission: each fill's slippage, impact and option spread on
`TradeRecord` against its pre-cost `reference_price`, their run totals on `BacktestResult`, their
per-pod split in `pod_costs`, and the reconciliation of the NAV change to gross P&L at reference
prices less every execution cost and net financing."""

import math
from collections import defaultdict
from collections.abc import Callable, Iterator
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from agentic_trading.agents.schemas import (
    BacktestResult,
    OptionContract,
    PodCostAttribution,
    PodCostDay,
    PortfolioDecision,
    TradeRecord,
)
from agentic_trading.backtest.engine import BacktestEngine
from agentic_trading.backtest.portfolio import OptionPosition, Portfolio
from agentic_trading.config import settings, temporary_overrides
from agentic_trading.data.loaders.option_marks import OptionMark
from agentic_trading.scoring.track_record import TrackRecord

# 25 warm-up bars give every simulated day a full ADV window and a daily vol.
_WARMUP = 25
_INDEX = pd.bdate_range("2024-01-02", periods=_WARMUP + 8)
_DAYS = [day.date() for day in _INDEX[_WARMUP:]]
_CASH = 1_000_000.0
_COMMISSION = 0.001
_SLIPPAGE_BPS = 10.0
_DAILY_RISK_FREE = 0.04 / 252
_EXTRA_COSTS = ("slippage_cost", "impact_cost", "spread_cost")
_COST_FIELDS = ("transaction_cost", *_EXTRA_COSTS)


class _InterruptError(Exception):
    pass


@pytest.fixture(autouse=True)
def _stubs(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Stub the engine's option marks and split factors; pin impact and commission settings."""
    monkeypatch.setattr("agentic_trading.backtest.engine.load_option_mark", _mark)
    monkeypatch.setattr(
        "agentic_trading.backtest.engine.lakehouse_loader.split_adjustment_factor",
        lambda *_args: 1.0,
    )
    with temporary_overrides(
        {
            "firm.market_impact_coefficient": 1.0,
            "firm.max_adv_participation": 0.05,
            "options.commission_per_contract": 0.65,
        }
    ):
        yield


def _quote(
    contract: OptionContract,
    as_of_date: date,
    mid: float,
    bid: float,
    ask: float,
    tradeable: bool = True,
    expired: bool = False,
) -> OptionMark:
    """An `OptionMark` with the given quote and placeholder greeks."""
    return OptionMark(
        contract=contract,
        as_of_date=as_of_date,
        snapshot_date=None if expired else as_of_date,
        mid=mid,
        bid=bid,
        ask=ask,
        implied_volatility=0.25,
        delta=0.5,
        gamma=0.02,
        vega=0.1,
        theta=-0.05,
        is_model_mark=False,
        is_expired=expired,
        tradeable=tradeable,
    )


def _mark(contract: OptionContract, as_of_date: date, spot: float) -> OptionMark:
    """Intrinsic plus time value, bid 0.15 below and ask 0.25 above mid; intrinsic at expiry."""
    if contract.option_type == "call":
        intrinsic = max(spot - contract.strike, 0.0)
    else:
        intrinsic = max(contract.strike - spot, 0.0)
    if as_of_date >= contract.expiry:
        return _quote(
            contract, as_of_date, intrinsic, intrinsic, intrinsic, tradeable=False, expired=True
        )
    mid = intrinsic + 1.5 + 0.05 * (contract.expiry - as_of_date).days
    return _quote(contract, as_of_date, mid, mid - 0.15, mid + 0.25)


def _decision(
    asset: str, target_weight: float, instrument: OptionContract | None = None
) -> PortfolioDecision:
    return PortfolioDecision(
        asset=asset,
        as_of_date=date(2024, 1, 2),
        target_weight=target_weight,
        contributing_signals=[],
        instrument=instrument,
    )


def _frame(assets: tuple[str, ...]) -> pd.DataFrame:
    """Open, close, log return and volume per asset over warm-up plus simulated days."""
    rng = np.random.default_rng(11)
    columns: dict[tuple[str, str], pd.Series] = {}
    for asset in assets:
        close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.015, len(_INDEX)))), _INDEX)
        columns[(asset, "open")] = close.shift(1).fillna(close.iloc[0])
        columns[(asset, "close")] = close
        columns[(asset, "log_return")] = np.log(close / close.shift(1))
        columns[(asset, "volume")] = pd.Series(rng.uniform(1.5e5, 2.5e5, len(_INDEX)), _INDEX)
    return pd.DataFrame(columns)


_CALL_EXPIRING = OptionContract(
    underlying="AAPL", expiry=_DAYS[5], strike=100.0, option_type="call"
)
_CALL_HELD = OptionContract(
    underlying="AAPL", expiry=_DAYS[-1] + timedelta(days=30), strike=100.0, option_type="call"
)

# MSFT's contributions are the same on day 3 and day 4, so its day-4 drift fill is charged to
# "commander".
_EQUITY_SERIES = {
    "pod_a": {
        "AAPL": [0.20, 0.25, 0.10, 0.10, 0.30, 0.15],
        "MSFT": [0.10, 0.00, 0.05, 0.20, 0.20, 0.10],
    },
    "pod_b": {
        "AAPL": [0.05, 0.05, 0.10, 0.00, 0.05, 0.10],
        "MSFT": [0.00, 0.10, 0.10, 0.05, 0.05, 0.10],
    },
}


def _option_series(contract: OptionContract) -> dict[str, dict[str, list[float]]]:
    """AAPL plus one call: bought on day 0, cut on day 2, no decision on it from day 5 on."""
    return {
        "pod_a": {
            "AAPL": [0.20, 0.25, 0.25, 0.15, 0.15, 0.10],
            contract.contract_id: [0.015, 0.015, 0.004, 0.004, 0.004],
        },
        "pod_b": {
            "AAPL": [0.05, 0.05, 0.10, 0.10, 0.00, 0.05],
            contract.contract_id: [0.005, 0.005, 0.002, 0.002, 0.002],
        },
    }


# name -> (pod contribution series, assets, execution lag)
_SCENARIOS: dict[str, tuple[dict[str, dict[str, list[float]]], tuple[str, ...], int]] = {
    "equity_same_day": (_EQUITY_SERIES, ("AAPL", "MSFT"), 0),
    "equity_next_open": (_EQUITY_SERIES, ("AAPL", "MSFT"), 1),
    "option_expires": (_option_series(_CALL_EXPIRING), ("AAPL",), 0),
    "option_held": (_option_series(_CALL_HELD), ("AAPL",), 0),
}


class _Scripted:
    """Targets each asset at the sum of the day's pod contributions and reports them. The last two
    simulated days are silent, so no fill lands after the final valuation at either lag."""

    def __init__(self, series: dict[str, dict[str, list[float]]]) -> None:
        length = max(len(values) for assets in series.values() for values in assets.values())
        assert length <= len(_DAYS) - 2
        self.script = {
            _DAYS[i]: {
                pod: {asset: values[i] for asset, values in assets.items() if i < len(values)}
                for pod, assets in series.items()
            }
            for i in range(length)
        }
        self.by_day: dict[date, dict[str, dict[str, float]]] = {}
        self.current: dict[str, dict[str, float]] = {}

    def decide(
        self,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
        scoreboard: object,
        instrument_returns: dict[str, float] | None = None,
    ) -> list[PortfolioDecision]:
        day = pd.Timestamp(as_of_date).date()
        self.current = self.by_day[day] = self.script.get(day, {})
        targets: dict[str, float] = defaultdict(float)
        for weights in self.current.values():
            for asset, weight in weights.items():
                targets[asset] += weight
        return [
            PortfolioDecision(
                asset=asset,
                as_of_date=day,
                target_weight=target,
                contributing_signals=[],
                instrument=(
                    OptionContract.parse(asset) if OptionContract.is_contract_id(asset) else None
                ),
            )
            for asset, target in targets.items()
        ]

    def pod_contributions(self) -> dict[str, dict[str, float]]:
        return {pod: dict(weights) for pod, weights in self.current.items()}


def _run(
    name: str,
    checkpoint_path: Path | None = None,
    resume: bool = False,
    on_day_complete: Callable[..., None] | None = None,
) -> tuple[BacktestResult, _Scripted, pd.DataFrame]:
    """Runs scenario `name` long only, with every cost on and a 4% risk-free rate."""
    series, assets, lag = _SCENARIOS[name]
    commander = _Scripted(series)
    frame = _frame(assets)
    engine = BacktestEngine(
        commander=commander,
        portfolio=Portfolio(
            cash=_CASH, positions={}, commission_rate=_COMMISSION, slippage_bps=_SLIPPAGE_BPS
        ),
        track_record=TrackRecord(),
        data_loader=lambda _assets, _start, _end: frame,
        start_date=_DAYS[0],
        end_date=_DAYS[-1],
        assets=list(assets),
        execution_lag_days=lag,
        checkpoint_path=checkpoint_path,
        resume=resume,
        risk_free_rate_series=pd.Series(_DAILY_RISK_FREE, index=_INDEX),
    )
    return engine.run(on_day_complete=on_day_complete), commander, frame


def _gross_pnl(result: BacktestResult, frame: pd.DataFrame) -> float:
    """Fills at `reference_price`, expiries at intrinsic, and holdings at the last close or mid."""
    multiplier = float(settings.options.contract_multiplier)
    last_day = result.portfolio_value_history[-1][0]
    held: dict[str, float] = defaultdict(float)
    pnl = 0.0
    for trade in result.trade_log:
        scale = multiplier if trade.instrument is not None else 1.0
        price = trade.price if trade.action == "expire" else trade.reference_price
        pnl -= trade.delta_shares * price * scale
        held[trade.asset] += trade.delta_shares
    for asset, quantity in held.items():
        if OptionContract.is_contract_id(asset):
            contract = OptionContract.parse(asset)
            spot = float(frame.loc[pd.Timestamp(last_day), (contract.underlying, "close")])
            pnl += quantity * _mark(contract, last_day, spot).mid * multiplier
        else:
            pnl += quantity * float(frame.loc[pd.Timestamp(last_day), (asset, "close")])
    return pnl


def _expected_pod_costs(
    result: BacktestResult, commander: _Scripted
) -> dict[str, dict[str, float]]:
    """Each fill's costs split pro rata to |contribution change| on the decision behind it."""
    days = sorted(commander.by_day)
    expected: dict[str, dict[str, float]] = defaultdict(lambda: dict.fromkeys(_COST_FIELDS, 0.0))
    for trade in result.trade_log:
        if trade.action not in ("buy", "sell"):
            continue
        index = days.index(trade.as_of_date)
        now = commander.by_day[trade.as_of_date]
        before = commander.by_day[days[index - 1]] if index else {}
        deltas = {
            pod: abs(
                now.get(pod, {}).get(trade.asset, 0.0) - before.get(pod, {}).get(trade.asset, 0.0)
            )
            for pod in {*now, *before}
        }
        total = sum(deltas.values())
        shares = (
            {pod: delta / total for pod, delta in deltas.items() if delta > 0.0}
            if total > 0.0
            else {"commander": 1.0}
        )
        for pod, share in shares.items():
            for field in _COST_FIELDS:
                expected[pod][field] += getattr(trade, field) * share
    return expected


def _stop_after(day: int) -> Callable[..., None]:
    def stop(day_index: int, *_args: object) -> None:
        if day_index == day:
            raise _InterruptError

    return stop


def test_cost_fields_default_to_zero_and_reject_negatives() -> None:
    """New fill and pod cost fields default to zero, and a negative fill cost does not validate."""
    base: dict[str, object] = {
        "as_of_date": date(2024, 1, 2),
        "asset": "AAPL",
        "action": "buy",
        "target_weight": 0.1,
        "previous_weight": 0.0,
        "target_shares": 10.0,
        "delta_shares": 10.0,
        "price": 100.1,
        "notional": 1_001.0,
        "reference_price": 100.0,
    }
    trade = TradeRecord.model_validate(base)
    assert trade.reference_price == 100.0
    assert [getattr(trade, field) for field in _EXTRA_COSTS] == [0.0, 0.0, 0.0]
    for field in _EXTRA_COSTS:
        with pytest.raises(ValidationError):
            TradeRecord.model_validate({**base, field: -1.0})

    attribution = PodCostAttribution()
    assert [getattr(attribution, field) for field in _EXTRA_COSTS] == [0.0, 0.0, 0.0]
    day = PodCostDay(as_of_date=date(2024, 1, 2), contributions=None)
    assert [getattr(day, field) for field in _EXTRA_COSTS] == [{}, {}, {}]


@pytest.mark.parametrize(
    ("positions", "target", "adv", "vol", "delta", "action"),
    [
        ({}, 0.5, None, None, 50.0, "buy"),
        ({"AAPL": 80.0}, 0.2, 10_000.0, 0.02, -60.0, "sell"),
        ({}, 0.5, 400.0, 0.01, 20.0, "buy"),
        ({"AAPL": 50.0}, 0.5, 10_000.0, 0.02, 0.0, "hold"),
    ],
    ids=["buy_without_volume", "sell_with_impact", "buy_capped_at_adv", "hold"],
)
def test_equity_fill_books_slippage_and_impact_against_the_reference_price(
    positions: dict[str, float],
    target: float,
    adv: float | None,
    vol: float | None,
    delta: float,
    action: str,
) -> None:
    """Slippage and impact are |delta| x pre-slippage price x bps; the price gap is their sum."""
    portfolio = Portfolio(
        cash=10_000.0 - 100.0 * sum(positions.values()),
        positions=dict(positions),
        commission_rate=_COMMISSION,
        slippage_bps=_SLIPPAGE_BPS,
    )

    trade = portfolio.apply_decision(
        _decision("AAPL", target),
        price=100.0,
        portfolio_value=10_000.0,
        adv_shares=adv,
        daily_vol=vol,
    )

    assert trade.action == action
    assert trade.delta_shares == pytest.approx(delta, abs=1e-12)
    assert trade.reference_price == 100.0
    impact_bps = (
        0.0
        if adv is None or vol is None or delta == 0.0
        else settings.firm.market_impact_coefficient * vol * math.sqrt(abs(delta) / adv) * 1e4
    )
    assert trade.impact_bps == pytest.approx(impact_bps, rel=1e-9, abs=1e-12)
    assert trade.slippage_cost == pytest.approx(abs(delta) * 100.0 * _SLIPPAGE_BPS / 1e4, abs=1e-12)
    assert trade.impact_cost == pytest.approx(abs(delta) * 100.0 * impact_bps / 1e4, abs=1e-12)
    assert trade.spread_cost == 0.0
    assert trade.delta_shares * (trade.price - trade.reference_price) == pytest.approx(
        trade.slippage_cost + trade.impact_cost, rel=1e-9, abs=1e-12
    )
    assert trade.transaction_cost == pytest.approx(
        abs(delta) * trade.price * _COMMISSION, rel=1e-12, abs=1e-12
    )
    if action == "buy" and adv is None:
        assert trade.slippage_cost == pytest.approx(5.0, rel=1e-12)
        assert trade.transaction_cost == pytest.approx(5.005, rel=1e-12)
    if action == "hold":
        assert [getattr(trade, field) for field in _COST_FIELDS] == [0.0, 0.0, 0.0, 0.0]


def test_option_fill_books_the_distance_to_mid_as_spread_cost() -> None:
    """Spread cost is |contracts| x |fill - mid| x multiplier; holds and expiries book none."""
    multiplier = float(settings.options.contract_multiplier)
    contract = OptionContract(
        underlying="AAPL", expiry=date(2024, 3, 15), strike=100.0, option_type="call"
    )
    contract_id = contract.contract_id
    quote = _quote(contract, date(2024, 1, 2), mid=2.0, bid=1.85, ask=2.25)

    def portfolio(contracts: float) -> Portfolio:
        positions = (
            {
                contract_id: OptionPosition(
                    contract=contract, contracts=contracts, entry_price=2.25, last_mid=2.0
                )
            }
            if contracts
            else {}
        )
        return Portfolio(
            cash=100_000.0,
            positions={},
            option_positions=positions,
            commission_rate=_COMMISSION,
            slippage_bps=_SLIPPAGE_BPS,
        )

    buy = portfolio(0.0).apply_option_decision(
        _decision(contract_id, 0.02, contract), quote, 100_000.0
    )
    sell = portfolio(10.0).apply_option_decision(
        _decision(contract_id, 0.0, contract), quote, 100_000.0
    )
    for trade, delta, spread in ((buy, 10.0, 250.0), (sell, -10.0, 150.0)):
        assert trade.delta_shares == pytest.approx(delta)
        assert trade.reference_price == 2.0
        assert trade.spread_cost == pytest.approx(spread, rel=1e-12)
        assert trade.spread_cost == pytest.approx(
            abs(delta) * abs(trade.price - 2.0) * multiplier, rel=1e-12
        )
        assert trade.slippage_cost == 0.0
        assert trade.impact_cost == 0.0
        assert trade.delta_shares * (trade.price - trade.reference_price) * multiplier == (
            pytest.approx(trade.spread_cost, rel=1e-9)
        )
        assert trade.transaction_cost == pytest.approx(
            10.0 * settings.options.commission_per_contract, rel=1e-12
        )

    untradeable = _quote(contract, date(2024, 1, 2), 2.0, 1.85, 2.25, tradeable=False)
    hold = portfolio(0.0).apply_option_decision(
        _decision(contract_id, 0.02, contract), untradeable, 100_000.0
    )
    assert hold.action == "hold"
    assert hold.reference_price == 2.0
    assert [getattr(hold, field) for field in _COST_FIELDS] == [0.0, 0.0, 0.0, 0.0]

    expired = _quote(contract, contract.expiry, 3.0, 3.0, 3.0, tradeable=False, expired=True)
    (settled,) = portfolio(10.0).settle_expired({contract_id: expired}, contract.expiry)
    assert settled.action == "expire"
    assert [getattr(settled, field) for field in _COST_FIELDS] == [0.0, 0.0, 0.0, 0.0]


@pytest.mark.parametrize("name", list(_SCENARIOS))
def test_every_fill_books_its_costs_and_the_run_totals_sum_them(name: str) -> None:
    """Engine fills reference the data's pre-cost price; `BacktestResult` totals sum the log."""
    result, _, frame = _run(name)
    lag = _SCENARIOS[name][2]
    multiplier = float(settings.options.contract_multiplier)

    assert any(trade.action in ("buy", "sell") for trade in result.trade_log)
    for trade in result.trade_log:
        if trade.action in ("hold", "expire"):
            assert [getattr(trade, field) for field in _EXTRA_COSTS] == [0.0, 0.0, 0.0]
            continue
        size = abs(trade.delta_shares)
        if trade.instrument is None:
            fill_day = _INDEX[_INDEX.get_loc(pd.Timestamp(trade.as_of_date)) + lag]
            reference = float(frame.loc[fill_day, (trade.asset, "open" if lag else "close")])
            assert trade.reference_price == pytest.approx(reference, rel=1e-12)
            assert trade.impact_bps > 0.0
            assert trade.slippage_cost == pytest.approx(
                size * reference * _SLIPPAGE_BPS / 1e4, rel=1e-9
            )
            assert trade.impact_cost == pytest.approx(
                size * reference * trade.impact_bps / 1e4, rel=1e-9
            )
            assert trade.spread_cost == 0.0
            assert trade.delta_shares * (trade.price - trade.reference_price) == pytest.approx(
                trade.slippage_cost + trade.impact_cost, rel=1e-9
            )
            assert trade.transaction_cost == pytest.approx(
                size * trade.price * _COMMISSION, rel=1e-9
            )
        else:
            spot = float(frame.loc[pd.Timestamp(trade.as_of_date), ("AAPL", "close")])
            mid = _mark(trade.instrument, trade.as_of_date, spot).mid
            assert trade.reference_price == pytest.approx(mid, rel=1e-12)
            assert trade.spread_cost == pytest.approx(
                size * abs(trade.price - mid) * multiplier, rel=1e-9
            )
            assert trade.slippage_cost == 0.0
            assert trade.impact_cost == 0.0
            assert trade.delta_shares * (trade.price - trade.reference_price) * multiplier == (
                pytest.approx(trade.spread_cost, rel=1e-9)
            )
            assert trade.transaction_cost == pytest.approx(
                size * settings.options.commission_per_contract, rel=1e-9
            )

    totals = {field: sum(getattr(t, field) for t in result.trade_log) for field in _COST_FIELDS}
    assert result.total_transaction_cost == pytest.approx(totals["transaction_cost"], rel=1e-9)
    assert result.total_slippage_cost == pytest.approx(totals["slippage_cost"], rel=1e-9)
    assert result.total_impact_cost == pytest.approx(totals["impact_cost"], rel=1e-9)
    assert result.total_spread_cost == pytest.approx(totals["spread_cost"], abs=1e-9)
    assert result.total_execution_cost == pytest.approx(sum(totals.values()), rel=1e-9)
    assert result.total_slippage_cost > 0.0
    assert result.total_impact_cost > 0.0
    if name.startswith("option"):
        assert result.total_spread_cost > 0.0
    else:
        assert result.total_spread_cost == 0.0


@pytest.mark.parametrize("name", list(_SCENARIOS))
def test_nav_change_reconciles_to_gross_pnl_less_execution_and_financing_costs(
    name: str,
) -> None:
    """NAV change = P&L at reference prices - execution cost - net financing, to 1e-6 of NAV."""
    result, _, frame = _run(name)
    nav_start = result.portfolio_value_history[0][1]
    nav_end = result.portfolio_value_history[-1][1]

    assert nav_start == _CASH
    assert result.total_transaction_cost > 0.0
    assert result.total_slippage_cost > 0.0
    assert result.total_impact_cost > 0.0
    assert result.total_financing_cost < 0.0
    if name == "option_expires":
        assert any(trade.action == "expire" for trade in result.trade_log)
    if name == "option_held":
        assert sum(t.delta_shares for t in result.trade_log if t.instrument is not None) > 0.0

    expected = _gross_pnl(result, frame) - result.total_execution_cost - result.total_financing_cost
    assert nav_end - nav_start == pytest.approx(expected, abs=1e-6 * nav_start)


@pytest.mark.parametrize("name", list(_SCENARIOS))
def test_pod_costs_split_every_cost_kind_pro_rata_and_sum_to_the_firm_totals(name: str) -> None:
    """Slippage, impact and spread split across pods like commission, and pods sum to totals."""
    result, commander, _ = _run(name)
    expected = _expected_pod_costs(result, commander)

    assert {"pod_a", "pod_b"} <= set(result.pod_costs)
    if name.startswith("equity"):
        assert result.pod_costs["commander"].slippage_cost > 0.0
        assert result.pod_costs["commander"].impact_cost > 0.0
    for pod in set(result.pod_costs) | set(expected):
        for field in _COST_FIELDS:
            actual = getattr(result.pod_costs[pod], field) if pod in result.pod_costs else 0.0
            assert actual == pytest.approx(
                expected[pod][field] if pod in expected else 0.0, rel=1e-9, abs=1e-9
            ), (pod, field)

    firm = {
        "transaction_cost": result.total_transaction_cost,
        "slippage_cost": result.total_slippage_cost,
        "impact_cost": result.total_impact_cost,
        "spread_cost": result.total_spread_cost,
    }
    for field, total in firm.items():
        pod_sum = sum(getattr(cost, field) for cost in result.pod_costs.values())
        assert pod_sum == pytest.approx(total, abs=1e-6 * _CASH), field


@pytest.mark.parametrize("name", ["equity_next_open", "option_expires"])
def test_resume_after_any_day_reproduces_the_uninterrupted_costs(name: str, tmp_path: Path) -> None:
    """A run resumed after any day gives the uninterrupted pod costs and cost totals."""
    expected, _, _ = _run(name)

    for k in range(len(_DAYS)):
        checkpoint_path = tmp_path / f"{name}_{k}.db"
        with pytest.raises(_InterruptError):
            _run(name, checkpoint_path, on_day_complete=_stop_after(k))

        resumed, _, _ = _run(name, checkpoint_path, resume=True)

        assert resumed.pod_costs == expected.pod_costs, f"interrupted after day {k}"
        assert resumed.trade_log == expected.trade_log, k
        assert resumed.portfolio_value_history == expected.portfolio_value_history, k
        assert [
            resumed.total_slippage_cost,
            resumed.total_impact_cost,
            resumed.total_spread_cost,
            resumed.total_execution_cost,
        ] == [
            expected.total_slippage_cost,
            expected.total_impact_cost,
            expected.total_spread_cost,
            expected.total_execution_cost,
        ], k
