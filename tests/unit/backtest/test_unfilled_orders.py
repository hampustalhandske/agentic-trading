# test_unfilled_orders.py

"""`BacktestEngine` records every decision it did not execute as an
`UnfilledOrder` on `BacktestResult.unfilled_orders` (equity with no usable
fill price, option with no mark or an untradeable mark), and
`extract_run_summary` reports their count and a count by reason."""

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from agentic_trading.agents.schemas import OptionContract, PortfolioDecision, UnfilledOrder
from agentic_trading.backtest.engine import BacktestEngine
from agentic_trading.backtest.portfolio import Portfolio
from agentic_trading.backtest.run_summary import extract_run_summary
from agentic_trading.config import temporary_overrides
from agentic_trading.data.loaders.option_marks import OptionMark
from agentic_trading.scoring.track_record import TrackRecord

_INDEX = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
_CONTRACT = OptionContract(
    underlying="AAPL", expiry=date(2024, 1, 10), strike=100.0, option_type="call"
)


class _Commander:
    """Returns the preset decisions for each day, or none."""

    def __init__(self, decisions: dict[date, list[PortfolioDecision]]) -> None:
        self._decisions = decisions

    def decide(
        self,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
        scoreboard: object,
        instrument_returns: dict[str, float] | None = None,
    ) -> list[PortfolioDecision]:
        return self._decisions.get(pd.Timestamp(as_of_date).date(), [])


def _decision(
    asset: str, day: pd.Timestamp, weight: float, instrument: OptionContract | None = None
) -> PortfolioDecision:
    return PortfolioDecision(
        asset=asset,
        as_of_date=day.date(),
        target_weight=weight,
        contributing_signals=[],
        instrument=instrument,
    )


def _equity_result(bad_open: float | None):
    """AAPL and MSFT at weight 0.2, one-day lag; MSFT's 2024-01-04 open is `bad_open`."""
    assets = ["AAPL", "MSFT"]
    columns = {}
    for asset in assets:
        close = pd.Series([100.0, 101.0, 102.0, 103.0], index=_INDEX)
        opens = close.copy()
        if asset == "MSFT" and bad_open is not None:
            opens.loc[pd.Timestamp("2024-01-04")] = bad_open
        columns[(asset, "open")] = opens
        columns[(asset, "close")] = close
        columns[(asset, "log_return")] = np.log(close / close.shift(1))
    frame = pd.DataFrame(columns)
    decisions = {day.date(): [_decision(a, day, 0.2) for a in assets] for day in _INDEX}
    return BacktestEngine(
        commander=_Commander(decisions),  # type: ignore[arg-type]
        portfolio=Portfolio(cash=100_000.0, positions={}, commission_rate=0.0, slippage_bps=0.0),
        track_record=TrackRecord(),
        data_loader=lambda _assets, _start, _end: frame,
        start_date=_INDEX[0].date(),
        end_date=_INDEX[-1].date(),
        assets=assets,
        execution_lag_days=1,
    ).run()


def _option_result(
    monkeypatch: pytest.MonkeyPatch,
    weights: dict[str, float],
    *,
    marked: bool = True,
    tradeable: bool = True,
) -> tuple:
    """One option contract at `weights` by ISO date; returns the result and mids served."""
    mids: dict[date, float] = {}

    def load_mark(contract: OptionContract, as_of_date: date, spot: float) -> OptionMark | None:
        if not marked:
            return None
        mids[as_of_date] = max(spot - contract.strike, 0.0) + 1.0
        mid = mids[as_of_date]
        return OptionMark(
            contract=contract,
            as_of_date=as_of_date,
            snapshot_date=as_of_date,
            mid=mid,
            bid=mid - 0.1,
            ask=mid + 0.1,
            implied_volatility=0.25,
            delta=0.5,
            gamma=0.02,
            vega=0.1,
            theta=-0.05,
            is_model_mark=False,
            is_expired=False,
            tradeable=tradeable,
        )

    monkeypatch.setattr("agentic_trading.backtest.engine.load_option_mark", load_mark)
    monkeypatch.setattr(
        "agentic_trading.backtest.engine.lakehouse_loader.split_adjustment_factor",
        lambda *_args: 1.0,
    )
    close = pd.Series([100.0, 102.0, 101.0, 104.0], index=_INDEX)
    frame = pd.DataFrame({"close": close, "log_return": np.log(close / close.shift(1))})
    decisions = {
        date.fromisoformat(day): [
            _decision(_CONTRACT.contract_id, pd.Timestamp(day), weight, _CONTRACT)
        ]
        for day, weight in weights.items()
    }
    with temporary_overrides({"options.max_notional_fraction": 1_000.0}):
        result = BacktestEngine(
            commander=_Commander(decisions),  # type: ignore[arg-type]
            portfolio=Portfolio(cash=10_000.0, positions={}, commission_rate=0.0, slippage_bps=0.0),
            track_record=TrackRecord(),
            data_loader=lambda _assets, _start, _end: frame,
            start_date=_INDEX[0].date(),
            end_date=_INDEX[-1].date(),
        ).run()
    return result, mids


def test_unfilled_order_schema() -> None:
    """`UnfilledOrder` carries its fields, allows a None price, and rejects an unknown reason."""
    order = UnfilledOrder(
        as_of_date=date(2024, 1, 4),
        asset="MSFT",
        target_weight=0.2,
        reason="no_option_mark",
        price=None,
    )
    assert order.price is None and order.reason == "no_option_mark"
    with pytest.raises(ValidationError):
        UnfilledOrder(
            as_of_date=date(2024, 1, 4), asset="MSFT", target_weight=0.2, reason="other", price=1.0
        )


@pytest.mark.parametrize("bad_open", [float("nan"), 0.0, -5.0])
def test_equity_without_fill_price_is_recorded(bad_open: float) -> None:
    """A NaN or non-positive fill price records one `no_fill_price` order with that price."""
    orders = _equity_result(bad_open).unfilled_orders

    assert len(orders) == 1
    order = orders[0]
    assert (order.asset, order.reason, order.target_weight) == ("MSFT", "no_fill_price", 0.2)
    assert order.as_of_date in {date(2024, 1, 3), date(2024, 1, 4)}
    assert math.isnan(order.price) if math.isnan(bad_open) else order.price == bad_open


def test_fully_filled_equity_run_has_no_unfilled_orders() -> None:
    """Executed decisions never appear in `unfilled_orders`."""
    assert _equity_result(None).unfilled_orders == []


def test_option_without_mark_is_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each option decision lacking a mark records a `no_option_mark` order with price None."""
    result, _ = _option_result(monkeypatch, {"2024-01-02": 0.05, "2024-01-03": 0.04}, marked=False)

    assert [(o.asset, o.reason, o.price, o.target_weight) for o in result.unfilled_orders] == [
        (_CONTRACT.contract_id, "no_option_mark", None, 0.05),
        (_CONTRACT.contract_id, "no_option_mark", None, 0.04),
    ]
    assert [o.as_of_date for o in result.unfilled_orders] == [date(2024, 1, 2), date(2024, 1, 3)]


def test_untradeable_option_mark_is_recorded_unless_target_matches_held(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An untradeable mark records its mid for a changed target, not a flat-on-flat one."""
    result, mids = _option_result(
        monkeypatch, {"2024-01-02": 0.0, "2024-01-03": 0.05}, tradeable=False
    )

    assert len(result.unfilled_orders) == 1
    order = result.unfilled_orders[0]
    assert (order.asset, order.reason, order.target_weight) == (
        _CONTRACT.contract_id,
        "untradeable_option_mark",
        0.05,
    )
    assert order.as_of_date == date(2024, 1, 3)
    assert order.price == pytest.approx(mids[date(2024, 1, 3)])
    assert result.trade_log == [] or all(t.delta_shares == 0.0 for t in result.trade_log)


def test_tradeable_option_run_has_no_unfilled_orders(monkeypatch: pytest.MonkeyPatch) -> None:
    """A filled option decision is not an unfilled order."""
    result, _ = _option_result(monkeypatch, {"2024-01-02": 0.05})

    assert any(t.action == "buy" for t in result.trade_log)
    assert result.unfilled_orders == []


def _block(orders: list[dict] | None) -> dict:
    block = {
        "start_date": "2018-04-02",
        "end_date": "2018-04-03",
        "portfolio_value_history": [["2018-04-02", 10_000.0], ["2018-04-03", 10_050.0]],
        "final_sharpe_ratio": 0.5,
        "final_max_drawdown": -0.01,
        "final_sortino_ratio": 0.6,
        "final_calmar_ratio": 0.4,
        "final_information_ratio": None,
        "final_hit_rate": 0.55,
        "final_profit_factor": 1.1,
        "signal_health": {},
        "source_performance": {},
    }
    if orders is not None:
        block["unfilled_orders"] = orders
    return block


def _order(reason: str) -> dict:
    return {
        "as_of_date": "2018-04-02",
        "asset": "AAPL",
        "target_weight": 0.1,
        "reason": reason,
        "price": None,
    }


def test_run_summary_counts_unfilled_orders_per_period_and_baseline() -> None:
    """Each period and baseline summary counts unfilled orders, in total and by reason."""
    orders = [_order("no_fill_price"), _order("no_fill_price"), _order("no_option_mark")]
    result = {
        "experiment": {},
        "market_data_source": "lakehouse data",
        "calibration": _block(None),
        "test": _block(orders),
        "baselines": {"equal_weight": _block(orders[:1])},
    }

    summary = extract_run_summary(result)

    assert summary["test"]["unfilled_orders"] == 3
    assert summary["test"]["unfilled_orders_by_reason"] == {"no_fill_price": 2, "no_option_mark": 1}
    assert summary["calibration"]["unfilled_orders"] == 0
    assert summary["calibration"]["unfilled_orders_by_reason"] == {}
    assert summary["baselines"]["equal_weight"]["unfilled_orders"] == 1
    assert summary["baselines"]["equal_weight"]["unfilled_orders_by_reason"] == {"no_fill_price": 1}
    assert "unfilled_orders" in summary["test"] and not isinstance(
        summary["test"]["unfilled_orders"], list
    )
