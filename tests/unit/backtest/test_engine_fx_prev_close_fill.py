# test_engine_fx_prev_close_fill.py

"""With a one-day execution lag, a spot FX decision whose fill-day open is
NaN or not positive fills at the previous session's close (FX trades
through the night, so that close is a price the order could have taken),
records `fill_basis="prev_close"` on the trade and pays the same costs an
open fill would. Without a usable previous close, and for every non-FX
asset, the order is recorded as unfilled with reason `no_fill_price`."""

import math
from collections.abc import Callable
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agentic_trading.agents.schemas import BacktestResult, PortfolioDecision, TradeRecord
from agentic_trading.backtest.engine import BacktestEngine
from agentic_trading.backtest.portfolio import Portfolio
from agentic_trading.data.instruments import unit_value_usd
from agentic_trading.scoring.track_record import TrackRecord

_INDEX = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
_BAD_DAY = pd.Timestamp("2024-01-04")
_PREVIOUS_DAY = pd.Timestamp("2024-01-03")
_CLOSES = [1.10, 1.12, 1.09, 1.11]
_WEIGHTS = [0.5, 0.2, 0.3, 0.4]
_SLIPPAGE_BPS = 7.0
_COMMISSION = 0.0005


class _InterruptError(Exception):
    pass


class _Commander:
    """Targets `_WEIGHTS[day]` in every asset daily."""

    def __init__(self, assets: list[str]) -> None:
        self.assets = assets

    def decide(
        self,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
        scoreboard: object,
        instrument_returns: dict[str, float] | None = None,
    ) -> list[PortfolioDecision]:
        day = pd.Timestamp(as_of_date)
        weight = _WEIGHTS[_INDEX.get_loc(day)]
        return [
            PortfolioDecision(
                asset=asset, as_of_date=day.date(), target_weight=weight, contributing_signals=[]
            )
            for asset in self.assets
        ]


def _frame(asset: str, *, bad_open: float, previous_close: float | None = None) -> pd.DataFrame:
    """Four sessions of `asset`, each open equal to its close except `_BAD_DAY`'s,
    which is `bad_open`; `previous_close` overrides `_PREVIOUS_DAY`'s close."""
    close = pd.Series(_CLOSES, index=_INDEX)
    if previous_close is not None:
        close.loc[_PREVIOUS_DAY] = previous_close
    opens = close.copy()
    opens.loc[_PREVIOUS_DAY] = _CLOSES[1]
    opens.loc[_BAD_DAY] = bad_open
    return pd.DataFrame(
        {
            (asset, "open"): opens,
            (asset, "close"): close,
            (asset, "log_return"): np.log(close / close.shift(1)),
        }
    )


def _engine(
    asset: str, frame: pd.DataFrame, checkpoint_path: Path | None = None, resume: bool = False
) -> BacktestEngine:
    return BacktestEngine(
        commander=_Commander([asset]),  # type: ignore[arg-type]
        portfolio=Portfolio(
            cash=1_000_000.0,
            positions={},
            commission_rate=_COMMISSION,
            slippage_bps=_SLIPPAGE_BPS,
        ),
        track_record=TrackRecord(),
        data_loader=lambda _assets, _start, _end: frame,
        start_date=_INDEX[0].date(),
        end_date=_INDEX[-1].date(),
        assets=[asset],
        execution_lag_days=1,
        checkpoint_path=checkpoint_path,
        resume=resume,
    )


def _run(asset: str, frame: pd.DataFrame) -> BacktestResult:
    return _engine(asset, frame).run()


def _bad_day_fill(result: BacktestResult) -> TradeRecord | None:
    """The trade filled on `_BAD_DAY`: the one for `_PREVIOUS_DAY`'s decision, the
    date a trade record carries."""
    trades = [trade for trade in result.trade_log if trade.as_of_date == _PREVIOUS_DAY.date()]
    assert len(trades) <= 1
    return trades[0] if trades else None


@pytest.mark.parametrize("bad_open", [float("nan"), 0.0, -1.0])
def test_fx_fill_with_unusable_open_takes_the_previous_close(bad_open: float) -> None:
    """EURUSD=X with a NaN or non-positive open on day t fills day t-1's decision at
    day t-1's close, records `prev_close`, and books no unfilled order."""
    result = _run("EURUSD=X", _frame("EURUSD=X", bad_open=bad_open))

    trade = _bad_day_fill(result)
    assert trade is not None
    assert trade.reference_price == _CLOSES[1]
    assert trade.fill_basis == "prev_close"
    assert trade.target_weight == _WEIGHTS[1]
    assert result.unfilled_orders == []
    assert all(math.isfinite(value) for _, value in result.portfolio_value_history)


def test_fx_prev_close_fill_pays_the_open_fill_cost_model() -> None:
    """The previous-close fill pays slippage on |delta| x the USD unit value at the fill
    price and commission on the slipped notional, exactly as an open fill does."""
    result = _run("EURUSD=X", _frame("EURUSD=X", bad_open=float("nan")))

    trade = _bad_day_fill(result)
    assert trade is not None
    reference_notional = abs(trade.delta_shares) * unit_value_usd("EURUSD=X", _CLOSES[1])
    assert trade.slippage_cost == pytest.approx(reference_notional * _SLIPPAGE_BPS / 10_000.0)
    slippage = _SLIPPAGE_BPS / 10_000.0
    price_move = 1.0 + slippage if trade.delta_shares > 0 else 1.0 - slippage
    assert trade.action == "sell"
    assert trade.price == pytest.approx(_CLOSES[1] * price_move)
    assert trade.notional == pytest.approx(reference_notional * price_move)
    assert trade.transaction_cost == pytest.approx(trade.notional * _COMMISSION)
    assert trade.impact_cost == 0.0


def test_fx_fills_with_a_usable_open_keep_the_open_basis() -> None:
    """Every other FX fill of the run takes its own open and records `open`."""
    result = _run("EURUSD=X", _frame("EURUSD=X", bad_open=float("nan")))

    others = [trade for trade in result.trade_log if trade.as_of_date != _PREVIOUS_DAY.date()]
    assert others
    assert all(trade.fill_basis == "open" for trade in others)


def test_equity_with_unusable_open_still_records_no_fill_price() -> None:
    """The same frame for an equity leaves day t unfilled with reason `no_fill_price`."""
    result = _run("AAPL", _frame("AAPL", bad_open=float("nan")))

    assert _bad_day_fill(result) is None
    assert [(order.as_of_date, order.asset, order.reason) for order in result.unfilled_orders] == [
        (_BAD_DAY.date(), "AAPL", "no_fill_price")
    ]


def test_future_with_unusable_open_still_records_no_fill_price() -> None:
    """A future keeps the fail-closed path too."""
    result = _run("CL=F", _frame("CL=F", bad_open=float("nan")))

    assert _bad_day_fill(result) is None
    assert [order.reason for order in result.unfilled_orders] == ["no_fill_price"]


def test_fx_without_a_previous_close_records_no_fill_price() -> None:
    """An FX pair whose open is NaN on day t and whose day t-1 close is missing is
    recorded as unfilled with reason `no_fill_price`; an older close is not used."""
    result = _run(
        "EURUSD=X",
        _frame("EURUSD=X", bad_open=float("nan"), previous_close=float("nan")),
    )

    assert _bad_day_fill(result) is None
    orders = [(order.as_of_date, order.reason) for order in result.unfilled_orders]
    assert orders == [(_BAD_DAY.date(), "no_fill_price")]


def test_fill_basis_defaults_to_open_and_serialises() -> None:
    """A trade built without `fill_basis` is an open fill and dumps the default."""
    trade = TradeRecord(
        as_of_date=date(2024, 1, 2),
        asset="AAPL",
        action="buy",
        target_weight=0.1,
        previous_weight=0.0,
        target_shares=10.0,
        delta_shares=10.0,
        price=100.0,
        notional=1_000.0,
    )

    assert trade.fill_basis == "open"
    assert trade.model_dump(mode="json")["fill_basis"] == "open"
    assert TradeRecord.model_validate(trade.model_dump(mode="json")) == trade


def _stop_after(day: int) -> Callable[..., None]:
    def stop(day_index: int, *_args: object) -> None:
        if day_index == day:
            raise _InterruptError

    return stop


def test_resume_across_a_prev_close_fill_is_byte_identical(tmp_path: Path) -> None:
    """Interrupted after any day and resumed from its checkpoint, a run holding a
    previous-close FX fill gives the uninterrupted run's `BacktestResult` JSON, the
    checkpointed trade keeping `fill_basis="prev_close"`."""
    frame = _frame("EURUSD=X", bad_open=float("nan"))
    expected = _engine("EURUSD=X", frame).run()
    expected_json = expected.model_dump_json()
    assert [trade.fill_basis for trade in expected.trade_log].count("prev_close") == 1

    for k in range(len(_INDEX) - 1):
        checkpoint_path = tmp_path / f"checkpoint_{k}.db"
        with pytest.raises(_InterruptError):
            _engine("EURUSD=X", frame, checkpoint_path).run(on_day_complete=_stop_after(k))

        resumed = _engine("EURUSD=X", frame, checkpoint_path, resume=True).run()

        resumed = resumed.model_copy(
            update={"unfactored_split_moves": expected.unfactored_split_moves}
        )
        assert resumed.model_dump_json() == expected_json, f"interrupted after day {k}"
