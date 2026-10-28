# test_engine_split_factor.py

"""`BacktestEngine` passes each option fill the split factor of the
contract's underlying on the day the fill is priced."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from agentic_trading.agents.schemas import OptionContract, PortfolioDecision
from agentic_trading.backtest.engine import BacktestEngine
from agentic_trading.backtest.portfolio import Portfolio
from agentic_trading.config import temporary_overrides
from agentic_trading.data.loaders.option_marks import OptionMark
from agentic_trading.scoring.track_record import TrackRecord

_CONTRACT = OptionContract(
    underlying="AAPL", expiry=date(2024, 3, 15), strike=100.0, option_type="call"
)
_DECISION_DAY = date(2024, 1, 2)
_NEXT_DAY = date(2024, 1, 3)


def _price_frame() -> pd.DataFrame:
    """Three days of AAPL bars; the first is a pad day before the run starts."""
    index = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    close = pd.Series([100.0, 102.0, 101.0], index=index)
    return pd.DataFrame(
        {"open": close, "close": close, "log_return": np.log(close / close.shift(1))}
    )


def _mark(contract: OptionContract, as_of_date: date, spot: float) -> OptionMark:
    """A tradeable two-sided mark one dollar over intrinsic."""
    mid = max(spot - contract.strike, 0.0) + 1.0
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
        tradeable=True,
    )


class _OneOptionDecisionCommander:
    """Buys 5% of NAV of `_CONTRACT` on `_DECISION_DAY`, nothing afterwards."""

    def decide(
        self,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
        scoreboard: object,
        instrument_returns: dict[str, float] | None = None,
    ) -> list[PortfolioDecision]:
        if pd.Timestamp(as_of_date).date() != _DECISION_DAY:
            return []
        return [
            PortfolioDecision(
                asset=_CONTRACT.contract_id,
                as_of_date=_DECISION_DAY,
                target_weight=0.05,
                contributing_signals=[],
                instrument=_CONTRACT,
            )
        ]


@pytest.mark.parametrize(("execution_lag_days", "fill_day"), [(0, _DECISION_DAY), (1, _NEXT_DAY)])
def test_option_fill_receives_the_underlyings_split_factor_on_the_fill_day(
    monkeypatch: pytest.MonkeyPatch, execution_lag_days: int, fill_day: date
) -> None:
    """The factor is looked up for the contract's underlying on the day
    the fill is priced (the decision day, or the next day under a one-day
    execution lag) and reaches `Portfolio.apply_option_decision`."""
    factor_calls: list[tuple[str, date]] = []

    def fake_factor(ticker: str, quote_date: date) -> float:
        factor_calls.append((ticker, quote_date))
        return 4.0

    applied_factors: list[float] = []
    real_apply = Portfolio.apply_option_decision

    def recording_apply(self: Portfolio, *args: object, **kwargs: object) -> object:
        applied_factors.append(kwargs["split_factor"])  # type: ignore[arg-type]
        return real_apply(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("agentic_trading.backtest.engine.load_option_mark", _mark)
    monkeypatch.setattr(
        "agentic_trading.backtest.engine.lakehouse_loader.split_adjustment_factor", fake_factor
    )
    monkeypatch.setattr(Portfolio, "apply_option_decision", recording_apply)

    portfolio = Portfolio(cash=100_000.0, positions={}, commission_rate=0.0, slippage_bps=0.0)
    engine = BacktestEngine(
        commander=_OneOptionDecisionCommander(),  # type: ignore[arg-type]
        portfolio=portfolio,
        track_record=TrackRecord(),
        data_loader=lambda _assets, _start, _end: _price_frame(),
        start_date=_DECISION_DAY,
        end_date=_NEXT_DAY,
        execution_lag_days=execution_lag_days,
    )
    with temporary_overrides({"options.max_notional_fraction": 1_000.0}):
        engine.run()

    assert factor_calls == [("AAPL", fill_day)]
    assert applied_factors == [4.0]
    assert _CONTRACT.contract_id in portfolio.option_positions
