# test_engine_unfillable_open.py

"""With a one-day execution lag, `BacktestEngine` skips a decision whose
fill-day open is NaN or not positive: shares and cash stay as they were,
the day is valued from the close, and a warning names the asset and date."""

import logging
import math
from collections.abc import Iterable, Iterator

import numpy as np
import pandas as pd
import pytest

from agentic_trading.agents.schemas import PortfolioDecision
from agentic_trading.backtest.engine import BacktestEngine
from agentic_trading.backtest.portfolio import Portfolio
from agentic_trading.scoring.track_record import TrackRecord

_INDEX = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
_BAD_DAY = pd.Timestamp("2024-01-04")
_NEXT_DAY = pd.Timestamp("2024-01-05")
_WEIGHTS = [0.5, 0.2, 0.2, 0.2]


class _LiveRecords:
    """Iterable over whatever `caplog` holds at iteration time."""

    def __init__(self, caplog: pytest.LogCaptureFixture) -> None:
        self._caplog = caplog

    def __iter__(self) -> Iterator[logging.LogRecord]:
        return iter(self._caplog.records)


@pytest.fixture
def warnings_seen(caplog: pytest.LogCaptureFixture) -> Iterator[Iterable[logging.LogRecord]]:
    """Routes the package logger (which does not propagate) into `caplog`
    and yields a view that reads `caplog.records` each time it is iterated
    (pytest replaces that list when the test phase starts)."""
    package_logger = logging.getLogger("agentic_trading")
    package_logger.addHandler(caplog.handler)
    previous_level = package_logger.level
    package_logger.setLevel(logging.DEBUG)
    yield _LiveRecords(caplog)
    package_logger.setLevel(previous_level)
    package_logger.removeHandler(caplog.handler)


def _frame(bad_opens: dict[str, float]) -> pd.DataFrame:
    """Four days of AAPL and MSFT bars; `bad_opens` overrides the open on `_BAD_DAY`."""
    columns = {}
    for asset in ("AAPL", "MSFT"):
        close = pd.Series([100.0, 101.0, 102.0, 103.0], index=_INDEX)
        opens = close.copy()
        if asset in bad_opens:
            opens.loc[_BAD_DAY] = bad_opens[asset]
        columns[(asset, "open")] = opens
        columns[(asset, "close")] = close
        columns[(asset, "log_return")] = np.log(close / close.shift(1))
    return pd.DataFrame(columns)


class _Commander:
    """Targets `_WEIGHTS[day]` in every asset daily, recording the book before the fills."""

    def __init__(self, portfolio: Portfolio, assets: list[str]) -> None:
        self.portfolio = portfolio
        self.assets = assets
        self.books: dict[pd.Timestamp, tuple[float, dict[str, float]]] = {}

    def decide(
        self,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
        scoreboard: object,
        instrument_returns: dict[str, float] | None = None,
    ) -> list[PortfolioDecision]:
        day = pd.Timestamp(as_of_date)
        self.books[day] = (self.portfolio.cash, dict(self.portfolio.positions))
        weight = _WEIGHTS[_INDEX.get_loc(day)]
        return [
            PortfolioDecision(
                asset=asset, as_of_date=day.date(), target_weight=weight, contributing_signals=[]
            )
            for asset in self.assets
        ]


def _run(bad_opens: dict[str, float]) -> tuple[Portfolio, _Commander, list[float]]:
    assets = ["AAPL", "MSFT"]
    portfolio = Portfolio(cash=100_000.0, positions={}, commission_rate=0.0, slippage_bps=0.0)
    commander = _Commander(portfolio, assets)
    result = BacktestEngine(
        commander=commander,  # type: ignore[arg-type]
        portfolio=portfolio,
        track_record=TrackRecord(),
        data_loader=lambda _assets, _start, _end: _frame(bad_opens),
        start_date=_INDEX[0].date(),
        end_date=_INDEX[-1].date(),
        assets=assets,
        execution_lag_days=1,
    ).run()
    return portfolio, commander, [value for _, value in result.portfolio_value_history]


@pytest.mark.parametrize("bad_open", [float("nan"), 0.0, -5.0])
def test_unfillable_open_skips_only_that_assets_fill_and_warns(
    warnings_seen: Iterable[logging.LogRecord], bad_open: float
) -> None:
    """A NaN or non-positive open skips that asset, fills the other, and warns."""
    portfolio, commander, values = _run({"MSFT": bad_open})

    before = commander.books[_BAD_DAY]
    after = commander.books[_NEXT_DAY]
    assert after[1]["MSFT"] == before[1]["MSFT"] > 0.0
    assert after[1]["AAPL"] != before[1]["AAPL"]
    assert math.isfinite(after[0])
    assert portfolio.positions["MSFT"] != after[1]["MSFT"]
    assert all(math.isfinite(shares) for shares in portfolio.positions.values())
    assert len(values) == len(_INDEX) and all(math.isfinite(value) for value in values)

    messages = [r.getMessage() for r in warnings_seen if r.levelno == logging.WARNING]
    assert any("MSFT" in m and "2024-01-04" in m for m in messages)
    assert not any("AAPL" in m and "2024-01-04" in m for m in messages)


def test_all_decisions_unfillable_leaves_cash_and_shares_unchanged(
    warnings_seen: Iterable[logging.LogRecord],
) -> None:
    """When every asset's open is NaN that day, cash and shares are exactly as before."""
    portfolio, commander, values = _run({"AAPL": float("nan"), "MSFT": float("nan")})

    assert commander.books[_NEXT_DAY] == commander.books[_BAD_DAY]
    assert all(math.isfinite(value) for value in values)
    assert math.isfinite(portfolio.cash)
    assert portfolio.positions != commander.books[_NEXT_DAY][1]
