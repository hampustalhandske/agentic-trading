# test_band_fill_cost.py

"""The no-trade band in `PortfolioConstructor.construct` clears the fill cost
model: its minimum move is `max(no_trade_buffer, cost_clearance_multiplier x
expected_fill_cost_fraction())`, where `expected_fill_cost_fraction` is the
commission plus slippage a linear fill pays per reference dollar, read from
`settings.firm` at call time."""

from datetime import date

import numpy as np
import pandas as pd
import pytest
from agentic_trading.execution_costs import expected_fill_cost_fraction

from agentic_trading.agents.portfolio_construction import PodView, PortfolioConstructor
from agentic_trading.agents.schemas import PortfolioDecision, TradeRecord
from agentic_trading.backtest.portfolio import Portfolio
from agentic_trading.config import settings, temporary_overrides

_ASSETS = ["AAPL", "MSFT"]
_AS_OF = pd.Timestamp("2024-06-03")
_PRICE = 100.0
_NAV = 10_000.0


def _views(signals: dict[str, float]) -> dict[str, PodView]:
    return {
        asset: PodView(net_signal=value, signal_count=1 if value else 0, confidence_sum=abs(value))
        for asset, value in signals.items()
    }


def _returns_history(assets: list[str], periods: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({asset: rng.normal(0.0005, 0.01, size=periods) for asset in assets})


def _book(
    previous_weights: dict[str, float] | None = None,
    constructor: PortfolioConstructor | None = None,
) -> dict[str, float]:
    """Sizes on a constructor with no prior call and the VaR cap off, so only the band can bind."""
    constructor = constructor or PortfolioConstructor(method="risk_parity")
    with temporary_overrides({"portfolio.component_var_max_share": 1.0}):
        return constructor.construct(
            _views({"AAPL": 0.4, "MSFT": -0.2}),
            _returns_history(_ASSETS),
            _AS_OF,
            previous_weights,
        )


def _moved(target: dict[str, float], change: float) -> dict[str, float]:
    """Previous weights `change` below the AAPL target and `change` above the MSFT target."""
    return {"AAPL": target["AAPL"] - change, "MSFT": target["MSFT"] + change}


def _fill(positions: dict[str, float], target_weight: float) -> TradeRecord:
    """An AAPL fill at `_PRICE` on a `_NAV` book with no volume data, at `settings.firm` costs."""
    portfolio = Portfolio(cash=_NAV - _PRICE * sum(positions.values()), positions=dict(positions))
    decision = PortfolioDecision(
        asset="AAPL",
        as_of_date=date(2024, 6, 3),
        target_weight=target_weight,
        contributing_signals=[],
    )
    return portfolio.apply_decision(decision, price=_PRICE, portfolio_value=_NAV)


def _charged_fraction(trade: TradeRecord) -> float:
    return (trade.slippage_cost + trade.transaction_cost) / (
        abs(trade.delta_shares) * trade.reference_price
    )


@pytest.mark.parametrize(
    ("commission_rate", "slippage_bps"),
    [(0.0005, 5.0), (0.002, 20.0), (0.0, 0.0)],
    ids=["defaults", "overridden", "zero_cost"],
)
def test_expected_fill_cost_fraction(commission_rate: float, slippage_bps: float) -> None:
    """Matches a no-volume buy's charge per reference dollar; a sell is charged 2cs less."""
    assert (settings.firm.commission_rate, settings.firm.slippage_bps) == (0.0005, 5.0)
    slippage = slippage_bps / 10_000.0
    expected = slippage + commission_rate * (1.0 + slippage)
    sell_expected = slippage + commission_rate * (1.0 - slippage)

    with temporary_overrides(
        {"firm.commission_rate": commission_rate, "firm.slippage_bps": slippage_bps}
    ):
        from_settings = expected_fill_cost_fraction()
        buy = _fill({}, 0.5)
        sell = _fill({"AAPL": 80.0}, 0.2)

    assert from_settings == pytest.approx(expected, rel=1e-12, abs=1e-15)
    assert expected_fill_cost_fraction(commission_rate, slippage_bps) == pytest.approx(
        expected, rel=1e-12, abs=1e-15
    )
    assert expected_fill_cost_fraction() == pytest.approx(0.0005 + 0.0005 * 1.0005, rel=1e-12)
    assert expected_fill_cost_fraction(commission_rate=commission_rate) == pytest.approx(
        0.0005 + commission_rate * 1.0005, rel=1e-12, abs=1e-15
    )

    assert (buy.action, buy.delta_shares, buy.impact_cost) == ("buy", pytest.approx(50.0), 0.0)
    assert _charged_fraction(buy) == pytest.approx(expected, rel=1e-12, abs=1e-15)
    assert (sell.action, sell.delta_shares, sell.impact_cost) == ("sell", pytest.approx(-60.0), 0.0)
    assert _charged_fraction(sell) == pytest.approx(sell_expected, rel=1e-12, abs=1e-15)
    assert expected - _charged_fraction(sell) == pytest.approx(
        2.0 * commission_rate * slippage, rel=1e-6, abs=1e-15
    )


def test_no_trade_band_at_default_costs() -> None:
    """A change just under 5 x the default fill cost is held; one just over it goes through."""
    assert settings.firm.commission_rate == 0.0005
    assert settings.firm.slippage_bps == 5.0
    assert settings.portfolio.cost_clearance_multiplier == 5.0
    assert settings.portfolio.no_trade_buffer == 0.005
    fill_cost = 5.0 / 10_000.0 + 0.0005 * (1.0 + 5.0 / 10_000.0)
    min_move = max(0.005, 5.0 * fill_cost)
    margin = (min_move - 0.005) / 2.0
    assert fill_cost == pytest.approx(0.00100025, rel=1e-12)
    assert min_move == pytest.approx(0.00500125, rel=1e-12)
    assert min_move - margin > settings.portfolio.no_trade_buffer

    target = _book()
    held_from = _moved(target, min_move - margin)
    traded_from = _moved(target, min_move + margin)

    assert _book(held_from) == pytest.approx(held_from, rel=1e-12)
    assert _book(traded_from) == pytest.approx(target, rel=1e-9)


@pytest.mark.parametrize(
    ("commission_rate", "slippage_bps"),
    [(0.0005, 20.0), (0.002, 5.0)],
    ids=["slippage_20bps", "commission_20bps"],
)
def test_no_trade_band_reads_firm_costs_at_construct_time(
    commission_rate: float, slippage_bps: float
) -> None:
    """Raised `firm` costs after construction hold a change the default band trades."""
    default_min_move = max(0.005, 5.0 * (0.0005 + 0.0005 * (1.0 + 0.0005)))
    slippage = slippage_bps / 10_000.0
    wide_min_move = max(0.005, 5.0 * (slippage + commission_rate * (1.0 + slippage)))
    change = 0.0075
    assert wide_min_move == pytest.approx(0.012505, rel=1e-12)
    assert default_min_move < change < wide_min_move

    target = _book()
    previous = _moved(target, change)
    at_default = PortfolioConstructor(method="risk_parity")
    widened = PortfolioConstructor(method="risk_parity")

    assert _book(previous, at_default) == pytest.approx(target, rel=1e-9)
    with temporary_overrides(
        {"firm.commission_rate": commission_rate, "firm.slippage_bps": slippage_bps}
    ):
        assert _book(previous, widened) == pytest.approx(previous, rel=1e-12)


def test_no_separate_band_cost_setting() -> None:
    """Neither `settings.portfolio` nor `PortfolioConstructor` carries a `transaction_cost_bps`."""
    assert "transaction_cost_bps" not in type(settings.portfolio).model_fields
    with pytest.raises(KeyError), temporary_overrides({"portfolio.transaction_cost_bps": 5.0}):
        pass
    with pytest.raises(TypeError):
        PortfolioConstructor(transaction_cost_bps=5.0)  # type: ignore[call-arg]
    assert not hasattr(PortfolioConstructor(), "transaction_cost_bps")
