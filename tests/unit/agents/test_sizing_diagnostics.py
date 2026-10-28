# test_sizing_diagnostics.py
import numpy as np
import pandas as pd
import pytest

from agentic_trading.agents.portfolio_construction import PodView, PortfolioConstructor
from agentic_trading.agents.schemas import SizingDiagnostics
from agentic_trading.config import temporary_overrides

AS_OF = pd.Timestamp("2024-06-01")
PIPELINE = [
    "vol_target_cap",
    "kelly",
    "gross_cap",
    "net_trim",
    "no_trade_band",
    "component_var_cap",
    "rebalance_calendar",
    "final_clip",
]


def _views(signals: dict[str, float]) -> dict[str, PodView]:
    return {
        asset: PodView(net_signal=value, signal_count=1 if value else 0, confidence_sum=abs(value))
        for asset, value in signals.items()
    }


def _returns(stds: dict[str, float], periods: int = 250, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({a: rng.normal(0.0, s, size=periods) for a, s in stds.items()})


def _quiet_constructor(**kwargs: object) -> PortfolioConstructor:
    """A constructor with the Kelly, cost and no-trade stages switched off."""
    params: dict[str, object] = {
        "method": "risk_parity",
        "leverage_cap": 6.0,
        "target_annualized_vol": 0.10,
        "transaction_cost_bps": 0.0,
        "no_trade_buffer": 0.0,
        "kelly_fraction_weight": 0.0,
    }
    params.update(kwargs)
    return PortfolioConstructor(**params)  # type: ignore[arg-type]


def _construct(
    constructor: PortfolioConstructor, stds: dict[str, float], signals: dict[str, float]
) -> SizingDiagnostics:
    constructor.construct(_views(signals), _returns(stds), AS_OF)
    diagnostics = constructor.last_diagnostics()
    assert diagnostics is not None
    return diagnostics


def test_last_diagnostics_is_none_before_any_call() -> None:
    """A fresh constructor has no diagnostics."""
    assert PortfolioConstructor().last_diagnostics() is None


def test_vol_target_cap_binds_in_calm_markets() -> None:
    """A calm universe hits the leverage cap, so the vol-target stage falls short of target."""
    constructor = PortfolioConstructor(
        method="risk_parity", leverage_cap=2.5, target_annualized_vol=0.15
    )
    diagnostics = _construct(
        constructor, {"AAPL": 0.0005, "MSFT": 0.0005}, {"AAPL": 0.3, "MSFT": -0.1}
    )

    assert diagnostics.path == "full"
    assert diagnostics.manager_id is None
    assert diagnostics.as_of_date == AS_OF.date()
    assert diagnostics.target_vol == pytest.approx(0.15)
    assert diagnostics.unlevered_vol is not None
    assert diagnostics.vol_after["vol_target_cap"] < diagnostics.target_vol
    assert diagnostics.binding_stage == "vol_target_cap"


def test_nothing_binds_leaves_final_vol_at_target() -> None:
    """With every limiting stage relaxed, final vol equals target and no stage binds."""
    with temporary_overrides(
        {"portfolio.component_var_max_share": 1.0, "portfolio.rebalance_calendar_days": 1}
    ):
        diagnostics = _construct(
            _quiet_constructor(), {"AAPL": 0.01, "MSFT": 0.01}, {"AAPL": 0.3, "MSFT": -0.1}
        )

    assert diagnostics.final_vol == pytest.approx(diagnostics.target_vol, rel=1e-6)
    assert diagnostics.binding_stage is None


def test_component_var_cap_binds_when_one_asset_dominates_risk() -> None:
    """A heavily weighted, high-vol asset trips the component-VaR cap as the largest vol drop."""
    with temporary_overrides(
        {"portfolio.component_var_max_share": 0.3, "portfolio.rebalance_calendar_days": 1}
    ):
        diagnostics = _construct(
            _quiet_constructor(),
            {"AAPL": 0.05, "MSFT": 0.005, "GOOGL": 0.005},
            {"AAPL": 5.0, "MSFT": 0.1, "GOOGL": 0.1},
        )

    assert diagnostics.binding_stage == "component_var_cap"


def test_view_direction_path_has_no_stage_vols() -> None:
    """Too little history sizes off views alone and records no per-stage vols."""
    constructor = _quiet_constructor()
    constructor.construct(
        _views({"AAPL": 0.4, "MSFT": -0.2}),
        _returns({"AAPL": 0.01, "MSFT": 0.01}, periods=2),
        AS_OF,
    )
    diagnostics = constructor.last_diagnostics()

    assert diagnostics is not None
    assert diagnostics.path == "view_direction"
    assert diagnostics.vol_after == {}
    assert diagnostics.binding_stage is None


def test_vol_after_follows_pipeline_order_and_final_vol() -> None:
    """Stage keys appear in pipeline order and final_vol is the last stage's vol."""
    stds = {"AAPL": 0.01, "MSFT": 0.015, "GOOGL": 0.02}
    signals = {"AAPL": 0.5, "MSFT": -0.3, "GOOGL": 0.2}
    for constructor in (PortfolioConstructor(), _quiet_constructor()):
        diagnostics = _construct(constructor, stds, signals)
        keys = list(diagnostics.vol_after)

        assert keys == [stage for stage in PIPELINE if stage in keys]
        assert keys
        assert diagnostics.final_vol == pytest.approx(list(diagnostics.vol_after.values())[-1])
        assert all(vol >= 0.0 for vol in diagnostics.vol_after.values())
