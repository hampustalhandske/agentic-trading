import math

import numpy as np
import pandas as pd
import pytest
from quantcore.performance.metrics import component_var
from quantcore.portfolio.optimization import risk_parity_weights

from agentic_trading.agents.portfolio_construction import PodView, PortfolioConstructor
from agentic_trading.config import settings, temporary_overrides


def _views(signals: dict[str, float]) -> dict[str, PodView]:
    return {
        asset: PodView(net_signal=value, signal_count=1 if value else 0, confidence_sum=abs(value))
        for asset, value in signals.items()
    }


def _returns_history(assets: list[str], periods: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {asset: rng.normal(0.0005, 0.01, size=periods) for asset in assets}
    return pd.DataFrame(data)


def test_constructor_produces_sane_weights_within_leverage_cap() -> None:
    assets = ["AAPL", "MSFT", "GOOGL"]
    constructor = PortfolioConstructor(method="risk_parity", leverage_cap=2.0)
    views = _views({"AAPL": 0.6, "MSFT": -0.2, "GOOGL": 0.1})

    result = constructor.construct(
        views=views,
        returns_history=_returns_history(assets),
        as_of_date=pd.Timestamp("2024-06-01"),
    )

    assert set(result) == set(assets)
    gross_leverage = sum(abs(weight) for weight in result.values())
    assert gross_leverage <= 2.0 + 1e-6
    for weight in result.values():
        assert math.isfinite(weight)
        assert not math.isnan(weight)


def test_constructor_skips_gracefully_with_too_few_observations() -> None:
    """Early backtest days can hand PortfolioConstructor a returns_history
    with too few rows (or none at all — the first simulated day's
    log_return is NaN and gets dropped) to estimate a covariance matrix.
    It must degrade to sizing off view direction alone (never raising,
    never sitting at an all-zero position despite having real views) —
    see `_bootstrap_weights_from_views`."""
    assets = ["AAPL", "MSFT"]
    constructor = PortfolioConstructor()
    views = _views({"AAPL": 0.4, "MSFT": -0.2})

    for periods in (0, 1, 3):
        result = constructor.construct(
            views=views,
            returns_history=_returns_history(assets, periods=periods),
            as_of_date=pd.Timestamp("2024-06-01"),
        )
        assert result == {"AAPL": 0.5, "MSFT": -0.5}


def test_constructor_skips_when_observations_equal_asset_count() -> None:
    """quantcore's ledoit_wolf_shrinkage requires strictly more observations
    (T) than assets (k) — a basket of 5 assets (the project's
    DEFAULT_BASKET) with exactly 5 days of returns history is a real case
    hit in an actual run: T=5, k=5 fails T > k even though 5 observations
    clears the flat MIN_OBSERVATIONS_FOR_COVARIANCE floor. Must degrade to
    the view-direction bootstrap, not raise."""
    assets = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM"]
    constructor = PortfolioConstructor()
    views = _views(dict.fromkeys(assets, 0.1))

    result = constructor.construct(
        views=views,
        returns_history=_returns_history(assets, periods=5),
        as_of_date=pd.Timestamp("2024-06-01"),
    )

    assert result == dict.fromkeys(assets, 0.2)


def test_constructor_supports_min_variance_and_mean_variance() -> None:
    assets = ["AAPL", "MSFT"]
    views = _views({"AAPL": 0.4, "MSFT": 0.2})
    returns_history = _returns_history(assets)

    for method in ("min_variance", "mean_variance", "risk_parity"):
        constructor = PortfolioConstructor(method=method)
        result = constructor.construct(
            views=views,
            returns_history=returns_history,
            as_of_date=pd.Timestamp("2024-06-01"),
        )
        assert set(result) == set(assets)
        assert all(math.isfinite(weight) for weight in result.values())


def test_leverage_tracks_target_annualized_vol_below_the_cap() -> None:
    """A calm, low-volatility universe should lever up toward
    target_annualized_vol rather than sitting at some fixed fraction of
    leverage_cap — the Bridgewater-style volatility-targeting mechanism,
    not a flat Kelly-esque fraction."""
    rng = np.random.default_rng(0)

    def _gross(daily_std: float) -> float:
        returns = pd.DataFrame(
            {
                "AAPL": rng.normal(0.0001, daily_std, 120),
                "MSFT": rng.normal(-0.0001, daily_std, 120),
            }
        )
        constructor = PortfolioConstructor(
            method="risk_parity", leverage_cap=2.5, target_annualized_vol=0.15
        )
        result = constructor.construct(
            views=_views({"AAPL": 0.3, "MSFT": -0.1}),
            returns_history=returns,
            as_of_date=pd.Timestamp("2024-06-01"),
        )
        return sum(abs(weight) for weight in result.values())

    calm_gross = _gross(0.0005)
    noisy_gross = _gross(0.01)

    # A very low-variance universe implies far more than 2.5x leverage
    # under pure vol-targeting, so leverage_cap is the backstop that binds;
    # a noisier universe gets less leverage.
    assert calm_gross <= 2.5 + 1e-6
    assert noisy_gross < calm_gross


def test_leverage_cap_must_be_within_max_leverage() -> None:
    from agentic_trading.agents.schemas import MAX_LEVERAGE

    with pytest.raises(ValueError):
        PortfolioConstructor(leverage_cap=MAX_LEVERAGE + 0.1)


def test_bootstrap_weights_zero_out_zero_view_assets() -> None:
    """An asset with no real analyst view (view == 0.0) must stay at
    exactly 0.0 in the insufficient-history bootstrap too — this never
    invents a position for an asset nothing has an opinion on."""
    constructor = PortfolioConstructor(leverage_cap=1.0)

    result = constructor._bootstrap_weights_from_views(
        _views({"AAPL": 0.5, "MSFT": 0.0, "GOOGL": -0.3})
    )

    assert result["MSFT"] == 0.0
    assert result["AAPL"] > 0.0
    assert result["GOOGL"] < 0.0
    assert result["AAPL"] == pytest.approx(-result["GOOGL"])
    assert sum(abs(w) for w in result.values()) == pytest.approx(1.0)


def test_bootstrap_weights_all_zero_when_every_view_is_zero() -> None:
    constructor = PortfolioConstructor()

    result = constructor._bootstrap_weights_from_views(_views({"AAPL": 0.0, "MSFT": 0.0}))

    assert result == {"AAPL": 0.0, "MSFT": 0.0}


def test_bootstrap_weights_respect_a_low_leverage_cap() -> None:
    constructor = PortfolioConstructor(leverage_cap=0.4)

    result = constructor._bootstrap_weights_from_views(_views({"AAPL": 0.5, "MSFT": -0.5}))

    assert sum(abs(w) for w in result.values()) == pytest.approx(0.4)


def test_net_leverage_cap_none_falls_back_to_leverage_cap() -> None:
    constructor = PortfolioConstructor(leverage_cap=2.0)

    assert constructor.net_leverage_cap == pytest.approx(2.0)


def test_net_leverage_cap_must_be_within_leverage_cap() -> None:
    with pytest.raises(ValueError):
        PortfolioConstructor(leverage_cap=1.0, net_leverage_cap=1.5)


def test_kelly_fraction_weight_must_be_within_unit_interval() -> None:
    with pytest.raises(ValueError):
        PortfolioConstructor(kelly_fraction_weight=1.5)
    with pytest.raises(ValueError):
        PortfolioConstructor(kelly_fraction_weight=-0.1)


def test_net_leverage_trim_only_touches_same_sign_positions() -> None:
    """A net-tilted book (long AAPL 1.2, short MSFT -0.3 as a partial
    hedge) trimmed to net_leverage_cap=0.5 must only shrink the
    over-net long leg — the short (offsetting) leg stays exactly as-is,
    since a uniform book-wide rescale would also cut the very gross
    exposure a pairs/factor book is supposed to run."""
    constructor = PortfolioConstructor(leverage_cap=2.0, net_leverage_cap=0.5)
    weights = np.array([1.2, -0.3])

    trimmed = constructor._apply_net_leverage_trim(weights)

    assert trimmed[1] == pytest.approx(-0.3)
    assert float(np.sum(trimmed)) == pytest.approx(0.5)


def test_net_leverage_trim_is_a_noop_within_the_cap() -> None:
    constructor = PortfolioConstructor(leverage_cap=2.0, net_leverage_cap=1.0)
    weights = np.array([0.3, -0.2])

    trimmed = constructor._apply_net_leverage_trim(weights)

    assert trimmed.tolist() == pytest.approx(weights.tolist())


def test_kelly_overlay_is_a_noop_when_disabled() -> None:
    constructor = PortfolioConstructor(kelly_fraction_weight=0.0)
    constructor._own_return_history = [
        (pd.Timestamp(f"2024-01-{d:02d}"), 0.02) for d in range(2, 22)
    ]

    assert constructor._kelly_overlay_scalar() == pytest.approx(1.0)


def test_kelly_overlay_stays_near_one_with_short_trailing_history() -> None:
    """Even an extreme, unbroken winning streak must not swing the Kelly
    overlay far from 1.0 while trailing history is still short relative
    to kelly_lookback_days — the confidence ramp, not the raw trailing
    Sharpe alone, is what should be gating this."""
    constructor = PortfolioConstructor(kelly_fraction_weight=0.5, kelly_lookback_days=60)
    constructor._own_return_history = [
        (pd.Timestamp(f"2024-01-{d:02d}"), 0.05) for d in range(2, 8)
    ]

    scalar = constructor._kelly_overlay_scalar()

    # confidence = 6/60 = 0.1, so the overlay can move at most
    # 0.5 * 0.1 = 0.05 away from 1.0 in either direction.
    assert 0.94 < scalar < 1.06


def test_kelly_overlay_scales_up_after_a_full_lookback_winning_streak() -> None:
    constructor = PortfolioConstructor(kelly_fraction_weight=0.5, kelly_lookback_days=60)
    # Alternating +3%/+1% (nonzero variance, unambiguously positive mean)
    # over a full lookback, rather than a perfectly constant series
    # (zero variance -> the std < 1e-12 no-op guard would fire instead).
    constructor._own_return_history = [
        (pd.Timestamp("2024-01-01") + pd.Timedelta(days=d), 0.03 if d % 2 == 0 else 0.01)
        for d in range(60)
    ]

    assert constructor._kelly_overlay_scalar() > 1.3


def test_kelly_overlay_scales_down_after_a_full_lookback_losing_streak() -> None:
    constructor = PortfolioConstructor(kelly_fraction_weight=0.5, kelly_lookback_days=60)
    constructor._own_return_history = [
        (pd.Timestamp("2024-01-01") + pd.Timedelta(days=d), -0.03 if d % 2 == 0 else -0.01)
        for d in range(60)
    ]

    assert constructor._kelly_overlay_scalar() < 0.7


def test_kelly_overlay_never_exceeds_bounds_even_at_a_pathological_sharpe() -> None:
    constructor = PortfolioConstructor(kelly_fraction_weight=1.0, kelly_lookback_days=10)
    constructor._own_return_history = [
        (pd.Timestamp("2024-01-01") + pd.Timedelta(days=d), 1.0) for d in range(10)
    ]
    # A pathological, perfectly-constant nonzero return series has zero
    # sample variance -> the overlay's own std < 1e-12 guard, not tanh's
    # bound, is what should be firing here.

    assert constructor._kelly_overlay_scalar() == pytest.approx(1.0)


def test_no_trade_buffer_snaps_a_small_delta_back_to_the_previous_weight() -> None:
    assets = ["AAPL", "MSFT"]
    constructor = PortfolioConstructor(method="risk_parity", no_trade_buffer=2.0)
    returns_history = _returns_history(assets)

    result = constructor.construct(
        views=_views({"AAPL": 0.4, "MSFT": -0.2}),
        returns_history=returns_history,
        as_of_date=pd.Timestamp("2024-06-01"),
        previous_weights={"AAPL": 0.0, "MSFT": 0.0},
    )

    # A no_trade_buffer wider than any weight this constructor could
    # ever produce (bounded by leverage_cap) should block essentially
    # any single-day rebalance, snapping both assets back to their
    # (zero) previous weight.
    assert result == {"AAPL": 0.0, "MSFT": 0.0}


def test_no_trade_buffer_allows_a_large_rebalance_through() -> None:
    assets = ["AAPL", "MSFT"]
    constructor = PortfolioConstructor(method="risk_parity", no_trade_buffer=1e-6)
    returns_history = _returns_history(assets)

    result = constructor.construct(
        views=_views({"AAPL": 0.4, "MSFT": -0.2}),
        returns_history=returns_history,
        as_of_date=pd.Timestamp("2024-06-01"),
        previous_weights={"AAPL": 0.0, "MSFT": 0.0},
    )

    assert result != {"AAPL": 0.0, "MSFT": 0.0}


def test_transaction_cost_bps_blocks_a_marginal_rebalance() -> None:
    assets = ["AAPL", "MSFT"]
    constructor = PortfolioConstructor(method="risk_parity", transaction_cost_bps=100_000.0)
    returns_history = _returns_history(assets)

    result = constructor.construct(
        views=_views({"AAPL": 0.4, "MSFT": -0.2}),
        returns_history=returns_history,
        as_of_date=pd.Timestamp("2024-06-01"),
        previous_weights={"AAPL": 0.0, "MSFT": 0.0},
    )

    assert result == {"AAPL": 0.0, "MSFT": 0.0}


def test_strong_short_view_produces_a_negative_weight() -> None:
    """A pod whose analysts are unanimously short an asset must be able to
    end up short it — the view is translated into a negative expected
    return against a zero prior, not drowned by an always-long prior."""
    assets = ["AAPL", "MSFT"]
    constructor = PortfolioConstructor(method="mean_variance", leverage_cap=2.0)
    views = {
        "AAPL": PodView(net_signal=-2.4, signal_count=3, confidence_sum=2.4),
        "MSFT": PodView(net_signal=0.0, signal_count=0, confidence_sum=0.0),
    }

    result = constructor.construct(
        views=views,
        returns_history=_returns_history(assets, periods=250),
        as_of_date=pd.Timestamp("2024-06-01"),
    )

    assert result["AAPL"] < 0.0


def test_assets_without_a_view_get_no_expected_return() -> None:
    """With a zero prior, an asset no analyst has an opinion on carries no
    expected return, so mean-variance gives it no directional weight."""
    assets = ["AAPL", "MSFT"]
    constructor = PortfolioConstructor(method="mean_variance", leverage_cap=2.0)
    views = {
        "AAPL": PodView(net_signal=0.0, signal_count=0, confidence_sum=0.0),
        "MSFT": PodView(net_signal=0.0, signal_count=0, confidence_sum=0.0),
    }
    result = constructor.construct(
        views=views,
        returns_history=_returns_history(assets, periods=250),
        as_of_date=pd.Timestamp("2024-06-01"),
    )
    assert all(abs(weight) < 1e-9 for weight in result.values())


def test_pod_view_information_coefficient_scales_the_black_litterman_posterior() -> None:
    """With a zero prior the BL posterior is linear in the view returns,
    so doubling a view's IC doubles its posterior expected return."""
    assets = ["AAPL", "MSFT"]
    constructor = PortfolioConstructor(method="mean_variance")
    returns_matrix = _returns_history(assets, periods=250).to_numpy()
    cov_matrix = np.cov(returns_matrix, rowvar=False)
    prior_ic = settings.portfolio.information_coefficient

    def posterior(ic: float | None) -> np.ndarray:
        views = {
            "AAPL": PodView(
                net_signal=0.8, signal_count=1, confidence_sum=0.8, information_coefficient=ic
            ),
            "MSFT": PodView(),
        }
        return constructor._black_litterman_expected_returns(assets, views, cov_matrix)

    baseline = posterior(None)
    assert baseline[0] == pytest.approx(posterior(prior_ic)[0])
    assert posterior(2.0 * prior_ic)[0] == pytest.approx(2.0 * baseline[0])
    assert posterior(0.0)[0] == pytest.approx(0.0)


def test_risk_parity_tilt_flips_sign_on_a_short_view_and_keeps_a_long_budget_on_none() -> None:
    assets = ["AAPL", "MSFT", "GOOGL"]
    constructor = PortfolioConstructor(method="risk_parity")
    returns_matrix = _returns_history(assets, periods=250).to_numpy()
    cov_matrix = np.cov(returns_matrix, rowvar=False)
    views = {
        "AAPL": PodView(net_signal=-0.9, signal_count=1, confidence_sum=0.9),
        "MSFT": PodView(),
        "GOOGL": PodView(net_signal=0.9, signal_count=1, confidence_sum=0.9),
    }
    z_scores = constructor._conviction_z_scores(assets, views)
    base_budget = np.asarray(risk_parity_weights(cov_matrix))
    tilt = settings.portfolio.risk_parity_view_tilt

    weights = constructor._risk_budget_weights(np.zeros(3), cov_matrix, z_scores)

    assert weights[0] < 0.0
    assert weights[1] > 0.0
    assert weights[2] > 0.0
    assert float(np.sum(np.abs(weights))) == pytest.approx(1.0)
    expected = base_budget * np.array([-(1.0 + 0.9 * tilt), 1.0, 1.0 + 0.9 * tilt])
    np.testing.assert_allclose(weights, expected / np.sum(np.abs(expected)))


def test_risk_parity_tilt_caps_conviction_at_one() -> None:
    assets = ["AAPL", "MSFT"]
    constructor = PortfolioConstructor(method="risk_parity")
    cov_matrix = np.cov(_returns_history(assets, periods=250).to_numpy(), rowvar=False)
    strong = constructor._risk_budget_weights(
        np.zeros(2),
        cov_matrix,
        constructor._conviction_z_scores(assets, _views({"AAPL": 5.0, "MSFT": 0.0})),
    )
    unit = constructor._risk_budget_weights(
        np.zeros(2),
        cov_matrix,
        constructor._conviction_z_scores(assets, _views({"AAPL": 1.0, "MSFT": 0.0})),
    )
    np.testing.assert_allclose(strong, unit)


def test_risk_parity_with_no_views_is_quantcores_plain_budget() -> None:
    assets = ["AAPL", "MSFT"]
    constructor = PortfolioConstructor(method="risk_parity")
    cov_matrix = np.cov(_returns_history(assets, periods=250).to_numpy(), rowvar=False)

    weights = constructor._risk_budget_weights(np.zeros(2), cov_matrix, np.zeros(2))

    np.testing.assert_allclose(weights, np.asarray(risk_parity_weights(cov_matrix)))


def test_cost_aware_sizing_is_on_by_default() -> None:
    constructor = PortfolioConstructor()

    assert constructor.transaction_cost_bps == settings.portfolio.transaction_cost_bps
    assert constructor.no_trade_buffer == settings.portfolio.no_trade_buffer
    assert constructor.transaction_cost_bps > 0.0
    assert constructor.no_trade_buffer > 0.0


def test_default_no_trade_band_still_trades_from_a_flat_book() -> None:
    assets = ["AAPL", "MSFT"]
    constructor = PortfolioConstructor(method="risk_parity")

    result = constructor.construct(
        views=_views({"AAPL": 0.4, "MSFT": -0.2}),
        returns_history=_returns_history(assets),
        as_of_date=pd.Timestamp("2024-06-01"),
        previous_weights={"AAPL": 0.0, "MSFT": 0.0},
    )

    assert any(abs(weight) > 0.0 for weight in result.values())


def test_rebalance_calendar_scheduled_day_lets_a_full_resize_through() -> None:
    """With `rebalance_calendar_days=1`, every call is a scheduled day, so
    a full resize should go through even though `previous_weights` is
    set — never snapped back to the drift-band override path."""
    assets = ["AAPL", "MSFT"]
    constructor = PortfolioConstructor(
        method="risk_parity", no_trade_buffer=1e-9, transaction_cost_bps=0.0
    )
    with temporary_overrides({"portfolio.rebalance_calendar_days": 1}):
        returns_history = _returns_history(assets)
        first = constructor.construct(
            views=_views({"AAPL": 0.4, "MSFT": -0.2}),
            returns_history=returns_history,
            as_of_date=pd.Timestamp("2024-06-01"),
        )
        second = constructor.construct(
            views=_views({"AAPL": -0.4, "MSFT": 0.2}),
            returns_history=returns_history,
            as_of_date=pd.Timestamp("2024-06-02"),
            previous_weights=first,
        )

    assert second["AAPL"] < 0.0
    assert second["MSFT"] > 0.0


def test_rebalance_calendar_non_scheduled_day_snaps_everything_back() -> None:
    """Off-cycle (`rebalance_calendar_days` not yet elapsed) and no asset
    drifted past the band: every asset must snap back to its held
    (`previous_weights`) value, not the newly-sized `final_weights`."""
    assets = ["AAPL", "MSFT"]
    constructor = PortfolioConstructor(
        method="risk_parity", no_trade_buffer=1e-9, transaction_cost_bps=0.0
    )
    with temporary_overrides({"portfolio.rebalance_calendar_days": 100}):
        returns_history = _returns_history(assets)
        first = constructor.construct(
            views=_views({"AAPL": 0.4, "MSFT": -0.2}),
            returns_history=returns_history,
            as_of_date=pd.Timestamp("2024-06-01"),
        )
        # A tiny nudge to the view keeps the resize within the drift band
        # for both assets, so the calendar should suppress it entirely.
        second = constructor.construct(
            views=_views({"AAPL": 0.41, "MSFT": -0.21}),
            returns_history=returns_history,
            as_of_date=pd.Timestamp("2024-06-02"),
            previous_weights=first,
        )

    assert second == first


def test_rebalance_calendar_drift_override_lets_only_the_drifted_asset_through() -> None:
    """A non-scheduled day where one asset's relative drift clears the
    band and the other doesn't: only the drifted asset takes its new
    weight, the other snaps back to `previous_weights`."""
    assets = ["AAPL", "MSFT"]
    constructor = PortfolioConstructor(
        method="risk_parity", no_trade_buffer=1e-9, transaction_cost_bps=0.0
    )
    with temporary_overrides(
        {"portfolio.rebalance_calendar_days": 100, "portfolio.rebalance_drift_band": 0.10}
    ):
        previous_weights = {"AAPL": 0.20, "MSFT": -0.10}
        # Seed a scheduled rebalance first so the second call is genuinely
        # off-cycle rather than treated as this pod's first-ever call.
        constructor._apply_rebalance_calendar(
            assets, np.array([0.20, -0.10]), None, pd.Timestamp("2024-06-01")
        )
        final_weights = np.array([0.21, -0.50])  # AAPL: 5% drift, MSFT: 400% drift

        adjusted = constructor._apply_rebalance_calendar(
            assets, final_weights, previous_weights, pd.Timestamp("2024-06-02")
        )

    assert adjusted[0] == pytest.approx(previous_weights["AAPL"])
    assert adjusted[1] == pytest.approx(final_weights[1])


def test_rebalance_calendar_first_ever_call_is_scheduled() -> None:
    """A pod's very first `construct()` call must be treated as a
    scheduled rebalance regardless of `previous_weights`, so it isn't
    stuck sizing at zero."""
    constructor = PortfolioConstructor()
    final_weights = np.array([0.3, -0.1])

    adjusted = constructor._apply_rebalance_calendar(
        ["AAPL", "MSFT"], final_weights, None, pd.Timestamp("2024-06-01")
    )

    assert adjusted.tolist() == pytest.approx(final_weights.tolist())
    assert constructor._last_rebalance_date == pd.Timestamp("2024-06-01")
    assert constructor._days_since_last_rebalance == 0


def test_component_var_cap_scales_down_a_concentrated_asset() -> None:
    """An asset whose component VaR share exceeds
    `component_var_max_share` must have its weight scaled down by the
    documented single-pass formula, meaningfully reducing (though, since
    this is a single proportional pass rather than an iterative solver,
    not necessarily eliminating entirely) its share of the pod's total
    component VaR — without touching any other asset's weight."""
    assets = ["AAPL", "MSFT", "GOOGL"]
    constructor = PortfolioConstructor()
    # AAPL carries almost all the risk (much larger weight, same vol scale).
    final_weights = np.array([1.8, 0.05, 0.05])
    cov_matrix = np.diag([0.0004, 0.0004, 0.0004])
    original_report = component_var(final_weights, cov_matrix, constructor.var_confidence_level)
    original_share = abs(original_report[0]) / float(np.sum(np.abs(original_report)))
    original_total = float(np.sum(np.abs(original_report)))
    cap = 0.40
    expected_scale = (cap * original_total) / abs(float(original_report[0]))

    with temporary_overrides({"portfolio.component_var_max_share": cap}):
        adjusted = constructor._apply_component_var_cap(assets, final_weights, cov_matrix)

    new_report = component_var(adjusted, cov_matrix, constructor.var_confidence_level)
    new_share = abs(new_report[0]) / float(np.sum(np.abs(new_report)))

    assert adjusted[0] == pytest.approx(final_weights[0] * expected_scale)
    assert adjusted[0] < final_weights[0]
    assert new_share < original_share
    assert adjusted[1] == pytest.approx(final_weights[1])
    assert adjusted[2] == pytest.approx(final_weights[2])


def test_component_var_cap_is_a_noop_for_a_balanced_book() -> None:
    """No asset over the cap: the component VaR step must leave every
    weight untouched."""
    assets = ["AAPL", "MSFT", "GOOGL"]
    constructor = PortfolioConstructor()
    final_weights = np.array([0.2, -0.2, 0.2])
    cov_matrix = np.diag([0.0004, 0.0004, 0.0004])

    adjusted = constructor._apply_component_var_cap(assets, final_weights, cov_matrix)

    assert adjusted.tolist() == pytest.approx(final_weights.tolist())


def test_default_no_trade_band_snaps_a_sub_band_delta_back() -> None:
    assets = ["AAPL", "MSFT"]
    constructor = PortfolioConstructor(method="risk_parity")
    returns_history = _returns_history(assets)
    views = _views({"AAPL": 0.4, "MSFT": -0.2})
    first = constructor.construct(
        views=views, returns_history=returns_history, as_of_date=pd.Timestamp("2024-06-01")
    )
    nudged = {asset: weight + 0.5 * constructor.no_trade_buffer for asset, weight in first.items()}

    second = constructor.construct(
        views=views,
        returns_history=returns_history,
        as_of_date=pd.Timestamp("2024-06-01"),
        previous_weights=nudged,
    )

    assert second == nudged
