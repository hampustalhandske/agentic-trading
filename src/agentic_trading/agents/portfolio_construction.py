# portfolio_construction.py

"""Every pod manager's own risk-and-sizing engine — Bridgewater-style risk
parity plus volatility targeting, not a global, optional Layer-4 stage.

Each `ManagerAgent` owns one `PortfolioConstructor` (see
`ManagerAgent.construct_portfolio()` in `agents/base.py`) and uses it to
turn its own analysts' combined views into a complete, covariance-aware
set of per-asset target weights, independent of every other pod. The
Commander never re-touches position sizing — it only allocates capital
across pods (see `commander.py`) and sums their already-risk-managed
portfolios."""

import math
from typing import cast

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from quantcore.performance.metrics import component_var
from quantcore.portfolio.black_litterman import black_litterman
from quantcore.portfolio.covariance import ledoit_wolf_shrinkage
from quantcore.portfolio.optimization import (
    min_variance_weights,
    risk_parity_weights,
    unconstrained_mean_variance_weights,
)

from agentic_trading.agents.schemas import MAX_LEVERAGE, SizingDiagnostics, SizingStage
from agentic_trading.config import settings
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)


class PodView(BaseModel):
    """One pod's combined analyst opinion on one asset, in signal units:
    `net_signal` is the sum of direction_sign x confidence over the pod's
    active signals, `signal_count` how many signals contributed, and
    `confidence_sum` the sum of their confidences regardless of sign —
    so `|net_signal| / confidence_sum` is how unanimous the pod is.
    `information_coefficient` is the confidence-weighted mean of the
    contributing analysts' shrunk trailing ICs (see
    `PerformanceScoreboard.shrunk_information_coefficient`); None means
    "use the firm-wide prior"."""

    model_config = ConfigDict(frozen=True)

    net_signal: float = 0.0
    signal_count: int = Field(default=0, ge=0)
    confidence_sum: float = Field(default=0.0, ge=0.0)
    information_coefficient: float | None = Field(default=None, ge=0.0)


# Every numeric default here (risk aversion, tau, view uncertainty,
# observation floors, vol target, Kelly lookback, cost-clearance hurdle)
# lives in `config.settings.portfolio`, with the reasoning for each value
# there. Constructor arguments left as `None` resolve to it at build time.


class PortfolioConstructor:
    """Eight-stage pipeline turning one pod's raw analyst views into a sized
    book: covariance estimate (Ledoit-Wolf shrinkage) -> equilibrium prior
    (`implied_equilibrium_returns`) -> Black-Litterman posterior using the
    pod's own views -> a risk budget (`method` selects min_variance/
    mean_variance/risk_parity, default mean_variance) -> volatility-targeting
    leverage (lever the whole book so its own realized annualized vol
    tracks `target_annualized_vol`, hard-clipped to `leverage_cap` as a
    sanity backstop, not the primary sizing lever — this is the
    Bridgewater All-Weather mechanism, not Kelly) -> component-VaR
    enforcement (any asset whose share of the pod's own total component VaR
    exceeds `settings.portfolio.component_var_max_share` has its weight
    scaled down by a single proportional pass — an approximation, not an
    iterative solver, same rigor level as the other single-pass scale-downs
    in this class) -> a rebalance calendar with a drift-band override (a
    full rebalance only executes every `settings.portfolio.
    rebalance_calendar_days` trading days; on an off-cycle day, only assets
    whose relative drift from `previous_weights` exceeds
    `settings.portfolio.rebalance_drift_band` are let through at their new
    weight, everything else snaps back to what's actually held) -> a final
    component-VaR risk report (logged only, computed on the truly final
    weights).

    `method` matters more than a tuning knob: `min_variance` sizes purely
    off `cov_matrix`, discarding the Black-Litterman posterior this class
    just spent four stages building from the pod's own views —
    `mean_variance` is the only one that reads `expected_returns`.
    `min_variance`/`mean_variance` are both called with `allow_short=True`
    (a quantcore-level parameter): a bearish enough view can size to a
    genuine negative weight, not just floor at zero, matching
    `PortfolioDecision.target_weight`'s own `[-MAX_LEVERAGE, MAX_LEVERAGE]`
    bounds and `AgentSignal.direction`'s first-class "short".
    `risk_parity` never reads `expected_returns` either, but it does read
    the views directly: quantcore's equal-risk budget is tilted by each
    asset's conviction and given the view's sign (see
    `_risk_budget_weights`), so a short view becomes a short budget.

    `views` are each pod's own raw, unbounded combined-analyst signal per
    asset (direction_sign * confidence, summed across that pod's active
    analyst signals) — not pre-clamped PortfolioDecisions, since the
    clamping now happens once, here, after real risk-budgeting, rather
    than before it independently per asset."""

    def __init__(
        self,
        method: str = "mean_variance",
        risk_aversion: float | None = None,
        tau: float | None = None,
        leverage_cap: float = MAX_LEVERAGE,
        target_annualized_vol: float | None = None,
        var_confidence_level: float | None = None,
        net_leverage_cap: float | None = None,
        kelly_fraction_weight: float = 0.0,
        kelly_lookback_days: int | None = None,
        transaction_cost_bps: float | None = None,
        no_trade_buffer: float | None = None,
    ) -> None:
        defaults = settings.portfolio
        transaction_cost_bps = (
            defaults.transaction_cost_bps if transaction_cost_bps is None else transaction_cost_bps
        )
        no_trade_buffer = defaults.no_trade_buffer if no_trade_buffer is None else no_trade_buffer
        risk_aversion = defaults.risk_aversion if risk_aversion is None else risk_aversion
        tau = defaults.tau if tau is None else tau
        target_annualized_vol = (
            defaults.target_annualized_vol
            if target_annualized_vol is None
            else target_annualized_vol
        )
        var_confidence_level = (
            defaults.var_confidence_level if var_confidence_level is None else var_confidence_level
        )
        kelly_lookback_days = (
            defaults.kelly_lookback_days if kelly_lookback_days is None else kelly_lookback_days
        )
        if not 0.0 <= leverage_cap <= MAX_LEVERAGE:
            raise ValueError(f"leverage_cap must be within [0, {MAX_LEVERAGE}]")
        if net_leverage_cap is not None and not 0.0 <= net_leverage_cap <= leverage_cap:
            raise ValueError(f"net_leverage_cap must be within [0, {leverage_cap}]")
        if not 0.0 <= kelly_fraction_weight <= 1.0:
            raise ValueError("kelly_fraction_weight must be within [0, 1]")
        self.method = method
        self.risk_aversion = risk_aversion
        self.tau = tau
        self.leverage_cap = leverage_cap
        self.target_annualized_vol = target_annualized_vol
        self.var_confidence_level = var_confidence_level
        self.net_leverage_cap = leverage_cap if net_leverage_cap is None else net_leverage_cap
        self.kelly_fraction_weight = kelly_fraction_weight
        self.kelly_lookback_days = kelly_lookback_days
        self.transaction_cost_bps = transaction_cost_bps
        self.no_trade_buffer = no_trade_buffer
        # This pod's own realized daily returns, from yesterday's sized
        # weights against today's real asset returns — same
        # last-weights-times-next-day-returns shape as Commander's own
        # _PodReturnTracker one layer up, but scoped to this one book.
        self._own_return_history: list[tuple[pd.Timestamp, float]] = []
        # Rebalance-calendar state: `_last_rebalance_date` is `None` until
        # this pod's first full/scheduled rebalance ever fires;
        # `_days_since_last_rebalance` counts simulated trading days
        # (one `construct()` call each) since that last full rebalance.
        self._last_rebalance_date: pd.Timestamp | None = None
        self._days_since_last_rebalance: int = 0
        # The owning manager's `agent_id`, set by `ManagerAgent` before each
        # `construct()` call; `None` for a constructor no manager runs.
        self.manager_id: str | None = None
        self._last_diagnostics: SizingDiagnostics | None = None

    def last_diagnostics(self) -> SizingDiagnostics | None:
        """The per-stage volatility diagnostics of the most recent
        `construct()` call. `None` before any call, and after a call that
        sized nothing (no views, or `returns_history` missing an asset)."""
        return self._last_diagnostics

    @staticmethod
    def _annualized_vol(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
        """Ex-ante annualised volatility `sqrt(w' Sigma w * trading_days_per_year)`
        of `weights` under the daily covariance `cov_matrix`."""
        daily_variance = float(weights @ cov_matrix @ weights)
        return math.sqrt(max(daily_variance, 0.0) * settings.firm.trading_days_per_year)

    @staticmethod
    def _binding_stage(
        target_vol: float, vol_after: dict[SizingStage, float]
    ) -> SizingStage | None:
        """The stage in `vol_after` whose vol fell furthest below the vol
        before it (`target_vol` before the first stage); the earliest such
        stage on a tie. `None` when no stage lowered vol by more than
        `1e-9 * target_vol`."""
        threshold = 1e-9 * target_vol
        binding: SizingStage | None = None
        largest_drop = threshold
        previous_vol = target_vol
        for stage, vol in vol_after.items():
            drop = previous_vol - vol
            if drop > largest_drop:
                binding, largest_drop = stage, drop
            previous_vol = vol
        return binding

    @staticmethod
    def _conviction_z_scores(assets: list[str], views: dict[str, PodView]) -> np.ndarray:
        """Per-asset conviction z-score: net signal scaled by
        `sqrt(signal_count)` (a sum of n unit-scale signals grows like
        sqrt(n)), capped at `view_zscore_cap`; 0.0 for an asset with no
        active view."""
        cap = settings.portfolio.view_zscore_cap
        return np.array(
            [
                float(
                    np.clip(
                        views[asset].net_signal / math.sqrt(max(views[asset].signal_count, 1)),
                        -cap,
                        cap,
                    )
                )
                if views[asset].signal_count > 0
                else 0.0
                for asset in assets
            ]
        )

    def _risk_budget_weights(
        self, expected_returns: np.ndarray, cov_matrix: np.ndarray, z_scores: np.ndarray
    ) -> np.ndarray:
        # `mean_variance`: the unconstrained long/short solution
        # w = Sigma^-1 mu / risk_aversion — no full-investment budget, so a
        # pod that is net short in its views is net short in its book, and
        # an asset with no view gets no weight. Gross exposure is set
        # afterwards by vol targeting and the leverage caps, not here.
        # `min_variance` keeps quantcore's
        # fully-invested, view-free solution. `risk_parity` starts from
        # quantcore's equal-risk-contribution budget and tilts each asset's
        # share by `risk_parity_view_tilt` x its conviction (|z| clipped to
        # 1), taking the view's sign, so a short view becomes a short risk
        # budget and a no-view asset keeps its plain long budget; gross is
        # renormalised to 1 so vol targeting sees the same scale.
        if self.method == "min_variance":
            return np.asarray(min_variance_weights(cov_matrix, allow_short=True))
        if self.method == "mean_variance":
            if not np.any(expected_returns):
                return np.zeros(len(expected_returns))
            return np.asarray(
                unconstrained_mean_variance_weights(
                    expected_returns, cov_matrix, self.risk_aversion
                )
            )
        if self.method == "risk_parity":
            budget = np.asarray(risk_parity_weights(cov_matrix))
            tilt = settings.portfolio.risk_parity_view_tilt
            conviction = np.clip(np.abs(z_scores), 0.0, 1.0)
            signs = np.where(z_scores < 0.0, -1.0, 1.0)
            tilted = budget * (1.0 + tilt * conviction) * signs
            gross = float(np.sum(np.abs(tilted)))
            return tilted / gross if gross > 0.0 else tilted
        raise ValueError(f"Unsupported portfolio construction method {self.method!r}")

    def _bootstrap_weights_from_views(self, views: dict[str, PodView]) -> dict[str, float]:
        """Equal-magnitude sizing by view sign alone, gross-capped at
        `min(leverage_cap, 1.0)` — the same "something simple and
        defensible beats an empty book" principle `Commander`'s own
        early-run bootstrap already uses one layer up (see
        `commander.py`), applied here for a pod that has real analyst
        views but not yet enough covariance history
        (`settings.portfolio.min_observations_for_covariance`) to risk-budget them
        properly. Never invents a signal — an asset with a zero view
        gets zero weight regardless. A wide basket (many assets) can
        otherwise need dozens of trading days of history before
        Ledoit-Wolf's `T > k` requirement is ever satisfied; sitting
        entirely in cash for that whole stretch despite having real,
        already-computed directional views is not how a real desk
        would handle a strategy that's simply new."""
        directional_assets = [asset for asset, view in views.items() if view.net_signal != 0.0]
        if not directional_assets:
            return dict.fromkeys(views, 0.0)
        per_asset_weight = min(self.leverage_cap, 1.0) / len(directional_assets)
        return {
            asset: (per_asset_weight if views[asset].net_signal > 0.0 else -per_asset_weight)
            if asset in directional_assets
            else 0.0
            for asset in views
        }

    def _view_direction_diagnostics(self, as_of_date: pd.Timestamp) -> SizingDiagnostics:
        """Diagnostics for a book sized by `_view_direction_book`: no
        covariance to measure vol with, so no per-stage vols."""
        return SizingDiagnostics(
            as_of_date=as_of_date.date(),
            manager_id=self.manager_id,
            path="view_direction",
            target_vol=self.target_annualized_vol,
            unlevered_vol=None,
            vol_after={},
            final_vol=None,
            binding_stage=None,
        )

    def _view_direction_book(
        self, assets: list[str], views: dict[str, PodView]
    ) -> dict[str, float]:
        """`_bootstrap_weights_from_views` followed by the net-leverage trim:
        the whole book this pod runs when it has no covariance it can
        risk-budget with."""
        bootstrap_weights = self._bootstrap_weights_from_views(views)
        trimmed = self._apply_net_leverage_trim(
            np.array([bootstrap_weights[asset] for asset in assets])
        )
        return dict(zip(assets, trimmed.tolist(), strict=True))

    def _record_own_realized_return(
        self,
        previous_weights: dict[str, float] | None,
        returns_history: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> None:
        """Appends this pod's own realized return for `as_of_date` —
        yesterday's sized weights dotted with today's real per-asset
        returns — to `_own_return_history`. No-op on the first call for a
        pod (no previous weights yet) or if `as_of_date` isn't in
        `returns_history`."""
        if not previous_weights or as_of_date not in returns_history.index:
            return
        day_returns = returns_history.loc[as_of_date]
        realized = 0.0
        for asset, weight in previous_weights.items():
            realized += weight * float(cast(float, day_returns.get(asset, 0.0)))
        self._own_return_history.append((as_of_date, realized))

    def _kelly_overlay_scalar(self) -> float:
        """A bounded, confidence-scaled overlay on `leverage`, derived
        from this pod's own trailing realized Sharpe — not raw Kelly
        (`f* = expected_return / variance`), and not a flat fraction
        either. `kelly_fraction_weight` is the *target* fraction; the
        fraction actually applied ramps linearly from 0 up to that
        target as trailing history accumulates
        (`len(_own_return_history) / kelly_lookback_days`), so a pod with
        almost no track record gets a scalar pinned near 1.0 (a no-op)
        rather than reacting to a noisy 2-3-observation Sharpe. Returns
        1.0 (no-op) if the overlay is disabled or there isn't yet enough
        history for a meaningful Sharpe estimate at all."""
        if (
            self.kelly_fraction_weight <= 0.0
            or len(self._own_return_history) < settings.portfolio.min_observations_for_covariance
        ):
            return 1.0
        returns = np.array([r for _, r in self._own_return_history])
        std = float(np.std(returns, ddof=1))
        if std < 1e-12:
            return 1.0
        trailing_sharpe = float(np.mean(returns)) / std
        confidence = min(1.0, len(returns) / self.kelly_lookback_days)
        effective_fraction = self.kelly_fraction_weight * confidence
        return 1.0 + effective_fraction * math.tanh(trailing_sharpe / 2.0)

    def _apply_net_leverage_trim(self, final_weights: np.ndarray) -> np.ndarray:
        """Trims only the same-sign (net-tilting) positions down to
        `net_leverage_cap`, leaving the offsetting/hedge leg untouched —
        a uniform book-wide rescale would also cut the very gross
        exposure a pairs/factor book is supposed to run."""
        net_exposure = float(np.sum(final_weights))
        if abs(net_exposure) <= self.net_leverage_cap or net_exposure == 0.0:
            return final_weights
        same_sign_mask = np.sign(final_weights) == np.sign(net_exposure)
        same_sign_sum = float(final_weights[same_sign_mask].sum())
        if same_sign_sum == 0.0:
            return final_weights
        excess = net_exposure - math.copysign(self.net_leverage_cap, net_exposure)
        final_weights = final_weights.copy()
        final_weights[same_sign_mask] *= (same_sign_sum - excess) / same_sign_sum
        return final_weights

    def _apply_component_var_cap(
        self, assets: list[str], final_weights: np.ndarray, cov_matrix: np.ndarray
    ) -> np.ndarray:
        """Scales down any single asset whose share of this pod's own total
        component VaR exceeds `settings.portfolio.component_var_max_share`,
        by `(component_var_max_share * total) / that asset's component` —
        a single proportional pass per over-cap asset, not an iterative
        re-solve, so a scale-down never itself re-triggers another asset's
        cap in the same call."""
        risk_report = component_var(final_weights, cov_matrix, self.var_confidence_level)
        total = float(np.sum(np.abs(risk_report)))
        if total <= 0.0:
            return final_weights
        cap = settings.portfolio.component_var_max_share
        shares = np.abs(risk_report) / total
        adjusted = final_weights.copy()
        for index, asset in enumerate(assets):
            if shares[index] > cap and risk_report[index] != 0.0:
                scale = (cap * total) / abs(float(risk_report[index]))
                adjusted[index] *= scale
                logger.info(
                    "PortfolioConstructor: %s component VaR share %.1f%% exceeded the "
                    "%.1f%% cap — scaled its weight down by %.3fx.",
                    asset,
                    shares[index] * 100.0,
                    cap * 100.0,
                    scale,
                )
        return adjusted

    def _apply_rebalance_calendar(
        self,
        assets: list[str],
        final_weights: np.ndarray,
        previous_weights: dict[str, float] | None,
        as_of_date: pd.Timestamp,
    ) -> np.ndarray:
        """Enforces a scheduled rebalance cadence with a drift-band
        override. A full rebalance (every asset in `final_weights` passes
        through unchanged) executes on this pod's first-ever call, on any
        call with no `previous_weights` yet, and once every
        `settings.portfolio.rebalance_calendar_days` trading days
        thereafter. On any other, off-cycle day, only the assets whose
        relative drift from `previous_weights` exceeds
        `settings.portfolio.rebalance_drift_band` (or that have no prior
        weight at all — a brand-new position always counts as drifted) are
        let through at their new weight; every other asset snaps back to
        its held (`previous_weights`) value for the day."""
        drift_epsilon = 1e-6
        self._days_since_last_rebalance += 1
        scheduled = (
            self._last_rebalance_date is None
            or not previous_weights
            or self._days_since_last_rebalance >= settings.portfolio.rebalance_calendar_days
        )
        if scheduled:
            self._last_rebalance_date = as_of_date
            self._days_since_last_rebalance = 0
            logger.info(
                "PortfolioConstructor: scheduled rebalance executed as of %s for %s.",
                as_of_date.date(),
                assets,
            )
            return final_weights

        assert previous_weights is not None
        adjusted = final_weights.copy()
        drifted_assets = []
        suppressed_assets = []
        for index, asset in enumerate(assets):
            if asset not in previous_weights:
                drifted_assets.append(asset)
                continue
            prior = previous_weights[asset]
            relative_drift = abs(final_weights[index] - prior) / max(abs(prior), drift_epsilon)
            if relative_drift > settings.portfolio.rebalance_drift_band:
                drifted_assets.append(asset)
            else:
                if final_weights[index] != prior:
                    suppressed_assets.append(asset)
                adjusted[index] = prior
        if drifted_assets:
            logger.info(
                "PortfolioConstructor: off-cycle drift-band override let %s through as of %s.",
                drifted_assets,
                as_of_date.date(),
            )
        if suppressed_assets:
            logger.info(
                "PortfolioConstructor: rebalance calendar snapped %s back to their held "
                "weight as of %s.",
                suppressed_assets,
                as_of_date.date(),
            )
        return adjusted

    def _black_litterman_expected_returns(
        self, assets: list[str], views: dict[str, PodView], cov_matrix: np.ndarray
    ) -> np.ndarray:
        """Posterior daily expected returns from the pod's views.

        Prior: zero (an alpha pod has no benchmark to be long by default,
        so an asset the pod has no opinion on has no expected return).
        Views: one per asset that has at least one active signal, in
        return units via the Grinold-Kahn rule
        `alpha_i = IC_i * sigma_i * z_i`, where `z_i` is the capped
        conviction z-score (`_conviction_z_scores`) and `IC_i` the view's
        own `information_coefficient`, falling back to the firm-wide prior
        `settings.portfolio.information_coefficient`. View
        uncertainty follows Idzorek: `omega_i = tau * Sigma_ii * (1 - c_i)
        / c_i` with `c_i = |net| / confidence_sum` clipped to
        [`view_confidence_floor`, `view_confidence_ceiling`], so `c_i` is
        the fraction of the way the posterior moves from the prior toward
        the view. Assets without a view are left out of P/Q entirely and
        keep the zero prior."""
        cfg = settings.portfolio
        viewed = [
            index
            for index, asset in enumerate(assets)
            if views[asset].signal_count > 0 and views[asset].net_signal != 0.0
        ]
        if not viewed:
            return np.zeros(len(assets))
        sigma = np.sqrt(np.maximum(np.diag(cov_matrix), 1e-18))
        z_scores = self._conviction_z_scores(assets, views)[viewed]
        information_coefficients = np.array(
            [
                cfg.information_coefficient
                if views[assets[i]].information_coefficient is None
                else views[assets[i]].information_coefficient
                for i in viewed
            ]
        )
        view_returns = information_coefficients * sigma[viewed] * z_scores
        confidences = np.array(
            [
                float(
                    np.clip(
                        abs(views[assets[i]].net_signal)
                        / max(views[assets[i]].confidence_sum, 1e-12),
                        cfg.view_confidence_floor,
                        cfg.view_confidence_ceiling,
                    )
                )
                for i in viewed
            ]
        )
        view_matrix = np.zeros((len(viewed), len(assets)))
        for row, index in enumerate(viewed):
            view_matrix[row, index] = 1.0
        omega = np.diag(self.tau * np.diag(cov_matrix)[viewed] * (1.0 - confidences) / confidences)
        posterior_returns, _posterior_cov = black_litterman(
            cov_matrix,
            np.zeros(len(assets)),
            view_matrix,
            view_returns,
            omega,
            self.risk_aversion,
            self.tau,
        )
        logger.debug(
            "PortfolioConstructor BL views z=%s ic=%s alpha=%s confidence=%s posterior=%s",
            z_scores.tolist(),
            information_coefficients.tolist(),
            view_returns.tolist(),
            confidences.tolist(),
            posterior_returns.tolist(),
        )
        return np.asarray(posterior_returns)

    def construct(
        self,
        views: dict[str, PodView],
        returns_history: pd.DataFrame,
        as_of_date: pd.Timestamp,
        previous_weights: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Returns `{asset: target_weight}` for every asset in `views`,
        each already within `[-leverage_cap, leverage_cap]` and the whole
        book's gross exposure within `leverage_cap` too. Returns every
        asset at weight 0.0 (never omits a key) if `returns_history` is
        missing an asset entirely (a real data gap, not just a
        not-enough-history-yet timing issue) — the caller still needs
        one `PortfolioDecision` per asset regardless.

        An asset whose returns do not vary at all over `returns_history`
        (every observation equal, e.g. a flat price series) has no
        measurable risk: it is sized at 0.0, with a warning, and left out
        of the covariance estimate and every stage after it. When no asset
        varies, the whole book is sized off view direction alone, as with
        too little history."""
        assets = list(views.keys())
        self._last_diagnostics = None
        if not assets:
            return {}

        if any(asset not in returns_history.columns for asset in assets):
            logger.warning(
                "PortfolioConstructor: returns_history missing columns for %s — sizing "
                "at 0 for this day.",
                [asset for asset in assets if asset not in returns_history.columns],
            )
            return dict.fromkeys(assets, 0.0)

        # ledoit_wolf_shrinkage requires strictly more observations (T) than
        # assets (k) — MIN_OBSERVATIONS_FOR_COVARIANCE alone is only a
        # statistical-reasonableness floor for small baskets; for a basket
        # of k >= MIN_OBSERVATIONS_FOR_COVARIANCE assets it understates the
        # true requirement (T=5 observations with k=5 assets still fails
        # quantcore's T > k check), so the actual threshold is whichever of
        # the two is larger.
        min_observations = max(settings.portfolio.min_observations_for_covariance, len(assets) + 1)
        if len(returns_history) < min_observations:
            logger.debug(
                "PortfolioConstructor: only %d observation(s) as of %s (need >= %d for %d "
                "assets) — sizing off view direction alone until then.",
                len(returns_history),
                as_of_date.date(),
                min_observations,
                len(assets),
            )
            self._last_diagnostics = self._view_direction_diagnostics(as_of_date)
            return self._view_direction_book(assets, views)

        returns_matrix = returns_history[assets].to_numpy(dtype=np.float64)
        flat_assets = [
            asset
            for asset, column in zip(assets, returns_matrix.T, strict=True)
            if np.all(column == column[0])
        ]
        if len(flat_assets) == len(assets):
            logger.warning(
                "PortfolioConstructor: no asset in %s has any return variation over %d "
                "observation(s) as of %s — no covariance to risk-budget with, sizing off "
                "view direction alone.",
                assets,
                len(returns_history),
                as_of_date.date(),
            )
            self._last_diagnostics = self._view_direction_diagnostics(as_of_date)
            return self._view_direction_book(assets, views)
        if flat_assets:
            logger.warning(
                "PortfolioConstructor: %s have no return variation over %d observation(s) as "
                "of %s — sized at 0.0 and left out of the covariance estimate.",
                flat_assets,
                len(returns_history),
                as_of_date.date(),
            )
            # Every remaining asset varies, so this nested call runs the full
            # pipeline below exactly once for the reduced book.
            varying_weights = self.construct(
                {asset: view for asset, view in views.items() if asset not in flat_assets},
                returns_history,
                as_of_date,
                previous_weights,
            )
            return {asset: varying_weights.get(asset, 0.0) for asset in assets}

        self._record_own_realized_return(previous_weights, returns_history, as_of_date)

        cov_matrix = ledoit_wolf_shrinkage(returns_matrix)
        posterior_returns = self._black_litterman_expected_returns(assets, views, cov_matrix)

        risk_budgeted_weights = self._risk_budget_weights(
            posterior_returns, cov_matrix, self._conviction_z_scores(assets, views)
        )
        # Sigma^-1 mu can legitimately put a highly correlated name on the
        # opposite side of its own view as a hedge; log it, never override it.
        for index, asset in enumerate(assets):
            view_sign = np.sign(views[asset].net_signal)
            if view_sign != 0.0 and np.sign(risk_budgeted_weights[index]) == -view_sign:
                logger.debug(
                    "PortfolioConstructor: %s sized against its own view as a hedge "
                    "(view=%.2f, weight=%.4f)",
                    asset,
                    views[asset].net_signal,
                    risk_budgeted_weights[index],
                )

        # Bridgewater-style volatility targeting: lever the whole
        # risk-budgeted book up or down so its own realized annualized
        # volatility tracks target_annualized_vol — more leverage in calm
        # markets, automatic de-risking in stressed ones — then
        # leverage_cap is the hard sanity backstop for when the vol
        # estimate itself is noisy, not the primary sizing mechanism.
        portfolio_daily_variance = float(risk_budgeted_weights @ cov_matrix @ risk_budgeted_weights)
        portfolio_annualized_vol = math.sqrt(
            max(portfolio_daily_variance, 1e-12) * settings.firm.trading_days_per_year
        )
        leverage = self.target_annualized_vol / portfolio_annualized_vol
        leverage = max(0.0, min(leverage, self.leverage_cap))
        # Ex-ante annualised vol of the book after each stage below, for
        # `last_diagnostics()`; reads the weights, never changes them.
        vol_after: dict[SizingStage, float] = {
            "vol_target_cap": self._annualized_vol(risk_budgeted_weights * leverage, cov_matrix)
        }

        kelly_scalar = self._kelly_overlay_scalar()
        leverage = leverage * kelly_scalar

        final_weights = risk_budgeted_weights * leverage
        vol_after["kelly"] = self._annualized_vol(final_weights, cov_matrix)
        gross_leverage = float(np.sum(np.abs(final_weights)))
        if gross_leverage > self.leverage_cap and gross_leverage > 0:
            final_weights = final_weights * (self.leverage_cap / gross_leverage)
        vol_after["gross_cap"] = self._annualized_vol(final_weights, cov_matrix)

        final_weights = self._apply_net_leverage_trim(final_weights)
        vol_after["net_trim"] = self._annualized_vol(final_weights, cov_matrix)

        if previous_weights and (self.transaction_cost_bps > 0.0 or self.no_trade_buffer > 0.0):
            cost_derived_min_move = (
                settings.portfolio.cost_clearance_multiplier * self.transaction_cost_bps / 10_000.0
            )
            min_move = max(self.no_trade_buffer, cost_derived_min_move)
            final_weights = np.array(
                [
                    previous_weights.get(asset, 0.0)
                    if abs(w - previous_weights.get(asset, 0.0)) < min_move
                    else w
                    for asset, w in zip(assets, final_weights, strict=True)
                ]
            )
        vol_after["no_trade_band"] = self._annualized_vol(final_weights, cov_matrix)

        # component_var (quantcore.performance.metrics) returns an all-zero
        # vector rather than dividing by zero when portfolio_std is ~0 —
        # e.g. the no-trade buffer legitimately leaving every asset at
        # 0.0 — so both calls below can run unconditionally.
        final_weights = self._apply_component_var_cap(assets, final_weights, cov_matrix)
        vol_after["component_var_cap"] = self._annualized_vol(final_weights, cov_matrix)
        final_weights = self._apply_rebalance_calendar(
            assets, final_weights, previous_weights, as_of_date
        )
        vol_after["rebalance_calendar"] = self._annualized_vol(final_weights, cov_matrix)

        risk_report = component_var(final_weights, cov_matrix, self.var_confidence_level)
        logger.info(
            "PortfolioConstructor (%s) manager=%s leverage=%.2f kelly_scalar=%.3f component "
            "VaR as of %s: %s",
            self.method,
            self.manager_id,
            leverage,
            kelly_scalar,
            as_of_date.date(),
            dict(zip(assets, risk_report.tolist(), strict=True)),
        )

        clipped_weights = np.clip(final_weights, -self.leverage_cap, self.leverage_cap)
        vol_after["final_clip"] = self._annualized_vol(clipped_weights, cov_matrix)
        self._last_diagnostics = SizingDiagnostics(
            as_of_date=as_of_date.date(),
            manager_id=self.manager_id,
            path="full",
            target_vol=self.target_annualized_vol,
            unlevered_vol=self._annualized_vol(risk_budgeted_weights, cov_matrix),
            vol_after=vol_after,
            final_vol=vol_after["final_clip"],
            binding_stage=self._binding_stage(self.target_annualized_vol, vol_after),
        )

        return {
            asset: float(max(min(weight, self.leverage_cap), -self.leverage_cap))
            for asset, weight in zip(assets, final_weights, strict=True)
        }
