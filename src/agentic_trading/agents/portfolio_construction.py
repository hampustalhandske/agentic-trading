# portfolio_construction.py

"""Optional Layer-4 stage after the Commander. Off by default — a no-op
that returns the Commander's decisions unchanged — so existing runs and
tests are unaffected unless explicitly opted in."""

import numpy as np
import pandas as pd
from quantcore.performance.metrics import component_var
from quantcore.portfolio.black_litterman import black_litterman, implied_equilibrium_returns
from quantcore.portfolio.covariance import ledoit_wolf_shrinkage
from quantcore.portfolio.optimization import (
    kelly_fraction,
    mean_variance_weights,
    min_variance_weights,
    risk_parity_weights,
)

from agentic_trading.agents.schemas import PortfolioDecision
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_RISK_AVERSION = 2.5
DEFAULT_TAU = 0.025
DEFAULT_VIEW_UNCERTAINTY_SCALE = 0.05
MIN_VIEW_CONFIDENCE = 1e-4
DEFAULT_VAR_CONFIDENCE_LEVEL = 0.95
MIN_OBSERVATIONS_FOR_COVARIANCE = 5


class PortfolioConstructor:
    """No-op (returns `decisions` unchanged) when `enabled=False`.
    Otherwise runs a six-stage pipeline: covariance estimate (Ledoit-Wolf
    shrinkage) → equilibrium prior (implied_equilibrium_returns) →
    Black-Litterman using the Commander's own decisions as views → a risk
    budget (`method` selects min_variance/mean_variance/risk_parity) →
    Kelly leverage scaling applied uniformly across the risk-budgeted
    weights → a component-VaR risk report (logged, not returned —
    `PortfolioDecision` has no field for it; a future schema addition
    could carry this on `BacktestResult` if it needs to be queryable
    rather than just logged).
    """

    def __init__(
        self,
        enabled: bool = False,
        method: str = "risk_parity",
        risk_aversion: float = DEFAULT_RISK_AVERSION,
        tau: float = DEFAULT_TAU,
        view_uncertainty_scale: float = DEFAULT_VIEW_UNCERTAINTY_SCALE,
        max_gross_leverage: float = 1.0,
        var_confidence_level: float = DEFAULT_VAR_CONFIDENCE_LEVEL,
    ) -> None:
        self.enabled = enabled
        self.method = method
        self.risk_aversion = risk_aversion
        self.tau = tau
        self.view_uncertainty_scale = view_uncertainty_scale
        self.max_gross_leverage = max_gross_leverage
        self.var_confidence_level = var_confidence_level

    def _risk_budget_weights(
        self, expected_returns: np.ndarray, cov_matrix: np.ndarray
    ) -> np.ndarray:
        if self.method == "min_variance":
            return np.asarray(min_variance_weights(cov_matrix))
        if self.method == "mean_variance":
            return np.asarray(
                mean_variance_weights(expected_returns, cov_matrix, self.risk_aversion)
            )
        if self.method == "risk_parity":
            return np.asarray(risk_parity_weights(cov_matrix))
        raise ValueError(f"Unsupported portfolio construction method {self.method!r}")

    def construct(
        self,
        decisions: list[PortfolioDecision],
        returns_history: pd.DataFrame,
        market_weights: dict[str, float],
        as_of_date: pd.Timestamp,
    ) -> list[PortfolioDecision]:
        if not self.enabled or not decisions:
            return decisions

        assets = [decision.asset for decision in decisions]
        if any(asset not in returns_history.columns for asset in assets):
            logger.warning(
                "PortfolioConstructor: returns_history missing columns for %s — skipping "
                "portfolio construction for this day.",
                [asset for asset in assets if asset not in returns_history.columns],
            )
            return decisions

        if len(returns_history) < MIN_OBSERVATIONS_FOR_COVARIANCE:
            logger.debug(
                "PortfolioConstructor: only %d observation(s) as of %s (need >= %d) — "
                "skipping portfolio construction for this day.",
                len(returns_history),
                as_of_date.date(),
                MIN_OBSERVATIONS_FOR_COVARIANCE,
            )
            return decisions

        returns_matrix = returns_history[assets].to_numpy(dtype=np.float64)
        cov_matrix = ledoit_wolf_shrinkage(returns_matrix)
        market_weight_vector = np.array([market_weights.get(asset, 0.0) for asset in assets])

        # implied_equilibrium_returns is the explicit "equilibrium prior"
        # pipeline stage the module docstring describes; black_litterman
        # below recomputes the same quantity internally from cov_matrix/
        # market_weight_vector/risk_aversion, so this call exists to make
        # that stage visible (logged), not to feed a separate input.
        equilibrium_returns = implied_equilibrium_returns(
            cov_matrix, market_weight_vector, self.risk_aversion
        )
        logger.debug("PortfolioConstructor equilibrium prior: %s", equilibrium_returns.tolist())

        # Views: one per asset, P is the identity (each decision is a direct
        # view on its own asset's expected excess return, taken as
        # target_weight itself — a simple, documented proxy for "expected
        # return implied by conviction" since the Commander doesn't expose a
        # calibrated expected-return number). View uncertainty
        # omega_i = view_uncertainty_scale / max(confidence_i, epsilon),
        # where confidence_i is the decision's own |target_weight| — a
        # stronger conviction (larger |target_weight|) gets a tighter
        # (more trusted) uncertainty band.
        num_assets = len(assets)
        view_matrix = np.eye(num_assets)
        view_returns = np.array([decision.target_weight for decision in decisions])
        confidences = np.array(
            [max(abs(decision.target_weight), MIN_VIEW_CONFIDENCE) for decision in decisions]
        )
        omega = np.diag(self.view_uncertainty_scale / confidences)

        posterior_returns, _posterior_cov = black_litterman(
            cov_matrix,
            market_weight_vector,
            view_matrix,
            view_returns,
            omega,
            self.risk_aversion,
            self.tau,
        )

        risk_budgeted_weights = self._risk_budget_weights(posterior_returns, cov_matrix)

        portfolio_expected_return = float(risk_budgeted_weights @ posterior_returns)
        portfolio_variance = float(risk_budgeted_weights @ cov_matrix @ risk_budgeted_weights)
        leverage = kelly_fraction(portfolio_expected_return, max(portfolio_variance, 1e-12))
        leverage = max(0.0, min(leverage, self.max_gross_leverage))

        final_weights = risk_budgeted_weights * leverage
        gross_leverage = float(np.sum(np.abs(final_weights)))
        if gross_leverage > self.max_gross_leverage and gross_leverage > 0:
            final_weights = final_weights * (self.max_gross_leverage / gross_leverage)

        risk_report = component_var(final_weights, cov_matrix, self.var_confidence_level)
        logger.info(
            "PortfolioConstructor (%s) component VaR as of %s: %s",
            self.method,
            as_of_date.date(),
            dict(zip(assets, risk_report.tolist(), strict=True)),
        )

        return [
            decision.model_copy(update={"target_weight": float(max(min(weight, 1.0), -1.0))})
            for decision, weight in zip(decisions, final_weights, strict=True)
        ]
