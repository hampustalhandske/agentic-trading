# regime_hmm_analyst.py

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import PrivateAttr
from quantcore.regime.hmm import hmm_fit, hmm_predict_proba

from agentic_trading.agents._market_math import RefitCache, clamp, market_slice
from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.schemas import AgentSignal, SignalStatus
from agentic_trading.config import settings
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

N_STATES = 2
# Degeneracy guards on a fitted two-state model: the bear state must be
# meaningfully more volatile than the bull state, each state must
# account for a non-trivial share of the sample, and each state must be
# persistent — otherwise the fit is splitting noise, not regimes.
MIN_VARIANCE_RATIO = 1.5
MIN_STATE_OCCUPANCY = 0.05
MIN_STATE_PERSISTENCE = 0.8


class RegimeHMMAnalyst(AnalystAgent):
    """Hamilton two-state Gaussian regime model on the trailing
    `settings.analysts.hmm_window` daily log returns. The higher-variance
    state is "bear" (lower mean breaks a tie), the other "bull". The
    signal is the one-step-ahead bull probability — the last smoothed
    posterior row pushed through the transition matrix — long above
    `hmm_action_threshold`, short below `1 - hmm_action_threshold`, flat
    in between, with confidence scaling linearly from 0 at 50/50 to 1 at
    certainty.

    The fit is refreshed at most once every
    `settings.analysts.hmm_refit_interval_days` simulated days per asset
    (see `RefitCache`); every day in between decodes the latest window
    against the cached parameters. A fit whose two states are not
    genuinely distinct (variance ratio below `MIN_VARIANCE_RATIO`, a
    state occupying under `MIN_STATE_OCCUPANCY` of the sample, or a
    transition-matrix diagonal below `MIN_STATE_PERSISTENCE`) is treated
    as no regime structure and reports flat."""

    agent_id: str = "regime_hmm_analyst"

    _refit_cache: RefitCache = PrivateAttr(
        default_factory=lambda: RefitCache(
            refit_interval_days=settings.analysts.hmm_refit_interval_days
        )
    )

    def _flat(
        self,
        asset: str,
        as_of_date: pd.Timestamp,
        rationale: str,
        status: SignalStatus = "active",
    ) -> AgentSignal:
        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction="flat",
            confidence=0.0,
            rationale=rationale,
            status=status,
        )

    def generate_signal(
        self,
        asset: str,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> AgentSignal:
        returns = market_slice(market_data, as_of_date, asset).log_returns[
            -settings.analysts.hmm_window :
        ]
        min_observations = settings.analysts.hmm_min_observations
        if returns.size < min_observations:
            return self._flat(
                asset,
                as_of_date,
                f"need >= {min_observations} observations for HMM regime fitting",
                "unavailable",
            )

        try:
            transition_matrix, means, variances, initial_probs = self._refit_cache.get_or_refit(
                asset, as_of_date, lambda: hmm_fit(returns, N_STATES)
            )
            posteriors = hmm_predict_proba(
                returns, transition_matrix, means, variances, initial_probs
            )
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            logger.debug(
                "%s: HMM fit/decode failed for %s (%s) — returning flat.",
                self.agent_id,
                asset,
                exc,
            )
            return self._flat(asset, as_of_date, "HMM fit/decode failed", "error")

        variances = np.asarray(variances, dtype=float)
        means = np.asarray(means, dtype=float)
        if variances[0] == variances[1]:
            bear_state = int(np.argmin(means))
        else:
            bear_state = int(np.argmax(variances))
        bull_state = 1 - bear_state

        low_variance = float(variances[bull_state])
        variance_ratio = (
            float(variances[bear_state]) / low_variance if low_variance > 0.0 else float("inf")
        )
        if variance_ratio < MIN_VARIANCE_RATIO:
            return self._flat(
                asset,
                as_of_date,
                f"degenerate HMM: state variance ratio {variance_ratio:.2f} < "
                f"{MIN_VARIANCE_RATIO} — no distinct volatility regimes",
                "unavailable",
            )
        occupancy = np.asarray(posteriors, dtype=float).mean(axis=0)
        if float(occupancy.min()) < MIN_STATE_OCCUPANCY:
            return self._flat(
                asset,
                as_of_date,
                f"degenerate HMM: minimum state occupancy {float(occupancy.min()):.3f} < "
                f"{MIN_STATE_OCCUPANCY} — one state barely occurs in the sample",
                "unavailable",
            )
        persistence = float(np.min(np.diag(np.asarray(transition_matrix, dtype=float))))
        if persistence < MIN_STATE_PERSISTENCE:
            return self._flat(
                asset,
                as_of_date,
                f"degenerate HMM: minimum state persistence {persistence:.3f} < "
                f"{MIN_STATE_PERSISTENCE} — states are not regimes",
                "unavailable",
            )

        p_next = np.asarray(posteriors[-1], dtype=float) @ np.asarray(
            transition_matrix, dtype=float
        )
        p_bull = float(p_next[bull_state])

        direction: Literal["long", "short", "flat"]
        if p_bull > settings.analysts.hmm_action_threshold:
            direction = "long"
        elif p_bull < 1.0 - settings.analysts.hmm_action_threshold:
            direction = "short"
        else:
            direction = "flat"

        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction=direction,
            confidence=clamp(2.0 * abs(p_bull - 0.5)),
            rationale=(
                f"HMM one-step bull probability={p_bull:.3f} "
                f"(bull mean={means[bull_state]:.5f}, bear mean={means[bear_state]:.5f}, "
                f"variance ratio={variance_ratio:.2f})"
            ),
        )
