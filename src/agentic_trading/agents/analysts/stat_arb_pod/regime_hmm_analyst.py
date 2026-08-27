# regime_hmm_analyst.py

from typing import Literal

import numpy as np
import pandas as pd
from quantcore.regime.hmm import hmm_fit, hmm_predict_proba

from agentic_trading.agents._market_math import clamp, market_slice
from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

MIN_OBSERVATIONS = 250


class RegimeHMMAnalyst(AnalystAgent):
    """Fits a Gaussian HMM on trailing daily returns and signals from the
    current smoothed state posterior. States are labeled by their fitted
    mean (lowest mean = "bear", highest = "bull", any middle state =
    "neutral/chop") rather than a fixed state index, since EM doesn't
    guarantee state ordering."""

    agent_id: str = "regime_hmm_analyst"

    n_states: int = 3
    _risk_window: int = 250

    def generate_signal(
        self,
        asset: str,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> AgentSignal:
        returns = market_slice(market_data, as_of_date, asset).log_returns[-self._risk_window :]
        if returns.size < MIN_OBSERVATIONS:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale=f"need >= {MIN_OBSERVATIONS} observations for HMM regime fitting",
            )

        try:
            transition_matrix, means, variances, initial_probs = hmm_fit(returns, self.n_states)
            posteriors = hmm_predict_proba(
                returns, transition_matrix, means, variances, initial_probs
            )
        except (RuntimeError, ValueError) as exc:
            logger.debug(
                "%s: HMM fit failed for %s (%s) — returning flat.", self.agent_id, asset, exc
            )
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="HMM fit failed",
            )

        state_order = np.argsort(means)
        bear_state, *_middle, bull_state = state_order
        current_posterior = posteriors[-1]
        current_state = int(np.argmax(current_posterior))

        direction: Literal["long", "short", "flat"]
        if current_state == bull_state:
            direction = "long"
        elif current_state == bear_state:
            direction = "short"
        else:
            direction = "flat"

        confidence = clamp(float(current_posterior[current_state]))
        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction=direction,
            confidence=confidence,
            rationale=(
                f"HMM state posterior={current_posterior[current_state]:.3f} (state means={means})"
            ),
        )
