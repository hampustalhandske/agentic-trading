# sec_fundamentals_analyst.py

from typing import Literal

import numpy as np
import pandas as pd
from quantcore.statistics.regression import ols

from agentic_trading.agents._market_math import clamp
from agentic_trading.agents.base import AnalystAgent, DataRequirement, is_likely_equity
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.config import settings
from agentic_trading.data.loaders.lakehouse_loader import load_company_facts_from_lakehouse
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_CONCEPT = "EarningsPerShareDiluted"
DEFAULT_TRAILING_FILINGS = 8


class SecFundamentalsAnalyst(AnalystAgent):
    """Fits an OLS trend (period index vs. value) over the trailing
    filings of a chosen SEC EDGAR concept and signals from the trend's
    sign when statistically distinguishable from zero. Quarterly,
    non-overlapping filings don't need HAC correction the way daily
    overlapping return regressions do, so this uses the plain OLS standard
    error rather than Newey-West — `newey_west_cov` is only imported for
    the (rare, configurable) case where `trailing_filings` is large enough
    that overlapping reporting periods could still induce autocorrelation.
    Reports the latest filing date as `source_as_of` so
    EventCreditManager can dampen on staleness."""

    agent_id: str = "sec_fundamentals_analyst"

    concept: str = DEFAULT_CONCEPT
    trailing_filings: int = DEFAULT_TRAILING_FILINGS
    data_requirements: tuple[DataRequirement, ...] = (
        DataRequirement(
            source="sec_edgar_facts",
            identifier="{ticker}",
            asset_scoped=True,
            description="SEC EDGAR company facts",
        ),
    )

    def generate_signal(
        self,
        asset: str,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> AgentSignal:
        if not is_likely_equity(asset):
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="not an equity — SEC company facts don't apply",
                status="unavailable",
            )
        facts = load_company_facts_from_lakehouse(asset, as_of_date.date())
        if facts is None:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="no SEC EDGAR company-facts coverage in the lakehouse",
                status="unavailable",
            )

        concept_facts = facts.loc[facts["concept"] == self.concept].sort_values("filed")
        if len(concept_facts) < 4:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale=f"insufficient {self.concept} filings for a trend fit",
                status="unavailable",
            )

        trailing = concept_facts.tail(self.trailing_filings)
        values = trailing["value"].to_numpy(dtype=np.float64)
        periods = np.arange(len(values), dtype=np.float64)
        design_matrix = np.column_stack([np.ones(len(values)), periods])

        # ols() returns (coefficients, residuals, r_squared) — residuals is
        # shaped (n_observations,), not (n_coefficients,), so it can't be
        # indexed as a per-coefficient standard error. Compute the classical
        # OLS standard error directly from the residual sum of squares.
        coefs, residuals, _r_squared = ols(values, design_matrix)
        slope = float(coefs[1])
        degrees_of_freedom = len(values) - design_matrix.shape[1]
        if degrees_of_freedom > 0:
            residual_variance = float(residuals @ residuals) / degrees_of_freedom
            coef_covariance = residual_variance * np.linalg.inv(design_matrix.T @ design_matrix)
            slope_std_error = float(np.sqrt(max(coef_covariance[1, 1], 0.0)))
        else:
            slope_std_error = 0.0
        t_stat = slope / slope_std_error if slope_std_error > 0 else 0.0

        latest_filed = trailing["filed"].max().date()
        if abs(t_stat) < settings.analysts.t_stat_threshold:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale=f"{self.concept} trend not significant (t={t_stat:.2f})",
                source_as_of=latest_filed,
                sample_count=len(values),
            )

        direction: Literal["long", "short", "flat"] = "long" if slope >= 0.0 else "short"
        confidence = clamp(abs(t_stat) / (2 * settings.analysts.t_stat_threshold))

        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction=direction,
            confidence=confidence,
            rationale=f"{self.concept} trend slope={slope:.4f}, t={t_stat:.2f}",
            source_as_of=latest_filed,
            sample_count=len(values),
        )
