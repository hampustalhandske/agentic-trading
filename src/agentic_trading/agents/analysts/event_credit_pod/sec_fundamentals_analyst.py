# sec_fundamentals_analyst.py

import math
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

# A fiscal quarter is 13 or 14 weeks (91 or 98 days); the band also admits
# calendar quarters and leaves out 6- and 9-month year-to-date facts.
_QUARTER_MIN_DAYS = 80
_QUARTER_MAX_DAYS = 100
# A fiscal year is 52 or 53 weeks (364 or 371 days), or a calendar year.
_ANNUAL_MIN_DAYS = 350
_ANNUAL_MAX_DAYS = 380
# Average days per quarter (365.25 / 4), the unit of the trend's time axis.
_DAYS_PER_QUARTER = 91.3125


def discrete_quarters(facts: pd.DataFrame, concept: str) -> pd.DataFrame:
    """One value per fiscal quarter of `concept`, built from company facts
    shaped like `load_company_facts_from_lakehouse` output (`concept`,
    `period_start`, `period_end`, `value`, `filed`, `available_date`).

    A fact spanning `_QUARTER_MIN_DAYS`..`_QUARTER_MAX_DAYS` days
    (inclusive) is a discrete quarter; longer year-to-date facts are
    ignored. A fiscal Q4 is rarely filed on its own, so when an annual
    fact (`_ANNUAL_MIN_DAYS`..`_ANNUAL_MAX_DAYS` days) has no discrete
    quarter ending on its `period_end`, Q4 is derived as the annual value
    minus the three discrete quarters lying inside the annual period —
    only when exactly three do. A derived row takes the annual fact's
    `period_end` and `filed`. When several facts of one kind end on the
    same day, the latest available one (ties: latest filed) is used.

    Returns columns `period_end`, `value`, `filed`, sorted by
    `period_end`; empty when `concept` has no quarters."""
    rows = facts.loc[facts["concept"] == concept]
    rows = rows.sort_values(["available_date", "filed"], kind="stable")
    span_days = (rows["period_end"] - rows["period_start"]).dt.days
    quarters = rows.loc[span_days.between(_QUARTER_MIN_DAYS, _QUARTER_MAX_DAYS)]
    quarters = quarters.drop_duplicates(subset="period_end", keep="last")
    annuals = rows.loc[span_days.between(_ANNUAL_MIN_DAYS, _ANNUAL_MAX_DAYS)]
    annuals = annuals.drop_duplicates(subset="period_end", keep="last")

    reported_ends = set(quarters["period_end"])
    derived: list[dict[str, object]] = []
    for annual in annuals.to_dict("records"):
        if annual["period_end"] in reported_ends:
            continue
        inside = quarters.loc[
            (quarters["period_start"] >= annual["period_start"])
            & (quarters["period_end"] <= annual["period_end"])
        ]
        if len(inside) != 3:
            continue
        derived.append(
            {
                "period_end": annual["period_end"],
                "value": float(annual["value"]) - float(inside["value"].sum()),
                "filed": annual["filed"],
            }
        )

    columns = ["period_end", "value", "filed"]
    result = quarters[columns]
    if derived:
        result = pd.concat([result, pd.DataFrame(derived, columns=columns)])
    return result.sort_values("period_end", kind="stable").reset_index(drop=True)


class SecFundamentalsAnalyst(AnalystAgent):
    """Fits an OLS trend of a chosen SEC EDGAR concept's quarterly values
    against time and signals from the trend's sign when it is
    statistically distinguishable from zero.

    The series is `discrete_quarters(facts, concept)`: discrete quarterly
    facts plus a fiscal Q4 derived from the annual fact where Q4 is not
    reported on its own. A derived Q4 of a per-share concept such as
    diluted EPS is approximate, because the share counts behind the
    annual and the three quarterly figures differ. The regression uses
    the latest `trailing_filings` quarters, with time measured in
    quarters since the first of them (`_DAYS_PER_QUARTER`), so a missing
    quarter leaves a gap rather than compressing the axis. Fewer than four
    quarters give an `unavailable` signal. Discrete quarters do not
    overlap, so the plain OLS standard error is used rather than a HAC
    estimate. Reports the latest filing date among the quarters used as
    `source_as_of` and their number as `sample_count`, so
    EventCreditManager can dampen on staleness."""

    agent_id: str = "sec_fundamentals_analyst"

    concept: str = DEFAULT_CONCEPT
    trailing_filings: int = DEFAULT_TRAILING_FILINGS
    # The latest `DEFAULT_TRAILING_FILINGS` quarters before the decision
    # date; an instance changing `trailing_filings` should also override
    # this.
    data_requirements: tuple[DataRequirement, ...] = (
        DataRequirement(
            source="sec_edgar_facts",
            identifier="{ticker}",
            asset_scoped=True,
            description="SEC EDGAR company facts",
            lookback_days=math.ceil(DEFAULT_TRAILING_FILINGS * _DAYS_PER_QUARTER),
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

        quarters = discrete_quarters(facts, self.concept)
        if len(quarters) < 4:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale=f"insufficient {self.concept} quarters for a trend fit",
                status="unavailable",
            )

        trailing = quarters.tail(self.trailing_filings)
        values = trailing["value"].to_numpy(dtype=np.float64)
        elapsed_days = (trailing["period_end"] - trailing["period_end"].iloc[0]).dt.days
        periods = elapsed_days.to_numpy(dtype=np.float64) / _DAYS_PER_QUARTER
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
