# sec_fundamentals_analyst.py

from typing import Literal

import numpy as np
import pandas as pd
from quantcore.statistics.regression import ols

from agentic_trading.agents._market_math import clamp
from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.data.loaders.fmp_loader import load_fmp_fundamentals
from agentic_trading.data.loaders.fundamentals_loader import load_company_facts
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_CONCEPT = "EarningsPerShareDiluted"
DEFAULT_TRAILING_FILINGS = 8
T_STAT_SIGNIFICANCE = 2.0

# SEC EDGAR's XBRL concept names have no free-tier lookup on FMP's
# income-statement/balance-sheet endpoints for every concept (there's no
# FMP equivalent of diluted EPS on those two endpoints) — only the
# concepts in this map have a genuine FMP substitute, so the fallback
# below only ever engages for those.
_FMP_CONCEPT_FALLBACK: dict[str, str] = {
    "Revenues": "revenue",
    "Assets": "totalAssets",
    "Liabilities": "totalLiabilities",
    "StockholdersEquity": "totalStockholdersEquity",
}


class SecFundamentalsAnalyst(AnalystAgent):
    """Fits an OLS trend (period index vs. value) over the trailing
    filings of a chosen SEC EDGAR concept and signals from the trend's
    sign when statistically distinguishable from zero. Quarterly,
    non-overlapping filings don't need HAC correction the way daily
    overlapping return regressions do, so this uses the plain OLS standard
    error rather than Newey-West — `newey_west_cov` is only imported for
    the (rare, configurable) case where `trailing_filings` is large enough
    that overlapping reporting periods could still induce autocorrelation.
    Appends `[source_as_of=...]` per EventCreditManager's staleness
    convention."""

    agent_id: str = "sec_fundamentals_analyst"

    concept: str = DEFAULT_CONCEPT
    trailing_filings: int = DEFAULT_TRAILING_FILINGS

    def _concept_facts(self, asset: str, as_of_date: pd.Timestamp) -> pd.DataFrame:
        """SEC EDGAR facts for `self.concept`, falling back to FMP's
        income-statement/balance-sheet endpoints (`_FMP_CONCEPT_FALLBACK`)
        when SEC EDGAR fails or has too few filings and an FMP equivalent
        of the concept exists."""
        try:
            facts = load_company_facts(asset, as_of_date)
            concept_facts = facts.loc[facts["concept"] == self.concept].sort_values("filed")
        except Exception as exc:
            logger.debug(
                "%s: SEC EDGAR lookup failed for %s (%s) — trying FMP fallback if available.",
                self.agent_id,
                asset,
                exc,
            )
            concept_facts = pd.DataFrame(columns=["concept", "value", "fiscal_period", "filed"])

        if len(concept_facts) >= 4:
            return concept_facts

        fmp_concept = _FMP_CONCEPT_FALLBACK.get(self.concept)
        if fmp_concept is None:
            return concept_facts

        try:
            fmp_facts = load_fmp_fundamentals(asset, as_of_date)
        except Exception as exc:
            logger.debug(
                "%s: FMP fallback lookup failed for %s (%s) — using SEC EDGAR result as-is.",
                self.agent_id,
                asset,
                exc,
            )
            return concept_facts

        fmp_concept_facts = fmp_facts.loc[fmp_facts["concept"] == fmp_concept].sort_values("filed")
        return fmp_concept_facts if len(fmp_concept_facts) >= 4 else concept_facts

    def generate_signal(
        self,
        asset: str,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> AgentSignal:
        concept_facts = self._concept_facts(asset, as_of_date)
        if len(concept_facts) < 4:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale=f"insufficient {self.concept} filings for a trend fit",
            )

        trailing = concept_facts.tail(self.trailing_filings)
        values = trailing["value"].to_numpy(dtype=np.float64)
        periods = np.arange(len(values), dtype=np.float64)
        design_matrix = np.column_stack([np.ones(len(values)), periods])

        coefs, std_errors, _r_squared = ols(values, design_matrix)
        slope = float(coefs[1])
        slope_std_error = float(std_errors[1])
        t_stat = slope / slope_std_error if slope_std_error > 0 else 0.0

        latest_filed = trailing["filed"].max().date()
        if abs(t_stat) < T_STAT_SIGNIFICANCE:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale=(
                    f"{self.concept} trend not significant (t={t_stat:.2f}) "
                    f"[source_as_of={latest_filed.isoformat()}]"
                ),
            )

        direction: Literal["long", "short", "flat"] = "long" if slope >= 0.0 else "short"
        confidence = clamp(abs(t_stat) / (2 * T_STAT_SIGNIFICANCE))

        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction=direction,
            confidence=confidence,
            rationale=(
                f"{self.concept} trend slope={slope:.4f}, t={t_stat:.2f} "
                f"[source_as_of={latest_filed.isoformat()}]"
            ),
        )
