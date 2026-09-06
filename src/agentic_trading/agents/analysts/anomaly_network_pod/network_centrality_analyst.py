# lead_lag_network_centrality_analyst.py

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import Field

from agentic_trading.agents._market_math import clamp, market_slice
from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_UNIVERSE: tuple[str, ...] = ("AAPL", "MSFT", "GOOGL", "AMZN", "JPM")
WINDOW = 60
LAG = 1
RETURN_CAP = 0.05


class LeadLagNetworkCentralityAnalyst(AnalystAgent):
    """The second analyst (after `fama_macbeth_analyst`) that legitimately
    needs more than one ticker's data: builds a rolling lag-1 cross-
    correlation graph across a configurable `universe` — edge (i, j) is
    the correlation between asset i's return at t-1 and asset j's return
    at t, i.e. how much i "leads" j — and derives two things from it for
    `asset`: a network-centrality read (the mean absolute lead-lag
    correlation between `asset` and every other node, a simple degree-
    centrality proxy — no graph library needed, since this reduces to a
    row/column average over a correlation matrix) used for confidence, and
    a directional forecast (each other asset's most recent return, weighted
    by how strongly it historically leads `asset`). Distinct methodology
    from the factor_pod regressions: this is a graph/network-science read
    of the cross-sectional return structure, not a linear factor model."""

    agent_id: str = "lead_lag_network_centrality_analyst"

    universe: tuple[str, ...] = Field(default=DEFAULT_UNIVERSE)

    def generate_signal(
        self,
        asset: str,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> AgentSignal:
        universe = self.universe if asset in self.universe else (*self.universe, asset)
        returns = {
            ticker: market_slice(market_data, as_of_date, ticker).log_returns for ticker in universe
        }

        min_length = min(series.size for series in returns.values())
        if min_length < WINDOW + LAG or asset not in returns:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale=(
                    f"insufficient cross-sectional history across universe {universe} — holding."
                ),
            )

        aligned = {ticker: series[-(WINDOW + LAG) :] for ticker, series in returns.items()}
        asset_current = aligned[asset][LAG:]

        lead_lag_corr: dict[str, float] = {}
        for ticker in universe:
            if ticker == asset:
                continue
            leader_lagged = aligned[ticker][:-LAG]
            with np.errstate(invalid="ignore"):
                corr = float(np.corrcoef(leader_lagged, asset_current)[0, 1])
            lead_lag_corr[ticker] = 0.0 if np.isnan(corr) else corr

        if not lead_lag_corr:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="universe contains no other tickers to build a lead-lag graph — holding.",
            )

        centrality = float(np.mean([abs(corr) for corr in lead_lag_corr.values()]))
        predicted_return = float(
            sum(corr * float(aligned[ticker][-1]) for ticker, corr in lead_lag_corr.items())
        )

        direction: Literal["long", "short", "flat"] = (
            "long" if predicted_return > 0.0 else "short" if predicted_return < 0.0 else "flat"
        )
        confidence = clamp(centrality) * clamp(abs(predicted_return) / RETURN_CAP)

        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction=direction,
            confidence=confidence,
            rationale=(
                f"lead-lag centrality={centrality:.3f}, predicted return={predicted_return:.5f} "
                f"across universe {universe} [sample_count={min_length}]"
            ),
        )
