# fama_macbeth_analyst.py

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import Field
from quantcore.statistics.regression import factor_loadings, fama_macbeth_regression

from agentic_trading.agents._market_math import clamp, market_slice
from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.data.loaders.factor_loader import load_fama_french_factors
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_UNIVERSE: tuple[str, ...] = ("AAPL", "MSFT", "GOOGL", "AMZN", "JPM")


class FamaMacbethAnalyst(AnalystAgent):
    """The one analyst whose signal legitimately needs more than one
    ticker's data: estimates per-asset factor loadings across a
    configurable cross-sectional `universe` (default: a small fixed
    basket, not necessarily including `asset`), runs a Fama-MacBeth
    cross-sectional regression for the estimated risk premia, and predicts
    `asset`'s expected return from its own loadings times those premia.
    Loads the extra cross-sectional history it needs internally via the
    existing point-in-time-safe loaders, rather than expecting the
    Commander to hand it a wider `market_data` slice."""

    agent_id: str = "fama_macbeth_analyst"

    universe: tuple[str, ...] = Field(default=DEFAULT_UNIVERSE)
    model: str = "3factor"
    _risk_window: int = 120

    def _asset_returns(
        self, market_data: pd.DataFrame, as_of_date: pd.Timestamp, ticker: str
    ) -> np.ndarray:
        return market_slice(market_data, as_of_date, ticker).log_returns[-self._risk_window :]

    def generate_signal(
        self,
        asset: str,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> AgentSignal:
        universe = self.universe if asset in self.universe else (*self.universe, asset)
        try:
            factors = load_fama_french_factors(as_of_date, model=self.model)
        except Exception as exc:
            logger.debug("%s: factor download failed (%s) — returning flat.", self.agent_id, exc)
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="Fama-French factor download failed",
            )
        factor_columns = [column for column in factors.columns if column != "RF"]

        universe_returns = {
            ticker: self._asset_returns(market_data, as_of_date, ticker) for ticker in universe
        }
        min_length = min([len(factors), *(series.size for series in universe_returns.values())])
        if min_length < 30 or asset not in universe_returns:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="insufficient cross-sectional history for a Fama-MacBeth regression",
            )

        aligned_factors = factors.iloc[-min_length:]
        factor_matrix = aligned_factors[factor_columns].to_numpy(dtype=np.float64)
        risk_free = aligned_factors["RF"].to_numpy(dtype=np.float64)

        betas_by_ticker: dict[str, np.ndarray] = {}
        panel_returns = []
        for ticker in universe:
            excess = universe_returns[ticker][-min_length:] - risk_free
            betas, _t_stats = factor_loadings(excess, factor_matrix)
            betas_by_ticker[ticker] = betas
            panel_returns.append(excess)

        panel = np.column_stack(panel_returns)
        beta_matrix = np.vstack([betas_by_ticker[ticker] for ticker in universe])
        premia, _t_stats = fama_macbeth_regression(panel, beta_matrix)

        predicted_return = float(np.dot(betas_by_ticker[asset], premia))
        realized_vol = float(np.std(universe_returns[asset][-min_length:], ddof=1))

        direction: Literal["long", "short", "flat"] = "long" if predicted_return >= 0.0 else "short"
        confidence = clamp(abs(predicted_return) / max(realized_vol, 1e-8))

        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction=direction,
            confidence=confidence,
            rationale=(
                f"Fama-MacBeth predicted return={predicted_return:.5f} across universe {universe}"
            ),
        )
