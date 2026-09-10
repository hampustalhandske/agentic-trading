# macro_var_spillover_analyst.py

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import Field
from quantcore.time_series.var_model import (
    granger_causality_test,
    impulse_response,
    var_fit,
    var_forecast,
)

from agentic_trading.agents._market_math import clamp, market_slice
from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.data.loaders.macro_loader import DGS3MO, DGS10, VIXCLS, load_fred_series
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

GRANGER_SIGNIFICANCE = 0.05
DEFAULT_N_LAGS = 2


class MacroVarSpilloverAnalyst(AnalystAgent):
    """Builds a small VAR system from the asset's returns plus configurable
    FRED macro series (default: VIX level and the 10y-3m yield curve
    slope), gates on macro Granger-causing the asset, and weights
    confidence by the impulse-response magnitude — a statistically
    significant but tiny-impact relationship shouldn't produce a loud
    signal."""

    agent_id: str = "macro_var_spillover_analyst"

    macro_series_ids: tuple[str, ...] = Field(default=(VIXCLS,))
    use_yield_curve_slope: bool = True
    n_lags: int = DEFAULT_N_LAGS
    _risk_window: int = 120

    def _macro_series(self, as_of_date: pd.Timestamp) -> list[pd.Series] | None:
        series_list: list[pd.Series] = []
        try:
            for series_id in self.macro_series_ids:
                series_list.append(load_fred_series(series_id, as_of_date))
            if self.use_yield_curve_slope:
                yield_curve = pd.concat(
                    [load_fred_series(DGS10, as_of_date), load_fred_series(DGS3MO, as_of_date)],
                    axis=1,
                    join="inner",
                )
                series_list.append(yield_curve.iloc[:, 0] - yield_curve.iloc[:, 1])
        except Exception as exc:
            logger.debug(
                "%s: FRED macro series lookup failed (%s) — returning None.", self.agent_id, exc
            )
            return None
        return series_list

    def generate_signal(
        self,
        asset: str,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> AgentSignal:
        asset_returns = market_slice(market_data, as_of_date, asset).log_returns[
            -self._risk_window :
        ]
        macro_series = self._macro_series(as_of_date)
        if asset_returns.size < 30 or macro_series is None:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="insufficient asset/macro history for a VAR spillover read",
            )

        history = market_data.loc[market_data.index <= as_of_date]
        if isinstance(history.columns, pd.MultiIndex):
            if asset not in history.columns.get_level_values(0):
                asset_history = pd.DataFrame()
            else:
                asset_history = history.xs(asset, axis=1, level=0)
        else:
            asset_history = history

        if "log_return" not in asset_history:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="no asset return history for a VAR spillover read",
            )

        aligned = pd.concat(
            [asset_history["log_return"], *macro_series], axis=1, join="inner"
        ).replace([np.inf, -np.inf], np.nan).dropna()
        system = aligned.to_numpy(dtype=np.float64)
        if system.shape[0] < 4 * self.n_lags + 10:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="insufficient overlapping history for a VAR spillover read",
            )

        try:
            _, p_value = granger_causality_test(system[:, 0], system[:, 1], self.n_lags)
            if p_value >= GRANGER_SIGNIFICANCE:
                return AgentSignal(
                    agent_id=self.agent_id,
                    asset=asset,
                    as_of_date=as_of_date.date(),
                    direction="flat",
                    confidence=0.0,
                    rationale=f"macro does not Granger-cause {asset} (p={p_value:.3f})",
                )

            coef_matrix, sigma_u = var_fit(system, self.n_lags)
            forecast = var_forecast(system, coef_matrix, self.n_lags, h=1)
            forecast_move = float(forecast[0, 0])

            impulse = impulse_response(coef_matrix, sigma_u, self.n_lags, n_periods=1)
            impact_magnitude = (
                float(np.abs(impulse[0, 0, 1]))
                if impulse.ndim == 3
                else float(np.abs(impulse[0]))
            )
        except (np.linalg.LinAlgError, RuntimeError, ValueError) as exc:
            logger.debug(
                "%s: VAR fit failed for %s on %s (%s) — returning flat.",
                self.agent_id,
                asset,
                as_of_date.date(),
                exc,
            )
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="VAR spillover fit failed on the available finite observations",
            )

        direction: Literal["long", "short", "flat"] = "long" if forecast_move >= 0.0 else "short"
        confidence = clamp(impact_magnitude)

        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction=direction,
            confidence=confidence,
            rationale=(
                f"macro Granger-causes {asset} (p={p_value:.3f}), forecast move={forecast_move:.4f}"
            ),
        )
