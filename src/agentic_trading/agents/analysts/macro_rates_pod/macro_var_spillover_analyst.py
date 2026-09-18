# macro_var_spillover_analyst.py

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import Field, PrivateAttr
from quantcore.time_series.var_model import (
    granger_causality_test,
    impulse_response,
    select_var_lag_order,
    var_fit,
    var_forecast,
)

from agentic_trading.agents._market_math import RefitCache, clamp, market_slice
from agentic_trading.agents.base import AnalystAgent, DataRequirement
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.config import settings
from agentic_trading.data.loaders.lakehouse_loader import load_fred_series_from_lakehouse
from agentic_trading.data.loaders.macro_loader import DGS3MO, DGS10, VIXCLS
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

# Search grid for select_var_lag_order when n_lags isn't explicitly
# pinned — a daily macro/asset-return system rarely benefits from a
# longer lag structure than this.
MAX_LAGS = 5
LAG_SELECTION_CRITERION = "aic"


class MacroVarSpilloverAnalyst(AnalystAgent):
    """Builds a small VAR system from the asset's returns plus configurable
    FRED macro series (default: VIX level and the 10y-3m yield curve
    slope), gates on macro Granger-causing the asset, and weights
    confidence by the impulse-response magnitude — a statistically
    significant but tiny-impact relationship shouldn't produce a loud
    signal. The lag order is selected fresh each refit via
    `quantcore.time_series.var_model.select_var_lag_order` (an AIC grid
    search over `MAX_LAGS`) unless `n_lags` is explicitly pinned. Lag
    selection, the Granger test, and the VAR fit are only re-run at most
    once every `_refit_cache.refit_interval_days` simulated days per
    asset — see `RefitCache`'s docstring; the forecast/impulse-response
    step still scores against every day's fresh data."""

    agent_id: str = "macro_var_spillover_analyst"

    macro_series_ids: tuple[str, ...] = Field(default=(VIXCLS,))
    use_yield_curve_slope: bool = True
    # None -> selected fresh each refit via select_var_lag_order (AIC grid
    # search over MAX_LAGS); pin to a fixed int to skip selection entirely.
    n_lags: int | None = None
    _refit_cache: RefitCache = PrivateAttr(default_factory=RefitCache)
    # Reflects the default configuration (macro_series_ids=(VIXCLS,),
    # use_yield_curve_slope=True) — an instance customizing either field
    # away from its default should also override this.
    data_requirements: tuple[DataRequirement, ...] = (
        DataRequirement(
            source="fred", identifier=VIXCLS, asset_scoped=False, description="FRED VIXCLS"
        ),
        DataRequirement(
            source="fred", identifier=DGS10, asset_scoped=False, description="FRED DGS10"
        ),
        DataRequirement(
            source="fred", identifier=DGS3MO, asset_scoped=False, description="FRED DGS3MO"
        ),
    )

    def _macro_series(self, as_of_date: pd.Timestamp) -> list[pd.Series] | None:
        series_list: list[pd.Series] = []
        for series_id in self.macro_series_ids:
            series = load_fred_series_from_lakehouse(series_id, as_of_date.date())
            if series is None:
                return None
            series_list.append(series)
        if self.use_yield_curve_slope:
            long_rate = load_fred_series_from_lakehouse(DGS10, as_of_date.date())
            short_rate = load_fred_series_from_lakehouse(DGS3MO, as_of_date.date())
            if long_rate is None or short_rate is None:
                return None
            yield_curve = pd.concat([long_rate, short_rate], axis=1, join="inner")
            series_list.append(yield_curve.iloc[:, 0] - yield_curve.iloc[:, 1])
        return series_list

    def generate_signal(
        self,
        asset: str,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> AgentSignal:
        asset_returns = market_slice(market_data, as_of_date, asset).log_returns[
            -settings.analysts.risk_window_medium :
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
                status="unavailable",
            )

        history = market_data.loc[market_data.index <= as_of_date]
        if isinstance(history.columns, pd.MultiIndex):
            if asset not in history.columns.get_level_values(0):
                asset_history = pd.DataFrame()
            else:
                asset_history = pd.DataFrame(history.xs(asset, axis=1, level=0))
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
                status="unavailable",
            )

        aligned = (
            pd.concat([asset_history["log_return"], *macro_series], axis=1, join="inner")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        system = aligned.to_numpy(dtype=np.float64)
        min_lags_assumed = self.n_lags if self.n_lags is not None else MAX_LAGS
        if system.shape[0] < 4 * min_lags_assumed + 10:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="insufficient overlapping history for a VAR spillover read",
                status="unavailable",
            )

        def _fit_var_system() -> tuple[int, float, np.ndarray | None, np.ndarray | None]:
            n_lags = (
                self.n_lags
                if self.n_lags is not None
                else select_var_lag_order(system, MAX_LAGS, LAG_SELECTION_CRITERION)
            )
            _, granger_p_value = granger_causality_test(system[:, 0], system[:, 1], n_lags)
            if granger_p_value >= settings.analysts.significance_level:
                return n_lags, granger_p_value, None, None
            fitted_coef_matrix, fitted_sigma_u = var_fit(system, n_lags)
            return n_lags, granger_p_value, fitted_coef_matrix, fitted_sigma_u

        try:
            n_lags, p_value, coef_matrix, sigma_u = self._refit_cache.get_or_refit(
                asset, as_of_date, _fit_var_system
            )
            if coef_matrix is None or sigma_u is None:
                return AgentSignal(
                    agent_id=self.agent_id,
                    asset=asset,
                    as_of_date=as_of_date.date(),
                    direction="flat",
                    confidence=0.0,
                    rationale=f"macro does not Granger-cause {asset} (p={p_value:.3f})",
                )

            forecast = var_forecast(system, coef_matrix, n_lags, h=1)
            forecast_move = float(forecast[0, 0])

            impulse = impulse_response(coef_matrix, sigma_u, n_lags, n_periods=1)
            impact_magnitude = (
                float(np.abs(impulse[0, 0, 1])) if impulse.ndim == 3 else float(np.abs(impulse[0]))
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
                status="error",
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
