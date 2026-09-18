# arima_forecast_analyst.py

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import PrivateAttr
from quantcore.time_series.arima import (
    arima_fit,
    arima_forecast,
    arima_residuals,
    ljung_box_test,
    select_arima_order,
)

from agentic_trading.agents._market_math import RefitCache, clamp, market_slice
from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.schemas import AgentSignal, SignalStatus
from agentic_trading.config import settings
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

LJUNG_BOX_MAX_LAGS = 10
LJUNG_BOX_SIGNIFICANCE = 0.05
# CSS estimation needs roughly a year of daily returns for stable
# low-order coefficient estimates.
ARIMA_WINDOW = 250
# Search grid for select_arima_order — small on purpose (a daily-return
# series rarely needs higher-order AR/MA terms, and a wider grid mostly
# just risks overfitting the trailing window's own noise). Every extra
# grid cell costs a full CSS fit.
MAX_P = 2
MAX_Q = 2
# Log price is I(1) by construction (its first difference is the return
# series), so the integration order is fixed rather than searched.
INTEGRATION_ORDER = 1
ORDER_SELECTION_CRITERION = "aic"


class ARIMAForecastAnalyst(AnalystAgent):
    """Fits a low-order ARIMA model on the last `ARIMA_WINDOW` trailing log
    prices and forecasts `forecast_horizon` days ahead. Two Ljung-Box
    gates bracket the fit: a pre-fit test on the return series (no serial
    dependence means nothing to forecast) and a post-fit test on the CSS
    residuals (`quantcore.time_series.arima.arima_residuals`; residual
    autocorrelation means the order is misspecified and the forecast is
    not trusted). The order is re-identified every
    `settings.analysts.arima_order_reselect_days` on the return series
    with d fixed at 1 (log price is I(1) by construction) and p, q in
    {0..MAX_P}, and coefficients re-estimated every
    `_refit_cache.refit_interval_days`; the `p`/`d`/`q` fields pin an
    order and skip selection."""

    agent_id: str = "arima_forecast_analyst"

    p: int | None = None
    d: int | None = None
    q: int | None = None
    forecast_horizon: int = 5
    _order_cache: RefitCache = PrivateAttr(
        default_factory=lambda: RefitCache(
            refit_interval_days=settings.analysts.arima_order_reselect_days
        )
    )
    _refit_cache: RefitCache = PrivateAttr(default_factory=RefitCache)

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

    def _select_order(self, returns: np.ndarray) -> tuple[int, int, int]:
        if self.p is not None and self.d is not None and self.q is not None:
            return self.p, self.d, self.q
        p, _, q = select_arima_order(returns, MAX_P, 0, MAX_Q, ORDER_SELECTION_CRITERION)
        return p, INTEGRATION_ORDER, q

    def generate_signal(
        self,
        asset: str,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> AgentSignal:
        slice_ = market_slice(market_data, as_of_date, asset)
        returns = slice_.log_returns[-ARIMA_WINDOW:]
        if slice_.close is None or returns.size < 30:
            return self._flat(
                asset, as_of_date, "insufficient history for an ARIMA fit", "unavailable"
            )

        pretest_returns = returns[-settings.analysts.arima_pretest_window :]
        n_lags = min(LJUNG_BOX_MAX_LAGS, pretest_returns.size // 5)
        _, p_value = ljung_box_test(pretest_returns, n_lags)
        if p_value >= LJUNG_BOX_SIGNIFICANCE:
            return self._flat(
                asset,
                as_of_date,
                f"no serial dependence to forecast (Ljung-Box p={p_value:.3f})",
                "unavailable",
            )

        log_prices = np.cumsum(returns)

        try:
            p, d, q = self._order_cache.get_or_refit(
                asset, as_of_date, lambda: self._select_order(returns)
            )
            ar_coefs, ma_coefs, _sigma2 = self._refit_cache.get_or_refit(
                asset, as_of_date, lambda: arima_fit(log_prices, p, d, q)
            )
        except (RuntimeError, ValueError) as exc:
            logger.debug(
                "%s: ARIMA order selection/fit failed for %s (%s) — returning flat.",
                self.agent_id,
                asset,
                exc,
            )
            return self._flat(asset, as_of_date, "ARIMA order selection/fit failed", "error")

        residuals = arima_residuals(log_prices, ar_coefs, ma_coefs, d)
        residual_lags = min(LJUNG_BOX_MAX_LAGS, max(residuals.size // 5, 1))
        _, residual_p_value = ljung_box_test(residuals, residual_lags)
        if residual_p_value < LJUNG_BOX_SIGNIFICANCE:
            return self._flat(
                asset,
                as_of_date,
                f"ARIMA({p},{d},{q}) residuals still autocorrelated "
                f"(Ljung-Box p={residual_p_value:.3f}) — misspecified fit",
                "unavailable",
            )

        forecast_path = arima_forecast(log_prices, ar_coefs, ma_coefs, d, self.forecast_horizon)
        forecast_move = float(forecast_path[-1] - log_prices[-1])
        daily_vol = float(np.std(returns, ddof=1))
        direction: Literal["long", "short", "flat"] = "long" if forecast_move >= 0.0 else "short"
        confidence = clamp(
            abs(forecast_move) / max(daily_vol * np.sqrt(self.forecast_horizon), 1e-8)
        )

        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction=direction,
            confidence=confidence,
            rationale=(
                f"ARIMA({p},{d},{q}) forecast move={forecast_move:.4f} "
                f"over {self.forecast_horizon}d"
            ),
        )
