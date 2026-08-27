# arima_forecast_analyst.py

from typing import Literal

import numpy as np
import pandas as pd
from quantcore.time_series.arima import arima_fit, arima_forecast, ljung_box_test

from agentic_trading.agents._market_math import clamp, market_slice
from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

LJUNG_BOX_LAGS = 10
LJUNG_BOX_SIGNIFICANCE = 0.05


class ARIMAForecastAnalyst(AnalystAgent):
    """Fits a low-order ARIMA model on trailing log prices, gated by a
    Ljung-Box residual-autocorrelation test — a misspecified fit (residuals
    still show significant autocorrelation) is distrusted rather than
    forecast from."""

    agent_id: str = "arima_forecast_analyst"

    p: int = 1
    d: int = 1
    q: int = 1
    forecast_horizon: int = 5
    _risk_window: int = 120

    def generate_signal(
        self,
        asset: str,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> AgentSignal:
        slice_ = market_slice(market_data, as_of_date, asset)
        returns = slice_.log_returns[-self._risk_window :]
        if slice_.close is None or returns.size < 30:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="insufficient history for an ARIMA fit",
            )

        log_prices = np.cumsum(returns)
        try:
            ar_coefs, ma_coefs, _sigma2 = arima_fit(log_prices, self.p, self.d, self.q)
        except (RuntimeError, ValueError) as exc:
            logger.debug(
                "%s: ARIMA fit failed for %s (%s) — returning flat.", self.agent_id, asset, exc
            )
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="ARIMA fit failed",
            )

        forecast_path = arima_forecast(
            log_prices, ar_coefs, ma_coefs, self.d, self.forecast_horizon
        )
        residuals = np.diff(log_prices)[-min(len(np.diff(log_prices)), 60) :]
        _, p_value = ljung_box_test(residuals, min(LJUNG_BOX_LAGS, residuals.size - 1))
        if p_value < LJUNG_BOX_SIGNIFICANCE:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale=(
                    f"ARIMA residuals still autocorrelated (Ljung-Box p={p_value:.3f}) "
                    "— distrusting fit"
                ),
            )

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
                f"ARIMA({self.p},{self.d},{self.q}) forecast move={forecast_move:.4f} "
                f"over {self.forecast_horizon}d"
            ),
        )
