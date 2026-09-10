import numpy as np
import pandas as pd

from agentic_trading.agents.analysts import (
    ARIMAForecastAnalyst,
    BlackScholesValuationAnalyst,
    CalendarSeasonalityAnalyst,
    CIRRateRegimeAnalyst,
    EGARCHAsymmetricVolAnalyst,
    FamaFrenchFactorAnalyst,
    FamaMacbethAnalyst,
    GBMMomentumAnalyst,
    GJRGARCHCrashRiskAnalyst,
    GreeksDeltaHedgeAnalyst,
    HestonVolSurfaceAnalyst,
    InsiderClusterBuyingAnalyst,
    KalmanHedgeRatioAnalyst,
    LeadLagNetworkCentralityAnalyst,
    MacroVarSpilloverAnalyst,
    MonteCarloValuationAnalyst,
    NeweyWestSignificanceAnalyst,
    NewsSentimentAnalyst,
    OptionsSkewSentimentAnalyst,
    OUMeanReversionAnalyst,
    PairsCointegrationAnalyst,
    RedditSentimentAnalyst,
    RegimeHMMAnalyst,
    RetailAttentionAnalyst,
    RiskMaximizingAnalyst,
    RiskMinimizingAnalyst,
    SecFundamentalsAnalyst,
    ShortInterestSqueezeAnalyst,
    WikipediaPageviewsAnalyst,
)
from agentic_trading.agents.base import AnalystAgent

ALL_ANALYST_CLASSES = (
    NewsSentimentAnalyst,
    RiskMaximizingAnalyst,
    RiskMinimizingAnalyst,
    BlackScholesValuationAnalyst,
    MonteCarloValuationAnalyst,
    GBMMomentumAnalyst,
    GreeksDeltaHedgeAnalyst,
    HestonVolSurfaceAnalyst,
    OUMeanReversionAnalyst,
    PairsCointegrationAnalyst,
    KalmanHedgeRatioAnalyst,
    RegimeHMMAnalyst,
    EGARCHAsymmetricVolAnalyst,
    GJRGARCHCrashRiskAnalyst,
    ARIMAForecastAnalyst,
    MacroVarSpilloverAnalyst,
    CIRRateRegimeAnalyst,
    FamaFrenchFactorAnalyst,
    FamaMacbethAnalyst,
    NeweyWestSignificanceAnalyst,
    SecFundamentalsAnalyst,
    ShortInterestSqueezeAnalyst,
    InsiderClusterBuyingAnalyst,
    RetailAttentionAnalyst,
    WikipediaPageviewsAnalyst,
    RedditSentimentAnalyst,
    CalendarSeasonalityAnalyst,
    LeadLagNetworkCentralityAnalyst,
    OptionsSkewSentimentAnalyst,
)


def test_analysts_imports() -> None:
    for analyst_cls in ALL_ANALYST_CLASSES:
        assert analyst_cls.__name__


def test_analysts_can_be_constructed_with_defaults() -> None:
    for analyst_cls in ALL_ANALYST_CLASSES:
        instance = analyst_cls()
        assert isinstance(instance.agent_id, str) and instance.agent_id


def test_analysts_behave_as_analysts() -> None:
    for analyst_cls in ALL_ANALYST_CLASSES:
        assert issubclass(analyst_cls, AnalystAgent)


def test_macro_var_spillover_drops_non_finite_rows(monkeypatch) -> None:
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    market_data = pd.DataFrame(
        {"close": np.arange(100.0, 130.0), "log_return": np.linspace(-0.01, 0.01, 30)},
        index=dates,
    )
    macro = pd.Series(np.linspace(10.0, 20.0, 30), index=dates)
    macro.iloc[5] = np.nan

    monkeypatch.setattr(
        "agentic_trading.agents.analysts.macro_rates_pod.macro_var_spillover_analyst.load_fred_series",
        lambda _series_id, _as_of_date: macro,
    )
    signal = MacroVarSpilloverAnalyst(use_yield_curve_slope=False).generate_signal(
        "AAPL", market_data, dates[-1]
    )

    assert signal.direction == "flat"
    assert signal.confidence == 0.0


def test_macro_var_spillover_handles_solver_failure(monkeypatch) -> None:
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    market_data = pd.DataFrame(
        {"close": np.arange(100.0, 130.0), "log_return": np.linspace(-0.01, 0.01, 30)},
        index=dates,
    )
    macro = pd.Series(np.linspace(10.0, 20.0, 30), index=dates)

    monkeypatch.setattr(
        "agentic_trading.agents.analysts.macro_rates_pod.macro_var_spillover_analyst.load_fred_series",
        lambda _series_id, _as_of_date: macro,
    )
    monkeypatch.setattr(
        "agentic_trading.agents.analysts.macro_rates_pod.macro_var_spillover_analyst.granger_causality_test",
        lambda *_args: (0.0, 0.01),
    )
    monkeypatch.setattr(
        "agentic_trading.agents.analysts.macro_rates_pod.macro_var_spillover_analyst.var_fit",
        lambda *_args: (_ for _ in ()).throw(np.linalg.LinAlgError("SVD did not converge")),
    )
    signal = MacroVarSpilloverAnalyst(use_yield_curve_slope=False).generate_signal(
        "AAPL", market_data, dates[-1]
    )

    assert signal.direction == "flat"
    assert signal.confidence == 0.0
