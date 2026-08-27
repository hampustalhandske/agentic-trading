from agentic_trading.agents.analysts import (
    ARIMAForecastAnalyst,
    BlackScholesValuationAnalyst,
    CIRRateRegimeAnalyst,
    EGARCHAsymmetricVolAnalyst,
    FamaFrenchFactorAnalyst,
    FamaMacbethAnalyst,
    GBMMomentumAnalyst,
    GJRGARCHCrashRiskAnalyst,
    GreeksDeltaHedgeAnalyst,
    HestonVolSurfaceAnalyst,
    KalmanHedgeRatioAnalyst,
    MacroVarSpilloverAnalyst,
    MonteCarloValuationAnalyst,
    NeweyWestSignificanceAnalyst,
    NewsSentimentAnalyst,
    OUMeanReversionAnalyst,
    PairsCointegrationAnalyst,
    RegimeHMMAnalyst,
    RiskMaximizingAnalyst,
    RiskMinimizingAnalyst,
    SecFundamentalsAnalyst,
    ShortInterestSqueezeAnalyst,
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
