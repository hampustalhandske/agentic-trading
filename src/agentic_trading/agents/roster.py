# roster.py

"""Default analyst/manager roster shared by the CLI and GUI entry points.

Splits 22 analysts across 6 pod managers, each pod themed around a
coherent quant strategy family, so the Commander sees six genuinely
different risk overlays rather than duplicate coverage of the same
signals under six names:

- **Risk pod** (RiskManager): directional/volatility analysts
  (risk_minimizing, risk_maximizing, gbm_momentum).
- **CVaR pod** (CVaRManager): options-valuation analysts
  (black_scholes_valuation, monte_carlo_valuation, greeks_delta_hedge,
  heston_vol_surface).
- **StatArb pod** (StatArbManager): mean-reversion/cointegration analysts
  (ou_mean_reversion, pairs_cointegration, kalman_hedge_ratio, regime_hmm).
- **Macro/Rates pod** (MacroRatesManager): vol-regime and macro-driven
  analysts (egarch_asymmetric_vol, gjr_garch_crash_risk, arima_forecast,
  macro_var_spillover, cir_rate_regime).
- **Factor pod** (FactorManager): cross-sectional/regression analysts
  (fama_french_factor, fama_macbeth, newey_west_significance).
- **Event/Credit pod** (EventCreditManager): event-driven analysts
  (news_sentiment, sec_fundamentals, short_interest_squeeze).
"""

from __future__ import annotations

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
from agentic_trading.agents.base import AnalystAgent, ManagerAgent
from agentic_trading.agents.managers import (
    CVaRManager,
    EventCreditManager,
    FactorManager,
    MacroRatesManager,
    RiskManager,
    StatArbManager,
)
from agentic_trading.scoring.weighting import Hedge

RISK_MANAGER_ANALYST_IDS: frozenset[str] = frozenset(
    {
        "risk_minimizing_analyst",
        "risk_maximizing_analyst",
        "gbm_momentum_analyst",
    }
)
CVAR_MANAGER_ANALYST_IDS: frozenset[str] = frozenset(
    {
        "monte_carlo_valuation_analyst",
        "black_scholes_valuation_analyst",
        "greeks_delta_hedge_analyst",
        "heston_vol_surface_analyst",
    }
)
STAT_ARB_MANAGER_ANALYST_IDS: frozenset[str] = frozenset(
    {
        "ou_mean_reversion_analyst",
        "pairs_cointegration_analyst",
        "kalman_hedge_ratio_analyst",
        "regime_hmm_analyst",
    }
)
MACRO_RATES_MANAGER_ANALYST_IDS: frozenset[str] = frozenset(
    {
        "egarch_asymmetric_vol_analyst",
        "gjr_garch_crash_risk_analyst",
        "arima_forecast_analyst",
        "macro_var_spillover_analyst",
        "cir_rate_regime_analyst",
    }
)
FACTOR_MANAGER_ANALYST_IDS: frozenset[str] = frozenset(
    {
        "fama_french_factor_analyst",
        "fama_macbeth_analyst",
        "newey_west_significance_analyst",
    }
)
EVENT_CREDIT_MANAGER_ANALYST_IDS: frozenset[str] = frozenset(
    {
        "news_sentiment_analyst",
        "sec_fundamentals_analyst",
        "short_interest_squeeze_analyst",
    }
)


def build_default_analysts() -> list[AnalystAgent]:
    """The full 22-analyst Layer-1 roster, in a stable, deterministic
    order: the original five first, then the 17 new ones grouped by pod in
    the same order the pods are described above."""
    return [
        # Original five
        RiskMinimizingAnalyst(),
        RiskMaximizingAnalyst(),
        NewsSentimentAnalyst(),
        MonteCarloValuationAnalyst(),
        BlackScholesValuationAnalyst(),
        # Risk pod
        GBMMomentumAnalyst(),
        # CVaR pod
        GreeksDeltaHedgeAnalyst(),
        HestonVolSurfaceAnalyst(),
        # StatArb pod
        OUMeanReversionAnalyst(),
        PairsCointegrationAnalyst(),
        KalmanHedgeRatioAnalyst(),
        RegimeHMMAnalyst(),
        # Macro/Rates pod
        EGARCHAsymmetricVolAnalyst(),
        GJRGARCHCrashRiskAnalyst(),
        ARIMAForecastAnalyst(),
        MacroVarSpilloverAnalyst(),
        CIRRateRegimeAnalyst(),
        # Factor pod
        FamaFrenchFactorAnalyst(),
        FamaMacbethAnalyst(),
        NeweyWestSignificanceAnalyst(),
        # Event/Credit pod
        SecFundamentalsAnalyst(),
        ShortInterestSqueezeAnalyst(),
    ]


def build_default_managers(analysts: list[AnalystAgent] | None = None) -> list[ManagerAgent]:
    """All 6 pod managers, each `managed_analysts` filtered from `analysts`
    (defaults to build_default_analysts()) via its own `_ANALYST_IDS`
    frozenset."""
    analysts = analysts if analysts is not None else build_default_analysts()

    def owned(analyst_ids: frozenset[str]) -> list[AnalystAgent]:
        return [analyst for analyst in analysts if analyst.agent_id in analyst_ids]

    return [
        RiskManager(managed_analysts=owned(RISK_MANAGER_ANALYST_IDS), weighting_strategy=Hedge()),
        CVaRManager(managed_analysts=owned(CVAR_MANAGER_ANALYST_IDS), weighting_strategy=Hedge()),
        StatArbManager(
            managed_analysts=owned(STAT_ARB_MANAGER_ANALYST_IDS), weighting_strategy=Hedge()
        ),
        MacroRatesManager(
            managed_analysts=owned(MACRO_RATES_MANAGER_ANALYST_IDS), weighting_strategy=Hedge()
        ),
        FactorManager(
            managed_analysts=owned(FACTOR_MANAGER_ANALYST_IDS), weighting_strategy=Hedge()
        ),
        EventCreditManager(
            managed_analysts=owned(EVENT_CREDIT_MANAGER_ANALYST_IDS), weighting_strategy=Hedge()
        ),
    ]
