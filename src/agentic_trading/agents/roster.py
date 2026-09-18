# roster.py

"""Default analyst/manager roster shared by the CLI and GUI entry points.

Splits analysts across 11 pod managers, laid out the way a real
multi-strategy desk would organize them — by *what a pod trades* and
*how it behaves* — so the Commander sees genuinely different risk
overlays, not duplicate coverage under different names, and no pod is
just 1-2 analysts trying to justify its own existence. Every pod's
`weighting_strategy`/sizing-method choice is deliberate (see the comment
above each manager in `build_default_managers()`); asset-class-scoped
pods (`ManagerAgent.assets`, see its own docstring) only ever price the
instruments they were actually built for, even inside a mixed
multi-asset universe:

**Equity desks — unscoped (except the options-scoped CVaR pod), share
the Commander's full traded universe:**
- **Risk pod** (RiskManager): directional/volatility-reducing +
  distress-signal analysts (risk_minimizing, gbm_momentum,
  dividend_payout — a dividend cut/raise is management literally
  spending or preserving real cash, a genuine risk/quality signal, not a
  price-derived statistic).
- **CVaR pod** (CVaRManager): options-valuation and vol-harvesting
  analysts (black_scholes_valuation, monte_carlo_valuation,
  greeks_delta_hedge, heston_vol_surface, risk_maximizing). Scoped
  (`ManagerAgent.assets`, same mechanism as the macro desks below) to
  `cvar_manager.OPTIONS_ASSETS` — the eleven underlyings with a real
  historical option chain in the lakehouse — so its analysts read
  **real market options data** (`_options_data.real_atm_options_by_expiration`)
  and never a model-implied substitute quote; a covered ticker with no
  snapshot on a given day just reports flat. risk_maximizing_analyst
  lives here, not in Risk, because it's a deliberate long-vol/
  vol-risk-premium bet rather than risk reduction — this pod already
  houses every other volatility-facing analyst.
- **StatArb pod** (StatArbManager): mean-reversion/cointegration analysts
  (ou_mean_reversion, pairs_cointegration, kalman_hedge_ratio, regime_hmm).
  Its walk-forward pair re-screening candidate set
  (`_pairs_screening.resolve_candidate_pairs`) combines real SEC SIC-code
  matches within the pod's own traded universe with whichever of the
  original hand-picked pairs that same universe also supports.
- **Factor pod** (FactorManager): cross-sectional/regression analysts
  (fama_french_factor, fama_macbeth, newey_west_significance,
  earnings_surprise) — SMB (size) and HML (value) loadings already come
  from fama_french_factor's own 3-factor model, so there is no separate
  dedicated size analyst; earnings_surprise reads real EPS-vs-consensus
  surprises (`dolthub_eps_history`/`dolthub_earnings_calendar`) for
  post-earnings-announcement drift, the academic-anomaly factor this pod
  was otherwise missing.
- **Event/Credit pod** (EventCreditManager): event-driven equity
  analysts (sec_fundamentals, short_interest_squeeze) — the "Credit"
  half of its original name is the separate Credit pod below; this
  stays equity-only.
- **Alt-Data pod** (AltDataManager): behavioral/alternative-data
  analysts that don't fit a stronger theme (calendar_seasonality,
  lead_lag_network_centrality, insider_cluster_buying on SEC Form 4
  open-market purchases), run as one desk the way real funds
  tend to group their small behavioral-signal books.

**Macro desks — each scoped (`ManagerAgent.assets`) to a real asset
class, none Black-Litterman-sized unless a genuine market-cap-style
equilibrium exists for that basket:**
- **Macro/Rates pod** (MacroRatesManager): vol-regime and macro-driven
  analysts on equities (egarch_asymmetric_vol, gjr_garch_crash_risk,
  arima_forecast, macro_var_spillover).
- **Rates pod** (RatesManager): scoped to a Treasury ETF duration
  ladder (SHY/IEF/TLT). cir_rate_regime_analyst (a short-rate regime
  read that needs a genuinely rate-sensitive instrument),
  cot_positioning_analyst_rates (real CFTC positioning on the matching
  Treasury-futures contract per ETF — ZT=F/ZN=F/ZB=F), and a fresh
  arima_forecast instance. Default Black-Litterman + mean-variance
  sizing — a bond ladder has a real, duration-weighted equilibrium.
- **Cross-Asset pod** (CrossAssetManager): scoped to gold/crude/
  silver/nat-gas/copper futures (GC=F/CL=F/SI=F/NG=F/HG=F), four major
  currency pairs (EURUSD=X/GBPUSD=X/USDJPY=X/AUDUSD=X), and BTC-USD/
  ETH-USD — three asset classes merged into one pod because they share
  the same sizing answer (risk_parity, no market-cap equilibrium exists
  for any of them) and each was too thin (2-3 analysts) to justify a
  separate pod on its own. One fresh instance each of GJR-GARCH crash
  risk, EGARCH asymmetric vol, HMM regime detection, and ARIMA forecast
  runs across the whole scope, plus two COT-positioning instances
  (real CFTC positioning; separate commodities/FX ticker conventions,
  no crypto category — those two just read "no coverage" for BTC-USD/
  ETH-USD).
- **Credit pod** (CreditManager): scoped to HYG/LQD (high-yield /
  investment-grade corporate bond ETFs) — the actual credit data
  "Event/Credit" never had. credit_spread_signal_analyst reads FRED's
  ICE BofA HY option-adjusted spread as a market-wide risk-appetite
  overlay (no free per-issuer ratings/CDS data exists), plus fresh
  egarch/regime_hmm instances on the two ETFs themselves. Risk-parity
  sized.
- **Tail-Hedge pod** (TailHedgeManager): scoped to SPY, a single
  systematic insurance sleeve — vix_tail_hedge_analyst reads `^VIX`'s
  own price history and produces a short-SPY signal scaled by how
  elevated VIX is. Funded by a fixed, protected Commander-level budget
  carve-out (`commander.TAIL_HEDGE_BUDGET_FRACTION`) rather than the
  normal trailing-Sharpe risk-budgeting every other pod gets, since a
  hedge that gets defunded by its own performance during a calm stretch
  defeats its own purpose. A linear proxy, not real convex options
  protection — see KNOWN_LIMITATIONS.md.

Every pod's own `leverage_cap`/`net_leverage_cap`/`target_annualized_vol`/
`kelly_fraction_weight` is set per-strategy (research-grounded real-desk
figures, not a shared default) in `config.settings.pod_sizing`, keyed by
manager agent_id and overridable per run — the comment above each
`PortfolioConstructor(...)` call in `build_default_managers()` below
carries the reasoning for that pod's row.
"""

from __future__ import annotations

from agentic_trading.agents.analysts import (
    ARIMAForecastAnalyst,
    BlackScholesValuationAnalyst,
    CalendarSeasonalityAnalyst,
    CIRRateRegimeAnalyst,
    CreditSpreadSignalAnalyst,
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
    OUMeanReversionAnalyst,
    PairsCointegrationAnalyst,
    RegimeHMMAnalyst,
    RiskMaximizingAnalyst,
    RiskMinimizingAnalyst,
    SecFundamentalsAnalyst,
    ShortInterestSqueezeAnalyst,
)
from agentic_trading.agents.analysts.factor_pod.earnings_surprise_analyst import (
    EarningsSurpriseAnalyst,
)
from agentic_trading.agents.analysts.macro_rates_pod.cot_positioning_analyst import (
    COTPositioningAnalyst,
)
from agentic_trading.agents.analysts.risk_pod.dividend_payout_analyst import DividendPayoutAnalyst
from agentic_trading.agents.analysts.tail_hedge_pod.vix_tail_hedge_analyst import (
    VixTailHedgeAnalyst,
)
from agentic_trading.agents.base import AnalystAgent, ManagerAgent
from agentic_trading.agents.managers import (
    AltDataManager,
    CreditManager,
    CrossAssetManager,
    CVaRManager,
    EventCreditManager,
    FactorManager,
    MacroRatesManager,
    RatesManager,
    RiskManager,
    StatArbManager,
    TailHedgeManager,
)
from agentic_trading.agents.managers.credit_manager import CREDIT_ASSETS
from agentic_trading.agents.managers.cross_asset_manager import (
    COMMODITIES_ASSETS,
    CROSS_ASSET_ASSETS,
    FX_ASSETS,
)
from agentic_trading.agents.managers.cvar_manager import OPTIONS_ASSETS
from agentic_trading.agents.managers.rates_manager import RATES_ASSETS
from agentic_trading.agents.managers.tail_hedge_manager import TAIL_HEDGE_ASSETS
from agentic_trading.agents.portfolio_construction import PortfolioConstructor
from agentic_trading.config import settings
from agentic_trading.scoring.weighting import (
    DrawdownAwareWeight,
    EqualWeight,
    Hedge,
    SharpeWeight,
    ShrinkageWeight,
)

RISK_MANAGER_ANALYST_IDS: frozenset[str] = frozenset(
    {
        "risk_minimizing_analyst",
        "gbm_momentum_analyst",
        "dividend_payout_analyst",
    }
)
CVAR_MANAGER_ANALYST_IDS: frozenset[str] = frozenset(
    {
        "monte_carlo_valuation_analyst",
        "black_scholes_valuation_analyst",
        "greeks_delta_hedge_analyst",
        "heston_vol_surface_analyst",
        "risk_maximizing_analyst",
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
    }
)
# cir_rate_regime_analyst moved here from Macro/Rates — see roster.py's
# module docstring for why (it needs a genuinely rate-sensitive
# instrument, not an arbitrary equity). cot_positioning_analyst_rates
# reads real CFTC positioning on the Treasury-futures proxy for each ETF
# (SHY/IEF/TLT -> ZT=F/ZN=F/ZB=F, see COTPositioningAnalyst's own
# docstring); arima_forecast_analyst_rates is a fresh generic instance.
RATES_MANAGER_ANALYST_IDS: frozenset[str] = frozenset(
    {
        "cir_rate_regime_analyst",
        "cot_positioning_analyst_rates",
        "arima_forecast_analyst_rates",
    }
)
# Commodities + FX + crypto share one pod (see roster.py's module
# docstring) — each generic vol/regime model runs once across the whole
# cross-asset scope instead of a separate duplicate instance per asset
# class; only the two COT analysts stay asset-class-specific, since COT
# positioning has no crypto category and a different ticker convention
# for commodities vs. FX/rate futures.
CROSS_ASSET_MANAGER_ANALYST_IDS: frozenset[str] = frozenset(
    {
        "gjr_garch_crash_risk_analyst_cross_asset",
        "egarch_asymmetric_vol_analyst_cross_asset",
        "regime_hmm_analyst_cross_asset",
        "arima_forecast_analyst_cross_asset",
        "cot_positioning_analyst_commodities",
        "cot_positioning_analyst_fx",
    }
)
CREDIT_MANAGER_ANALYST_IDS: frozenset[str] = frozenset(
    {
        "credit_spread_signal_analyst",
        "egarch_asymmetric_vol_analyst_credit",
        "regime_hmm_analyst_credit",
    }
)
FACTOR_MANAGER_ANALYST_IDS: frozenset[str] = frozenset(
    {
        "fama_french_factor_analyst",
        "fama_macbeth_analyst",
        "newey_west_significance_analyst",
        "earnings_surprise_analyst",
    }
)
EVENT_CREDIT_MANAGER_ANALYST_IDS: frozenset[str] = frozenset(
    {
        "sec_fundamentals_analyst",
        "short_interest_squeeze_analyst",
    }
)
# One "behavioral/alternative signal" desk — small buckets real funds
# tend to run as one desk rather than two.
ALT_DATA_MANAGER_ANALYST_IDS: frozenset[str] = frozenset(
    {
        "calendar_seasonality_analyst",
        "lead_lag_network_centrality_analyst",
        "insider_cluster_buying_analyst",
    }
)
TAIL_HEDGE_MANAGER_ANALYST_IDS: frozenset[str] = frozenset(
    {
        "vix_tail_hedge_analyst",
    }
)


# Pairs for PairsCointegrationAnalyst/KalmanHedgeRatioAnalyst, produced by
# scripts/screen_cointegrated_pairs.py: same-sector candidates from the
# 20-ticker research basket, tested for Engle-Granger cointegration over a
# formation window that ends before that basket's own test period starts,
# so pair selection never sees the data it will later be evaluated on.
# Both tickers in a pair are wired symmetrically, since either side is a
# legitimate view on the same relationship. This is only the starting
# wiring: `StatArbManager._maybe_rescreen_pairs` re-screens the candidate
# set quarterly inside the backtest and replaces it (partners and hedge
# ratios) with whatever passes its admit/keep/drop hysteresis.
_COINTEGRATED_PAIRS: tuple[tuple[str, str], ...] = (
    ("MSFT", "ADBE"),
    ("MA", "JPM"),
    ("VZ", "NFLX"),
    ("BA", "CAT"),
)
_PARTNER_TICKERS_BY_ASSET: dict[str, tuple[str, ...]] = {}
_PARTNER_TICKER_BY_ASSET: dict[str, str] = {}
for _ticker_a, _ticker_b in _COINTEGRATED_PAIRS:
    _PARTNER_TICKERS_BY_ASSET[_ticker_a] = (_ticker_b,)
    _PARTNER_TICKERS_BY_ASSET[_ticker_b] = (_ticker_a,)
    _PARTNER_TICKER_BY_ASSET[_ticker_a] = _ticker_b
    _PARTNER_TICKER_BY_ASSET[_ticker_b] = _ticker_a


def build_default_analysts(
    excluded_analyst_ids: frozenset[str] = frozenset(),
) -> list[AnalystAgent]:
    """The full 38-analyst Layer-1 roster, in a stable, deterministic
    order: the original four first, then the rest grouped by pod in the
    same order the pods are described above.

    `excluded_analyst_ids` drops any analyst by `agent_id` without
    touching its code — the tool for a run that wants to try the roster
    with a given analyst switched off without deleting anything."""
    analysts = [
        # Original four
        RiskMinimizingAnalyst(),
        RiskMaximizingAnalyst(),
        MonteCarloValuationAnalyst(),
        BlackScholesValuationAnalyst(),
        # Risk pod
        GBMMomentumAnalyst(),
        DividendPayoutAnalyst(),
        # CVaR pod
        GreeksDeltaHedgeAnalyst(),
        HestonVolSurfaceAnalyst(),
        # StatArb pod
        OUMeanReversionAnalyst(),
        PairsCointegrationAnalyst(partner_tickers_by_asset=dict(_PARTNER_TICKERS_BY_ASSET)),
        KalmanHedgeRatioAnalyst(partner_ticker_by_asset=dict(_PARTNER_TICKER_BY_ASSET)),
        RegimeHMMAnalyst(),
        # Macro/Rates pod
        EGARCHAsymmetricVolAnalyst(),
        GJRGARCHCrashRiskAnalyst(),
        ARIMAForecastAnalyst(),
        MacroVarSpilloverAnalyst(),
        # Rates pod — cir_rate_regime_analyst moved here (see module
        # docstring): same class, same agent_id, just a different pod
        # wiring so it lands on Treasury ETFs instead of an arbitrary
        # equity.
        CIRRateRegimeAnalyst(),
        COTPositioningAnalyst(agent_id="cot_positioning_analyst_rates"),
        ARIMAForecastAnalyst(agent_id="arima_forecast_analyst_rates"),
        # Cross-Asset pod (commodities + FX + crypto) — one fresh instance
        # of each already-generic, asset-agnostic model, given a distinct
        # agent_id so its track record doesn't collide with the
        # Macro/Rates- or StatArb-pod instances of the same classes; each
        # runs across the pod's whole scope rather than duplicating one
        # instance per asset class. cot_positioning_analyst_commodities/
        # _fx's default cot_ticker_by_asset already matches this pod's
        # tickers directly (GC=F, EURUSD=X, etc. — no remapping needed);
        # crypto has no COT category at all, so those two just read as
        # "no coverage" for BTC-USD/ETH-USD, the same graceful fallback
        # any out-of-scope asset already gets.
        GJRGARCHCrashRiskAnalyst(agent_id="gjr_garch_crash_risk_analyst_cross_asset"),
        EGARCHAsymmetricVolAnalyst(agent_id="egarch_asymmetric_vol_analyst_cross_asset"),
        RegimeHMMAnalyst(agent_id="regime_hmm_analyst_cross_asset"),
        ARIMAForecastAnalyst(agent_id="arima_forecast_analyst_cross_asset"),
        COTPositioningAnalyst(
            agent_id="cot_positioning_analyst_commodities", covered_assets=COMMODITIES_ASSETS
        ),
        COTPositioningAnalyst(agent_id="cot_positioning_analyst_fx", covered_assets=FX_ASSETS),
        # Credit pod
        CreditSpreadSignalAnalyst(),
        EGARCHAsymmetricVolAnalyst(agent_id="egarch_asymmetric_vol_analyst_credit"),
        RegimeHMMAnalyst(agent_id="regime_hmm_analyst_credit"),
        # Factor pod
        FamaFrenchFactorAnalyst(),
        FamaMacbethAnalyst(),
        NeweyWestSignificanceAnalyst(),
        EarningsSurpriseAnalyst(),
        # Event/Credit pod
        SecFundamentalsAnalyst(),
        ShortInterestSqueezeAnalyst(),
        # Alt-Data pod
        CalendarSeasonalityAnalyst(),
        InsiderClusterBuyingAnalyst(),
        LeadLagNetworkCentralityAnalyst(),
        # Tail-Hedge pod
        VixTailHedgeAnalyst(),
    ]
    return [analyst for analyst in analysts if analyst.agent_id not in excluded_analyst_ids]


def build_default_managers(
    analysts: list[AnalystAgent] | None = None,
    excluded_manager_ids: frozenset[str] = frozenset(),
) -> list[ManagerAgent]:
    """All 11 pod managers, each `managed_analysts` filtered from `analysts`
    (defaults to build_default_analysts()) via its own `_ANALYST_IDS`
    frozenset. Each manager's `weighting_strategy` is wrapped in
    `ShrinkageWeight` (see `scoring/weighting.py`) so a thin-data analyst
    can't swing that manager's trust in it on a handful of noisy
    observations, and each owns its own `PortfolioConstructor` (risk
    budgeting + volatility targeting), with `leverage_cap`/
    `net_leverage_cap`/`target_annualized_vol`/`kelly_fraction_weight`
    set per-pod to research-grounded real-desk figures for that pod's
    own strategy (see each manager's own comment below and
    `PortfolioConstructor`'s own docstring for what each parameter
    does) — every pod sizes its own book independently; the Commander
    only allocates capital across pods (see `commander.py`).

    `excluded_manager_ids` drops a whole pod by `agent_id`, the manager
    counterpart to `build_default_analysts()`'s `excluded_analyst_ids` —
    a manager's owned analysts simply stop running for this build, since
    the Commander only ever iterates the managers it's given.

    Each pod's `weighting_strategy` choice is deliberate, not
    interchangeable — see the comment above each manager below for the
    domain reason. `scale_factor` (each manager's own method) already
    answers "does today's specific signal look trustworthy right now";
    `weighting_strategy` answers the different question "has this analyst
    been reliably right lately," and the two pods' own scoring histories
    (accumulated at very different rates/shapes depending on how often
    that pod's analysts actually fire) are what justify picking a
    different trailing-track-record shape per pod."""
    analysts = analysts if analysts is not None else build_default_analysts()

    def owned(analyst_ids: frozenset[str]) -> list[AnalystAgent]:
        return [analyst for analyst in analysts if analyst.agent_id in analyst_ids]

    managers = [
        # Risk-parametric analysts (risk_minimizing/risk_maximizing) are
        # directional bets sized off a real-time VaR/CVaR/vol read — the
        # natural way to judge whether such a call was actually
        # risk-efficient, not just a lucky raw return, is trailing Sharpe.
        # Directional long/short equity book: leverage_cap=3.0 matches the
        # ~2.9x gross norm real long/short equity funds run; net_leverage_cap
        # stays close to gross since directional exposure is the point;
        # target_annualized_vol=0.12 sits in the institutional 12% band.
        RiskManager(
            managed_analysts=owned(RISK_MANAGER_ANALYST_IDS),
            weighting_strategy=ShrinkageWeight(SharpeWeight()),
            portfolio_constructor=PortfolioConstructor(
                **settings.pod_sizing["risk_manager"].model_dump(),
            ),
        ),
        # This pod's mandate is volatility itself — most of its analysts
        # are tail-risk control, and risk_maximizing_analyst is the
        # deliberate opposite (vol-harvesting) bet, but every one of them
        # sinks or swims on how well it read a vol regime. An analyst
        # currently mid-drawdown is directly failing at that read, so
        # DrawdownAwareWeight's explicit drawdown penalty (beyond what an
        # average-based Sharpe captures) is the fit, not an arbitrary
        # alternative to it.
        # Options book: leverage_cap/net_leverage_cap kept below the equity
        # ceiling since the signals themselves already embed real
        # gamma/vega leverage — stacking notional leverage on top of that
        # would double-count it; kelly_fraction_weight reduced for the same
        # reason.
        CVaRManager(
            managed_analysts=owned(CVAR_MANAGER_ANALYST_IDS),
            weighting_strategy=ShrinkageWeight(DrawdownAwareWeight()),
            portfolio_constructor=PortfolioConstructor(
                **settings.pod_sizing["cvar_manager"].model_dump(),
            ),
            assets=OPTIONS_ASSETS,
        ),
        # Mean-reversion/cointegration relationships decay — a stale pair
        # (see pairs_cointegration_analyst's own docstring) needs trust to
        # shift fast once it stops working, not average out slowly over a
        # long lookback. Hedge's exponentiated weighting reacts quickest of
        # the available strategies; a high eta makes that reaction sharper.
        # Market-neutral pairs book: leverage_cap=6.0 matches the real
        # ~400-650% gross norm quant market-neutral/stat-arb funds run
        # (thin relative-value edges need leverage to matter); net_leverage_cap
        # stays tight at 0.5 since offsetting legs are supposed to cancel
        # most net risk by construction; target_annualized_vol=0.06 matches
        # a relative-value book's typically low absolute vol target.
        StatArbManager(
            managed_analysts=owned(STAT_ARB_MANAGER_ANALYST_IDS),
            weighting_strategy=ShrinkageWeight(Hedge(eta=1.5)),
            portfolio_constructor=PortfolioConstructor(
                **settings.pod_sizing["stat_arb_manager"].model_dump(),
            ),
        ),
        # Macro/rate regimes (hiking cycles, ZIRP eras) persist for months —
        # a short lookback would overreact to noise inside one regime,
        # and this pod's few-but-large tail moves would distort a raw
        # cumulative-return (Hedge) ranking more than a risk-adjusted one.
        # A longer 90-day Sharpe lookback fits a slower-moving domain.
        # Macro/vol-regime overlay on equities: leverage_cap=5.0/
        # net_leverage_cap=3.0 sit within the real 3-8x global-macro
        # leverage range (mostly derivatives margin, not borrowed cash);
        # net stays meaningfully below gross but the overlay is still
        # clearly directional.
        MacroRatesManager(
            managed_analysts=owned(MACRO_RATES_MANAGER_ANALYST_IDS),
            weighting_strategy=ShrinkageWeight(SharpeWeight(lookback_days=90)),
            portfolio_constructor=PortfolioConstructor(
                **settings.pod_sizing["macro_rates_manager"].model_dump(),
            ),
        ),
        # Same 90-day-Sharpe reasoning as Macro/Rates — rate regimes are
        # slow-moving, and this pod's whole point is cir_rate_regime_analyst
        # finally landing on a real duration ladder instead of an
        # arbitrary equity. Default (BL + mean-variance) sizing: a real,
        # duration-weighted equilibrium concept exists for a bond ladder.
        # Treasury duration ladder: leverage_cap=5.0/net_leverage_cap=4.0
        # match the real 4-5x fixed-income relative-value norm (individual
        # rate mispricings are small relative to notional); net close to
        # gross since a real curve view is directional by construction;
        # target_annualized_vol=0.05 matches Treasuries' inherently low vol.
        RatesManager(
            managed_analysts=owned(RATES_MANAGER_ANALYST_IDS),
            weighting_strategy=ShrinkageWeight(SharpeWeight(lookback_days=90)),
            portfolio_constructor=PortfolioConstructor(
                **settings.pod_sizing["rates_manager"].model_dump(),
            ),
            assets=RATES_ASSETS,
        ),
        # No market-cap-style equilibrium exists for commodity futures,
        # currency pairs, or crypto — risk_parity sizing (volatility
        # alone), not Black-Litterman, across all three. EqualWeight
        # trust: generic, individually weak, largely uncorrelated
        # analysts, same reasoning as Alt-Data.
        # Mixed-vol basket between macro and equity directionality on the
        # leverage axis; commodities/crypto pull target_annualized_vol up
        # to 0.13.
        CrossAssetManager(
            managed_analysts=owned(CROSS_ASSET_MANAGER_ANALYST_IDS),
            weighting_strategy=ShrinkageWeight(EqualWeight()),
            portfolio_constructor=PortfolioConstructor(
                method="risk_parity",
                **settings.pod_sizing["cross_asset_manager"].model_dump(),
            ),
            assets=CROSS_ASSET_ASSETS,
        ),
        # A 2-name book (HYG/LQD) driven by one market-wide signal — no
        # need for an equilibrium prior or a trailing-performance ranking
        # across more than one analyst; EqualWeight is a no-op here but
        # keeps this pod's wiring consistent with every other manager.
        # Credit desks run materially lower leverage than rates/macro given
        # illiquidity in stress. HYG/LQD now have real lakehouse price
        # history, so kelly_fraction_weight matches the fund-wide
        # quarter-Kelly default rather than staying disabled.
        CreditManager(
            managed_analysts=owned(CREDIT_MANAGER_ANALYST_IDS),
            weighting_strategy=ShrinkageWeight(EqualWeight()),
            portfolio_constructor=PortfolioConstructor(
                method="risk_parity",
                **settings.pod_sizing["credit_manager"].model_dump(),
            ),
            assets=CREDIT_ASSETS,
        ),
        # Fama-French/Fama-MacBeth/Newey-West test well-established, largely
        # orthogonal academic factor exposures whose premia rotate in and
        # out of favor for structural reasons, not analyst skill —
        # trailing-performance weighting here would just chase factor
        # rotation. EqualWeight keeps the pod diversified across factors.
        # Cross-sectional long/short factor books are market-neutral by
        # design — same "cancel most risk, lever the thin edge" logic as
        # StatArb; tight net_leverage_cap=0.5 enforces the neutrality.
        FactorManager(
            managed_analysts=owned(FACTOR_MANAGER_ANALYST_IDS),
            weighting_strategy=ShrinkageWeight(EqualWeight()),
            portfolio_constructor=PortfolioConstructor(
                **settings.pod_sizing["factor_manager"].model_dump(),
            ),
        ),
        # Event-driven catalysts (an EPS trend, a short squeeze) take a
        # filing/reporting cycle or more to play out — DrawdownAwareWeight's
        # ~quarter-long default lookback gives a catalyst time to resolve
        # before its trailing weight gets pulled down for a losing streak.
        # Event-driven/merger-arb-style leverage_cap=4.0 matches the real
        # 3-5x norm for that style; net_leverage_cap=2.5 keeps some net
        # directionality since deal-completion risk isn't perfectly hedged,
        # unlike a pure market-neutral book.
        EventCreditManager(
            managed_analysts=owned(EVENT_CREDIT_MANAGER_ANALYST_IDS),
            weighting_strategy=ShrinkageWeight(DrawdownAwareWeight()),
            portfolio_constructor=PortfolioConstructor(
                **settings.pod_sizing["event_credit_manager"].model_dump(),
            ),
        ),
        # Alt-data/behavioral signals (calendar anomalies, lead-lag network
        # centrality) are individually weak, largely uncorrelated in what
        # they capture, and sparse (ShrinkageWeight's own target case) — a
        # trailing-performance formula here would mostly fit noise, so this
        # pod leans on scale_factor's staleness/sparsity read instead of
        # ranking analysts against each other. Weakest, most speculative
        # signal set of the 11 pods — kept near a conservative
        # directional-equity ceiling rather than the higher relative-value
        # multiples above.
        AltDataManager(
            managed_analysts=owned(ALT_DATA_MANAGER_ANALYST_IDS),
            weighting_strategy=ShrinkageWeight(EqualWeight()),
            portfolio_constructor=PortfolioConstructor(
                **settings.pod_sizing["alt_data_manager"].model_dump(),
            ),
        ),
        # Systematic VIX-triggered short-SPY insurance sleeve, not a
        # leverage-seeking pod — sized by its own signal plus the
        # Commander's fixed budget carve-out (see
        # commander.TAIL_HEDGE_BUDGET_FRACTION), so this constructor's
        # leverage_cap only needs to comfortably cover a single-instrument
        # short position. kelly_fraction_weight=0.0: a hedge's sizing must
        # not depend on its own (usually negative-carry) trailing Sharpe.
        TailHedgeManager(
            managed_analysts=owned(TAIL_HEDGE_MANAGER_ANALYST_IDS),
            weighting_strategy=ShrinkageWeight(EqualWeight()),
            portfolio_constructor=PortfolioConstructor(
                **settings.pod_sizing["tail_hedge_manager"].model_dump(),
            ),
            assets=TAIL_HEDGE_ASSETS,
        ),
    ]
    return [manager for manager in managers if manager.agent_id not in excluded_manager_ids]
