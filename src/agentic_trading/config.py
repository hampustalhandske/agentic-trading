"""Every tunable strategy constant in one place, overridable per run.

`settings` is the single process-wide `StrategyConfig` instance every
analyst, manager, the Commander, and `PortfolioConstructor` read from
*at use time* (inside `generate_signal`/`scale_factor`/`construct`/
`decide`, or in a constructor), never at import time — so an override
applied before a roster is built, or inside `temporary_overrides`, is
what the run actually trades on.

Three ways to override without editing source:

- CLI: `agentic-trading-experiment ... --set commander.drawdown_trip_threshold=-0.15
  --set pod_sizing.stat_arb_manager.leverage_cap=4.0` (repeatable).
- Script sweep: `with temporary_overrides({"analysts.z_entry_threshold": 1.5}): run()`
  restores the previous values on exit, so several runs in one process
  don't leak settings into each other.
- Direct: `apply_overrides({...})` for a permanent process-wide change.

The lakehouse API location and token are not strategy constants: the
loader reads `LAKEHOUSE_API_URL` and `LAKEHOUSE_API_TOKEN` from the
process environment or the repo-root `.env` through `_env_value`.

Keys are dotted paths: `<group>.<field>`, or
`pod_sizing.<manager_agent_id>.<field>` for the per-pod table. Values
are validated by the group's Pydantic model (type and range), so a typo
or an out-of-range number fails fast instead of silently running.

Every value carries a one-line reason. Anything not derived from data is
labelled a policy choice. `MAX_LEVERAGE` is the one exception to the
use-time rule: Pydantic field bounds on `PortfolioDecision`/`TradeRecord`
are fixed at import, so `firm.max_leverage` must be overridden before
`agentic_trading.agents.schemas` is first imported to take effect there.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

_GROUP_CONFIG = ConfigDict(validate_assignment=True, extra="forbid")

_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


def _env_value(name: str) -> str | None:
    """`name` from the process environment, else from a `KEY=VALUE` line in
    the repo-root `.env` file (comments and blank lines ignored); `None`
    when neither has it."""
    if name in os.environ:
        return os.environ[name]
    if not _ENV_FILE.exists():
        return None
    for line in _ENV_FILE.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        if key.strip() == name:
            return value.strip().strip("'\"")
    return None


class FirmConfig(BaseModel):
    model_config = _GROUP_CONFIG

    # Hard system-wide ceiling on any single target weight; sized to cover
    # the most leveraged real-desk pod (market-neutral stat-arb at 6x gross).
    max_leverage: float = Field(default=6.0, gt=0.0)
    # Sqrt-time annualization base for every daily-vol -> annual-vol step.
    trading_days_per_year: int = Field(default=252, gt=0)
    # Firm-level limits applied to the combined book after the Commander's
    # allocation: gross exposure, any single name, and one-day 95% VaR as a
    # fraction of NAV (a multi-strategy desk's headline risk line).
    max_gross_exposure: float = Field(default=4.0, gt=0.0)
    max_single_name_weight: float = Field(default=0.25, gt=0.0)
    max_daily_var: float = Field(default=0.02, gt=0.0)
    var_confidence_level: float = Field(default=0.95, gt=0.5, lt=1.0)
    # Daily-return window behind the firm VaR covariance: one full vol cycle.
    var_lookback_days: int = Field(default=250, ge=30)
    # Execution assumptions shared by every entry point: 5 bps commission,
    # 5 bps slippage, decide on the close and fill at the next open.
    commission_rate: float = Field(default=0.0005, ge=0.0)
    slippage_bps: float = Field(default=5.0, ge=0.0)
    execution_lag_days: int = Field(default=1, ge=0, le=1)
    # Financing: stock borrow on short notional (general-collateral names
    # cost ~50 bps/yr) and a spread over the risk-free rate on any
    # borrowed cash (negative cash balance).
    short_borrow_rate_annual: float = Field(default=0.005, ge=0.0)
    margin_spread_annual: float = Field(default=0.015, ge=0.0)
    # Square-root market impact (Almgren et al.): impact = coefficient x
    # daily vol x sqrt(shares traded / ADV), added to the fill price. 1.0 is
    # the mid-range empirical estimate for US equities.
    market_impact_coefficient: float = Field(default=1.0, ge=0.0)
    # Firm-level book beta to SPY (weighted sum of each traded asset's own
    # beta, from the same trailing covariance the VaR limit already
    # estimates) may not exceed this; a real multi-strategy book targets
    # low net market exposure, not directional beta.
    max_portfolio_beta: float = Field(default=1.0, ge=0.0)
    # Stress test: a Ledoit-Wolf correlation matrix stressed toward crisis
    # correlations (pairwise correlations pulled toward this value — the
    # standard "correlation breakdown" a real risk desk assumes in a
    # systemic shock) with volatility scaled up, and the resulting one-day
    # loss capped at `max_stress_loss` of NAV.
    stress_correlation_shock: float = Field(default=0.90, ge=0.0, le=1.0)
    stress_vol_multiplier: float = Field(default=2.0, gt=0.0)
    max_stress_loss: float = Field(default=0.15, gt=0.0)
    # A fill is capped at this fraction of the asset's average daily volume
    # (partial fill; the rest is re-targeted next day). 5% is a common
    # desk participation limit.
    max_adv_participation: float = Field(default=0.05, gt=0.0, le=1.0)
    # Trailing window for ADV and daily vol used by impact and participation.
    adv_window_days: int = Field(default=20, ge=2)
    # Starting NAV of every run. A book must be large enough that a
    # fractional-percent option budget buys whole contracts (one SPY put
    # is ~$1,000 of premium); $10M is a small multi-strategy sleeve.
    starting_cash: float = Field(default=10_000_000.0, gt=0.0)


class OptionsConfig(BaseModel):
    """Listed-option execution and expression conventions."""

    model_config = _GROUP_CONFIG

    # Listed equity options settle 100 shares per contract.
    contract_multiplier: int = Field(default=100, ge=1)
    # Retail-institutional midpoint for per-contract commission, in dollars.
    commission_per_contract: float = Field(default=0.65, ge=0.0)
    # Fills cross the quoted spread: buys at the ask, sells at the bid. A
    # contract with no bid is untradeable and never entered or marked.
    # A quote older than this many days cannot be traded against.
    max_snapshot_age_days: int = Field(default=10, ge=1)
    # A held contract keeps being re-marked from its last quote (Black-
    # Scholes at that implied vol) for up to this long after the chain
    # stops carrying its expiry; the historical chain lists only three
    # expiries per snapshot and rotates the near ones weekly.
    max_mark_age_days: int = Field(default=45, ge=1)
    # DoltHub publishes a chain dated t one to four days after t, so an
    # agent deciding on day t only sees a snapshot dated on or before t
    # minus this many calendar days. Applies to every decision and
    # contract-selection read of the chain; marks and fills are not lagged.
    publication_lag_days: int = Field(default=1, ge=0)
    # Reg-T style maintenance margin on a short option: premium plus this
    # fraction of underlying notional, financed at the margin rate.
    short_margin_fraction: float = Field(default=0.20, ge=0.0, le=1.0)
    # Short premium outstanding may not exceed this fraction of NAV
    # (unbounded-loss exposure is capped at the firm level).
    max_short_premium_fraction: float = Field(default=0.05, ge=0.0, le=1.0)
    # Target tenor for every expressed position (30 days: the liquid,
    # CBOE-style constant-maturity point the chain always carries).
    target_tenor_days: int = Field(default=30, ge=5)
    # An existing position is only re-traded when the target differs from
    # it by more than this fraction of the held contracts (a close to zero
    # always goes through); crossing a 20%-relative option spread for a
    # 5% size change is a losing trade.
    rebalance_band: float = Field(default=0.5, ge=0.0, le=1.0)
    # Every fill needs a two-sided quote and at least this much premium per
    # share: below it the per-contract commission and the spread dominate
    # and a "position" is just a stream of fees.
    min_premium_per_share: float = Field(default=0.05, gt=0.0)
    # No single contract may control more underlying notional than this
    # fraction of NAV (contracts x multiplier x spot), so a decaying
    # premium can never be met with an unbounded contract count.
    max_notional_fraction: float = Field(default=1.5, gt=0.0)
    # Roll a position once fewer than this many days remain to expiry:
    # gamma and expiry-week distortions dominate inside a week.
    roll_min_dte: int = Field(default=7, ge=1)
    # The hedge uses the 60-day point: the chain's longest listed expiry
    # persists across snapshots, so the put stays quoted until it rolls,
    # and 1-3 month puts are the institutional tail-hedge tenor.
    tail_hedge_tenor_days: int = Field(default=60, ge=5)
    # A directional option leg only switches side after the pod's
    # underlying view has held the opposite sign for this many consecutive
    # days, and only opens above this weight; daily sign flips would
    # otherwise round-trip the option spread every day.
    expression_confirmation_days: int = Field(default=3, ge=1)
    min_expression_weight: float = Field(default=0.005, ge=0.0)
    # Protective-put tail hedge: strike as a fraction of spot (5% OTM is
    # the common institutional tail-hedge point) and the protection level
    # the sizing targets.
    tail_hedge_strike_moneyness: float = Field(default=0.95, gt=0.0, le=1.0)
    # The hedge protects this fraction of NAV in underlying notional; the
    # premium budget caps what that costs, so as a put decays the sleeve
    # holds a fixed number of contracts rather than buying more.
    tail_hedge_notional_fraction: float = Field(default=1.0, gt=0.0)
    tail_hedge_target_protection: float = Field(default=0.10, gt=0.0, le=1.0)
    # Short straddles on a rich variance risk premium need at least this
    # confidence from the VRP analyst; below it the view is expressed as a
    # long single leg only.
    short_straddle_min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class ManagerRiskConfig(BaseModel):
    """Shared by every pod manager's `scale_factor` overlay."""

    model_config = _GROUP_CONFIG

    # RiskMetrics daily decay constant for the EWMA-variance overlays.
    ewma_lambda: float = Field(default=0.94, gt=0.0, lt=1.0)
    # Alt-data attention signals go stale within ~2 weeks (policy choice).
    alt_data_staleness_half_life_days: float = Field(default=14.0, gt=0.0)
    # Event/filing-driven signals stay relevant for about a reporting month.
    event_staleness_half_life_days: float = Field(default=30.0, gt=0.0)
    # Vol-regime overlay: z-score of today's log EWMA vol against its own
    # trailing path — one full vol cycle of history so "normal" is measured
    # across calm and stressed stretches, not just the last quarter.
    vol_regime_window: int = Field(default=250, ge=30)
    # Below this much history the z-score's std is too noisy to trust; the
    # overlay is a no-op (factor 1.0) until then.
    min_vol_regime_observations: int = Field(default=120, ge=10)
    # Logistic dampening centred 1.5 sigma above normal vol (factor 0.5
    # there), slope 2: no dampening in normal regimes, sharp in a spike.
    vol_zscore_midpoint: float = Field(default=1.5)
    vol_zscore_slope: float = Field(default=2.0, gt=0.0)
    # A pod is never dampened below 10% — a full zero would silently mute
    # the analysts' track record during the very stress they are scored on.
    min_scale_factor: float = Field(default=0.10, ge=0.0, le=1.0)
    # Below this many observations an alt-data read is treated as thin.
    alt_data_min_sample_count: float = Field(default=5.0, gt=0.0)
    # Network-centrality statistics need a longer sample to mean anything.
    anomaly_min_sample_count: float = Field(default=20.0, gt=0.0)
    # Factor analysts already encode significance in their own confidence,
    # so the factor overlay is deliberately allowed to dampen only 10%.
    factor_min_scale_factor: float = Field(default=0.9, ge=0.0, le=1.0)
    factor_max_scale_factor: float = Field(default=1.0, ge=0.0, le=1.0)
    # ~1 trading quarter between cointegration re-screens; pairs decay on a
    # quarterly timescale and daily re-testing is both slow and noisy.
    pairs_refit_interval_days: int = Field(default=63, ge=1)


class CommanderConfig(BaseModel):
    model_config = _GROUP_CONFIG

    # Rolling window of each pod's realized daily return for risk budgeting.
    pod_returns_lookback_days: int = Field(default=90, ge=2)
    # Below this a pod is sized off its bootstrap weight, not a vol estimate.
    min_observations_for_risk_budget: int = Field(default=5, ge=2)
    # Absolute floor on the inverse-vol divisor (fallback path only).
    min_trailing_vol: float = Field(default=1e-8, gt=0.0)
    # Peer-relative vol floor so one abnormally calm pod can't crowd out the rest.
    min_relative_vol_floor: float = Field(default=0.2, gt=0.0, le=1.0)
    # Circuit breaker trips at -10% pod drawdown from its own peak (policy choice).
    drawdown_trip_threshold: float = Field(default=-0.10, lt=0.0)
    # Hysteresis: must recover to -3% before the breaker resets.
    drawdown_recovery_threshold: float = Field(default=-0.03, le=0.0)
    # A tripped pod keeps 25% of its allocation, never 0 — a zero-capital cut
    # is a death spiral a pod could never earn its way out of.
    drawdown_cut_multiplier: float = Field(default=0.25, gt=0.0, lt=1.0)
    # Fixed 2% of gross permanently reserved for the tail hedge (policy choice).
    tail_hedge_budget_fraction: float = Field(default=0.02, ge=0.0, lt=1.0)


class PortfolioConstructionConfig(BaseModel):
    model_config = _GROUP_CONFIG

    # Black-Litterman / mean-variance risk aversion; textbook mid-range value.
    risk_aversion: float = Field(default=2.5, gt=0.0)
    # BL prior-uncertainty scalar; He & Litterman's canonical 0.025.
    tau: float = Field(default=0.025, gt=0.0)
    # Grinold-Kahn information coefficient turning a conviction z-score
    # into a daily expected return: alpha = IC * sigma * z. 0.05 is the
    # prior (a realistic skill level: 0.02 weak, 0.10 excellent); each
    # analyst's measured trailing IC is shrunk toward it with the weight
    # of `ic_shrinkage_observations` pseudo-observations.
    information_coefficient: float = Field(default=0.05, gt=0.0, le=1.0)
    ic_lookback_days: int = Field(default=250, ge=20)
    ic_shrinkage_observations: int = Field(default=60, ge=1)
    # Measured IC is capped: above this a trailing correlation is luck or a
    # data artefact, not skill.
    max_measured_ic: float = Field(default=0.15, gt=0.0, le=1.0)
    # Risk-parity pods tilt each asset's risk budget by this fraction of
    # its conviction z-score (clipped to +-1) and take the view's sign, so
    # analyst opinion reaches position size without abandoning the
    # covariance-based budget.
    risk_parity_view_tilt: float = Field(default=0.5, ge=0.0, le=1.0)
    # Cost-aware sizing: a rebalance must move a weight by more than the
    # no-trade band, and clear `cost_clearance_multiplier` x the round-trip
    # cost, before it goes through. 5 bps matches the engine's slippage.
    transaction_cost_bps: float = Field(default=5.0, ge=0.0)
    no_trade_buffer: float = Field(default=0.005, ge=0.0)
    # Rebalance calendar: a pod only re-trades on a scheduled cadence, or
    # off-cycle when an asset's weight has drifted from its last executed
    # value by more than the drift band — the standard "periodic plus
    # drift-triggered" hybrid real desks use so a book isn't re-traded on
    # every noise tick. 5 trading days = weekly; 10% relative drift is a
    # common institutional tolerance band.
    rebalance_calendar_days: int = Field(default=5, ge=1)
    rebalance_drift_band: float = Field(default=0.10, ge=0.0)
    # No single asset may account for more than this share of a pod's own
    # total component VaR (quantcore's component_var, which already sums
    # to the pod's total); above it that asset's weight is scaled down.
    # 40% means no one name may dominate the pod's own risk budget.
    component_var_max_share: float = Field(default=0.40, gt=0.0, le=1.0)
    # Cap on the per-asset conviction z-score (sum of signals / sqrt(n)) so
    # five unanimous analysts can't produce a 5-sigma view.
    view_zscore_cap: float = Field(default=2.0, gt=0.0)
    # Idzorek view confidence = |net signal| / sum of confidences, clipped:
    # a fully split pod still moves 10% toward its net view, a unanimous
    # pod 90%; never 0 or 1, which would degenerate the BL posterior.
    view_confidence_floor: float = Field(default=0.1, gt=0.0, lt=1.0)
    view_confidence_ceiling: float = Field(default=0.9, gt=0.0, lt=1.0)
    # Confidence level of the logged (not sizing-consumed) component-VaR report.
    var_confidence_level: float = Field(default=0.95, gt=0.5, lt=1.0)
    # Below this a pod bootstraps off view direction alone; the effective
    # floor is max(this, n_assets + 1) since Ledoit-Wolf needs T > k.
    min_observations_for_covariance: int = Field(default=5, ge=2)
    # Fallback vol target when a pod does not set its own (every default pod does).
    target_annualized_vol: float = Field(default=0.15, gt=0.0)
    # The Kelly overlay ramps from 0 to the pod's target fraction over this
    # many trailing days; a Sharpe from a handful of points is noise.
    kelly_lookback_days: int = Field(default=60, ge=1)
    # No-trade buffer's cost-derived minimum move = this x cost_bps / 10,000;
    # a fixed economic-hurdle approximation, not a real impact model.
    cost_clearance_multiplier: float = Field(default=5.0, ge=0.0)


class PodSizingConfig(BaseModel):
    """One pod's own `PortfolioConstructor` sizing parameters."""

    model_config = _GROUP_CONFIG

    leverage_cap: float = Field(gt=0.0)
    net_leverage_cap: float = Field(ge=0.0)
    target_annualized_vol: float = Field(gt=0.0)
    kelly_fraction_weight: float = Field(ge=0.0, le=1.0)


class AnalystConfig(BaseModel):
    """Layer-1 thresholds shared by more than one analyst, or worth sweeping."""

    model_config = _GROUP_CONFIG

    # Three trailing-window tiers: one quarter for fast vol/tail reads, half
    # a year for regressions that need more degrees of freedom, and a year
    # for cointegration/HMM/rate-regime fits that need a full cycle.
    risk_window_short: int = Field(default=60, ge=2)
    risk_window_medium: int = Field(default=120, ge=2)
    risk_window_long: int = Field(default=250, ge=2)
    # Tail cutoff for the parametric VaR/CVaR analysts (RiskMetrics convention).
    var_confidence_level: float = Field(default=0.95, gt=0.5, lt=1.0)
    # GARCH-family MLE (EGARCH/GJR) needs ~500+ observations to be
    # reliable; three years is used, with a hard floor below which the
    # analyst stays flat rather than trusting a degenerate fit.
    risk_window_garch: int = Field(default=750, ge=50)
    min_garch_observations: int = Field(default=250, ge=30)
    # Asymmetric-GARCH fits are re-run monthly, not daily: the parameters
    # move slowly and each MLE is the most expensive call in the pod.
    garch_refit_interval_days: int = Field(default=21, ge=1)
    # Accepted GARCH persistence (alpha + beta + gamma/2, or EGARCH beta):
    # above this the process is effectively integrated and the path is
    # not a usable forecast.
    max_garch_persistence: float = Field(default=0.995, gt=0.0, lt=1.0)
    # Vol-regime z-score (log conditional vol vs its own trailing 250-day
    # path) thresholds for the risk analysts: calm below -0.5 sigma earns
    # the low-vol premium; a 2-sigma spike carries negative expected return.
    vol_zscore_long_threshold: float = Field(default=-0.5)
    vol_zscore_short_threshold: float = Field(default=2.0)
    # Fama-French: monthly expected-return-over-vol ratio below which the
    # view is noise, and at which confidence is full (0.30 ≈ a 1.0 annual
    # information ratio).
    factor_zscore_flat: float = Field(default=0.05, ge=0.0)
    factor_zscore_full: float = Field(default=0.30, gt=0.0)
    # Two-sigma entry / half-sigma exit for the stat-arb spread z-scores
    # (Gatev-Goetzmann-Rouwenhorst convention).
    z_entry_threshold: float = Field(default=2.0, gt=0.0)
    z_exit_threshold: float = Field(default=0.5, ge=0.0)
    # Pair screening: two-year formation (Engle-Granger has low power at
    # one year), admit at 5%, keep while below 10% (hysteresis), require a
    # tradeable half-life, at most five pairs ranked by half-life.
    pair_formation_days: int = Field(default=504, ge=60)
    pair_admit_pvalue: float = Field(default=0.05, gt=0.0, lt=1.0)
    pair_keep_pvalue: float = Field(default=0.10, gt=0.0, lt=1.0)
    pair_min_half_life_days: float = Field(default=5.0, gt=0.0)
    pair_max_half_life_days: float = Field(default=60.0, gt=0.0)
    pair_drop_half_life_days: float = Field(default=120.0, gt=0.0)
    pair_min_correlation: float = Field(default=0.5, ge=-1.0, le=1.0)
    max_pairs: int = Field(default=5, ge=1)
    # Regime HMM: two Hamilton states on three years (a one-year window
    # rarely contains two regimes), monthly refit, act above 70% posterior.
    hmm_window: int = Field(default=750, ge=100)
    hmm_min_observations: int = Field(default=500, ge=50)
    hmm_refit_interval_days: int = Field(default=21, ge=1)
    hmm_action_threshold: float = Field(default=0.70, gt=0.5, lt=1.0)
    # Variance risk premium (30-day ATM IV minus model vol) in vol points
    # needed to act; 8 more points earn full confidence.
    vrp_threshold: float = Field(default=0.02, gt=0.0)
    vrp_full_confidence_span: float = Field(default=0.08, gt=0.0)
    # Greeks: DTE-normalised ATM gamma is z-scored against its own trailing
    # 60 snapshots (30 minimum); +-1 sigma separates pinning from trending.
    gamma_history_window: int = Field(default=60, ge=10)
    gamma_min_history: int = Field(default=30, ge=5)
    gamma_zscore_threshold: float = Field(default=1.0, gt=0.0)
    # Insider cluster buying (Cohen, Malloy & Pomorski): distinct insiders
    # making open-market purchases within one month; two is the smallest
    # cluster, five buyers is full conviction; confidence is capped because
    # the signal is sparse and its premium is documented mostly in small caps.
    insider_window_days: int = Field(default=30, ge=5)
    insider_min_buyers: int = Field(default=2, ge=1)
    insider_full_confidence_buyers: int = Field(default=5, ge=1)
    insider_max_confidence: float = Field(default=0.5, gt=0.0, le=1.0)
    # ARIMA: Box-Jenkins practice re-identifies the order rarely and
    # re-estimates often; the pre-fit Ljung-Box window is the last ~6 months.
    arima_order_reselect_days: int = Field(default=63, ge=1)
    arima_pretest_window: int = Field(default=120, ge=30)
    # Spread z-score window: one quarter, matching the pairs literature.
    z_window: int = Field(default=60, ge=2)
    # Cap on any z-score-to-confidence mapping so one outlier can't pin
    # confidence at 1.0 forever.
    z_cap: float = Field(default=3.0, gt=0.0)
    # |t| = 2 ≈ 5% two-sided significance, the usual regression cutoff.
    t_stat_threshold: float = Field(default=2.0, gt=0.0)
    # 5% level for the Engle-Granger cointegration and Granger-causality tests.
    significance_level: float = Field(default=0.05, gt=0.0, lt=1.0)
    # GJR-GARCH: a forecast variance 1.5x the trailing sample flags crash risk.
    vol_jump_threshold: float = Field(default=1.5, gt=1.0)
    # VIX z-score that maps to full hedge confidence (policy choice).
    vix_zscore_full_confidence: float = Field(default=3.0, gt=0.0)
    # One trading year of VIX history for its own z-score baseline.
    vix_lookback_days: int = Field(default=252, ge=2)
    # Below this much VIX history the hedge stays flat rather than guessing.
    min_vix_observations: int = Field(default=30, ge=2)
    # The hedge sleeve is always on at this confidence so protection is
    # owned before a stress, not bought after VIX has already spiked.
    tail_hedge_base_confidence: float = Field(default=0.15, ge=0.0, le=1.0)


class StrategyConfig(BaseModel):
    model_config = _GROUP_CONFIG

    firm: FirmConfig = Field(default_factory=FirmConfig)
    options: OptionsConfig = Field(default_factory=OptionsConfig)
    managers: ManagerRiskConfig = Field(default_factory=ManagerRiskConfig)
    commander: CommanderConfig = Field(default_factory=CommanderConfig)
    portfolio: PortfolioConstructionConfig = Field(default_factory=PortfolioConstructionConfig)
    analysts: AnalystConfig = Field(default_factory=AnalystConfig)
    # Per-pod sizing, keyed by manager `agent_id`; the research-grounded
    # reasoning for each row lives next to the manager in `roster.py`.
    pod_sizing: dict[str, PodSizingConfig] = Field(
        default_factory=lambda: {
            "risk_manager": PodSizingConfig(
                leverage_cap=3.0,
                net_leverage_cap=2.0,
                target_annualized_vol=0.12,
                kelly_fraction_weight=0.25,
            ),
            "cvar_manager": PodSizingConfig(
                leverage_cap=2.0,
                net_leverage_cap=1.5,
                target_annualized_vol=0.10,
                kelly_fraction_weight=0.15,
            ),
            "stat_arb_manager": PodSizingConfig(
                leverage_cap=6.0,
                net_leverage_cap=0.5,
                target_annualized_vol=0.06,
                kelly_fraction_weight=0.25,
            ),
            "macro_rates_manager": PodSizingConfig(
                leverage_cap=5.0,
                net_leverage_cap=3.0,
                target_annualized_vol=0.11,
                kelly_fraction_weight=0.25,
            ),
            "rates_manager": PodSizingConfig(
                leverage_cap=5.0,
                net_leverage_cap=4.0,
                target_annualized_vol=0.05,
                kelly_fraction_weight=0.25,
            ),
            "cross_asset_manager": PodSizingConfig(
                leverage_cap=3.5,
                net_leverage_cap=2.5,
                target_annualized_vol=0.13,
                kelly_fraction_weight=0.25,
            ),
            "credit_manager": PodSizingConfig(
                leverage_cap=2.0,
                net_leverage_cap=1.75,
                target_annualized_vol=0.04,
                kelly_fraction_weight=0.25,
            ),
            "factor_manager": PodSizingConfig(
                leverage_cap=4.0,
                net_leverage_cap=0.5,
                target_annualized_vol=0.08,
                kelly_fraction_weight=0.25,
            ),
            "event_credit_manager": PodSizingConfig(
                leverage_cap=4.0,
                net_leverage_cap=2.5,
                target_annualized_vol=0.11,
                kelly_fraction_weight=0.25,
            ),
            "alt_data_manager": PodSizingConfig(
                leverage_cap=2.5,
                net_leverage_cap=2.0,
                target_annualized_vol=0.09,
                kelly_fraction_weight=0.25,
            ),
            "tail_hedge_manager": PodSizingConfig(
                leverage_cap=1.5,
                net_leverage_cap=1.5,
                target_annualized_vol=0.10,
                kelly_fraction_weight=0.0,
            ),
        }
    )


settings = StrategyConfig()


def _resolve(path: str) -> tuple[BaseModel, str]:
    parts = path.split(".")
    if len(parts) < 2:
        raise KeyError(
            f"override key must be '<group>.<field>' or 'pod_sizing.<pod>.<field>': {path!r}"
        )
    target: Any = settings
    for part in parts[:-1]:
        if isinstance(target, dict):
            if part not in target:
                raise KeyError(f"unknown pod {part!r} in override {path!r}")
            target = target[part]
        elif part in type(target).model_fields:
            target = getattr(target, part)
        else:
            raise KeyError(f"unknown config group {part!r} in override {path!r}")
    if not isinstance(target, BaseModel) or parts[-1] not in type(target).model_fields:
        raise KeyError(f"unknown config field {parts[-1]!r} in override {path!r}")
    return target, parts[-1]


def apply_overrides(overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Set each dotted key on `settings` (validated) and return the values
    they replaced, so a caller can restore them."""
    previous: dict[str, Any] = {}
    for path, value in overrides.items():
        target, field = _resolve(path)
        previous[path] = getattr(target, field)
        setattr(target, field, value)
    return previous


def parse_override(text: str) -> tuple[str, str]:
    """Split a CLI-style `group.field=value` token; the value stays a string
    and is coerced by the group's own field validator."""
    key, separator, value = text.partition("=")
    if not separator or not key.strip():
        raise ValueError(f"override must look like group.field=value, got {text!r}")
    return key.strip(), value.strip()


@contextmanager
def temporary_overrides(overrides: Mapping[str, Any]) -> Iterator[StrategyConfig]:
    """Apply `overrides` for the duration of the block, then restore."""
    previous = apply_overrides(overrides)
    try:
        yield settings
    finally:
        apply_overrides(previous)


def reset_to_defaults() -> None:
    fresh = StrategyConfig()
    for group in type(settings).model_fields:
        setattr(settings, group, getattr(fresh, group))
