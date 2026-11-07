# schemas.py

import re
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agentic_trading.config import settings

# Set by the analyst that built the signal: "active" is a market opinion,
# including flat at confidence 0 when the model has the data and declines
# to bet; "unavailable" means the data needed for one is not there; and
# "error" means the model or a loader failed. `AnalystAgent` builds each
# non-opinion signal through `unavailable_signal`, `abstain_signal` or
# `error_signal`.
SignalStatus = Literal["active", "unavailable", "error"]

# A pod's state on one day. "unavailable": none of the pod's decisions
# carries an active signal (its data or analysts produced nothing).
# "no_position": at least one active signal, yet every decision has weight
# 0. "holding": anything else.
PodState = Literal["holding", "no_position", "unavailable"]

# The firm-wide gross-leverage ceiling every pod manager's own
# PortfolioConstructor is hard-clipped to (a Bridgewater-style
# volatility-targeting sanity backstop, not the primary sizing lever —
# see portfolio_construction.py). Since the Commander only ever combines
# already-risk-managed pod portfolios by capital allocations that sum to
# 1.0, no per-asset weight downstream can exceed this regardless of how
# the pods combine. Set high enough to cover the most leveraged real-desk
# pod (market-neutral stat-arb, which real quant books run at 400-650%+
# gross) without artificially clipping any pod below its own strategy's
# realistic ceiling.
MAX_LEVERAGE = settings.firm.max_leverage


class AgentSignal(BaseModel):
    """A single analyst's directional view on a single asset, for one day.

    A signal at `confidence` 0 carries no view, so validation sets its
    `direction` to "flat" whatever the analyst passed; every other field,
    `status` included, is kept as given."""

    agent_id: str
    asset: str
    as_of_date: date
    direction: Literal["long", "short", "flat"]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    status: SignalStatus = "active"
    # Typed provenance for sparse/lagged sources: the date the underlying
    # data was published (`source_as_of`) and how many observations the
    # read rests on (`sample_count`); managers dampen on both.
    source_as_of: date | None = None
    sample_count: int | None = None

    @model_validator(mode="after")
    def _zero_confidence_is_flat(self) -> "AgentSignal":
        if self.confidence == 0.0:
            self.direction = "flat"
        return self


class ManagerSignal(BaseModel):
    """An analyst's signal after being scaled by its owning manager.

    Same direction as the original AgentSignal — a manager dampens
    confidence, it does not override what an analyst believes. Traceable
    back to both the manager that scaled it and the analyst that
    originated it, which is what the Commander/scoring system will need
    to correctly attribute performance later. `raw_confidence` carries the
    analyst's own unscaled confidence alongside the manager-scaled
    `confidence`, so downstream attribution can report both a
    system-contribution number and a raw-skill number.
    """

    manager_id: str
    source_agent_id: str
    asset: str
    as_of_date: date
    direction: Literal["long", "short", "flat"]
    confidence: float = Field(ge=0.0, le=1.0)
    raw_confidence: float = Field(ge=0.0, le=1.0)
    scale_applied: float = Field(ge=0.0)
    rationale: str
    status: SignalStatus = "active"
    # Typed provenance for sparse/lagged sources: the date the underlying
    # data was published (`source_as_of`) and how many observations the
    # read rests on (`sample_count`); managers dampen on both.
    source_as_of: date | None = None
    sample_count: int | None = None


_CONTRACT_ID_PATTERN = re.compile(r"^(\S+) (\d{4}-\d{2}-\d{2}) ([CP]) (\d+(?:\.\d+)?)$")


class OptionContract(BaseModel):
    """One listed equity option: `underlying`, `expiry`, `strike` and
    `option_type`. Immutable and hashable, so it doubles as a dictionary
    key; `contract_id` (`"AAPL 2024-03-15 C 185.00"`) is the string form
    every ledger, blotter and decision keys the contract by."""

    model_config = ConfigDict(frozen=True)

    underlying: str
    expiry: date
    strike: float = Field(gt=0.0)
    option_type: Literal["call", "put"]

    @property
    def contract_id(self) -> str:
        letter = "C" if self.option_type == "call" else "P"
        return f"{self.underlying} {self.expiry.isoformat()} {letter} {self.strike:.2f}"

    @classmethod
    def parse(cls, contract_id: str) -> "OptionContract":
        """The contract a `contract_id` string names; raises `ValueError`
        for anything not in the `"<underlying> <YYYY-MM-DD> <C|P> <strike>"`
        form."""
        match = _CONTRACT_ID_PATTERN.match(contract_id)
        if match is None:
            raise ValueError(f"not an option contract id: {contract_id!r}")
        underlying, expiry, letter, strike = match.groups()
        return cls(
            underlying=underlying,
            expiry=date.fromisoformat(expiry),
            strike=float(strike),
            option_type="call" if letter == "C" else "put",
        )

    @staticmethod
    def is_contract_id(asset: str) -> bool:
        return _CONTRACT_ID_PATTERN.match(asset) is not None


def _check_instrument_matches_asset(asset: str, instrument: OptionContract | None) -> None:
    if instrument is not None and asset != instrument.contract_id:
        raise ValueError(
            f"asset {asset!r} does not match instrument contract_id {instrument.contract_id!r}"
        )


class PortfolioDecision(BaseModel):
    """The Commander's final allocation decision for one asset, one day.

    Traceable back to every ManagerSignal that fed into it, so the
    decision is fully auditable — not just "why this weight" but "which
    signals, from which managers and analysts, produced it."

    `instrument` is `None` for an equity. When set, `asset` must equal
    `instrument.contract_id` and `target_weight` is the signed premium
    notional over NAV (long positive, short negative) rather than a
    share-notional weight.

    `hold_contracts` marks an option row whose contract the book should
    keep at the count it already holds: with a position held and a
    non-zero `target_weight`, the book trades nothing whatever the
    weight, mark or NAV. A `target_weight` of 0.0 still closes the
    position, and with nothing held the row sizes from `target_weight`
    as an unflagged one does.
    """

    asset: str
    as_of_date: date
    target_weight: float = Field(ge=-MAX_LEVERAGE, le=MAX_LEVERAGE)
    contributing_signals: list[ManagerSignal]
    instrument: OptionContract | None = None
    hold_contracts: bool = False

    @model_validator(mode="after")
    def _instrument_matches_asset(self) -> "PortfolioDecision":
        _check_instrument_matches_asset(self.asset, self.instrument)
        return self

    @property
    def agent_weights(self) -> dict[str, float]:
        """Convenience view deriving {source_agent_id: weight applied} from
        contributing_signals, for callers that only need per-agent
        attribution rather than the full ManagerSignal objects."""
        return {
            signal.source_agent_id: signal.scale_applied for signal in self.contributing_signals
        }


class PodTarget(BaseModel):
    """One pod's part of the Commander's final target for one instrument on
    one day (`Commander.pod_targets`).

    `pod_id` is the manager's `agent_id`. `target_weight` is the pod's
    contribution after firm limits, the short-premium cap and the leverage
    clamp, so the targets of every pod for an instrument sum to that
    instrument's `PortfolioDecision.target_weight`; for an option it is a
    signed premium weight. `contributing_signals` are the pod's own signals
    for the instrument. `instrument` is set on an option row, and `asset`
    is then its `contract_id`. `delta_equiv_weight` is set on an option row
    with a mark: `target_weight x delta x spot / mid`, the signed delta
    exposure as a fraction of NAV (a long put is negative); `None`
    otherwise. `mark_as_of` is the date of the option-chain snapshot that
    mark rests on and `mark_mid` its mid (the decision day's close mark),
    both `None` without one.

    `held` is `True` when the pod keeps a target it set on an earlier day
    rather than setting one: on a day its instrument has no bar
    (`carried`) while its weight is the carried target (a firm limit that
    scales it makes it a new decision), when its own sized weight is
    unchanged from the day
    before (its no-trade band held it), and on a held option row whose
    contract count is kept (`hold_contracts`). `carried` is `True` on a
    day the instrument (or an option's underlying) has no bar: the pod's
    target is its last decided one, carried forward."""

    pod_id: str
    asset: str
    as_of_date: date
    target_weight: float = Field(ge=-MAX_LEVERAGE, le=MAX_LEVERAGE)
    contributing_signals: list[ManagerSignal]
    instrument: OptionContract | None = None
    delta_equiv_weight: float | None = None
    mark_as_of: date | None = None
    mark_mid: float | None = None
    held: bool = False
    carried: bool = False
    hold_contracts: bool = False

    @model_validator(mode="after")
    def _instrument_matches_asset(self) -> "PodTarget":
        _check_instrument_matches_asset(self.asset, self.instrument)
        return self


class TradeRecord(BaseModel):
    """One asset's realized position change for one day, produced by
    `Portfolio.apply_decision()`. The single source of truth for "what did
    the portfolio actually do" — both the CLI and the GUI's trade-blotter
    display are built from this rather than each re-deriving fill mechanics
    independently.

    For an option fill (`instrument` set, `asset == instrument.contract_id`)
    `target_shares`/`delta_shares` are in contracts, `price` is the
    per-share premium, `contracts` is the signed contract delta of the
    fill and `premium` its signed cash flow (negative when premium is
    paid, positive when received). An "expire" record is the settlement
    of a position at intrinsic value on or after its expiry. A "roll"
    record is a held future's round trip from its front contract to the
    next (`Portfolio.charge_rolls`): the position is unchanged, `notional`
    is the notional of both legs, and its costs are the legs' commission
    and slippage.

    `reference_price` is the price the fill is measured against before
    any execution cost: the market price handed to the fill for an
    equity, the mark's mid for an option, intrinsic value for an expiry
    (`None` on a record that does not carry one). `transaction_cost` is
    the commission. `slippage_cost` and `impact_cost` are an equity
    fill's |delta_shares| x `reference_price` x `slippage_bps` and
    x `impact_bps`, over 10,000 (for a future or spot FX fill,
    |delta_shares| x the USD value of one unit at `reference_price`
    instead; see `data/instruments.unit_value_usd`); `spread_cost` is an option fill's
    |contracts| x |price - reference_price| x the contract multiplier.
    Every cost is in currency, already in the fill's cash flow, and 0 on
    a hold or expiry."""

    as_of_date: date
    asset: str
    action: Literal["buy", "sell", "hold", "expire", "roll"]
    target_weight: float = Field(ge=-MAX_LEVERAGE, le=MAX_LEVERAGE)
    previous_weight: float
    target_shares: float
    delta_shares: float
    price: float = Field(ge=0.0)
    notional: float = Field(ge=0.0)
    reference_price: float | None = Field(ge=0.0, default=None)
    transaction_cost: float = Field(ge=0.0, default=0.0)
    slippage_cost: float = Field(ge=0.0, default=0.0)
    impact_cost: float = Field(ge=0.0, default=0.0)
    spread_cost: float = Field(ge=0.0, default=0.0)
    # Square-root market impact paid on this fill, in basis points of
    # price; 0 when the asset has no volume data.
    impact_bps: float = Field(ge=0.0, default=0.0)
    instrument: OptionContract | None = None
    contracts: float = 0.0
    premium: float = 0.0

    @model_validator(mode="after")
    def _instrument_matches_asset(self) -> "TradeRecord":
        _check_instrument_matches_asset(self.asset, self.instrument)
        return self


class PerformanceRecord(BaseModel):
    """One day's realized, attributable performance for one analyst or
    manager, on one asset. attributable_return is computed as
    direction_sign * confidence * realized_return, where confidence is the
    manager-scaled, trailing-weight-scaled ManagerSignal.confidence — a
    system-contribution number, not raw analyst skill (an analyst trusted
    early keeps compounding a bigger multiplier on this value regardless
    of whether its underlying calls got better or worse). raw_attributable_return
    is the same formula using ManagerSignal.raw_confidence instead — the
    analyst's own unscaled confidence, with no manager scale_factor or
    trailing-track-record weight applied — for reading an analyst's
    directional-call quality independent of how much the system came to
    trust it. This class only defines the data shape, not the computation."""

    source_id: str
    source_type: Literal["analyst", "manager"]
    asset: str
    as_of_date: date
    attributable_return: float
    raw_attributable_return: float
    # direction_sign x raw confidence: the signed forecast itself, so a
    # trailing information coefficient (its correlation with
    # `realized_return`) can be measured per source.
    signal_score: float = 0.0
    realized_return: float = 0.0


class AgentWeight(BaseModel):
    """One agent's portfolio-influence weight for one day, as produced by
    a WeightingStrategy.

    Not bounded above by 1.0: WeightingStrategy.get_weights() (see
    scoring/weighting.py) normalizes weights to average 1.0 across
    sources rather than sum to 1.0, so an outperforming source's weight
    legitimately exceeds 1.0.
    """

    agent_id: str
    as_of_date: date
    weight: float = Field(ge=0.0, le=1.0)


class CapitalUsage(BaseModel):
    """How much of the firm's capital sat idle on one valuation day.

    `idle_pod_share` is the fraction of firm capital allocated to pods
    that held no position that day (`None` when the commander does not
    report it); it is the sum of `unavailable_pod_share` (pods whose data
    or analysts produced no active signal) and `no_position_pod_share`
    (pods whose active signals produced no position). `pod_states` maps
    each pod to its `PodState` that day, empty when the commander does not
    report it. `cash_share` is cash divided by net asset value after the
    day's fills and financing."""

    as_of_date: date
    idle_pod_share: float | None
    cash_share: float
    unavailable_pod_share: float | None = None
    no_position_pod_share: float | None = None
    pod_states: dict[str, PodState] = {}


class PodPerformance(BaseModel):
    """One pod's result over a run, from the pod's own daily return series
    (its unscaled book at its own risk, before the Commander's allocation).

    `observations` is the number of daily returns. `cumulative_return` is
    prod(1 + r) - 1. `annualized_vol` is the sample standard deviation
    (ddof=1) times sqrt(trading days per year), 0.0 below 2 observations.
    `sharpe_ratio` is the annualized Sharpe ratio of r with no risk-free
    subtraction, since a pod's book is a sleeve over cash; `None` below 2
    observations or at zero vol. `max_drawdown` is the maximum peak-to-trough
    drawdown of the compounded series, 0.0 below 2 observations."""

    observations: int
    cumulative_return: float
    annualized_vol: float
    sharpe_ratio: float | None
    max_drawdown: float


class PodCostAttribution(BaseModel):
    """One pod's share of a run's fill costs, and how much its book moved.

    A pod's contribution to an asset is the Commander's allocation to the
    pod times the pod's target weight, before firm limits (a tail-hedge
    option row counts at its face value). Each cost and the premium of
    each fill a decision produced are split across pods pro rata to
    |change in contribution| to the filled asset on that decision,
    against the commander's previous decision. A fill on which no pod's
    contribution changed, every fill of a commander that reports no
    contributions, and the transaction cost of an expiry settlement are
    charged to the `"commander"` entry.

    `transaction_cost` is the commission charged to the pod;
    `slippage_cost`, `impact_cost` and `spread_cost` are its share of the
    fills' costs of those names (see `TradeRecord`), and `execution_cost`
    is the sum of all four. `option_premium_paid` is the net option
    premium charged to it (paid minus received; expiry settlements
    excluded). `turnover` is the
    sum over decisions and assets of |change in contribution|, times
    trading days per year over the number of valuation days. `trading_days`
    counts the decisions on which any of the pod's contributions changed.
    `sign_changes` counts, per asset, each time a nonzero contribution
    takes the opposite sign of the last nonzero one (a pass through zero
    counts once). The `"commander"` entry has zero `turnover`,
    `trading_days` and `sign_changes`."""

    transaction_cost: float = 0.0
    slippage_cost: float = 0.0
    impact_cost: float = 0.0
    spread_cost: float = 0.0
    execution_cost: float = 0.0
    option_premium_paid: float = 0.0
    turnover: float = 0.0
    trading_days: int = 0
    sign_changes: int = 0


class PodCostDay(BaseModel):
    """One valuation day's inputs to `PodCostAttribution`.

    `contributions` maps pod id to asset to the pod's contribution on that
    day's decision, with zero contributions left out (a pod with none maps
    to `{}`); `None` when the commander does not report contributions.
    `transaction_cost` (commission), `slippage_cost`, `impact_cost`,
    `spread_cost` and `option_premium_paid` are what that day's fills
    charged each pod id (and `"commander"`), with ids charged nothing left
    out."""

    as_of_date: date
    contributions: dict[str, dict[str, float]] | None
    transaction_cost: dict[str, float] = Field(default_factory=dict)
    slippage_cost: dict[str, float] = Field(default_factory=dict)
    impact_cost: dict[str, float] = Field(default_factory=dict)
    spread_cost: dict[str, float] = Field(default_factory=dict)
    option_premium_paid: dict[str, float] = Field(default_factory=dict)


class UnfactoredSplitMove(BaseModel):
    """A split-sized one-day move in a security's adjusted log returns on a
    date with no adjustment-factor row for the security. `ratio` is the
    day's price over the previous day's (0.5 for an unadjusted 2-for-1
    split)."""

    security_id: int
    as_of_date: date
    ratio: float


class ShortWarmup(BaseModel):
    """An instrument of a run whose first kept bar (`first_bar`) comes
    after the first session of the run's warm-up, so it enters the run's
    first decision day with only `warmup_sessions` sessions of history:
    the calendar's sessions from its first bar up to, not including, the
    run's first session."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    instrument: str
    first_bar: date
    warmup_sessions: int = Field(ge=0)


class UnfilledOrder(BaseModel):
    """A decision the backtest engine could not execute on `as_of_date`, the
    day the order failed to fill.

    `reason` is `no_fill_price` (an equity whose fill price was missing,
    NaN or non-positive; `price` is the value seen), `no_option_mark` (an
    option with no mark that day; `price` is None) or
    `untradeable_option_mark` (an option whose mark was not tradeable while
    the target differed from the held position; `price` is the mark's mid).
    """

    as_of_date: date
    asset: str
    target_weight: float
    reason: Literal["no_fill_price", "no_option_mark", "untradeable_option_mark"]
    price: float | None = None


# The sizing stages `PortfolioConstructor.construct` applies after the risk
# budget, in pipeline order.
SizingStage = Literal[
    "vol_target_cap",
    "kelly",
    "gross_cap",
    "net_trim",
    "no_trade_band",
    "component_var_cap",
    "rebalance_calendar",
    "final_clip",
]


class SizingDiagnostics(BaseModel):
    """Ex-ante annualised volatility of one pod's book after each sizing
    stage of one `PortfolioConstructor.construct` call.

    Every vol is `sqrt(w' Sigma w * trading_days_per_year)`, with `Sigma`
    the shrunk daily covariance the constructor estimated for the assets
    that vary; assets left out of it carry weight 0. `path` is `"full"`
    when the book was risk-budgeted off that covariance and
    `"view_direction"` when it was sized off view direction alone (too
    little history, or no asset varies); a `"view_direction"` call has an
    empty `vol_after` and `None` for `unlevered_vol`, `final_vol` and
    `binding_stage`.

    `target_vol` is the constructor's annualised vol target.
    `unlevered_vol` is the vol of the risk-budgeted weights before any
    leverage. `vol_after` maps each stage to the vol of the weights it
    produced, in pipeline order (see `SizingStage`); a stage that did not
    change the weights carries the previous stage's vol. `final_vol` is the
    vol of the returned weights, the last entry of `vol_after`.
    `binding_stage` is the stage with the largest drop in vol from the
    stage before it, `target_vol` standing in before the first stage;
    `None` when no drop exceeds `1e-9 * target_vol`. `manager_id` is the
    owning manager's `agent_id`, `None` for a constructor no manager runs."""

    as_of_date: date
    manager_id: str | None = None
    path: Literal["full", "view_direction"]
    target_vol: float
    unlevered_vol: float | None
    vol_after: dict[SizingStage, float]
    final_vol: float | None
    binding_stage: SizingStage | None


class ConstructorState(BaseModel):
    """What one `PortfolioConstructor` carries from one `construct` call to
    the next: the pod's own realised daily returns (`own_return_history`,
    read by the Kelly overlay), the day of its last full rebalance (`None`
    before the first) and the trading days since then."""

    own_return_history: list[tuple[date, float]] = Field(default_factory=list)
    last_rebalance_date: date | None = None
    days_since_last_rebalance: int = 0


class TailHedgeState(BaseModel):
    """What the tail-hedge manager carries between days: the put it holds
    (`None` when it holds none), that put's premium weight as a fraction of
    its uncapped weight, and the (open date, premium weight) of every put
    opened, which the premium cap's trailing window reads."""

    held_contract_id: str | None = None
    held_scale: float = 1.0
    commitments: list[tuple[date, float]] = Field(default_factory=list)


class ManagerState(BaseModel):
    """Everything one manager carries between days that moves its next
    decision: yesterday's sized weights (`previous_weights`, read by the
    no-trade band), its portfolio constructor's state, and the tail-hedge
    state for the tail-hedge manager (`None` for every other manager).
    Checkpointed per manager in `CommanderRiskState.manager_states`."""

    previous_weights: dict[str, float] = Field(default_factory=dict)
    constructor: ConstructorState = Field(default_factory=ConstructorState)
    tail_hedge: TailHedgeState | None = None


class BacktestResult(BaseModel):
    """Summary of a completed backtest run.

    References
    ----------
    Sharpe, W. F. (1966). Mutual Fund Performance. Journal of Business,
    39(1), 119-138.
    Sharpe, W. F. (1994). The Sharpe Ratio. Journal of Portfolio
    Management, 21(1), 49-58.
    See REFERENCES.md.
    """

    start_date: date
    end_date: date
    portfolio_value_history: list[tuple[date, float]]
    final_sharpe_ratio: float
    final_max_drawdown: float
    final_sortino_ratio: float
    final_calmar_ratio: float
    final_information_ratio: float | None = None
    final_hit_rate: float
    final_profit_factor: float
    # Every day's decisions, or with `history_horizon_days` set only the
    # last days the engine held in memory (the current and the pending
    # day's); the run's checkpoint then holds every day's.
    decisions_by_day: dict[date, list[PortfolioDecision]] = Field(default_factory=dict)
    # The engine's history horizon in calendar days, `None` for a run that
    # kept its whole history in memory.
    history_horizon_days: int | None = None
    trade_log: list[TradeRecord] = Field(default_factory=list)
    # Decisions the engine skipped rather than executed, in fill order.
    unfilled_orders: list[UnfilledOrder] = Field(default_factory=list)
    # One entry per valuation day, in date order.
    capital_usage: list[CapitalUsage] = Field(default_factory=list)
    # Each pod's sizing diagnostics per valuation day, in date then pod order.
    sizing_diagnostics: list[SizingDiagnostics] = Field(default_factory=list)
    signal_health: dict[str, dict[str, int]] = Field(default_factory=dict)
    market_data_source: str | None = None
    source_performance: dict[str, dict[str, float]] = Field(default_factory=dict)
    # Cumulative commission paid across `trade_log`, and cumulative net
    # financing over the run: stock borrow on short notional plus margin
    # interest on negative cash and short-option margin, minus the
    # risk-free interest earned on positive cash (negative when the credit
    # is larger). Both are already reflected in portfolio value.
    total_transaction_cost: float = 0.0
    total_financing_cost: float = 0.0
    # Cumulative slippage, market impact and option spread cost across
    # `trade_log` (see `TradeRecord`), and `total_execution_cost`, their
    # sum with `total_transaction_cost`: every cost of filling at a price
    # other than the reference price, plus commission. All are already
    # reflected in portfolio value.
    total_slippage_cost: float = 0.0
    total_impact_cost: float = 0.0
    total_spread_cost: float = 0.0
    total_execution_cost: float = 0.0
    # Net premium outflow across every option buy/sell fill in `trade_log`
    # (premium paid minus premium received; expiry settlements excluded).
    total_option_premium_paid: float = 0.0
    # Each pod's genuine realized daily return (prior-day pod weights x
    # today's asset returns), keyed by manager id, over the whole run.
    pod_return_history: dict[str, list[tuple[date, float]]] = Field(default_factory=dict)
    # Per-pod return, vol, Sharpe and drawdown computed from `pod_return_history`.
    pod_performance: dict[str, PodPerformance] = Field(default_factory=dict)
    # Fill costs, option premium and turnover per pod id, plus a
    # `"commander"` entry for costs no pod's contribution change explains.
    # Each cost kind over every entry sums to its `total_*` field (commission
    # to `total_transaction_cost`), and premium to `total_option_premium_paid`.
    pod_costs: dict[str, PodCostAttribution] = Field(default_factory=dict)
    # Split-sized moves the run's lakehouse reads found with no adjustment
    # factor, one per (security, date), sorted by security then date.
    unfactored_split_moves: list[UnfactoredSplitMove] = Field(default_factory=list)
