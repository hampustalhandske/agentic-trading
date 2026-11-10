"""
# base.py
-----------
Contains the abstract base classes for all agents in the system, both analysts and managers.
"""

import copy
import io
import math
import pickle
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from datetime import date
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field, PrivateAttr

from agentic_trading.agents._market_math import wide_log_returns
from agentic_trading.agents._stable_value import qualified_name, stable_value
from agentic_trading.agents.portfolio_construction import PodView, PortfolioConstructor
from agentic_trading.agents.schemas import (
    AgentSignal,
    ManagerSignal,
    ManagerState,
    PortfolioDecision,
    SignalStatus,
    SizingDiagnostics,
    VolatilityEstimator,
)
from agentic_trading.config import settings
from agentic_trading.logging_config import get_logger
from agentic_trading.scoring.scoreboard import PerformanceScoreboard
from agentic_trading.scoring.weighting import WeightingStrategy

_DIRECTION_SIGN: dict[str, float] = {"long": 1.0, "short": -1.0, "flat": 0.0}

logger = get_logger(__name__)

# The safe fallback for any agent that fails to produce a real decision
SAFE_FALLBACK_DIRECTION: Literal["flat"] = "flat"
SAFE_FALLBACK_CONFIDENCE = 0.0


def _pod_information_coefficient(
    active: list[ManagerSignal],
    confidence_sum: float,
    shrunk_ic: Callable[[str], float],
) -> float | None:
    """Confidence-weighted mean of the contributing analysts' shrunk ICs,
    floored at 0.0 so a pod whose analysts have been reliably wrong
    carries no view rather than an inverted one; None with no
    contributing signals. Signals whose confidences sum to zero fall
    back to an unweighted mean."""
    if not active:
        return None
    if confidence_sum > 0.0:
        weighted = sum(signal.confidence * shrunk_ic(signal.source_agent_id) for signal in active)
        return max(0.0, weighted / confidence_sum)
    return max(0.0, sum(shrunk_ic(signal.source_agent_id) for signal in active) / len(active))


def _rebuilt_exception(cls: type[BaseException], args: tuple[Any, ...]) -> BaseException:
    """An instance of exception class `cls` with `args`, made without
    calling `cls.__init__`; its other attributes are set by unpickling."""
    exception = cls.__new__(cls, *args)
    exception.args = args
    return exception


class _CarriedStatePickler(pickle.Pickler):
    """Pickles exceptions by class, `args` and attributes rather than by
    calling the class with `args`, so an exception whose `__init__` takes
    more than its message (quantcore's `ConvergenceError`, which a
    `RefitCache` holds as a non-converged fit) unpickles. Quantcore gap:
    `ConvergenceError` defines no `__reduce__` of its own."""

    def reducer_override(self, obj: Any) -> Any:
        if isinstance(obj, BaseException):
            return _rebuilt_exception, (type(obj), obj.args), obj.__dict__ or None
        return NotImplemented


class Agent(BaseModel, ABC):
    """Shared base for every agent in the system — analysts and managers alike"""

    agent_id: str

    model_config = {"arbitrary_types_allowed": True}
    # Needed for concrete subclasses that will eventually hold non-Pydantic

    # `built_parameters()`, recorded once the instance is built.
    _built_parameters: dict[str, Any] = PrivateAttr(default_factory=dict)

    def model_post_init(self, context: Any, /) -> None:
        cls = type(self)
        self._built_parameters = {
            "agent_id": self.agent_id,
            "class": qualified_name(cls),
            "parameters": {name: stable_value(getattr(self, name)) for name in cls.model_fields},
        }

    def built_parameters(self) -> dict[str, Any]:
        """This agent as it was built, as JSON values: its `agent_id`, its
        class as `<module>.<qualname>` and every field (`stable_value`), a
        non-pydantic value such as a `PortfolioConstructor` by its class and
        public attributes. Read at construction, so later changes to the
        instance's fields or their contents do not reach it. A copy."""
        return copy.deepcopy(self._built_parameters)

    def carried_state(self) -> bytes:
        """An opaque snapshot of every pydantic private attribute this
        instance holds (`__pydantic_private__`, `None` for a class with
        none), as it stands now; later changes to the instance do not
        reach it. The bytes are a pickle: pass them only to
        `restore_carried_state`, and only from a checkpoint this program
        wrote, never from untrusted input."""
        buffer = io.BytesIO()
        _CarriedStatePickler(buffer, protocol=pickle.HIGHEST_PROTOCOL).dump(
            self.__pydantic_private__
        )
        return buffer.getvalue()

    def restore_carried_state(self, state: bytes) -> None:
        """Replaces this instance's private attributes with a fresh copy of
        those in `state`, a `carried_state()` snapshot of an instance of
        the same class. `state` is unpickled: it must come from a
        checkpoint this program wrote, never from untrusted input. Raises
        `ValueError` when the snapshot names other private attributes than
        this instance has."""
        private = pickle.loads(state)
        expected = set(self.__pydantic_private__ or {})
        if set(private or {}) != expected:
            raise ValueError(
                f"{self.agent_id}: carried state holds private attributes "
                f"{sorted(private or {})}, this instance holds {sorted(expected)}"
            )
        object.__setattr__(self, "__pydantic_private__", private)


# Bond/credit/index ETFs traded by the Rates/Credit/Tail-Hedge pods
# (agents/managers/rates_manager.py, credit_manager.py,
# tail_hedge_manager.py) — plain tickers with no `=`/`-`,
# so the naming-convention check in `is_likely_equity` alone can't catch
# them. Hardcoded here (rather than imported from those manager modules)
# to avoid a base.py <-> managers import cycle; small and stable enough
# not to need a shared registry.
_KNOWN_NON_EQUITY_TICKERS = frozenset({"SHY", "IEF", "TLT", "HYG", "LQD", "SPY"})


def is_likely_equity(ticker: str) -> bool:
    """True unless `ticker` uses one of this project's established
    non-equity ticker-naming conventions — a futures/FX suffix
    (`=F`/`=X`) or a crypto-pair suffix (`-USD` etc.) — or is one of the
    known bond/credit ETFs that don't follow that convention but also
    aren't an operating company with SEC financial-statement filings.
    Not a formal asset-class classification (see `data/ticker_catalog.py`
    for that, though its static catalog is far from exhaustive) — just
    enough to keep an equity-only analyst or `DataRequirement` (SEC
    filings, FINRA short interest) from firing on an asset class it was
    never designed for, on an unscoped pod that sees every asset in the
    run."""
    return (
        "=" not in ticker and "-" not in ticker and ticker.upper() not in _KNOWN_NON_EQUITY_TICKERS
    )


class DataRequirement(BaseModel):
    """One real-data dependency an analyst needs beyond `market_data`
    (guaranteed present via the backtest's own lakehouse-only price
    load — see `backtest/preflight.py`, which checks every declared
    requirement's real coverage against a run's tickers/date range
    before the simulation starts).

    `asset_scoped=False` means the requirement doesn't vary by ticker
    (e.g. one FRED series every asset's analyst call shares); `True`
    means it must be checked per run ticker, with `identifier` as a
    `"{ticker}"` template the preflight substitutes.

    `lookback_days` is how many calendar days before a decision date the
    analyst reads this source, so the preflight requires coverage from
    `start_date - lookback_days`; `0` means only data dated on or before
    each decision date itself. `None` means the analyst reads the source
    as far back as the run's price history, so coverage is required from
    the start of the run's warm-up."""

    source: Literal[
        "fred",
        "ken_french_factors",
        "sec_edgar_facts",
        "finra_short_interest",
        "sec_insider_transactions",
        "options_chain",
        "cftc_cot",
        "earnings",
        "dividends",
        "sp500_membership",
        "sec_sic",
    ]
    identifier: str
    asset_scoped: bool
    description: str
    lookback_days: int | None = Field(default=None, ge=0)


def calendar_days_for_sessions(sessions: int) -> int:
    """Calendar days spanned by `sessions` trading sessions (or business-day
    observations, such as a daily FRED series), at
    `settings.firm.trading_days_per_year` sessions per 365.25-day year,
    rounded up — the `lookback_days` of a requirement read as a trailing
    window of `sessions` observations."""
    return math.ceil(sessions * 365.25 / settings.firm.trading_days_per_year)


class AnalystAgent(Agent):
    """Layer-1 analyst agent scaffold, every concrete analyst implements this"""

    # Declarative real-data dependencies beyond `market_data`: every lake
    # source the analyst reads other than prices, which the backtest's own
    # price load already guarantees. Empty for an analyst that reads only
    # `market_data`.
    data_requirements: tuple[DataRequirement, ...] = ()

    def covers_asset(self, asset: str) -> bool:
        """Whether this instance produces a real signal for `asset`. The
        preflight checks an `asset_scoped` requirement only against run
        tickers for which this is True. True for every asset unless an
        analyst limits an instance to a set of assets and overrides it."""
        return True

    def _status_signal(
        self,
        status: SignalStatus,
        asset: str,
        as_of_date: pd.Timestamp,
        rationale: str,
        source_as_of: date | None,
        sample_count: int | None,
        volatility_estimator: VolatilityEstimator | None = None,
    ) -> AgentSignal:
        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction="flat",
            confidence=0.0,
            rationale=rationale,
            status=status,
            source_as_of=source_as_of,
            sample_count=sample_count,
            volatility_estimator=volatility_estimator,
        )

    def unavailable_signal(
        self,
        asset: str,
        as_of_date: pd.Timestamp,
        rationale: str,
        *,
        source_as_of: date | None = None,
        sample_count: int | None = None,
    ) -> AgentSignal:
        """A flat, 0-confidence signal with status "unavailable", for when the
        data the analyst needs is not there: no coverage, too little history,
        no bar. `source_as_of` and `sample_count` are the provenance of
        whatever data was read, when known."""
        return self._status_signal(
            "unavailable", asset, as_of_date, rationale, source_as_of, sample_count
        )

    def abstain_signal(
        self,
        asset: str,
        as_of_date: pd.Timestamp,
        rationale: str,
        *,
        source_as_of: date | None = None,
        sample_count: int | None = None,
        volatility_estimator: VolatilityEstimator | None = None,
    ) -> AgentSignal:
        """A flat, 0-confidence signal with status "active", for when the model
        has the data and declines to bet on it: no serial dependence, a
        degenerate or misspecified fit, a signal under its threshold.
        `source_as_of`, `sample_count` and `volatility_estimator` are the
        provenance of the data and volatility read the model declined on."""
        return self._status_signal(
            "active",
            asset,
            as_of_date,
            rationale,
            source_as_of,
            sample_count,
            volatility_estimator,
        )

    def error_signal(
        self,
        asset: str,
        as_of_date: pd.Timestamp,
        rationale: str,
        *,
        source_as_of: date | None = None,
        sample_count: int | None = None,
    ) -> AgentSignal:
        """A flat, 0-confidence signal with status "error", for when the model
        or a loader fails, a non-converged fit included. `source_as_of` and
        `sample_count` are the provenance of the data read before the
        failure, when known."""
        return self._status_signal(
            "error", asset, as_of_date, rationale, source_as_of, sample_count
        )

    @abstractmethod
    def generate_signal(
        self,
        asset: str,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> "AgentSignal":
        """Produce one directional view for one asset, using data available as of `as_of_date`"""


class ManagerAgent(Agent):
    """Layer-2 manager agent scaffold, every concrete manager implements this"""

    # The list of analysts this manager owns and scales for
    managed_analysts: list[AnalystAgent]
    # The weighting strategy this manager uses to scale each analyst's signal confidence
    weighting_strategy: WeightingStrategy
    # This pod's own risk-and-sizing engine (Bridgewater-style risk parity
    # + volatility targeting) — every manager owns and runs one
    # independently; the Commander never re-touches position sizing.
    portfolio_constructor: PortfolioConstructor
    # `None` (every existing equity pod): this manager acts on whatever
    # asset list the Commander passes it — today, always the whole
    # backtest's traded universe. Set to a fixed tuple for an
    # asset-class-scoped pod (e.g. a Rates pod that should only ever
    # price SHY/IEF/TLT even when the Commander's universe also includes
    # equities/FX/commodities) — `construct_portfolio` then restricts
    # itself to the intersection of that tuple and whatever assets are
    # actually present that day, and reports every other asset flat
    # (0.0 weight, no contributing signals) rather than pricing something
    # this pod was never built for.
    assets: tuple[str, ...] | None = None

    # Yesterday's sized weights, fed into portfolio_constructor.construct()
    # so it can compute this pod's own realized return (for the Kelly
    # overlay) and apply the no-trade buffer/cost-clearance logic. Same
    # one-instance-dict-tracking-yesterday's-weights shape Commander's own
    # _PodReturnTracker uses one layer up.
    _previous_weights: dict[str, float] = PrivateAttr(default_factory=dict)

    def collect_signals(
        self,
        asset: str,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> list["AgentSignal"]:
        """Gathers one AgentSignal from every analyst this manager owns.
        When `asset` has a missing close (NaN) on some row of a
        (ticker, field) `market_data`, each analyst sees only the rows on
        which `asset` has a close, every column kept, so no signal on
        `asset` rests on a row without its bar."""
        if (asset, "close") in market_data.columns:
            closes = market_data[(asset, "close")]
            if bool(closes.isna().any()):
                market_data = market_data.loc[closes.notna()]
        signals = []
        for analyst in self.managed_analysts:
            try:
                signals.append(analyst.generate_signal(asset, market_data, as_of_date))
            except Exception:
                logger.warning(
                    "Analyst %s failed to generate a signal for %s on %s — defaulting to "
                    "flat/0-confidence (hold) for this asset/day.",
                    analyst.agent_id,
                    asset,
                    as_of_date.date(),
                    exc_info=True,
                )
                signals.append(
                    AgentSignal(
                        agent_id=analyst.agent_id,
                        asset=asset,
                        as_of_date=as_of_date.date(),
                        direction=SAFE_FALLBACK_DIRECTION,
                        confidence=SAFE_FALLBACK_CONFIDENCE,
                        rationale="generate_signal() raised an exception — signal unavailable.",
                        status="error",
                    )
                )
        return signals

    @abstractmethod
    def scale_factor(
        self,
        signal: "AgentSignal",
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> float:
        """The multiplier this manager applies to one analyst's signal confidence"""

    def evaluate(
        self,
        signals: list["AgentSignal"],
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
        scoreboard: "PerformanceScoreboard",
    ) -> list["ManagerSignal"]:
        """Scales a batch of already-collected analyst signals (typically
        produced by a prior call to collect_signals) into ManagerSignals.

        Applies two distinct, multiplicative adjustments to each signal's
        confidence: scale_factor() (this manager's own domain judgment,
        implemented per concrete subclass) and this manager's
        weighting_strategy, evaluated against `scoreboard` to reflect each
        source analyst's trailing track record. Concrete managers
        implement scale_factor, not evaluate — this method is the same
        mechanical wiring for every manager type.
        """
        # WeightingStrategy/PerformanceScoreboard/TrackRecord are all keyed by
        # plain `date` (see scoring/), while `as_of_date` here is a
        # pd.Timestamp used for market-data slicing — normalize at this
        # boundary rather than leaking a Timestamp into date-typed comparisons.
        weights = self.weighting_strategy.get_weights(
            scoreboard, [signal.agent_id for signal in signals], as_of_date.date()
        )
        scaled = []
        for signal in signals:
            status = signal.status
            try:
                factor = self.scale_factor(signal, market_data, as_of_date)
            except Exception:
                logger.warning(
                    "Manager %s's scale_factor() failed for %s's signal on %s (%s) — "
                    "defaulting to a 0 factor, excluding this signal's contribution "
                    "(hold) rather than crashing the run.",
                    self.agent_id,
                    signal.agent_id,
                    signal.asset,
                    as_of_date.date(),
                    exc_info=True,
                )
                factor = 0.0
                status = "error"
            combined = factor * weights[signal.agent_id]
            scaled.append(
                ManagerSignal(
                    manager_id=self.agent_id,
                    source_agent_id=signal.agent_id,
                    asset=signal.asset,
                    as_of_date=signal.as_of_date,
                    direction=signal.direction,
                    confidence=min(signal.confidence * combined, 1.0),
                    raw_confidence=signal.confidence,
                    scale_applied=combined,
                    rationale=f"{signal.rationale} [scaled x{combined:.2f} by {self.agent_id}]",
                    status=status,
                    source_as_of=signal.source_as_of,
                    sample_count=signal.sample_count,
                    volatility_estimator=signal.volatility_estimator,
                )
            )
        return scaled

    def construct_portfolio(
        self,
        assets: list[str],
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
        scoreboard: "PerformanceScoreboard",
        *,
        carried: Mapping[str, list["ManagerSignal"]] | None = None,
    ) -> list["PortfolioDecision"]:
        """Turns this pod's own analysts' signals into a complete,
        risk-managed set of per-asset target weights — this manager's own
        portfolio, sized independently of every other pod via
        `self.portfolio_constructor` (covariance-aware, within its own
        leverage_cap). The Commander only ever combines already-sized pod
        portfolios by capital allocation (see `commander.py`); it never
        re-touches position sizing itself.

        Each asset's `PodView` fed to `portfolio_constructor` is this pod's
        own combined analyst opinion — direction_sign * confidence summed
        across every active, non-flat ManagerSignal for that asset, plus
        how many signals contributed, their total confidence, and the
        confidence-weighted mean of the contributing analysts' shrunk
        trailing information coefficients — left in signal units; the
        constructor converts it to a return view.

        If `self.assets` is set, only that subset of `assets` is actually
        priced — every other asset gets a flat 0.0-weight decision with
        no contributing signals, the same shape an insufficient-history
        day already produces, so an asset-class-scoped pod (see
        `self.assets`' own docstring) never runs its analysts against an
        instrument it was never built for.

        `carried` maps each asset of `assets` without a bar on `as_of_date`
        to the signals this pod's decision on it last carried. No analyst
        is asked about such an asset: those signals form its view, so the
        constructor sizes the other assets as it would with the asset
        present, and its target is this pod's previous sized weight for it
        (0 without one), carried forward into tomorrow's previous weights."""
        scoped_assets = (
            [asset for asset in assets if asset in self.assets]
            if self.assets is not None
            else assets
        )

        # Scoreboard/TrackRecord are keyed by plain `date` — same boundary
        # conversion as `evaluate`. One shrunk IC per analyst per day,
        # shared across every asset it signals on.
        scoring_date = as_of_date.date()
        analyst_ics: dict[str, float] = {}

        def shrunk_ic(analyst_id: str) -> float:
            if analyst_id not in analyst_ics:
                analyst_ics[analyst_id] = scoreboard.shrunk_information_coefficient(
                    analyst_id, scoring_date
                )
            return analyst_ics[analyst_id]

        views: dict[str, PodView] = {}
        signals_by_asset: dict[str, list[ManagerSignal]] = {}
        carried = carried or {}
        for asset in scoped_assets:
            if asset in carried:
                manager_signals = list(carried[asset])
            else:
                analyst_signals = self.collect_signals(asset, market_data, as_of_date)
                manager_signals = self.evaluate(
                    analyst_signals, market_data, as_of_date, scoreboard
                )
            signals_by_asset[asset] = manager_signals
            active = [
                signal
                for signal in manager_signals
                if signal.status == "active" and signal.direction != "flat"
            ]
            confidence_sum = sum(signal.confidence for signal in active)
            views[asset] = PodView(
                net_signal=sum(
                    _DIRECTION_SIGN[signal.direction] * signal.confidence for signal in active
                ),
                signal_count=len(active),
                confidence_sum=confidence_sum,
                information_coefficient=_pod_information_coefficient(
                    active, confidence_sum, shrunk_ic
                ),
            )

        returns_history = wide_log_returns(scoped_assets, market_data)
        self.portfolio_constructor.manager_id = self.agent_id
        sized_weights = self.portfolio_constructor.construct(
            views, returns_history, as_of_date, self._previous_weights
        )
        for asset in scoped_assets:
            if asset in carried:
                sized_weights[asset] = self._previous_weights.get(asset, 0.0)
        self._previous_weights = dict(sized_weights)

        return [
            PortfolioDecision(
                asset=asset,
                as_of_date=as_of_date.date(),
                target_weight=sized_weights.get(asset, 0.0),
                contributing_signals=signals_by_asset.get(asset, []),
            )
            for asset in assets
        ]

    def last_sizing_diagnostics(self) -> SizingDiagnostics | None:
        """The per-stage volatility diagnostics of this pod's most recent
        `portfolio_constructor.construct()` call, with `manager_id` set to
        this manager's `agent_id`; `None` before any call or when the
        constructor recorded none. Describes the equity book the constructor
        sized, before any option rows a subclass adds."""
        diagnostics = self.portfolio_constructor.last_diagnostics()
        if diagnostics is None:
            return None
        return diagnostics.model_copy(update={"manager_id": self.agent_id})

    def manager_state(self) -> ManagerState:
        """A copy of what this manager carries between days: yesterday's
        sized weights and its portfolio constructor's state, with
        `tail_hedge` `None`."""
        return ManagerState(
            previous_weights=dict(self._previous_weights),
            constructor=self.portfolio_constructor.state(),
        )

    def restore_manager_state(self, state: ManagerState) -> None:
        """Replaces this manager's carried state with a copy of `state`.
        Raises `ValueError` when `state` carries tail-hedge state, which
        only the tail-hedge manager holds."""
        if state.tail_hedge is not None:
            raise ValueError(f"{self.agent_id} holds no tail-hedge state")
        self._restore_shared_state(state)

    def _restore_shared_state(self, state: ManagerState) -> None:
        """Restores the state every manager carries: yesterday's sized
        weights and the constructor's state."""
        self._previous_weights = dict(state.previous_weights)
        self.portfolio_constructor.restore_state(state.constructor)
