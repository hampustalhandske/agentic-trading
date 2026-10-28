# commander.py

from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from datetime import date
from typing import Literal, cast

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from quantcore.portfolio.covariance import ledoit_wolf_shrinkage
from quantcore.portfolio.optimization import risk_parity_weights
from quantcore.risk.var import value_at_risk

from agentic_trading.agents._market_math import (
    raw_spot,
    single_asset_from_attrs,
    wide_log_returns,
)
from agentic_trading.agents.base import ManagerAgent
from agentic_trading.agents.schemas import (
    MAX_LEVERAGE,
    ManagerSignal,
    OptionContract,
    PortfolioDecision,
)
from agentic_trading.config import settings
from agentic_trading.data.loaders import option_marks
from agentic_trading.logging_config import get_logger
from agentic_trading.scoring.scoreboard import PerformanceScoreboard
from agentic_trading.scoring.weighting import WeightingStrategy

logger = get_logger(__name__)

# Every numeric knob (pod-return lookback, risk-budget observation floor,
# vol floors, drawdown breaker thresholds, tail-hedge budget) lives in
# `config.settings.commander`, with the reasoning for each value there.

# The agent_id the tail-hedge pod's manager is registered under in
# roster.py — must match exactly, since this constant is how Commander
# recognizes which pod gets the fixed insurance budget below rather than
# a risk-parity-sized share.
TAIL_HEDGE_MANAGER_ID = "tail_hedge_manager"


class CommanderRiskState(BaseModel):
    """Everything stateful a `Commander` carries between simulated days
    that isn't recoverable from `market_data` alone: which pods the
    drawdown breaker currently has cut, each pod's trailing realized
    return series, and each pod's most recent target-weight vector (what
    tomorrow's pod return is measured against). Serializable through
    `model_dump(mode="json")`/`model_validate` so a run can be
    checkpointed and resumed with `Commander.restore_risk_state`."""

    tripped_pods: list[str] = Field(default_factory=list)
    pod_return_history: dict[str, list[tuple[date, float]]] = Field(default_factory=dict)
    last_pod_weights: dict[str, dict[str, float]] = Field(default_factory=dict)


def _construct_pod_portfolio(
    manager: ManagerAgent,
    assets: list[str],
    market_data: pd.DataFrame,
    as_of_date: pd.Timestamp,
    scoreboard: PerformanceScoreboard,
) -> tuple[str, list[PortfolioDecision]]:
    return manager.agent_id, manager.construct_portfolio(
        assets, market_data, as_of_date, scoreboard
    )


def _today_log_returns(market_data: pd.DataFrame, as_of_date: pd.Timestamp) -> dict[str, float]:
    """The realized log return for every asset on `as_of_date`, read
    straight from `market_data` (already point-in-time sliced up to and
    including this day) — the real, tradeable return `_PodReturnTracker`
    weights by each pod's own prior-day position, not an attribution
    score."""
    if isinstance(market_data.columns, pd.MultiIndex):
        assets = list(dict.fromkeys(market_data.columns.get_level_values(0)))
        returns = {}
        for asset in assets:
            value = market_data.loc[as_of_date, (asset, "log_return")]
            returns[asset] = 0.0 if pd.isna(value) else float(cast(float, value))
        return returns
    asset = single_asset_from_attrs(market_data)
    value = market_data.loc[as_of_date, "log_return"]
    return {asset: 0.0 if pd.isna(value) else float(cast(float, value))}


class _PodReturnTracker:
    """Maintains a real, weighted daily return series per pod — not an
    attribution score. Each call to `record_and_get_history` uses
    *yesterday's* stored pod weight vector (each pod's own combined
    per-asset target weight from the previous simulated day) against
    *today's* actual realized asset return to compute one genuine return
    per pod: `sum(prev_weight[asset] * actual_return_today[asset])`. This
    is dimensionally an actual return, not a confidence-scaled skill
    score, so a trailing-vol estimate built from it — "how volatile has
    this pod's book really been" — means what it looks like it means.

    The first call for any pod has no prior weight vector yet, so it
    records nothing for that pod until the second call onward — one
    day of missing history at the very start of a run, not an ongoing
    gap."""

    def __init__(self, lookback_days: int) -> None:
        self.lookback_days = lookback_days
        self._last_weights: dict[str, dict[str, float]] = {}
        self._history: dict[str, list[tuple[date, float]]] = {}

    @property
    def history(self) -> dict[str, list[tuple[date, float]]]:
        return {manager_id: list(entries) for manager_id, entries in self._history.items()}

    @property
    def last_weights(self) -> dict[str, dict[str, float]]:
        return {manager_id: dict(weights) for manager_id, weights in self._last_weights.items()}

    def restore(
        self,
        history: dict[str, list[tuple[date, float]]],
        last_weights: dict[str, dict[str, float]],
    ) -> None:
        self._history = {manager_id: list(entries) for manager_id, entries in history.items()}
        self._last_weights = {
            manager_id: dict(weights) for manager_id, weights in last_weights.items()
        }

    def record_and_get_history(
        self,
        pod_decisions: list[tuple[str, list[PortfolioDecision]]],
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
        instrument_returns: dict[str, float] | None = None,
    ) -> dict[str, pd.Series]:
        today_returns = _today_log_returns(market_data, as_of_date)
        # Option contracts are not in `market_data`; the engine supplies
        # their mark-to-mark log returns keyed by contract id.
        if instrument_returns:
            today_returns.update(instrument_returns)
        today = as_of_date.date()
        cutoff = today - pd.Timedelta(days=self.lookback_days)

        for manager_id, prev_weights in self._last_weights.items():
            if not prev_weights:
                # This pod produced zero decisions yesterday (e.g. an
                # asset-class-scoped pod outside today's traded universe,
                # or every analyst still below its own minimum-history
                # bar) — skip, rather than recording a fabricated 0.0
                # "return" that would understate this pod's real
                # trailing volatility once it does start trading.
                continue
            realized = sum(
                weight * today_returns.get(asset, 0.0) for asset, weight in prev_weights.items()
            )
            series = self._history.setdefault(manager_id, [])
            series.append((today, realized))
            self._history[manager_id] = [(d, r) for d, r in series if d >= cutoff]

        self._last_weights = {
            manager_id: {decision.asset: decision.target_weight for decision in decisions}
            for manager_id, decisions in pod_decisions
        }

        return {
            manager_id: pd.Series(
                [r for _, r in entries], index=[d for d, _ in entries], dtype=float
            )
            for manager_id, entries in self._history.items()
        }


def _cross_pod_risk_parity_weights(
    pod_return_history: dict[str, pd.Series], qualifying: set[str]
) -> dict[str, float] | None:
    """Real cross-pod risk parity: builds the qualifying pods' actual
    return-covariance matrix (from their genuine realized daily returns,
    not a fabricated attribution score — see `_PodReturnTracker`),
    shrinks it with `ledoit_wolf_shrinkage`, and sizes with
    `risk_parity_weights` — the same two functions each pod's own
    `PortfolioConstructor` already uses one layer down, reused here
    *across* pods. A pod whose overlapping returns do not vary at all
    (every day the same, e.g. a pod holding nothing) has no measurable
    risk: it is left out, with a log line, and has no key in the result.
    Returns `None` (caller falls back to the simpler per-pod inverse-vol
    mechanism) if there are fewer than 2 qualifying pods, if there isn't
    strictly more overlapping history (`T`) than qualifying pods (`k`) —
    `ledoit_wolf_shrinkage`'s own `T > k` requirement, same threshold
    shape as `PortfolioConstructor`'s `MIN_OBSERVATIONS_FOR_COVARIANCE`
    check one layer down — or if fewer than 2 pods with varying returns
    remain."""
    if len(qualifying) < 2:
        return None
    overlapping = pd.DataFrame({m: pod_return_history[m] for m in qualifying}).dropna()
    if len(overlapping) <= len(qualifying):
        return None
    flat_pods = [
        manager_id
        for manager_id in overlapping.columns
        if bool(np.all(overlapping[manager_id] == overlapping[manager_id].iloc[0]))
    ]
    if flat_pods:
        logger.info(
            "Commander: %s have no return variation over %d overlapping day(s) — left out "
            "of cross-pod risk parity.",
            flat_pods,
            len(overlapping),
        )
        overlapping = overlapping.drop(columns=flat_pods)
        if overlapping.shape[1] < 2:
            return None
    cov_matrix = ledoit_wolf_shrinkage(overlapping.to_numpy(dtype=np.float64))
    weights = risk_parity_weights(cov_matrix)
    return dict(zip(overlapping.columns, weights.tolist(), strict=True))


def _risk_budget_weights(
    pod_return_history: dict[str, pd.Series],
    manager_ids: list[str],
    scoreboard: PerformanceScoreboard,
    as_of: date,
    weighting_strategy: WeightingStrategy,
) -> dict[str, float]:
    """Allocates capital across pods, tilted by trailing Sharpe
    (`weighting_strategy`) in every case. Once at least 2 pods have
    enough overlapping trailing-return history for a real covariance
    matrix (`_cross_pod_risk_parity_weights`), those pods are sized by
    genuine cross-pod risk parity — equalizing each pod's *contribution*
    to combined risk, not just its own standalone vol — instead of
    treating every pod's vol as independent. Below that threshold (early
    in a run, or fewer than 2 pods with enough history), falls back to
    per-pod inverse-vol sizing with a peer-relative vol floor, the same
    graceful degradation `PortfolioConstructor._bootstrap_weights_from_views`
    already uses one layer down. A pod with no trailing history at all
    yet gets `weighting_strategy`'s own trailing-performance weight
    directly, in both cases — a per-pod bootstrap, not an all-or-nothing
    gate. A qualifying pod that risk parity leaves out for having no
    return variation is sized as an average risk-parity pod: its
    trailing-performance weight times `1 / (pods in risk parity)`. Every
    pod that produces any real signal at all ends
    up with a nonzero share; the whole allocation is normalized together so gross
    weight still sums to 1.0. Returns `{}` only if every pod's weight
    comes out exactly zero (e.g. a fully untested `weighting_strategy`),
    the signal `Commander.decide()` uses as a last-resort equal-weight
    fallback."""
    bootstrap = weighting_strategy.get_weights(scoreboard, manager_ids, as_of)
    qualifying = {
        manager_id
        for manager_id in manager_ids
        if len(pod_return_history.get(manager_id, pd.Series(dtype=float)))
        >= settings.commander.min_observations_for_risk_budget
    }

    risk_parity = _cross_pod_risk_parity_weights(pod_return_history, qualifying)
    if risk_parity is not None:
        average_share = 1.0 / len(risk_parity)
        raw_weights = {
            manager_id: (
                risk_parity.get(manager_id, average_share) * bootstrap.get(manager_id, 0.0)
                if manager_id in qualifying
                else bootstrap.get(manager_id, 0.0)
            )
            for manager_id in manager_ids
        }
    else:
        trailing_vols = {
            manager_id: float(np.std(pod_return_history[manager_id], ddof=1))
            for manager_id in qualifying
        }
        peer_vol_floor = (
            max(
                float(np.median(list(trailing_vols.values())))
                * settings.commander.min_relative_vol_floor,
                settings.commander.min_trailing_vol,
            )
            if trailing_vols
            else settings.commander.min_trailing_vol
        )
        raw_weights = {
            manager_id: (
                bootstrap.get(manager_id, 0.0) / max(trailing_vols[manager_id], peer_vol_floor)
                if manager_id in qualifying
                else bootstrap.get(manager_id, 0.0)
            )
            for manager_id in manager_ids
        }

    gross = sum(abs(weight) for weight in raw_weights.values())
    if gross == 0:
        return {}
    return {manager_id: weight / gross for manager_id, weight in raw_weights.items()}


def _current_drawdown(series: pd.Series) -> float:
    """The pod's current drawdown from its own trailing peak, in the
    same equity/running-peak/drawdown convention
    `PerformanceScoreboard.max_drawdown` uses (always <= 0) — evaluated
    at the series' most recent point, not the worst point over the whole
    window, since the breaker's hysteresis needs the pod's state *right
    now*."""
    equity = series.cumsum()
    running_peak = equity.cummax()
    return float((equity - running_peak).iloc[-1])


class _DrawdownBreaker:
    """A per-pod circuit breaker with hysteresis: a pod whose current
    drawdown breaches `trip_threshold` has its Commander-level allocation
    cut to `cut_multiplier` (never fully zeroed — a permanent zero-capital
    cut is a death spiral a pod could never earn its way out of) and
    stays cut until its drawdown claws back inside `recovery_threshold`
    (not merely back above `trip_threshold`), preventing flapping right
    at the line. Stateful across calls — one instance persists for the
    life of a `Commander`."""

    def __init__(
        self,
        trip_threshold: float | None = None,
        recovery_threshold: float | None = None,
        cut_multiplier: float | None = None,
    ) -> None:
        breaker = settings.commander
        trip_threshold = (
            breaker.drawdown_trip_threshold if trip_threshold is None else trip_threshold
        )
        recovery_threshold = (
            breaker.drawdown_recovery_threshold
            if recovery_threshold is None
            else recovery_threshold
        )
        cut_multiplier = (
            breaker.drawdown_cut_multiplier if cut_multiplier is None else cut_multiplier
        )
        if not trip_threshold < recovery_threshold <= 0.0:
            raise ValueError("trip_threshold must be < recovery_threshold <= 0.0")
        if not 0.0 < cut_multiplier < 1.0:
            raise ValueError("cut_multiplier must be within (0, 1)")
        self.trip_threshold = trip_threshold
        self.recovery_threshold = recovery_threshold
        self.cut_multiplier = cut_multiplier
        self._tripped: set[str] = set()

    @property
    def tripped(self) -> set[str]:
        return set(self._tripped)

    def restore(self, tripped: set[str]) -> None:
        self._tripped = set(tripped)

    def multipliers(
        self, pod_return_history: dict[str, pd.Series], manager_ids: list[str]
    ) -> dict[str, float]:
        result: dict[str, float] = {}
        for manager_id in manager_ids:
            series = pod_return_history.get(manager_id, pd.Series(dtype=float))
            drawdown = _current_drawdown(series) if len(series) else 0.0
            if manager_id in self._tripped:
                if drawdown >= self.recovery_threshold:
                    self._tripped.discard(manager_id)
                    result[manager_id] = 1.0
                else:
                    result[manager_id] = self.cut_multiplier
            elif drawdown <= self.trip_threshold:
                self._tripped.add(manager_id)
                result[manager_id] = self.cut_multiplier
            else:
                result[manager_id] = 1.0
        return result


def _apply_firm_limits(
    weights: dict[str, float], market_data: pd.DataFrame, as_of_date: pd.Timestamp
) -> dict[str, float]:
    """The firm-level risk line on a combined per-asset book, applied in
    order: (a) each |weight| is clipped to `settings.firm.max_single_name_weight`;
    (b) if gross exposure exceeds `settings.firm.max_gross_exposure` the
    whole book is scaled down proportionally; (c) if the book's one-day
    parametric VaR at `settings.firm.var_confidence_level` — from a
    Ledoit-Wolf covariance of the trailing `settings.firm.var_lookback_days` daily
    log returns of the traded assets — exceeds `settings.firm.max_daily_var`
    the whole book is scaled down by `max_daily_var / VaR`; (d) if the
    book's beta to SPY (the weighted sum of each traded asset's own beta,
    computed from that same trailing returns/covariance) exceeds
    `settings.firm.max_portfolio_beta` in absolute value, the whole book
    is scaled down by `max_portfolio_beta / |beta|`; (e) a correlation-
    breakdown stress test rebuilds the covariance with every off-diagonal
    correlation pulled toward `settings.firm.stress_correlation_shock`
    (never away from it) and every variance scaled by
    `settings.firm.stress_vol_multiplier ** 2`, and if the resulting
    one-day parametric loss exceeds `settings.firm.max_stress_loss` the
    whole book is scaled down by `max_stress_loss / stressed_loss`. Steps
    (c)-(e) are all skipped together when there are fewer than
    `settings.portfolio.min_observations_for_covariance + n_assets`
    observations, since a covariance from less history than that isn't a
    risk estimate worth acting on. Step (d) is additionally skipped on
    its own, independent of history depth, whenever `market_data` carries
    no SPY close/log_return series in this run's frame at all. Every
    binding limit is logged."""
    firm = settings.firm
    limited = {
        asset: max(min(weight, firm.max_single_name_weight), -firm.max_single_name_weight)
        for asset, weight in weights.items()
    }
    if any(abs(limited[asset] - weights[asset]) > 1e-12 for asset in weights):
        clipped = sorted(asset for asset in weights if abs(limited[asset] - weights[asset]) > 1e-12)
        logger.info(
            "%s: firm single-name limit %.3f binds on %s",
            as_of_date.date(),
            firm.max_single_name_weight,
            ", ".join(clipped),
        )

    gross = sum(abs(weight) for weight in limited.values())
    if gross > firm.max_gross_exposure:
        scale = firm.max_gross_exposure / gross
        limited = {asset: weight * scale for asset, weight in limited.items()}
        logger.info(
            "%s: firm gross limit %.3f binds (gross %.3f, scaled by %.4f)",
            as_of_date.date(),
            firm.max_gross_exposure,
            gross,
            scale,
        )

    traded = [asset for asset, weight in limited.items() if weight != 0.0]
    if not traded:
        return limited
    returns = wide_log_returns(traded, market_data)
    returns = returns.loc[returns.index <= as_of_date].tail(firm.var_lookback_days)
    min_observations = settings.portfolio.min_observations_for_covariance + len(traded)
    if len(returns) < min_observations:
        logger.debug(
            "%s: firm VaR limit skipped, %d observations < %d required",
            as_of_date.date(),
            len(returns),
            min_observations,
        )
        return limited
    weight_vector = np.array([limited[asset] for asset in traded], dtype=np.float64)
    returns_matrix = returns.to_numpy(dtype=np.float64)
    cov_matrix = ledoit_wolf_shrinkage(returns_matrix)
    portfolio_std = float(np.sqrt(max(weight_vector @ cov_matrix @ weight_vector, 0.0)))
    var = value_at_risk(
        returns_matrix @ weight_vector,
        firm.var_confidence_level,
        mean=0.0,
        volatility=portfolio_std,
    )
    if var > firm.max_daily_var:
        scale = firm.max_daily_var / var
        limited = {asset: weight * scale for asset, weight in limited.items()}
        logger.info(
            "%s: firm VaR limit %.4f binds (one-day %.0f%% VaR %.4f, scaled by %.4f)",
            as_of_date.date(),
            firm.max_daily_var,
            firm.var_confidence_level * 100,
            var,
            scale,
        )

    has_spy = isinstance(market_data.columns, pd.MultiIndex) and "SPY" in set(
        market_data.columns.get_level_values(0)
    )
    if not has_spy:
        logger.debug(
            "%s: firm beta limit skipped, no SPY series in this run's frame", as_of_date.date()
        )
    else:
        if "SPY" in traded:
            spy_returns = returns["SPY"]
            beta_returns = returns
        else:
            spy_returns = wide_log_returns(["SPY"], market_data)["SPY"].reindex(returns.index)
            beta_returns = returns.assign(SPY=spy_returns).dropna()
        min_beta_observations = settings.portfolio.min_observations_for_covariance + len(
            beta_returns.columns
        )
        if len(beta_returns) < min_beta_observations:
            logger.debug(
                "%s: firm beta limit skipped, %d observations < %d required",
                as_of_date.date(),
                len(beta_returns),
                min_beta_observations,
            )
        else:
            beta_cov = (
                cov_matrix
                if "SPY" in traded
                else ledoit_wolf_shrinkage(beta_returns.to_numpy(dtype=np.float64))
            )
            spy_index = beta_returns.columns.get_loc("SPY")
            spy_variance = float(beta_cov[spy_index, spy_index])
            asset_betas = {
                asset: float(beta_cov[beta_returns.columns.get_loc(asset), spy_index])
                / spy_variance
                for asset in traded
            }
            book_beta = sum(limited[asset] * asset_betas[asset] for asset in traded)
            if abs(book_beta) > firm.max_portfolio_beta:
                scale = firm.max_portfolio_beta / abs(book_beta)
                limited = {asset: weight * scale for asset, weight in limited.items()}
                logger.info(
                    "%s: firm beta limit %.3f binds (book beta %.4f, scaled by %.4f)",
                    as_of_date.date(),
                    firm.max_portfolio_beta,
                    book_beta,
                    scale,
                )

    variances = np.diag(cov_matrix)
    # A zero-variance asset has zero stressed variance, so its correlations
    # never enter the stressed covariance; they are 0 rather than 0 / 0.
    volatility_products = np.sqrt(np.outer(variances, variances))
    corr_matrix = np.divide(
        cov_matrix,
        volatility_products,
        out=np.zeros_like(cov_matrix),
        where=volatility_products > 0.0,
    )
    shocked_corr = np.sign(corr_matrix) * np.maximum(
        np.abs(corr_matrix), firm.stress_correlation_shock
    )
    np.fill_diagonal(shocked_corr, 1.0)
    stressed_variances = variances * firm.stress_vol_multiplier**2
    stressed_cov = shocked_corr * np.sqrt(np.outer(stressed_variances, stressed_variances))
    np.fill_diagonal(stressed_cov, stressed_variances)
    stressed_weight_vector = np.array([limited[asset] for asset in traded], dtype=np.float64)
    stressed_portfolio_std = float(
        np.sqrt(max(stressed_weight_vector @ stressed_cov @ stressed_weight_vector, 0.0))
    )
    stressed_loss = value_at_risk(
        returns_matrix @ stressed_weight_vector,
        firm.var_confidence_level,
        mean=0.0,
        volatility=stressed_portfolio_std,
    )
    if stressed_loss > firm.max_stress_loss:
        scale = firm.max_stress_loss / stressed_loss
        limited = {asset: weight * scale for asset, weight in limited.items()}
        logger.info(
            "%s: firm stress limit %.3f binds (stressed one-day loss %.4f, scaled by %.4f)",
            as_of_date.date(),
            firm.max_stress_loss,
            stressed_loss,
            scale,
        )
    else:
        logger.debug(
            "%s: firm stress test one-day loss %.4f (cap %.3f)",
            as_of_date.date(),
            stressed_loss,
            firm.max_stress_loss,
        )
    return limited


def _delta_equivalent_book(
    equity_weight: dict[str, float],
    option_rows: dict[str, tuple[OptionContract, float]],
    market_data: pd.DataFrame,
    as_of_date: pd.Timestamp,
) -> dict[str, float]:
    """The combined book in equity-equivalent weights: each option row's
    premium weight contributes `weight x delta x spot / mid` (its delta
    exposure as a fraction of NAV) to its underlying, or the premium
    weight itself when no mark or spot is available. An option on an
    underlying outside `equity_weight` has no equity row to load and is
    left out."""
    book = dict(equity_weight)
    for contract, weight in option_rows.values():
        underlying = contract.underlying
        if underlying not in book:
            logger.debug(
                "%s: %s's underlying is outside the traded universe — not in the firm book",
                as_of_date.date(),
                contract.contract_id,
            )
            continue
        spot = raw_spot(market_data, as_of_date, underlying)
        mark = (
            option_marks.load_option_mark(contract, as_of_date.date(), spot)
            if spot is not None and spot > 0.0
            else None
        )
        if mark is None or mark.mid <= 0.0 or spot is None:
            book[underlying] += weight
        else:
            book[underlying] += weight * mark.delta * spot / mark.mid
    return book


def _apply_short_premium_cap(
    option_rows: dict[str, tuple[OptionContract, float]], as_of_date: pd.Timestamp
) -> dict[str, tuple[OptionContract, float]]:
    """Scales every short option row (negative premium weight) down
    proportionally so their total premium stays within
    `settings.options.max_short_premium_fraction` of NAV."""
    cap = settings.options.max_short_premium_fraction
    short_premium = sum(-weight for _, weight in option_rows.values() if weight < 0.0)
    if short_premium <= cap:
        return option_rows
    scale = cap / short_premium
    logger.info(
        "%s: firm short-premium cap %.3f binds (short premium %.3f, scaled by %.4f)",
        as_of_date.date(),
        cap,
        short_premium,
        scale,
    )
    return {
        contract_id: (contract, weight * scale if weight < 0.0 else weight)
        for contract_id, (contract, weight) in option_rows.items()
    }


class Commander:
    """Layer 3, pure capital allocator across pods — it never sizes a
    single position itself. Each manager (pod) already produced its own
    complete, risk-managed portfolio via `ManagerAgent.construct_portfolio()`
    (covariance-aware, within its own leverage_cap, using Black-Litterman
    correctly since a pod's own book is real assets with a real
    market-cap-style equilibrium); the Commander's job is to decide how
    much of the fund's capital each pod gets.

    That allocation is plain risk-budgeting, not a second Black-Litterman
    layer: `_PodReturnTracker` turns each pod's own prior-day weight
    vector plus today's real realized asset returns into a genuine daily
    return series per pod, and `_risk_budget_weights` sizes each pod
    inversely to its own trailing volatility, tilted by trailing Sharpe
    (via `self.weighting_strategy` — the same abstraction analysts are
    weighted by within a pod, reused one level up rather than
    reinvented). Pods aren't tradable assets and have no real
    market-cap-weighted equilibrium a Black-Litterman prior could mean
    anything against, so this allocator never builds one.

    Any pod still below `settings.commander.min_observations_for_risk_budget` trailing
    observations (early in a run, or simply untraded so far) is sized
    off `self.weighting_strategy`'s trailing-performance weight
    directly rather than a vol-scaled one — a per-pod bootstrap, not an
    all-or-nothing gate; see `_risk_budget_weights`. `decide()` only
    falls back to an unweighted equal split across every pod in the
    rare case that comes out exactly zero for all of them."""

    def __init__(
        self,
        managers: list["ManagerAgent"],
        weighting_strategy: "WeightingStrategy",
        pod_returns_lookback_days: int | None = None,
        parallel_workers: int = 1,
        parallel_backend: Literal["thread", "process"] = "thread",
        drawdown_breaker: "_DrawdownBreaker | None" = None,
    ) -> None:
        if parallel_workers < 1:
            raise ValueError("parallel_workers must be at least 1")
        if pod_returns_lookback_days is None:
            pod_returns_lookback_days = settings.commander.pod_returns_lookback_days
        self.managers = managers
        self.weighting_strategy = weighting_strategy
        self.pod_returns_lookback_days = pod_returns_lookback_days
        self.parallel_workers = parallel_workers
        self.parallel_backend = parallel_backend
        self._executor: Executor | None = None
        self._pod_return_tracker = _PodReturnTracker(pod_returns_lookback_days)
        self.drawdown_breaker = (
            drawdown_breaker if drawdown_breaker is not None else _DrawdownBreaker()
        )

    def _get_executor(self) -> Executor | None:
        if self.parallel_workers == 1:
            return None
        if self._executor is None:
            executor_type = (
                ProcessPoolExecutor if self.parallel_backend == "process" else ThreadPoolExecutor
            )
            self._executor = executor_type(max_workers=self.parallel_workers)
        return self._executor

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def risk_state(self) -> CommanderRiskState:
        return CommanderRiskState(
            tripped_pods=sorted(self.drawdown_breaker.tripped),
            pod_return_history=self._pod_return_tracker.history,
            last_pod_weights=self._pod_return_tracker.last_weights,
        )

    def restore_risk_state(self, state: CommanderRiskState) -> None:
        self.drawdown_breaker.restore(set(state.tripped_pods))
        self._pod_return_tracker.restore(state.pod_return_history, state.last_pod_weights)

    def decide(
        self,
        market_data: "pd.DataFrame",
        as_of_date: "pd.Timestamp",
        scoreboard: "PerformanceScoreboard",
        instrument_returns: dict[str, float] | None = None,
    ) -> list["PortfolioDecision"]:
        """Produces one PortfolioDecision per asset covered by this
        Commander's managers, for `as_of_date`: each manager builds its
        own complete pod portfolio independently (in parallel, across the
        configured worker pool), then this method allocates capital
        across pods and sums their weighted decisions — never touching an
        individual position's sizing math itself. Every `ManagerSignal`
        that fed into any pod's decision for an asset is carried forward
        onto the combined decision, so the result stays fully auditable.
        """
        if market_data.empty or not self.managers:
            return []

        if isinstance(market_data.columns, pd.MultiIndex):
            assets = list(dict.fromkeys(market_data.columns.get_level_values(0)))
        else:
            assets = [single_asset_from_attrs(market_data)]

        manager_ids = [manager.agent_id for manager in self.managers]
        # See ManagerAgent.evaluate()'s comment: scoring/ is keyed by plain
        # `date`, so normalize the pd.Timestamp before it reaches that layer.
        as_of = as_of_date.date()

        executor = self._get_executor()
        if executor is None:
            pod_decisions = [
                (
                    manager.agent_id,
                    manager.construct_portfolio(assets, market_data, as_of_date, scoreboard),
                )
                for manager in self.managers
            ]
        else:
            futures = [
                executor.submit(
                    _construct_pod_portfolio, manager, assets, market_data, as_of_date, scoreboard
                )
                for manager in self.managers
            ]
            pod_decisions = [future.result() for future in futures]

        pod_return_history = self._pod_return_tracker.record_and_get_history(
            pod_decisions, market_data, as_of_date, instrument_returns
        )

        # The tail-hedge pod (if present) gets a fixed, protected budget
        # carved out before the risk-parity allocation runs — see
        # TAIL_HEDGE_BUDGET_FRACTION's own comment. The other pods are
        # allocated exactly as before, then renormalized to what's left.
        has_tail_hedge = TAIL_HEDGE_MANAGER_ID in manager_ids
        risk_budget_ids = (
            [manager_id for manager_id in manager_ids if manager_id != TAIL_HEDGE_MANAGER_ID]
            if has_tail_hedge
            else manager_ids
        )
        pod_allocations = _risk_budget_weights(
            pod_return_history, risk_budget_ids, scoreboard, as_of, self.weighting_strategy
        )
        if not pod_allocations and risk_budget_ids:
            # Every pod's weight came out exactly zero (e.g. a
            # weighting_strategy with nothing to go on yet for any pod)
            # — true last resort, an unweighted equal split, rather than
            # allocating no capital anywhere.
            pod_allocations = {
                manager_id: 1.0 / len(risk_budget_ids) for manager_id in risk_budget_ids
            }
        if has_tail_hedge:
            remaining_gross = 1.0 - settings.commander.tail_hedge_budget_fraction
            pod_allocations = {
                manager_id: weight * remaining_gross
                for manager_id, weight in pod_allocations.items()
            }
            pod_allocations[TAIL_HEDGE_MANAGER_ID] = settings.commander.tail_hedge_budget_fraction

        # Drawdown breaker: cuts a breached pod's allocation, not
        # renormalized back to gross 1.0 afterward (the freed-up capital
        # isn't reshuffled onto the rest of the book; total deployed
        # gross just shrinks, which is itself the desired de-risking).
        # The tail-hedge pod's fixed budget is exempt — a hedge's
        # funding must not depend on its own trailing P&L.
        breaker_ids = [
            manager_id for manager_id in manager_ids if manager_id != TAIL_HEDGE_MANAGER_ID
        ]
        for manager_id, multiplier in self.drawdown_breaker.multipliers(
            pod_return_history, breaker_ids
        ).items():
            pod_allocations[manager_id] = pod_allocations.get(manager_id, 0.0) * multiplier

        # The tail-hedge pod's book is kept aside while the firm limits
        # run on everything else, then added back: its fixed insurance
        # budget must not be scaled away by a gross/VaR breach the rest
        # of the book caused. Option decisions stay their own rows, keyed
        # by contract id. A core pod's option row is scaled by the pod's
        # allocation like its equity rows; the tail-hedge pod sizes its
        # put premium against firm NAV from the same budget fraction its
        # allocation is, so its option rows pass through at face value.
        # MAX_LEVERAGE stays the final backstop.
        core_weight: dict[str, float] = dict.fromkeys(assets, 0.0)
        hedge_weight: dict[str, float] = dict.fromkeys(assets, 0.0)
        core_options: dict[str, tuple[OptionContract, float]] = {}
        hedge_options: dict[str, tuple[OptionContract, float]] = {}
        combined_signals: dict[str, list[ManagerSignal]] = {asset: [] for asset in assets}
        for manager_id, decisions in pod_decisions:
            allocation = pod_allocations.get(manager_id, 0.0)
            is_hedge = manager_id == TAIL_HEDGE_MANAGER_ID
            book = hedge_weight if is_hedge else core_weight
            option_book = hedge_options if is_hedge else core_options
            for decision in decisions:
                combined_signals.setdefault(decision.asset, []).extend(
                    decision.contributing_signals
                )
                if decision.instrument is None:
                    book[decision.asset] += allocation * decision.target_weight
                    continue
                weight = decision.target_weight if is_hedge else allocation * decision.target_weight
                _, held = option_book.get(decision.asset, (decision.instrument, 0.0))
                option_book[decision.asset] = (decision.instrument, held + weight)

        # Firm limits run on the delta-equivalent book; whatever scaling
        # they impose on an underlying applies to its option rows too.
        equivalent = _delta_equivalent_book(core_weight, core_options, market_data, as_of_date)
        limited = _apply_firm_limits(equivalent, market_data, as_of_date)
        scale = {
            asset: limited[asset] / equivalent[asset] if equivalent[asset] != 0.0 else 1.0
            for asset in assets
        }
        limited_core = {asset: core_weight[asset] * scale[asset] for asset in assets}
        limited_options = _apply_short_premium_cap(
            {
                contract_id: (contract, weight * scale.get(contract.underlying, 1.0))
                for contract_id, (contract, weight) in core_options.items()
            },
            as_of_date,
        )
        combined_weight = {asset: limited_core[asset] + hedge_weight[asset] for asset in assets}

        def _clamp(weight: float) -> float:
            return max(min(weight, MAX_LEVERAGE), -MAX_LEVERAGE)

        return [
            PortfolioDecision(
                asset=asset,
                as_of_date=as_of_date.date(),
                target_weight=_clamp(combined_weight[asset]),
                contributing_signals=combined_signals[asset],
            )
            for asset in assets
        ] + [
            PortfolioDecision(
                asset=contract_id,
                as_of_date=as_of_date.date(),
                target_weight=_clamp(weight),
                contributing_signals=combined_signals.get(contract_id, []),
                instrument=contract,
            )
            for contract_id, (contract, weight) in {**limited_options, **hedge_options}.items()
        ]
