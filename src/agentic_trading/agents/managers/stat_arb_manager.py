# stat_arb_manager.py

import pandas as pd
from pydantic import PrivateAttr

from agentic_trading.agents._market_math import RefitCache, vol_regime_scale_factor
from agentic_trading.agents.analysts.stat_arb_pod._pairs_screening import (
    PairVerdict,
    rescreen_pairs,
    resolve_candidate_pairs,
)
from agentic_trading.agents.analysts.stat_arb_pod.kalman_hedge_ratio_analyst import (
    KalmanHedgeRatioAnalyst,
)
from agentic_trading.agents.analysts.stat_arb_pod.pairs_cointegration_analyst import (
    PairsCointegrationAnalyst,
)
from agentic_trading.agents.base import ManagerAgent
from agentic_trading.agents.schemas import AgentSignal, PortfolioDecision
from agentic_trading.config import settings
from agentic_trading.logging_config import get_logger
from agentic_trading.scoring.scoreboard import PerformanceScoreboard

logger = get_logger(__name__)


# A held pair survives a re-screen while its half-life stays below this
# (looser than the admit ceiling, so a pair is not churned by a modest
# slowdown); above `pair_drop_half_life_days` it is dropped immediately.
KEEP_MAX_HALF_LIFE_DAYS = 90.0
# Consecutive failed re-screens a held pair is allowed before it is dropped.
MAX_CONSECUTIVE_FAILURES = 2


class StatArbManager(ManagerAgent):
    """Scales analyst confidence down for elevated spread volatility. Uses
    the asset's own recent return series as an EWMA-variance proxy for
    spread vol — the manager doesn't have direct access to each analyst's
    specific spread construction (OU spread, cointegration spread, Kalman
    hedge-ratio spread), so it reads risk at the manager level instead.

    Also owns this pod's walk-forward pair selection: before building
    each day's portfolio, `_maybe_rescreen_pairs` checks (via
    `_pairs_refit_cache`, a quarterly cadence) whether it's time to re-run
    the Engle-Granger screen (`_pairs_screening.rescreen_pairs`) over
    every candidate pair (`resolve_candidate_pairs`: real SIC matches
    within the pod's own traded universe plus whichever hand-picked
    candidates that universe supports) and, only on a stale day, applies
    the admit/keep/drop hysteresis:

    - a pair not currently held is admitted when its p-value is below
      `pair_admit_pvalue` and its half-life lies within
      `[pair_min_half_life_days, pair_max_half_life_days]`;
    - a held pair is kept while its p-value stays below `pair_keep_pvalue`
      and its half-life below `KEEP_MAX_HALF_LIFE_DAYS`; a half-life above
      `pair_drop_half_life_days` drops it at once, any other failure only
      counts against it and it is dropped after `MAX_CONSECUTIVE_FAILURES`
      failed re-screens in a row;
    - the surviving set is ranked by half-life (fastest reversion first)
      and truncated to `max_pairs`.

    The resulting partner wiring and Engle-Granger hedge ratios are pushed
    into `PairsCointegrationAnalyst` (`partner_tickers_by_asset`,
    `hedge_ratio_by_pair`) and `KalmanHedgeRatioAnalyst`
    (`partner_ticker_by_asset`) on whichever of its own `managed_analysts`
    are those types. An asset with no held pair falls into both analysts'
    "no partner ticker(s) configured" flat path."""

    agent_id: str = "stat_arb_manager"

    _pairs_refit_cache: RefitCache = PrivateAttr(
        default_factory=lambda: RefitCache(
            refit_interval_days=settings.managers.pairs_refit_interval_days
        )
    )
    _held_pairs: dict[tuple[str, str], PairVerdict] = PrivateAttr(default_factory=dict)
    _failed_rescreens: dict[tuple[str, str], int] = PrivateAttr(default_factory=dict)

    def _apply_hysteresis(
        self, verdicts: dict[tuple[str, str], PairVerdict], as_of_date: pd.Timestamp
    ) -> dict[tuple[str, str], PairVerdict]:
        """The next held set (ranked by half-life, at most `max_pairs`)
        given this re-screen's `verdicts`, updating `_failed_rescreens`."""
        limits = settings.analysts
        next_held: dict[tuple[str, str], PairVerdict] = {}
        next_failures: dict[tuple[str, str], int] = {}

        for pair, last_verdict in self._held_pairs.items():
            verdict = verdicts.get(pair)
            if verdict is not None and verdict.half_life > limits.pair_drop_half_life_days:
                logger.info(
                    "%s: dropping %s as of %s — half-life %.1fd exceeds %.0fd.",
                    self.agent_id,
                    pair,
                    as_of_date.date(),
                    verdict.half_life,
                    limits.pair_drop_half_life_days,
                )
                continue
            if (
                verdict is not None
                and verdict.p_value < limits.pair_keep_pvalue
                and verdict.half_life < KEEP_MAX_HALF_LIFE_DAYS
            ):
                next_held[pair] = verdict
                continue
            failures = self._failed_rescreens.get(pair, 0) + 1
            if failures >= MAX_CONSECUTIVE_FAILURES:
                logger.info(
                    "%s: dropping %s as of %s after %d consecutive failed re-screens (%s).",
                    self.agent_id,
                    pair,
                    as_of_date.date(),
                    failures,
                    verdict if verdict is not None else "no verdict",
                )
                continue
            next_held[pair] = verdict if verdict is not None else last_verdict
            next_failures[pair] = failures

        for pair, verdict in verdicts.items():
            if pair in self._held_pairs:
                continue
            if (
                verdict.p_value < limits.pair_admit_pvalue
                and limits.pair_min_half_life_days
                <= verdict.half_life
                <= limits.pair_max_half_life_days
            ):
                next_held[pair] = verdict

        ranked = sorted(next_held.items(), key=lambda item: item[1].half_life)
        held = dict(ranked[: limits.max_pairs])
        self._failed_rescreens = {pair: n for pair, n in next_failures.items() if pair in held}
        return held

    def _wire_analysts(self, held: dict[tuple[str, str], PairVerdict]) -> None:
        partner_tickers_by_asset: dict[str, tuple[str, ...]] = {}
        partner_ticker_by_asset: dict[str, str] = {}
        hedge_ratio_by_pair: dict[tuple[str, str], float] = {}
        for (ticker_a, ticker_b), verdict in held.items():
            other = ticker_b if verdict.dependent == ticker_a else ticker_a
            if verdict.dependent in partner_ticker_by_asset or other in partner_ticker_by_asset:
                continue
            partner_tickers_by_asset[ticker_a] = (ticker_b,)
            partner_tickers_by_asset[ticker_b] = (ticker_a,)
            partner_ticker_by_asset[ticker_a] = ticker_b
            partner_ticker_by_asset[ticker_b] = ticker_a
            hedge_ratio_by_pair[(verdict.dependent, other)] = verdict.beta
            hedge_ratio_by_pair[(other, verdict.dependent)] = (
                1.0 / verdict.beta if verdict.beta != 0.0 else 1.0
            )

        for analyst in self.managed_analysts:
            if isinstance(analyst, PairsCointegrationAnalyst):
                analyst.partner_tickers_by_asset = dict(partner_tickers_by_asset)
                analyst.hedge_ratio_by_pair = dict(hedge_ratio_by_pair)
            if isinstance(analyst, KalmanHedgeRatioAnalyst):
                analyst.partner_ticker_by_asset = dict(partner_ticker_by_asset)

    def _maybe_rescreen_pairs(
        self, assets: list[str], market_data: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> None:
        def _refresh() -> tuple[tuple[str, str], ...]:
            candidate_pairs = resolve_candidate_pairs(tuple(assets))
            verdicts = rescreen_pairs(market_data, as_of_date, candidate_pairs)
            logger.info(
                "%s: quarterly pair re-screen as of %s — %d/%d candidates passed pre-filters: %s",
                self.agent_id,
                as_of_date.date(),
                len(verdicts),
                len(candidate_pairs),
                {
                    pair: (
                        f"p={v.p_value:.3f}, half-life={v.half_life:.1f}d, "
                        f"beta={v.beta:.3f} ({v.dependent} dependent)"
                    )
                    for pair, v in verdicts.items()
                },
            )
            held = self._apply_hysteresis(verdicts, as_of_date)
            self._held_pairs = held
            if not held:
                logger.warning(
                    "%s: quarterly pair re-screen as of %s holds 0 pairs out of %d candidates — "
                    "this pod will flat-signal on every asset until the next re-screen.",
                    self.agent_id,
                    as_of_date.date(),
                    len(candidate_pairs),
                )
            else:
                logger.info(
                    "%s: holding %d pair(s) as of %s (ranked by half-life): %s",
                    self.agent_id,
                    len(held),
                    as_of_date.date(),
                    {pair: f"{v.half_life:.1f}d" for pair, v in held.items()},
                )
            self._wire_analysts(held)
            return tuple(held)

        self._pairs_refit_cache.get_or_refit("pairs", as_of_date, _refresh)

    def construct_portfolio(
        self,
        assets: list[str],
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
        scoreboard: PerformanceScoreboard,
    ) -> list[PortfolioDecision]:
        self._maybe_rescreen_pairs(assets, market_data, as_of_date)
        return super().construct_portfolio(assets, market_data, as_of_date, scoreboard)

    def scale_factor(
        self,
        signal: AgentSignal,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> float:
        """Volatility-regime dampening for `signal.asset` (see
        `_market_math.vol_regime_scale_factor`): near 1.0 in a normal
        regime, halved at a 1.5-sigma vol spike."""
        return vol_regime_scale_factor(market_data, as_of_date, signal.asset)
