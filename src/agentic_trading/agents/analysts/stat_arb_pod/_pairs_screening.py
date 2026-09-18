# _pairs_screening.py

"""Walk-forward re-screening of StatArb's cointegrated pairs — the same
Engle-Granger + ADF test `scripts/screen_cointegrated_pairs.py` runs once,
offline, over a fixed formation window, but driven by the backtest's own
simulated `as_of_date` against a rolling trailing window instead.
`scripts/screen_cointegrated_pairs.py` stays the tool for an initial/
manual wiring pass (its printed `_COINTEGRATED_PAIRS` tuple is what
`roster.py` starts with); this module is what keeps that wiring from
going stale for the rest of a multi-year run — cointegration between
individual equities is known to decay, and a pair screened once at the
start of an 8-year backtest and never re-checked can spend years trading
a relationship that stopped holding partway through.

`StatArbManager` drives this via a `RefitCache` gate at a quarterly
cadence and owns the admit/keep/drop hysteresis — this module has no
scheduling or admission opinion of its own, it just answers "what does
the test say about each candidate pair as of today," given whatever
`market_data` the caller already has in hand (no separate network fetch
— the backtest's own point-in-time-sliced market data already has
everything a trailing-window re-test needs)."""

import numpy as np
import pandas as pd
from pydantic import BaseModel
from quantcore.statistics.cointegration import adf_test, engle_granger_test, ou_half_life

from agentic_trading.agents._market_math import market_slice
from agentic_trading.config import settings
from agentic_trading.data.loaders.lakehouse_loader import same_sic_candidate_pairs
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

# The same 10 hand-picked same-sector candidates
# scripts/screen_cointegrated_pairs.py's SECTOR_PAIRS uses. `resolve_
# candidate_pairs` combines these with `same_sic_candidate_pairs`'s real
# SEC SIC-code matches over the current traded universe rather than
# relying on either alone — a hand-picked "looks same-sector to me" call
# and a real SIC-code match can disagree (e.g. MA/JPM, hand-picked as
# "Financials/Payments," don't actually share a SIC code).
CANDIDATE_PAIRS: tuple[tuple[str, str], ...] = (
    ("AAPL", "AMZN"),
    ("AVGO", "NVDA"),
    ("MSFT", "ADBE"),
    ("MA", "JPM"),
    ("JNJ", "PFE"),
    ("COP", "CVX"),
    ("SBUX", "WMT"),
    ("VZ", "NFLX"),
    ("BA", "CAT"),
    ("PLD", "AMT"),
)
# Each leg's log price must itself look non-stationary (ADF p above this)
# before a cointegration framing applies — a stationary leg makes the
# Engle-Granger residual test trivially pass. 10% is the conventional
# level for a pre-screen: lenient enough to keep genuine random walks.
LEG_STATIONARITY_PVALUE = 0.10
# Trailing window for the return-correlation pre-filter.
CORRELATION_WINDOW_DAYS = 252


class PairVerdict(BaseModel):
    """One candidate pair's Engle-Granger result over the formation window.

    `dependent` is the ticker regressed on the other (the direction with
    the lower p-value of the two); `beta` is that direction's OLS
    cointegrating coefficient, so the tested spread is
    `log_p(dependent) - beta * log_p(other)`; `half_life` is that spread's
    OU mean-reversion half-life in trading days."""

    p_value: float
    beta: float
    dependent: str
    half_life: float


def resolve_candidate_pairs(assets: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    """Real SIC-classification-based candidate pairs (`same_sic_candidate_pairs`)
    restricted to whatever `assets` this pod is actually trading today,
    combined with whichever hand-picked `CANDIDATE_PAIRS` entries are also
    within `assets` — real SIC matches are never crowded out by the
    hand-picked list, but a single SIC match also doesn't discard every
    other hand-picked candidate this universe still supports."""
    asset_set = set(assets)
    sic_pairs = tuple(same_sic_candidate_pairs(list(assets)))
    fallback_pairs = tuple(
        (ticker_a, ticker_b)
        for ticker_a, ticker_b in CANDIDATE_PAIRS
        if ticker_a in asset_set and ticker_b in asset_set
    )
    seen: set[frozenset[str]] = set()
    combined: list[tuple[str, str]] = []
    for pair in (*sic_pairs, *fallback_pairs):
        key = frozenset(pair)
        if key not in seen:
            seen.add(key)
            combined.append(pair)
    return tuple(combined)


def _screen_pair(
    market_data: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    as_of_date: pd.Timestamp,
    formation_days: int,
) -> PairVerdict | None:
    """Pre-filters (enough overlapping history, both legs non-stationary,
    trailing return correlation above `pair_min_correlation`), then
    Engle-Granger in both directions, keeping the direction with the lower
    p-value. Returns None when any pre-filter or the test itself rejects
    the pair."""
    returns_a = market_slice(market_data, as_of_date, ticker_a).log_returns[-formation_days:]
    returns_b = market_slice(market_data, as_of_date, ticker_b).log_returns[-formation_days:]
    length = min(returns_a.size, returns_b.size)
    if length < formation_days // 2:
        return None

    returns_a = returns_a[-length:]
    returns_b = returns_b[-length:]
    correlation_window = min(CORRELATION_WINDOW_DAYS, length)
    correlation = float(
        np.corrcoef(returns_a[-correlation_window:], returns_b[-correlation_window:])[0, 1]
    )
    if not np.isfinite(correlation) or correlation <= settings.analysts.pair_min_correlation:
        return None

    log_prices_a = np.cumsum(returns_a)
    log_prices_b = np.cumsum(returns_b)
    try:
        _, p_a = adf_test(log_prices_a)
        _, p_b = adf_test(log_prices_b)
        if p_a <= LEG_STATIONARITY_PVALUE or p_b <= LEG_STATIONARITY_PVALUE:
            return None
        beta_ab, _, p_ab = engle_granger_test(log_prices_a, log_prices_b)
        beta_ba, _, p_ba = engle_granger_test(log_prices_b, log_prices_a)
        if p_ab <= p_ba:
            dependent, beta, p_value = ticker_a, float(beta_ab), float(p_ab)
            spread = log_prices_a - beta * log_prices_b
        else:
            dependent, beta, p_value = ticker_b, float(beta_ba), float(p_ba)
            spread = log_prices_b - beta * log_prices_a
        half_life = float(ou_half_life(spread))
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
        logger.debug(
            "Pair re-screen failed for %s/%s on %s (%s) — rejecting.",
            ticker_a,
            ticker_b,
            as_of_date.date(),
            exc,
        )
        return None
    if not np.isfinite(half_life) or half_life <= 0.0:
        return None
    return PairVerdict(p_value=p_value, beta=beta, dependent=dependent, half_life=half_life)


def rescreen_pairs(
    market_data: pd.DataFrame,
    as_of_date: pd.Timestamp,
    candidate_pairs: tuple[tuple[str, str], ...] = CANDIDATE_PAIRS,
    formation_days: int | None = None,
) -> dict[tuple[str, str], PairVerdict]:
    """Engle-Granger verdict for every `candidate_pairs` entry over the
    trailing `formation_days` (default `settings.analysts.pair_formation_
    days`) of `market_data` ending at `as_of_date`. A pair that fails a
    pre-filter or whose test errors is omitted rather than given a
    verdict — callers apply their own admit/keep thresholds to the
    p-values and half-lives of the pairs that remain."""
    window = formation_days if formation_days is not None else settings.analysts.pair_formation_days
    verdicts: dict[tuple[str, str], PairVerdict] = {}
    for ticker_a, ticker_b in candidate_pairs:
        verdict = _screen_pair(market_data, ticker_a, ticker_b, as_of_date, window)
        if verdict is not None:
            verdicts[(ticker_a, ticker_b)] = verdict
    return verdicts
