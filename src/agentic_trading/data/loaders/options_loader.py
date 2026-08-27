# options_loader.py

"""Live options-chain snapshots via yfinance.

No free source publishes historical options chains, so `load_option_chain`
only works for `as_of_date == today` — a past `as_of_date` raises
`NotImplementedError` rather than silently returning today's chain under a
false date, which would be a point-in-time violation. Consequence: nothing
that drives a historical backtest may depend on this loader; it exists only
for a possible future live/paper-trading mode. The CVaR-pod analysts price
theoretical options with model-estimated volatility instead — see
`agents/analysts/cvar_pod/_theoretical_options.py`.
"""

from datetime import date

import pandas as pd
import yfinance as yf

from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_TARGET_TENOR_DAYS = 30


def _nearest_expiration(expirations: tuple[str, ...], as_of: date, target_tenor_days: int) -> str:
    if not expirations:
        raise ValueError("no option expirations available for this ticker")
    return min(
        expirations,
        key=lambda expiration: abs(
            (date.fromisoformat(expiration) - as_of).days - target_tenor_days
        ),
    )


def load_option_chain(
    asset: str,
    as_of_date: pd.Timestamp,
    target_tenor_days: int = DEFAULT_TARGET_TENOR_DAYS,
) -> pd.DataFrame:
    """Live option-chain snapshot for `asset`, at the expiration nearest
    `target_tenor_days` out. Returns strike, bid, ask, mid (computed as
    (bid+ask)/2, not Yahoo's stale `lastPrice`), open interest, and volume.
    Does not use Yahoo's `impliedVolatility` field — it's frequently stale
    or wrong.
    """
    today = date.today()
    requested_date = as_of_date.date()
    if requested_date < today:
        raise NotImplementedError(
            f"load_option_chain requested for {requested_date}, which is in the past — no "
            "free source has historical options chains. This loader only supports "
            "as_of_date == today; a backtest must not call it."
        )

    ticker = yf.Ticker(asset)
    expiration = _nearest_expiration(ticker.options, today, target_tenor_days)
    logger.info("Loading option chain for %s, expiration %s", asset, expiration)
    chain = ticker.option_chain(expiration)

    frames = []
    for option_type, raw in (("call", chain.calls), ("put", chain.puts)):
        frame = pd.DataFrame(
            {
                "option_type": option_type,
                "strike": raw["strike"],
                "bid": raw["bid"],
                "ask": raw["ask"],
                "mid": (raw["bid"] + raw["ask"]) / 2,
                "open_interest": raw["openInterest"],
                "volume": raw["volume"],
            }
        )
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    result.attrs["expiration"] = expiration
    return result
