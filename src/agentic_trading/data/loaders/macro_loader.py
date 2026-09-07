# macro_loader.py

"""FRED macro/rates series, point-in-time via FRED's own vintage support.

Pulling with `realtime_end=as_of_date` means FRED itself only returns the
vintage of each observation that existed as of that date — this reuses
FRED's own point-in-time support rather than re-implementing publication-lag
logic by hand.
"""

import pandas as pd
from fredapi import Fred

from agentic_trading.data.cache import cached_fetch
from agentic_trading.data.loaders._env import require_env
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

DGS3MO = "DGS3MO"
DGS2 = "DGS2"
DGS10 = "DGS10"
VIXCLS = "VIXCLS"
CPIAUCSL = "CPIAUCSL"

SUPPORTED_SERIES: frozenset[str] = frozenset({DGS3MO, DGS2, DGS10, VIXCLS, CPIAUCSL})


def _fetch_fred_series(series_id: str, as_of_date: pd.Timestamp) -> pd.DataFrame:
    fred = Fred(api_key=require_env("FRED_API_KEY"))
    logger.debug("Loading FRED series %s as of %s", series_id, as_of_date.date())
    # realtime_start defaults to 1776-07-04 if omitted, which for a daily
    # series exceeds FRED's 2000-vintage-per-request cap. Pinning it to the
    # same date as realtime_end asks for exactly the one vintage that
    # existed as of as_of_date, which is also the semantically correct
    # point-in-time query.
    series: pd.Series = fred.get_series(
        series_id, realtime_start=as_of_date.date(), realtime_end=as_of_date.date()
    )
    return series.to_frame(name="value")


def load_fred_series(series_id: str, as_of_date: pd.Timestamp) -> pd.Series:
    """Returns the FRED series `series_id`, restricted to the vintage that
    existed as of `as_of_date` (via FRED's `realtime_end`)."""
    if series_id not in SUPPORTED_SERIES:
        raise ValueError(
            f"Unsupported FRED series {series_id!r}; supported: {sorted(SUPPORTED_SERIES)}"
        )

    frame = cached_fetch(
        source="fred",
        identifier=series_id,
        as_of_date=as_of_date.date(),
        fetch_fn=lambda: _fetch_fred_series(series_id, as_of_date),
    )
    return frame["value"]
