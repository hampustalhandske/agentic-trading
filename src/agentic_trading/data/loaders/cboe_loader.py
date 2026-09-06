# cboe_loader.py

"""CBOE total put/call ratio — a free daily CSV of aggregate market-wide
options positioning (calls, puts, total volume, and the put/call ratio
itself), no key required. Distinct from the CVaR pod's per-name
theoretical option pricing: this reads real aggregate market positioning,
not a modeled price.

Gates on the bar's own date, like price data — the full CSV is fetched
once and filtered to `<= as_of_date`, the same bulk-then-filter pattern
`factor_loader.py` uses for a bulk source."""

import io
from datetime import date

import pandas as pd
import requests

from agentic_trading.data.cache import cached_fetch
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

_URL = "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/totalpc.csv"


def _fetch_put_call_ratio() -> pd.DataFrame:
    logger.info("Downloading CBOE total put/call ratio history")
    response = requests.get(_URL, timeout=30)
    response.raise_for_status()
    lines = response.text.splitlines()
    header_index = next(
        index for index, line in enumerate(lines) if line.upper().startswith("DATE")
    )
    frame = pd.read_csv(io.StringIO("\n".join(lines[header_index:])))
    frame.columns = [column.strip().upper() for column in frame.columns]
    frame["DATE"] = pd.to_datetime(frame["DATE"])
    frame = frame.set_index("DATE").sort_index()
    frame = frame.rename(
        columns={
            "CALLS": "calls",
            "PUTS": "puts",
            "TOTAL": "total",
            "P/C RATIO": "put_call_ratio",
        }
    )
    keep = [column for column in ("calls", "puts", "total", "put_call_ratio") if column in frame]
    return frame[keep]


def load_put_call_ratio(as_of_date: pd.Timestamp) -> pd.DataFrame:
    """CBOE total put/call ratio history, filtered to `<= as_of_date`."""
    frame = cached_fetch(
        source="cboe",
        identifier="total_put_call_ratio",
        as_of_date=date.today(),
        fetch_fn=_fetch_put_call_ratio,
    )
    return frame.loc[frame.index <= as_of_date]
