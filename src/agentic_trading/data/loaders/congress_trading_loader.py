# congress_trading_loader.py

"""Congressional trading disclosures (Senate + House Stock Watcher, free
community-maintained bulk JSON dumps of STOCK Act filings, no key).

CRITICAL point-in-time trap: a disclosure's `transaction_date` (when the
trade happened) is typically 30-45 days *earlier* than its
`disclosure_date` (when the filing became public) — the STOCK Act gives
members up to 45 days to report. Gating on `transaction_date` would leak a
trade into a backtest weeks before the market could have known about it.
This loader gates on `disclosure_date` exclusively, the same "gate on
publication, not the event" rule `fundamentals_loader.py` and
`short_interest_loader.py` already enforce for their sources."""

import functools
from datetime import date

import pandas as pd
import requests

from agentic_trading.data.cache import cached_fetch
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

_SENATE_URL = (
    "https://senate-stock-watcher-data.s3-us-west-2.amazonaws.com/aggregate/all_transactions.json"
)
_HOUSE_URL = (
    "https://house-stock-watcher-data.s3-us-west-2.amazonaws.com/data/all_transactions.json"
)

_SOURCE_URLS: dict[str, str] = {"senate": _SENATE_URL, "house": _HOUSE_URL}


def _filer_name(frame: pd.DataFrame) -> pd.Series:
    for column in ("representative", "senator"):
        if column in frame.columns:
            return frame[column].astype(str)
    first = frame.get("first_name", pd.Series("", index=frame.index)).fillna("")
    last = frame.get("last_name", pd.Series("", index=frame.index)).fillna("")
    return (first + " " + last).str.strip()


def _fetch_disclosures(chamber: str) -> pd.DataFrame:
    url = _SOURCE_URLS[chamber]
    logger.info("Downloading %s Stock Watcher disclosures", chamber)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    raw = response.json()
    frame = pd.DataFrame(raw)
    columns = ["ticker", "filer", "transaction_date", "disclosure_date", "transaction_type"]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    disclosure_column = "disclosure_date" if "disclosure_date" in frame.columns else "date_recieved"
    result = pd.DataFrame(
        {
            "ticker": frame.get("ticker", pd.Series(dtype=str)).astype(str).str.upper(),
            "filer": _filer_name(frame),
            "transaction_date": pd.to_datetime(
                frame.get("transaction_date", pd.Series(dtype=str)), errors="coerce"
            ),
            "disclosure_date": pd.to_datetime(
                frame.get(disclosure_column, pd.Series(dtype=str)), errors="coerce"
            ),
            "transaction_type": frame.get("type", pd.Series(dtype=str)),
        }
    )
    result["chamber"] = chamber
    return result.dropna(subset=["disclosure_date"])


def load_congressional_trades(
    ticker: str,
    as_of_date: pd.Timestamp,
    chambers: tuple[str, ...] = ("senate", "house"),
) -> pd.DataFrame:
    """Congressional trading disclosures mentioning `ticker` across
    `chambers`, filtered to `disclosure_date <= as_of_date` — never
    `transaction_date`, which is filed on a 30-45 day delay under the
    STOCK Act."""
    frames = [
        cached_fetch(
            source="congress_trading",
            identifier=chamber,
            as_of_date=date.today(),
            fetch_fn=functools.partial(_fetch_disclosures, chamber),
        )
        for chamber in chambers
    ]
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if combined.empty:
        return combined
    combined = combined.loc[combined["ticker"] == ticker.upper()]
    return combined.loc[combined["disclosure_date"] <= as_of_date]
