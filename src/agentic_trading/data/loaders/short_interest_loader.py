# short_interest_loader.py

"""FINRA biweekly short interest, gated on the file's dissemination date.

FINRA settles short interest on a settlement date but doesn't disseminate
the file until several days later — gating on the settlement date instead
of the dissemination date would let a backtest see a number before it was
actually publishable, the same point-in-time hazard as SEC filings.

FINRA's API returns this endpoint as CSV text (`Content-Type: text/plain`)
regardless of the request body, not JSON, and its rows carry no explicit
dissemination-date field — only `settlementDate`. Per FINRA's published
short-interest settlement/dissemination calendar, dissemination trails
settlement by about 8 business days; that lag is used here as a documented
estimate, not an exact value pulled from the API."""

import io
from datetime import date

import pandas as pd
import requests

from agentic_trading.data.cache import cached_fetch
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

_FINRA_SHORT_INTEREST_URL = (
    "https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest"
)

_DISSEMINATION_LAG_BUSINESS_DAYS = 8

_OUTPUT_COLUMNS = [
    "settlement_date",
    "dissemination_date",
    "short_interest_shares",
    "days_to_cover",
    "avg_daily_volume",
]


def _fetch_short_interest_history(ticker: str) -> pd.DataFrame:
    logger.info("Downloading FINRA short interest history for %s", ticker)
    response = requests.post(
        _FINRA_SHORT_INTEREST_URL,
        json={
            "compareFilters": [
                {"compareType": "EQUAL", "fieldName": "symbolCode", "fieldValue": ticker}
            ],
            "limit": 10_000,
        },
        timeout=30,
    )
    response.raise_for_status()
    frame = pd.read_csv(io.StringIO(response.text))
    if frame.empty:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)
    frame = frame.rename(
        columns={
            "settlementDate": "settlement_date",
            "currentShortPositionQuantity": "short_interest_shares",
            "daysToCoverQuantity": "days_to_cover",
            "averageDailyVolumeQuantity": "avg_daily_volume",
        }
    )
    frame["settlement_date"] = pd.to_datetime(frame["settlement_date"])
    frame["dissemination_date"] = frame["settlement_date"] + pd.tseries.offsets.BDay(
        _DISSEMINATION_LAG_BUSINESS_DAYS
    )
    return frame[_OUTPUT_COLUMNS]


def load_short_interest(ticker: str, as_of_date: pd.Timestamp) -> pd.DataFrame:
    """Short interest history for `ticker`, filtered to
    `dissemination_date <= as_of_date` — the point-in-time gate for this
    loader."""
    history = cached_fetch(
        source="finra_short_interest",
        identifier=ticker.upper(),
        as_of_date=date.today(),
        fetch_fn=lambda: _fetch_short_interest_history(ticker),
    )
    if history.empty:
        return history
    return history.loc[history["dissemination_date"] <= as_of_date]
