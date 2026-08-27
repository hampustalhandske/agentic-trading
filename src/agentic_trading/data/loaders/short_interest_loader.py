# short_interest_loader.py

"""FINRA biweekly short interest, gated on the file's dissemination date.

FINRA settles short interest on a settlement date but doesn't disseminate
the file until several days later — gating on the settlement date instead
of the dissemination date would let a backtest see a number before it was
actually publishable, the same point-in-time hazard as SEC filings."""

from datetime import date

import pandas as pd
import requests

from agentic_trading.data.cache import cached_fetch
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

_FINRA_SHORT_INTEREST_URL = (
    "https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest"
)


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
    raw = response.json()
    frame = pd.DataFrame(raw)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "settlement_date",
                "dissemination_date",
                "short_interest_shares",
                "days_to_cover",
                "avg_daily_volume",
            ]
        )
    frame = frame.rename(
        columns={
            "settlementDate": "settlement_date",
            "disseminationDate": "dissemination_date",
            "currentShortPositionQuantity": "short_interest_shares",
            "daysToCoverQuantity": "days_to_cover",
            "averageDailyVolumeQuantity": "avg_daily_volume",
        }
    )
    frame["settlement_date"] = pd.to_datetime(frame["settlement_date"])
    frame["dissemination_date"] = pd.to_datetime(frame["dissemination_date"])
    return frame[
        [
            "settlement_date",
            "dissemination_date",
            "short_interest_shares",
            "days_to_cover",
            "avg_daily_volume",
        ]
    ]


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
