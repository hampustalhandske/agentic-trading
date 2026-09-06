# fmp_loader.py

"""Financial Modeling Prep fundamentals — an alternative/supplement to the
SEC EDGAR company-facts loader (`fundamentals_loader.py`) for names where
full-text XBRL parsing is awkward. Same point-in-time trap as SEC EDGAR:
gates on each statement's `fillingDate`, never the fiscal period it
reports on."""

from datetime import date

import pandas as pd
import requests

from agentic_trading.data.cache import cached_fetch
from agentic_trading.data.loaders._env import require_env
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

_BASE_URL = "https://financialmodelingprep.com/api/v3"

_INCOME_STATEMENT_CONCEPTS: tuple[str, ...] = ("revenue", "netIncome")
_BALANCE_SHEET_CONCEPTS: tuple[str, ...] = (
    "totalAssets",
    "totalLiabilities",
    "totalStockholdersEquity",
)


def _fetch_statement(ticker: str, endpoint: str, concepts: tuple[str, ...]) -> pd.DataFrame:
    api_key = require_env("FMP_API_KEY")
    logger.info("Downloading FMP %s for %s", endpoint, ticker)
    params: dict[str, str | int] = {"period": "quarter", "limit": 40, "apikey": api_key}
    response = requests.get(
        f"{_BASE_URL}/{endpoint}/{ticker}",
        params=params,
        timeout=30,
    )
    response.raise_for_status()
    raw = response.json()
    rows: list[dict[str, object]] = []
    for statement in raw:
        for concept in concepts:
            if concept not in statement:
                continue
            rows.append(
                {
                    "concept": concept,
                    "value": statement[concept],
                    "fiscal_period": f"{statement.get('calendarYear')}{statement.get('period')}",
                    "filed": statement.get("fillingDate"),
                }
            )
    frame = pd.DataFrame(rows, columns=["concept", "value", "fiscal_period", "filed"])
    if not frame.empty:
        frame["filed"] = pd.to_datetime(frame["filed"])
    return frame


def _fetch_fmp_fundamentals(ticker: str) -> pd.DataFrame:
    income = _fetch_statement(ticker, "income-statement", _INCOME_STATEMENT_CONCEPTS)
    balance = _fetch_statement(ticker, "balance-sheet-statement", _BALANCE_SHEET_CONCEPTS)
    return pd.concat([income, balance], ignore_index=True)


def load_fmp_fundamentals(ticker: str, as_of_date: pd.Timestamp) -> pd.DataFrame:
    """One row per statement/concept observed for `ticker` from FMP's
    income-statement and balance-sheet endpoints, filtered to
    `filed <= as_of_date` — mirrors `fundamentals_loader.load_company_facts`'s
    shape (`concept, value, fiscal_period, filed`) for easy substitution."""
    frame = cached_fetch(
        source="fmp",
        identifier=f"fundamentals_{ticker.upper()}",
        as_of_date=date.today(),
        fetch_fn=lambda: _fetch_fmp_fundamentals(ticker),
    )
    if frame.empty:
        return frame
    return frame.loc[frame["filed"] <= as_of_date]
