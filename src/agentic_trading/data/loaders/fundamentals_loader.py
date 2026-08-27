# fundamentals_loader.py

"""SEC EDGAR company facts, gated on each fact's `filed` date.

`filed <= as_of_date` is the point-in-time gate for this entire loader — a
10-Q's fiscal period ends weeks before it's actually filed, so gating on the
period instead of the filing date would be a lookahead bug."""

from datetime import date

import pandas as pd
import requests

from agentic_trading.data.cache import cached_fetch
from agentic_trading.data.loaders._env import require_env
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
_COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

CONCEPTS: tuple[str, ...] = (
    "Revenues",
    "EarningsPerShareDiluted",
    "StockholdersEquity",
    "Assets",
    "Liabilities",
)


def _headers() -> dict[str, str]:
    return {"User-Agent": require_env("SEC_EDGAR_USER_AGENT")}


def _fetch_ticker_map() -> pd.DataFrame:
    logger.info("Downloading SEC EDGAR ticker/CIK map")
    response = requests.get(_TICKER_MAP_URL, headers=_headers(), timeout=30)
    response.raise_for_status()
    raw = response.json()
    return pd.DataFrame(
        {
            "ticker": [entry["ticker"] for entry in raw.values()],
            "cik": [entry["cik_str"] for entry in raw.values()],
        }
    )


def _resolve_cik(ticker: str) -> int:
    mapping = cached_fetch(
        source="sec_edgar",
        identifier="ticker_cik_map",
        as_of_date=date.today(),
        fetch_fn=_fetch_ticker_map,
    )
    matches = mapping.loc[mapping["ticker"].str.upper() == ticker.upper(), "cik"]
    if matches.empty:
        raise ValueError(f"No SEC EDGAR CIK found for ticker {ticker!r}")
    return int(matches.iloc[0])


def _fetch_company_facts(cik: int) -> pd.DataFrame:
    url = _COMPANY_FACTS_URL.format(cik=cik)
    logger.info("Downloading SEC EDGAR company facts for CIK %010d", cik)
    response = requests.get(url, headers=_headers(), timeout=30)
    response.raise_for_status()
    raw = response.json()

    rows: list[dict[str, object]] = []
    us_gaap = raw.get("facts", {}).get("us-gaap", {})
    for concept in CONCEPTS:
        concept_data = us_gaap.get(concept)
        if concept_data is None:
            continue
        for unit_values in concept_data.get("units", {}).values():
            for entry in unit_values:
                rows.append(
                    {
                        "concept": concept,
                        "value": entry.get("val"),
                        "fiscal_period": f"{entry.get('fy')}{entry.get('fp')}",
                        "filed": entry.get("filed"),
                    }
                )
    frame = pd.DataFrame(rows, columns=["concept", "value", "fiscal_period", "filed"])
    if not frame.empty:
        frame["filed"] = pd.to_datetime(frame["filed"])
    return frame


def load_company_facts(ticker: str, as_of_date: pd.Timestamp) -> pd.DataFrame:
    """One row per filing/concept observed for `ticker`, filtered to
    `filed <= as_of_date`. Missing concepts are omitted, not raised on —
    not every filer tags every concept identically."""
    cik = _resolve_cik(ticker)
    frame = cached_fetch(
        source="sec_edgar",
        identifier=f"companyfacts_{ticker.upper()}",
        as_of_date=date.today(),
        fetch_fn=lambda: _fetch_company_facts(cik),
    )
    if frame.empty:
        return frame
    return frame.loc[frame["filed"] <= as_of_date]
