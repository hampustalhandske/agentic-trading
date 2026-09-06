# sec_fulltext_loader.py

"""SEC EDGAR full-text search (the `efts.sec.gov/LATEST/search-index`
endpoint that backs the public EDGAR full-text search UI), used here to
find Form 4 insider-transaction filings and 10-K/10-Q filings mentioning a
company. Returns filing-level metadata (form type, filed date, accession
number) — not per-transaction detail (buy/sell code, share count), which
would require fetching and parsing each filing's own XML separately; a
caller that needs that detail fetches it per accession number this loader
returns.

Gates on `filed`, the filing's own publish date — the same rule
`fundamentals_loader.py` applies to SEC EDGAR company facts, for the same
reason (a filing is never visible before it's actually filed)."""

from datetime import date

import pandas as pd
import requests

from agentic_trading.data.cache import cached_fetch
from agentic_trading.data.loaders._env import require_env
from agentic_trading.data.loaders.fundamentals_loader import _resolve_cik
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"


def _headers() -> dict[str, str]:
    return {"User-Agent": require_env("SEC_EDGAR_USER_AGENT")}


def _fetch_filings(cik: int, forms: str) -> pd.DataFrame:
    logger.info("Searching SEC EDGAR full-text index for CIK %010d forms=%s", cik, forms)
    response = requests.get(
        _SEARCH_URL,
        params={"ciks": f"{cik:010d}", "forms": forms},
        headers=_headers(),
        timeout=30,
    )
    response.raise_for_status()
    hits = response.json().get("hits", {}).get("hits", [])
    rows = [
        {
            "accession_no": hit.get("_id", "").split(":")[0],
            "form_type": ",".join(hit.get("_source", {}).get("forms", [])),
            "entity_name": ",".join(hit.get("_source", {}).get("display_names", [])),
            "filed": hit.get("_source", {}).get("file_date"),
        }
        for hit in hits
    ]
    frame = pd.DataFrame(rows, columns=["accession_no", "form_type", "entity_name", "filed"])
    if not frame.empty:
        frame["filed"] = pd.to_datetime(frame["filed"])
    return frame


def load_filings(ticker: str, as_of_date: pd.Timestamp, forms: str = "4") -> pd.DataFrame:
    """Filing-level metadata for `ticker`'s `forms` (SEC form-type filter,
    e.g. `"4"` for insider transactions or `"10-K,10-Q"`), filtered to
    `filed <= as_of_date`."""
    cik = _resolve_cik(ticker)
    frame = cached_fetch(
        source="sec_fulltext",
        identifier=f"{ticker.upper()}_{forms}",
        as_of_date=date.today(),
        fetch_fn=lambda: _fetch_filings(cik, forms),
    )
    if frame.empty:
        return frame
    return frame.loc[frame["filed"] <= as_of_date]
