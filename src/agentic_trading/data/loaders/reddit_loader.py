# reddit_loader.py

"""Reddit mention-volume via a script OAuth app (no per-post sentiment
scoring here — that lives in `agents/analysts`, this loader only returns
raw post metadata). Reddit's search endpoint only surfaces a limited
recent-history window (on the order of the last ~1000 results per query),
not a full multi-year archive — a real, documented limitation of this
source, not a bug.

Gates on each post's own `created_utc` timestamp: a post only counts if it
existed by `as_of_date`, enforced by filtering the search results rather
than trusting the query itself to never return something newer."""

import functools
from datetime import UTC, date, datetime

import pandas as pd
import requests

from agentic_trading.data.cache import cached_fetch
from agentic_trading.data.loaders._env import require_env
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

_TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
_SEARCH_URL = "https://oauth.reddit.com/r/{subreddit}/search"
_USER_AGENT = "agentic-trading-research/0.1 (data loader)"

DEFAULT_SUBREDDITS: tuple[str, ...] = ("wallstreetbets", "stocks")


def _get_access_token() -> str:
    client_id = require_env("REDDIT_CLIENT_ID")
    client_secret = require_env("REDDIT_CLIENT_SECRET")
    response = requests.post(
        _TOKEN_URL,
        data={"grant_type": "client_credentials"},
        auth=(client_id, client_secret),
        headers={"User-Agent": _USER_AGENT},
        timeout=30,
    )
    response.raise_for_status()
    return str(response.json()["access_token"])


def _fetch_mentions(ticker: str, subreddit: str) -> pd.DataFrame:
    token = _get_access_token()
    logger.info("Downloading Reddit mentions of %s in r/%s", ticker, subreddit)
    params: dict[str, str | int] = {"q": ticker, "restrict_sr": "on", "sort": "new", "limit": 100}
    response = requests.get(
        _SEARCH_URL.format(subreddit=subreddit),
        params=params,
        headers={"Authorization": f"bearer {token}", "User-Agent": _USER_AGENT},
        timeout=30,
    )
    response.raise_for_status()
    children = response.json().get("data", {}).get("children", [])
    rows = [
        {
            "created_utc": datetime.fromtimestamp(child["data"]["created_utc"], tz=UTC),
            "score": child["data"].get("score", 0),
            "num_comments": child["data"].get("num_comments", 0),
        }
        for child in children
    ]
    frame = pd.DataFrame(rows, columns=["created_utc", "score", "num_comments"])
    if not frame.empty:
        frame["created_utc"] = pd.to_datetime(frame["created_utc"]).dt.tz_localize(None)
    return frame


def load_mention_counts(
    ticker: str,
    as_of_date: pd.Timestamp,
    subreddits: tuple[str, ...] = DEFAULT_SUBREDDITS,
) -> pd.DataFrame:
    """Per-post Reddit mentions of `ticker` across `subreddits`, filtered
    to `created_utc <= as_of_date`. Aggregate to a daily mention count
    downstream with `result.groupby(result["created_utc"].dt.date).size()`."""
    frames = []
    for subreddit in subreddits:
        frame = cached_fetch(
            source="reddit",
            identifier=f"{ticker.upper()}_{subreddit}",
            as_of_date=date.today(),
            fetch_fn=functools.partial(_fetch_mentions, ticker, subreddit),
        )
        if not frame.empty:
            frame = frame.loc[frame["created_utc"] <= as_of_date].assign(subreddit=subreddit)
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["created_utc", "score", "num_comments", "subreddit"])
    return pd.concat(frames, ignore_index=True)
