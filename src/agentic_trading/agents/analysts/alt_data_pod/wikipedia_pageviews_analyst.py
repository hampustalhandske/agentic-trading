# wikipedia_pageviews_analyst.py

import numpy as np
import pandas as pd
from pydantic import Field

from agentic_trading.agents._market_math import clamp
from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.data.loaders.wikipedia_loader import load_pageviews
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

MIN_HISTORY_POINTS = 14
Z_CAP = 3.0

DEFAULT_ARTICLE_BY_TICKER: dict[str, str] = {
    "AAPL": "Apple_Inc.",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet_Inc.",
    "AMZN": "Amazon_(company)",
    "JPM": "JPMorgan_Chase",
}


class WikipediaPageviewsAnalyst(AnalystAgent):
    """Wikipedia page-view spike detection via
    `data/loaders/wikipedia_loader.py`: same attention-driven-momentum idea
    as `retail_attention_analyst`'s Google Trends z-score, but from an
    independently collected data source — Wikipedia's reader base skews
    differently from Google's search population, so the two proxies can
    diverge in informative ways, and agreement between two independent
    attention signals is stronger evidence than either alone (the
    ensemble-diversity reason both are included rather than just one).
    Requires a ticker→article-title mapping since Wikipedia has no ticker
    lookup API; unmapped tickers hold flat rather than guessing a
    URL-unsafe title."""

    agent_id: str = "wikipedia_pageviews_analyst"

    article_by_ticker: dict[str, str] = Field(
        default_factory=lambda: dict(DEFAULT_ARTICLE_BY_TICKER)
    )

    def generate_signal(
        self,
        asset: str,
        market_data: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> AgentSignal:
        article = self.article_by_ticker.get(asset.upper())
        if article is None:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale=f"no Wikipedia article mapping for {asset} — holding.",
            )

        try:
            pageviews = load_pageviews(article, as_of_date)
        except Exception as exc:
            logger.debug(
                "%s: Wikimedia pageviews lookup failed for %s (%s) — holding.",
                self.agent_id,
                asset,
                exc,
            )
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="Wikimedia pageviews lookup failed — holding.",
            )

        series = pageviews["views"].dropna() if "views" in pageviews.columns else pd.Series()
        if len(series) < MIN_HISTORY_POINTS:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale="insufficient Wikipedia pageview history for a z-score read — holding.",
            )

        values = series.to_numpy(dtype=np.float64)
        baseline, latest = values[:-1], values[-1]
        std = float(np.std(baseline, ddof=1))
        z_score = 0.0 if std == 0.0 else float((latest - np.mean(baseline)) / std)
        latest_date = series.index[-1]
        sample_count = len(series)

        if z_score <= 0.0:
            return AgentSignal(
                agent_id=self.agent_id,
                asset=asset,
                as_of_date=as_of_date.date(),
                direction="flat",
                confidence=0.0,
                rationale=(
                    f"pageviews not elevated (z={z_score:.2f}) "
                    f"[source_as_of={pd.Timestamp(latest_date).date().isoformat()}] "
                    f"[sample_count={sample_count}]"
                ),
            )

        confidence = clamp(min(z_score, Z_CAP) / Z_CAP)
        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction="long",
            confidence=confidence,
            rationale=(
                f"pageview spike (z={z_score:.2f}) — Wikipedia attention long bias "
                f"[source_as_of={pd.Timestamp(latest_date).date().isoformat()}] "
                f"[sample_count={sample_count}]"
            ),
        )
