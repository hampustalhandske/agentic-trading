from unittest.mock import patch

import pandas as pd

from agentic_trading.agents.analysts.alt_data_pod.congressional_trading_analyst import (
    CongressionalTradingAnalyst,
)
from agentic_trading.agents.analysts.alt_data_pod.insider_cluster_buying_analyst import (
    InsiderClusterBuyingAnalyst,
)
from agentic_trading.agents.analysts.alt_data_pod.reddit_sentiment_analyst import (
    RedditSentimentAnalyst,
)
from agentic_trading.agents.analysts.alt_data_pod.retail_attention_analyst import (
    RetailAttentionAnalyst,
)
from agentic_trading.agents.analysts.alt_data_pod.wikipedia_pageviews_analyst import (
    WikipediaPageviewsAnalyst,
)

AS_OF = pd.Timestamp("2024-06-20")


@patch(
    "agentic_trading.agents.analysts.alt_data_pod.congressional_trading_analyst.load_congressional_trades"
)
def test_congressional_trading_analyst_long_on_purchase_imbalance(mock_load) -> None:
    mock_load.return_value = pd.DataFrame(
        {
            "ticker": ["GME"] * 3,
            "filer": ["A", "B", "C"],
            "disclosure_date": pd.to_datetime(["2024-06-01", "2024-06-05", "2024-06-10"]),
            "transaction_type": ["Purchase", "Purchase", "Sale (Full)"],
        }
    )

    signal = CongressionalTradingAnalyst().generate_signal("GME", pd.DataFrame(), AS_OF)

    assert signal.direction == "long"
    assert signal.confidence > 0.0
    assert "[sample_count=3]" in signal.rationale


@patch(
    "agentic_trading.agents.analysts.alt_data_pod.congressional_trading_analyst.load_congressional_trades"
)
def test_congressional_trading_analyst_flat_with_no_disclosures(mock_load) -> None:
    mock_load.return_value = pd.DataFrame(
        columns=["ticker", "filer", "disclosure_date", "transaction_type"]
    )

    signal = CongressionalTradingAnalyst().generate_signal("GME", pd.DataFrame(), AS_OF)

    assert signal.direction == "flat"
    assert signal.confidence == 0.0


@patch("agentic_trading.agents.analysts.alt_data_pod.insider_cluster_buying_analyst.load_filings")
def test_insider_cluster_buying_analyst_long_on_filing_cluster(mock_load) -> None:
    mock_load.return_value = pd.DataFrame(
        {
            "accession_no": ["0001", "0002", "0003"],
            "form_type": ["4", "4", "4"],
            "entity_name": ["Issuer, Insider A", "Issuer, Insider B", "Issuer, Insider C"],
            "filed": pd.to_datetime(["2024-06-12", "2024-06-14", "2024-06-18"]),
        }
    )

    signal = InsiderClusterBuyingAnalyst().generate_signal("AAPL", pd.DataFrame(), AS_OF)

    assert signal.direction == "long"
    assert signal.confidence > 0.0
    assert "[sample_count=3]" in signal.rationale


@patch("agentic_trading.agents.analysts.alt_data_pod.insider_cluster_buying_analyst.load_filings")
def test_insider_cluster_buying_analyst_flat_below_cluster_threshold(mock_load) -> None:
    mock_load.return_value = pd.DataFrame(
        {
            "accession_no": ["0001"],
            "form_type": ["4"],
            "entity_name": ["Issuer, Insider A"],
            "filed": pd.to_datetime(["2024-06-18"]),
        }
    )

    signal = InsiderClusterBuyingAnalyst().generate_signal("AAPL", pd.DataFrame(), AS_OF)

    assert signal.direction == "flat"
    assert signal.confidence == 0.0


@patch("agentic_trading.agents.analysts.alt_data_pod.retail_attention_analyst.load_search_interest")
def test_retail_attention_analyst_long_on_search_spike(mock_load) -> None:
    baseline = [10.0, 11.0] * 10
    index = pd.date_range("2024-05-01", periods=21, freq="D")
    mock_load.return_value = pd.DataFrame({"interest": [*baseline, 90.0]}, index=index)

    signal = RetailAttentionAnalyst().generate_signal("AAPL", pd.DataFrame(), AS_OF)

    assert signal.direction == "long"
    assert signal.confidence > 0.0


@patch("agentic_trading.agents.analysts.alt_data_pod.retail_attention_analyst.load_search_interest")
def test_retail_attention_analyst_flat_without_spike(mock_load) -> None:
    index = pd.date_range("2024-05-01", periods=20, freq="D")
    mock_load.return_value = pd.DataFrame({"interest": [10.0] * 20}, index=index)

    signal = RetailAttentionAnalyst().generate_signal("AAPL", pd.DataFrame(), AS_OF)

    assert signal.direction == "flat"
    assert signal.confidence == 0.0


@patch("agentic_trading.agents.analysts.alt_data_pod.wikipedia_pageviews_analyst.load_pageviews")
def test_wikipedia_pageviews_analyst_long_on_view_spike(mock_load) -> None:
    baseline = [1000.0, 1100.0] * 10
    index = pd.date_range("2024-05-01", periods=21, freq="D")
    mock_load.return_value = pd.DataFrame({"views": [*baseline, 9000.0]}, index=index)

    signal = WikipediaPageviewsAnalyst().generate_signal("AAPL", pd.DataFrame(), AS_OF)

    assert signal.direction == "long"
    assert signal.confidence > 0.0


def test_wikipedia_pageviews_analyst_flat_for_unmapped_ticker() -> None:
    signal = WikipediaPageviewsAnalyst().generate_signal("ZZZZ", pd.DataFrame(), AS_OF)

    assert signal.direction == "flat"
    assert signal.confidence == 0.0
    assert "no Wikipedia article mapping" in signal.rationale


@patch("agentic_trading.agents.analysts.alt_data_pod.reddit_sentiment_analyst.load_mention_counts")
def test_reddit_sentiment_analyst_long_on_positive_average_score(mock_load) -> None:
    mock_load.return_value = pd.DataFrame(
        {
            "created_utc": pd.to_datetime(
                ["2024-06-18", "2024-06-18", "2024-06-19", "2024-06-19", "2024-06-20"]
            ),
            "score": [10, 20, 30, 15, 25],
            "num_comments": [1, 2, 3, 1, 2],
        }
    )

    signal = RedditSentimentAnalyst().generate_signal("GME", pd.DataFrame(), AS_OF)

    assert signal.direction == "long"
    assert signal.confidence > 0.0
    assert "[sample_count=5]" in signal.rationale


@patch("agentic_trading.agents.analysts.alt_data_pod.reddit_sentiment_analyst.load_mention_counts")
def test_reddit_sentiment_analyst_flat_when_too_sparse(mock_load) -> None:
    mock_load.return_value = pd.DataFrame(
        {
            "created_utc": pd.to_datetime(["2024-06-20"]),
            "score": [10],
            "num_comments": [1],
        }
    )

    signal = RedditSentimentAnalyst().generate_signal("GME", pd.DataFrame(), AS_OF)

    assert signal.direction == "flat"
    assert signal.confidence == 0.0
