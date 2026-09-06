from unittest.mock import patch

import numpy as np
import pandas as pd

from agentic_trading.agents.analysts.anomaly_network_pod.calendar_seasonality_analyst import (
    CalendarSeasonalityAnalyst,
)
from agentic_trading.agents.analysts.anomaly_network_pod.network_centrality_analyst import (
    DEFAULT_UNIVERSE,
    LeadLagNetworkCentralityAnalyst,
)
from agentic_trading.agents.analysts.anomaly_network_pod.options_skew_sentiment_analyst import (
    OptionsSkewSentimentAnalyst,
)


def _calendar_market_data() -> pd.DataFrame:
    index = pd.bdate_range("2020-01-01", "2024-06-19")
    weekday_return = {0: 0.01, 1: -0.005, 2: 0.0, 3: 0.002, 4: -0.001}
    returns = [weekday_return[day] for day in index.dayofweek]
    return pd.DataFrame({"close": 100.0, "log_return": returns}, index=index)


def test_calendar_seasonality_analyst_reads_weekday_effect() -> None:
    market_data = _calendar_market_data()
    # 2024-06-19 is a Wednesday; the next business day (Thursday) carries a
    # +0.002 historical mean under the synthetic weekday map above.
    as_of_date = pd.Timestamp("2024-06-19")

    signal = CalendarSeasonalityAnalyst().generate_signal("AAPL", market_data, as_of_date)

    assert signal.direction == "long"
    assert signal.confidence > 0.0
    assert "day-of-week" in signal.rationale


def test_calendar_seasonality_analyst_flat_with_insufficient_history() -> None:
    index = pd.bdate_range("2024-06-01", "2024-06-19")
    market_data = pd.DataFrame({"close": 100.0, "log_return": 0.001}, index=index)

    signal = CalendarSeasonalityAnalyst().generate_signal(
        "AAPL", market_data, pd.Timestamp("2024-06-19")
    )

    assert signal.direction == "flat"
    assert signal.confidence == 0.0


def _lead_lag_market_data(n: int = 71) -> pd.DataFrame:
    index = pd.bdate_range("2024-01-01", periods=n)
    msft_returns = np.array([0.01 if i % 2 == 0 else -0.01 for i in range(n)])
    aapl_returns = np.zeros(n)
    aapl_returns[1:] = msft_returns[:-1]

    frames = {}
    for ticker in DEFAULT_UNIVERSE:
        if ticker == "MSFT":
            returns = msft_returns
        elif ticker == "AAPL":
            returns = aapl_returns
        else:
            returns = np.zeros(n)
        frames[ticker] = pd.DataFrame({"close": 100.0, "log_return": returns}, index=index)
    return pd.concat(frames, axis=1)


def test_lead_lag_network_centrality_analyst_follows_its_leader() -> None:
    market_data = _lead_lag_market_data()
    as_of_date = market_data.index[-1]

    signal = LeadLagNetworkCentralityAnalyst().generate_signal("AAPL", market_data, as_of_date)

    assert signal.direction == "long"
    assert signal.confidence > 0.0


def test_lead_lag_network_centrality_analyst_flat_with_insufficient_history() -> None:
    market_data = _lead_lag_market_data(n=10)
    as_of_date = market_data.index[-1]

    signal = LeadLagNetworkCentralityAnalyst().generate_signal("AAPL", market_data, as_of_date)

    assert signal.direction == "flat"
    assert signal.confidence == 0.0


@patch(
    "agentic_trading.agents.analysts.anomaly_network_pod.options_skew_sentiment_analyst."
    "load_put_call_ratio"
)
def test_options_skew_sentiment_analyst_contrarian_long_on_elevated_ratio(mock_load) -> None:
    index = pd.bdate_range("2024-05-01", periods=21)
    values = [0.7] * 20 + [1.6]
    mock_load.return_value = pd.DataFrame({"put_call_ratio": values}, index=index)

    signal = OptionsSkewSentimentAnalyst().generate_signal(
        "SPY", pd.DataFrame(), pd.Timestamp("2024-06-01")
    )

    assert signal.direction == "long"
    assert signal.confidence > 0.0


@patch(
    "agentic_trading.agents.analysts.anomaly_network_pod.options_skew_sentiment_analyst."
    "load_put_call_ratio"
)
def test_options_skew_sentiment_analyst_contrarian_short_on_depressed_ratio(mock_load) -> None:
    index = pd.bdate_range("2024-05-01", periods=21)
    values = [0.7] * 20 + [0.1]
    mock_load.return_value = pd.DataFrame({"put_call_ratio": values}, index=index)

    signal = OptionsSkewSentimentAnalyst().generate_signal(
        "SPY", pd.DataFrame(), pd.Timestamp("2024-06-01")
    )

    assert signal.direction == "short"
    assert signal.confidence > 0.0
