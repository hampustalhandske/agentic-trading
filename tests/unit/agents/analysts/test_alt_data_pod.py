from unittest.mock import patch

import numpy as np
import pandas as pd

from agentic_trading.agents.analysts.alt_data_pod.calendar_seasonality_analyst import (
    CalendarSeasonalityAnalyst,
)
from agentic_trading.agents.analysts.alt_data_pod.network_centrality_analyst import (
    DEFAULT_UNIVERSE,
    LeadLagNetworkCentralityAnalyst,
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


def _lead_lag_universe_returns(n: int = 71) -> dict[str, np.ndarray]:
    msft_returns = np.array([0.01 if i % 2 == 0 else -0.01 for i in range(n)])
    aapl_returns = np.zeros(n)
    aapl_returns[1:] = msft_returns[:-1]

    returns = {}
    for ticker in DEFAULT_UNIVERSE:
        if ticker == "MSFT":
            returns[ticker] = msft_returns
        elif ticker == "AAPL":
            returns[ticker] = aapl_returns
        else:
            returns[ticker] = np.zeros(n)
    return returns


@patch(
    "agentic_trading.agents.analysts.alt_data_pod.network_centrality_analyst."
    "load_sp500_constituents_asof"
)
@patch(
    "agentic_trading.agents.analysts.alt_data_pod.network_centrality_analyst."
    "load_fixed_universe_returns"
)
def test_lead_lag_network_centrality_analyst_follows_its_leader(mock_load, mock_members) -> None:
    """No real S&P membership available — falls back to the fixed
    DEFAULT_UNIVERSE, the same behaviour this test has always exercised."""
    mock_members.return_value = None
    mock_load.return_value = _lead_lag_universe_returns()

    signal = LeadLagNetworkCentralityAnalyst().generate_signal(
        "AAPL", pd.DataFrame(), pd.Timestamp("2024-06-19")
    )

    assert signal.direction == "long"
    assert signal.confidence > 0.0


@patch(
    "agentic_trading.agents.analysts.alt_data_pod.network_centrality_analyst."
    "load_sp500_constituents_asof"
)
@patch(
    "agentic_trading.agents.analysts.alt_data_pod.network_centrality_analyst."
    "load_fixed_universe_returns"
)
def test_lead_lag_network_centrality_analyst_flat_with_insufficient_history(
    mock_load, mock_members
) -> None:
    mock_members.return_value = None
    mock_load.return_value = _lead_lag_universe_returns(n=10)

    signal = LeadLagNetworkCentralityAnalyst().generate_signal(
        "AAPL", pd.DataFrame(), pd.Timestamp("2024-06-19")
    )

    assert signal.direction == "flat"
    assert signal.confidence == 0.0


@patch(
    "agentic_trading.agents.analysts.alt_data_pod.network_centrality_analyst."
    "load_sp500_constituents_asof"
)
@patch(
    "agentic_trading.agents.analysts.alt_data_pod.network_centrality_analyst."
    "load_fixed_universe_returns"
)
def test_lead_lag_network_centrality_analyst_flat_when_universe_download_fails(
    mock_load, mock_members
) -> None:
    mock_members.return_value = None
    mock_load.side_effect = RuntimeError("network down")

    signal = LeadLagNetworkCentralityAnalyst().generate_signal(
        "AAPL", pd.DataFrame(), pd.Timestamp("2024-06-19")
    )

    assert signal.direction == "flat"
    assert signal.confidence == 0.0
    assert "lookup failed" in signal.rationale


@patch(
    "agentic_trading.agents.analysts.alt_data_pod.network_centrality_analyst."
    "load_price_history_from_lakehouse"
)
@patch(
    "agentic_trading.agents.analysts.alt_data_pod.network_centrality_analyst."
    "load_sp500_constituents_asof"
)
def test_lead_lag_network_centrality_analyst_prefers_real_sp500_membership(
    mock_members, mock_prices
) -> None:
    """When the lakehouse resolves real, point-in-time S&P 500 membership
    with enough usable names, that broad universe is used instead of the
    5-name DEFAULT_UNIVERSE fallback."""
    broad_universe = [f"TCK{i}" for i in range(35)]
    mock_members.return_value = broad_universe
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2023-01-01", periods=260)
    frames = {}
    for ticker in [*broad_universe, "AAPL"]:
        returns = rng.normal(0.0, 0.01, len(dates))
        frames[ticker] = pd.DataFrame(
            {"open": 100.0, "close": 100.0, "log_return": returns}, index=dates
        )
    mock_prices.return_value = pd.concat(frames, axis=1)

    signal = LeadLagNetworkCentralityAnalyst().generate_signal(
        "AAPL", pd.DataFrame(), pd.Timestamp("2024-01-15")
    )

    # The real 36-name broad universe was fetched (proving the broad path
    # ran, not the 5-name DEFAULT_UNIVERSE fallback) and the signal
    # completed against it.
    mock_prices.assert_called_once()
    assert mock_members.call_count == 1
    assert signal.status == "active"
    # A broad-universe ticker (never part of DEFAULT_UNIVERSE) shows up in
    # the rationale's printed universe — proof the real 36-name fetch was
    # used, not the 5-name fixed fallback.
    assert "TCK0" in signal.rationale
    assert not set(DEFAULT_UNIVERSE) - {"AAPL"} & set(broad_universe)
