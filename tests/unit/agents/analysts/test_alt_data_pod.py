from unittest.mock import patch

import numpy as np
import pandas as pd

from agentic_trading.agents.analysts.alt_data_pod.calendar_seasonality_analyst import (
    CalendarSeasonalityAnalyst,
)
from agentic_trading.agents.analysts.alt_data_pod.network_centrality_analyst import (
    LeadLagNetworkCentralityAnalyst,
)
from agentic_trading.data.loaders.lakehouse_loader import IndexMember


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


_LEAD_LAG_TICKERS = ("AAPL", "MSFT", *(f"TCK{i}" for i in range(33)))
_LEAD_LAG_MEMBERS = [
    IndexMember(security_id=i, ticker=ticker) for i, ticker in enumerate(_LEAD_LAG_TICKERS)
]


def _lead_lag_security_returns(n: int = 71) -> dict[int, pd.Series]:
    """Dated returns by security, the shape `load_security_log_returns`
    gives: AAPL repeats MSFT's return one day later, every other member is
    flat."""
    dates = pd.bdate_range(end="2024-06-19", periods=n)
    msft_returns = np.array([0.01 if i % 2 == 0 else -0.01 for i in range(n)])
    aapl_returns = np.zeros(n)
    aapl_returns[1:] = msft_returns[:-1]

    returns = {}
    for member in _LEAD_LAG_MEMBERS:
        if member.ticker == "MSFT":
            values = msft_returns
        elif member.ticker == "AAPL":
            values = aapl_returns
        else:
            values = np.zeros(n)
        returns[member.security_id] = pd.Series(values, index=dates)
    return returns


@patch(
    "agentic_trading.agents.analysts.alt_data_pod.network_centrality_analyst."
    "load_security_log_returns"
)
@patch(
    "agentic_trading.agents.analysts.alt_data_pod.network_centrality_analyst."
    "load_sp500_member_securities_asof"
)
def test_lead_lag_network_centrality_analyst_follows_its_leader(mock_members, mock_load) -> None:
    mock_members.return_value = _LEAD_LAG_MEMBERS
    mock_load.return_value = _lead_lag_security_returns()

    signal = LeadLagNetworkCentralityAnalyst().generate_signal(
        "AAPL", pd.DataFrame(), pd.Timestamp("2024-06-19")
    )

    assert signal.status == "active"
    assert signal.direction == "long"
    assert signal.confidence > 0.0


@patch(
    "agentic_trading.agents.analysts.alt_data_pod.network_centrality_analyst."
    "load_security_log_returns"
)
@patch(
    "agentic_trading.agents.analysts.alt_data_pod.network_centrality_analyst."
    "load_sp500_member_securities_asof"
)
def test_lead_lag_network_centrality_analyst_unavailable_with_insufficient_history(
    mock_members, mock_load
) -> None:
    """Members with too little history leave no cross-section: unavailable."""
    mock_members.return_value = _LEAD_LAG_MEMBERS
    mock_load.return_value = _lead_lag_security_returns(n=10)

    signal = LeadLagNetworkCentralityAnalyst().generate_signal(
        "AAPL", pd.DataFrame(), pd.Timestamp("2024-06-19")
    )

    assert signal.status == "unavailable"
    assert signal.direction == "flat"
    assert signal.confidence == 0.0


@patch(
    "agentic_trading.agents.analysts.alt_data_pod.network_centrality_analyst."
    "load_sp500_member_securities_asof"
)
def test_lead_lag_network_centrality_analyst_unavailable_without_membership(
    mock_members,
) -> None:
    mock_members.return_value = None

    signal = LeadLagNetworkCentralityAnalyst().generate_signal(
        "AAPL", pd.DataFrame(), pd.Timestamp("2024-06-19")
    )

    assert signal.status == "unavailable"
    assert signal.direction == "flat"
    assert signal.confidence == 0.0


@patch(
    "agentic_trading.agents.analysts.alt_data_pod.network_centrality_analyst."
    "load_security_log_returns"
)
@patch(
    "agentic_trading.agents.analysts.alt_data_pod.network_centrality_analyst."
    "load_sp500_member_securities_asof"
)
def test_lead_lag_network_centrality_analyst_unavailable_for_a_non_member(
    mock_members, mock_load
) -> None:
    """An asset outside the day's S&P 500 membership has no place in the graph."""
    mock_members.return_value = _LEAD_LAG_MEMBERS
    mock_load.return_value = _lead_lag_security_returns()

    signal = LeadLagNetworkCentralityAnalyst().generate_signal(
        "SPY", pd.DataFrame(), pd.Timestamp("2024-06-19")
    )

    assert signal.status == "unavailable"
    assert signal.direction == "flat"
    assert signal.confidence == 0.0


@patch(
    "agentic_trading.agents.analysts.alt_data_pod.network_centrality_analyst."
    "load_security_log_returns"
)
@patch(
    "agentic_trading.agents.analysts.alt_data_pod.network_centrality_analyst."
    "load_sp500_member_securities_asof"
)
def test_lead_lag_network_centrality_analyst_errors_when_returns_download_fails(
    mock_members, mock_load
) -> None:
    mock_members.return_value = _LEAD_LAG_MEMBERS
    mock_load.side_effect = RuntimeError("network down")

    signal = LeadLagNetworkCentralityAnalyst().generate_signal(
        "AAPL", pd.DataFrame(), pd.Timestamp("2024-06-19")
    )

    assert signal.status == "error"
    assert signal.direction == "flat"
    assert signal.confidence == 0.0
    assert "lookup failed" in signal.rationale


@patch(
    "agentic_trading.agents.analysts.alt_data_pod.network_centrality_analyst."
    "load_security_log_returns"
)
@patch(
    "agentic_trading.agents.analysts.alt_data_pod.network_centrality_analyst."
    "load_sp500_member_securities_asof"
)
def test_lead_lag_network_centrality_analyst_builds_on_sp500_membership(
    mock_members, mock_prices
) -> None:
    """The graph is built across the point-in-time S&P 500 membership
    priced by security."""
    broad_universe = [f"TCK{i}" for i in range(35)]
    members = [
        IndexMember(security_id=i, ticker=ticker)
        for i, ticker in enumerate([*broad_universe, "AAPL"])
    ]
    mock_members.return_value = members
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2023-01-01", periods=260)
    mock_prices.return_value = {
        member.security_id: pd.Series(rng.normal(0.0, 0.01, len(dates)), index=dates)
        for member in members
    }

    signal = LeadLagNetworkCentralityAnalyst().generate_signal(
        "AAPL", pd.DataFrame(), pd.Timestamp("2024-01-15")
    )

    # The 36-name membership was fetched once and the signal completed
    # against it.
    mock_prices.assert_called_once()
    assert mock_members.call_count == 1
    assert signal.status == "active"
    # A member ticker shows up in the rationale's printed universe.
    assert "TCK0" in signal.rationale
