from datetime import timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from agentic_trading.data.loaders.options_loader import load_option_chain


@patch("agentic_trading.data.loaders.options_loader.yf.Ticker")
def test_past_as_of_date_raises_not_implemented(mock_ticker_cls: MagicMock) -> None:
    """No free source has historical options chains — a past as_of_date
    must raise rather than silently returning today's chain."""
    past_date = pd.Timestamp.today().normalize() - timedelta(days=5)

    with pytest.raises(NotImplementedError):
        load_option_chain("AAPL", past_date)

    mock_ticker_cls.assert_not_called()


@patch("agentic_trading.data.loaders.options_loader.yf.Ticker")
def test_today_returns_calls_and_puts(mock_ticker_cls: MagicMock) -> None:
    today = pd.Timestamp.today().normalize()
    expiration = (today + timedelta(days=30)).date().isoformat()
    mock_ticker = mock_ticker_cls.return_value
    mock_ticker.options = (expiration,)
    chain = MagicMock()
    chain.calls = pd.DataFrame(
        {"strike": [100.0], "bid": [1.0], "ask": [1.2], "openInterest": [10], "volume": [5]}
    )
    chain.puts = pd.DataFrame(
        {"strike": [100.0], "bid": [0.9], "ask": [1.1], "openInterest": [8], "volume": [3]}
    )
    mock_ticker.option_chain.return_value = chain

    result = load_option_chain("AAPL", today)

    assert set(result["option_type"]) == {"call", "put"}
    assert result["mid"].iloc[0] == pytest.approx(1.1)
    assert "impliedVolatility" not in result.columns
