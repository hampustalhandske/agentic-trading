"""Unit tests for agentic_trading.data.loader.

Mock call-shape convention chosen for this test file: `load_price_history`
is expected to call
`yfinance.download(asset, start=start, end=end, multi_level_index=False,
progress=False)` — a single positional ticker plus `start`/`end`/
`multi_level_index`/`progress` keyword arguments. `multi_level_index=False`
is required because yfinance 1.6.0 defaults `download()` to
`multi_level_index=True`, which returns MultiIndex `(Price, Ticker)`
columns even for a single ticker; passing `multi_level_index=False`
explicitly is what makes yfinance itself collapse to flat columns.
`progress=False` suppresses yfinance's own "N of M completed" progress bar,
which it otherwise prints straight to stdout regardless of any logging
configuration. The mocked frame mirrors yfinance's real single-ticker
output shape *when called with `multi_level_index=False`*: a
`DatetimeIndex` named "Date" and capitalized OHLCV columns
(Open/High/Low/Close/Volume). The implementer should match this exact call
shape so these tests pass without being rewritten around the
implementation.
"""

import math
from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from agentic_trading.data.loader import load_price_history, load_price_history_with_fallback


def _mock_yfinance_frame() -> pd.DataFrame:
    index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    index.name = "Date"
    return pd.DataFrame(
        {
            "Open": [99.0, 104.0, 103.0],
            "High": [101.0, 106.0, 104.0],
            "Low": [98.0, 103.0, 101.0],
            "Close": [100.0, 105.0, 102.9],
            "Volume": [1000, 1200, 900],
        },
        index=index,
    )


@patch("agentic_trading.data.loader.yf.download")
def test_calls_yfinance_with_ticker_and_date_range(mock_download: MagicMock) -> None:
    """load_price_history must forward the exact ticker and date range to yfinance."""
    mock_download.return_value = _mock_yfinance_frame()
    start, end = date(2024, 1, 2), date(2024, 1, 4)

    load_price_history("AAPL", start, end)

    mock_download.assert_called_once_with(
        "AAPL", start=start, end=end, multi_level_index=False, progress=False
    )


@patch("agentic_trading.data.loader.yf.download")
def test_result_has_datetime_index_and_expected_columns(mock_download: MagicMock) -> None:
    """The returned frame must be indexed by date and expose close/log_return columns."""
    mock_download.return_value = _mock_yfinance_frame()

    result = load_price_history("AAPL", date(2024, 1, 2), date(2024, 1, 4))

    assert isinstance(result.index, pd.DatetimeIndex)
    assert "close" in result.columns
    assert "log_return" in result.columns
    assert len(result) == 3


@patch("agentic_trading.data.loader.yf.download")
def test_log_return_matches_log_price_ratio_not_simple_return(mock_download: MagicMock) -> None:
    """Returns must be log(P_t / P_{t-1}), the convention quantcore's GBM solver assumes."""
    mock_download.return_value = _mock_yfinance_frame()

    result = load_price_history("AAPL", date(2024, 1, 2), date(2024, 1, 4))

    expected_log_returns = [math.log(105.0 / 100.0), math.log(102.9 / 105.0)]
    simple_returns = [(105.0 - 100.0) / 100.0, (102.9 - 105.0) / 105.0]

    actual = result["log_return"].to_numpy()[1:]
    assert actual == pytest.approx(expected_log_returns)
    assert not (actual == pytest.approx(simple_returns))


@patch("agentic_trading.data.loader.yf.download")
def test_first_row_log_return_is_nan_and_row_is_kept(mock_download: MagicMock) -> None:
    """The first day has no prior close, so log_return is NaN there rather than dropped."""
    mock_download.return_value = _mock_yfinance_frame()

    result = load_price_history("AAPL", date(2024, 1, 2), date(2024, 1, 4))

    assert len(result) == 3
    assert math.isnan(result["log_return"].iloc[0])
    assert result["close"].iloc[0] == pytest.approx(100.0)
    assert result.index[0] == pd.Timestamp("2024-01-02")


@patch("agentic_trading.data.loader.yf.download")
def test_empty_yfinance_response_raises_value_error(mock_download: MagicMock) -> None:
    """An invalid ticker yields an empty frame from yfinance; this must raise, not pass through."""
    mock_download.return_value = pd.DataFrame()

    with pytest.raises(ValueError):
        load_price_history("NOT_A_REAL_TICKER", date(2024, 1, 2), date(2024, 1, 4))


@patch("agentic_trading.data.loader.yf.download")
def test_with_fallback_returns_multiindex_for_a_single_ticker(
    mock_download: MagicMock,
) -> None:
    """Always MultiIndex (ticker, field) columns, even for one ticker, so
    every downstream consumer only has to handle one shape."""
    mock_download.return_value = _mock_yfinance_frame()

    frame, source_label = load_price_history_with_fallback(
        ["AAPL"], date(2024, 1, 2), date(2024, 1, 4)
    )

    assert isinstance(frame.columns, pd.MultiIndex)
    assert "AAPL" in frame.columns.get_level_values(0)
    assert source_label == "live yfinance data"


@patch("agentic_trading.data.loader.yf.download")
def test_with_fallback_uses_synthetic_data_when_yfinance_raises(
    mock_download: MagicMock,
) -> None:
    mock_download.side_effect = ConnectionError("simulated network failure")

    frame, source_label = load_price_history_with_fallback(
        ["AAPL", "MSFT"], date(2024, 1, 2), date(2024, 1, 12)
    )

    assert source_label == "synthetic fallback data"
    assert isinstance(frame.columns, pd.MultiIndex)
    assert set(frame.columns.get_level_values(0)) == {"AAPL", "MSFT"}
    assert not frame.empty


@patch("agentic_trading.data.loader.yf.download")
def test_with_fallback_uses_synthetic_data_when_yfinance_returns_empty(
    mock_download: MagicMock,
) -> None:
    mock_download.return_value = pd.DataFrame()

    frame, source_label = load_price_history_with_fallback(
        ["AAPL"], date(2024, 1, 2), date(2024, 1, 12)
    )

    assert source_label == "synthetic fallback data"
    assert not frame.empty


def test_synthetic_fallback_is_deterministic_for_the_same_ticker() -> None:
    """Not randomness in the trading sense — same ticker, same series."""
    from agentic_trading.data.loader import _synthetic_price_history

    first = _synthetic_price_history("AAPL", date(2024, 1, 2), date(2024, 1, 12))
    second = _synthetic_price_history("AAPL", date(2024, 1, 2), date(2024, 1, 12))

    pd.testing.assert_frame_equal(first, second)
