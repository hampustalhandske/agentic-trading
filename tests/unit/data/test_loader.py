"""Unit tests for agentic_trading.data.loader.

The lakehouse query is mocked at the module boundary
(`agentic_trading.data.loader.load_price_history_from_lakehouse`) — no
DuckDB/parquet access in this suite.
"""

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from agentic_trading.data.loader import load_price_history_with_fallback


def _mock_lakehouse_frame(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    index = pd.bdate_range(start, end)
    frames = {
        ticker: pd.DataFrame(
            {"close": [100.0 + i for i in range(len(index))], "log_return": 0.0}, index=index
        )
        for ticker in tickers
    }
    return pd.concat(frames, axis=1)


@patch("agentic_trading.data.loader.load_price_history_from_lakehouse")
def test_with_fallback_returns_multiindex_for_a_single_ticker(
    mock_lakehouse: MagicMock,
) -> None:
    """Always MultiIndex (ticker, field) columns, even for one ticker, so
    every downstream consumer only has to handle one shape."""
    mock_lakehouse.return_value = _mock_lakehouse_frame(
        ["AAPL"], date(2024, 1, 2), date(2024, 1, 4)
    )

    frame, source_label = load_price_history_with_fallback(
        ["AAPL"], date(2024, 1, 2), date(2024, 1, 4)
    )

    assert isinstance(frame.columns, pd.MultiIndex)
    assert "AAPL" in frame.columns.get_level_values(0)
    assert source_label == "lakehouse data"


@patch("agentic_trading.data.loader.load_price_history_from_lakehouse")
def test_with_fallback_raises_when_lakehouse_has_no_coverage(
    mock_lakehouse: MagicMock,
) -> None:
    """This project never calls a live price API or fabricates data during
    a run — an uncovered ticker is a loud failure, not a silent
    degradation to synthetic data."""
    mock_lakehouse.return_value = None

    with pytest.raises(ValueError, match="AAPL"):
        load_price_history_with_fallback(["AAPL"], date(2024, 1, 2), date(2024, 1, 12))


@patch("agentic_trading.data.loader.load_price_history_from_lakehouse")
def test_with_fallback_raises_when_lakehouse_covers_only_some_tickers(
    mock_lakehouse: MagicMock,
) -> None:
    mock_lakehouse.return_value = _mock_lakehouse_frame(
        ["AAPL"], date(2024, 1, 2), date(2024, 1, 12)
    )

    with pytest.raises(ValueError, match="MSFT"):
        load_price_history_with_fallback(["AAPL", "MSFT"], date(2024, 1, 2), date(2024, 1, 12))


@patch("agentic_trading.data.loader.load_price_history_from_lakehouse")
def test_with_fallback_widens_the_query_start_by_warmup_days(
    mock_lakehouse: MagicMock,
) -> None:
    mock_lakehouse.return_value = _mock_lakehouse_frame(
        ["AAPL"], date(2023, 12, 1), date(2024, 1, 12)
    )

    load_price_history_with_fallback(["AAPL"], date(2024, 1, 2), date(2024, 1, 12), warmup_days=10)

    _, query_start, query_end = mock_lakehouse.call_args.args
    assert query_start == date(2023, 12, 23)
    assert query_end == date(2024, 1, 12)


def test_with_fallback_rejects_negative_warmup_days() -> None:
    with pytest.raises(ValueError, match="warmup_days"):
        load_price_history_with_fallback(["AAPL"], date(2024, 1, 2), date(2024, 1, 12), -1)
