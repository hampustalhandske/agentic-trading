# test_ohlc_sanity.py

"""`load_price_history_from_lakehouse` blanks the `open` of any
`instrument_prices` row whose raw open lies outside its raw [low, high]
range (or whose low exceeds its high), keeps the rest of the row, and logs
one warning per call naming each affected ticker with its masked-row count."""

import logging
import re
from collections.abc import Iterable, Iterator
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import pytest

from agentic_trading.data.loaders import lakehouse_loader

_ADJ_RATIO = 0.5
_DAYS = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]


class _LiveRecords:
    """Iterable over whatever `caplog` holds at iteration time."""

    def __init__(self, caplog: pytest.LogCaptureFixture) -> None:
        self._caplog = caplog

    def __iter__(self) -> Iterator[logging.LogRecord]:
        return iter(self._caplog.records)


@pytest.fixture
def records_seen(caplog: pytest.LogCaptureFixture) -> Iterator[Iterable[logging.LogRecord]]:
    """Routes the package logger (which does not propagate) into `caplog`
    and yields a view that reads `caplog.records` each time it is iterated
    (pytest replaces that list when the test phase starts)."""
    package_logger = logging.getLogger("agentic_trading")
    package_logger.addHandler(caplog.handler)
    previous_level = package_logger.level
    package_logger.setLevel(logging.DEBUG)
    yield _LiveRecords(caplog)
    package_logger.setLevel(previous_level)
    package_logger.removeHandler(caplog.handler)


def _row(ticker: str, day: str, open_: float, high: float, low: float) -> tuple[Any, ...]:
    return (ticker, day, open_, high, low, 10.0, 10.0 * _ADJ_RATIO, 1_000_000)


def _seed(lake: Any, rows: list[tuple[Any, ...]]) -> None:
    lake.add(
        "instrument_prices",
        pd.DataFrame(
            rows,
            columns=["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"],
        ),
        date_columns=("date",),
    )


def _load(tickers: list[str]) -> pd.DataFrame:
    frame = lakehouse_loader.load_price_history_from_lakehouse(
        tickers, date(2024, 1, 1), date(2024, 1, 31)
    )
    assert frame is not None
    return frame


def test_open_outside_raw_range_is_masked_and_the_rest_of_the_row_kept(
    fake_lake: Any, records_seen: Iterable[logging.LogRecord]
) -> None:
    """Out-of-range and low>high opens become NaN; boundaries and other fields survive."""
    eur = [
        _row("EURUSD=X", _DAYS[0], 10.0, 11.0, 9.0),
        _row("EURUSD=X", _DAYS[1], 11.5, 11.0, 9.0),
        _row("EURUSD=X", _DAYS[2], 8.5, 11.0, 9.0),
        _row("EURUSD=X", _DAYS[3], 9.0, 11.0, 9.0),
        _row("EURUSD=X", _DAYS[4], 11.0, 11.0, 9.0),
    ]
    gbp = [
        _row("GBPUSD=X", _DAYS[0], 10.0, 9.0, 11.0),
        _row("GBPUSD=X", _DAYS[1], 10.0, 11.0, 9.0),
    ]
    aapl = [_row("AAPL", day, 10.0, 11.0, 9.0) for day in _DAYS[:2]]
    _seed(fake_lake, eur + gbp + aapl)

    frame = _load(["EURUSD=X", "GBPUSD=X", "AAPL"])

    eur_open = frame[("EURUSD=X", "open")]
    assert eur_open.isna().tolist() == [False, True, True, False, False]
    assert eur_open.dropna().tolist() == pytest.approx(
        [10.0 * _ADJ_RATIO, 9.0 * _ADJ_RATIO, 11.0 * _ADJ_RATIO]
    )
    gbp_open = frame[("GBPUSD=X", "open")].dropna()
    assert gbp_open.index.tolist() == [pd.Timestamp(_DAYS[1])]
    assert gbp_open.iloc[0] == pytest.approx(10.0 * _ADJ_RATIO)
    assert frame[("AAPL", "open")].dropna().tolist() == pytest.approx([5.0, 5.0])

    for ticker in ("EURUSD=X", "GBPUSD=X"):
        assert frame[(ticker, "close")].dropna().eq(10.0 * _ADJ_RATIO).all()
        assert frame[(ticker, "raw_close")].dropna().eq(10.0).all()
        assert frame[(ticker, "volume")].dropna().eq(1_000_000).all()
    assert frame[("EURUSD=X", "close")].notna().all()
    assert np.isfinite(frame[("EURUSD=X", "log_return")].iloc[1:]).all()
    assert frame[("GBPUSD=X", "close")].loc[pd.Timestamp(_DAYS[0])] == pytest.approx(5.0)

    warnings = [r for r in records_seen if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "AAPL" not in message
    assert re.search(r"EURUSD=X\D{0,6}2\b", message)
    assert re.search(r"GBPUSD=X\D{0,6}1\b", message)


def test_clean_prices_log_no_warning(
    fake_lake: Any, records_seen: Iterable[logging.LogRecord]
) -> None:
    """A call with no masked row emits no WARNING."""
    _seed(fake_lake, [_row("AAPL", day, 10.0, 11.0, 9.0) for day in _DAYS])

    frame = _load(["AAPL"])

    assert frame[("AAPL", "open")].notna().all()
    assert not [r for r in records_seen if r.levelno >= logging.WARNING]
