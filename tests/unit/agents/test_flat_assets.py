# test_flat_assets.py

"""`PortfolioConstructor.construct` sizes an asset with no return variation
at 0.0 with a warning, leaves it out of the covariance estimate, and sizes
the whole book off view direction when no asset varies."""

import logging
import math
from collections.abc import Iterable, Iterator

import numpy as np
import pandas as pd
import pytest

from agentic_trading.agents.portfolio_construction import PodView, PortfolioConstructor

AS_OF = pd.Timestamp("2024-06-03")
METHODS = ["mean_variance", "risk_parity", "min_variance"]


class _LiveRecords:
    """Iterable over whatever `caplog` holds at iteration time."""

    def __init__(self, caplog: pytest.LogCaptureFixture) -> None:
        self._caplog = caplog

    def __iter__(self) -> Iterator[logging.LogRecord]:
        return iter(self._caplog.records)


@pytest.fixture
def records_seen(caplog: pytest.LogCaptureFixture) -> Iterator[Iterable[logging.LogRecord]]:
    """Routes the non-propagating package logger into `caplog`; yields a live view."""
    package_logger = logging.getLogger("agentic_trading")
    package_logger.addHandler(caplog.handler)
    previous_level = package_logger.level
    package_logger.setLevel(logging.DEBUG)
    yield _LiveRecords(caplog)
    package_logger.setLevel(previous_level)
    package_logger.removeHandler(caplog.handler)


def _views(signals: dict[str, float]) -> dict[str, PodView]:
    return {
        asset: PodView(net_signal=value, signal_count=1 if value else 0, confidence_sum=abs(value))
        for asset, value in signals.items()
    }


def _returns_history(assets: list[str], periods: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({asset: rng.normal(0.0005, 0.01, size=periods) for asset in assets})


@pytest.mark.parametrize("method", METHODS)
def test_flat_asset_is_zero_and_leaves_other_weights_unchanged(
    method: str, records_seen: Iterable[logging.LogRecord]
) -> None:
    """A flat asset is 0.0, keyed, named in a warning; the others match a book without it."""
    varying = ["AAPL", "MSFT", "GOOGL"]
    signals = {"AAPL": 0.6, "MSFT": -0.3, "GOOGL": 0.2}
    history = _returns_history(varying)
    mixed_history = history.assign(FLAT=np.zeros(len(history)))

    mixed = PortfolioConstructor(method=method).construct(
        _views({**signals, "FLAT": 0.5}), mixed_history, AS_OF
    )
    reference = PortfolioConstructor(method=method).construct(_views(signals), history, AS_OF)

    assert set(mixed) == {*varying, "FLAT"}
    assert mixed["FLAT"] == 0.0
    assert any(
        record.levelno == logging.WARNING and "FLAT" in record.getMessage()
        for record in records_seen
    )
    assert any(abs(weight) > 0.0 for weight in reference.values())
    for asset in varying:
        assert mixed[asset] == pytest.approx(reference[asset])


@pytest.mark.parametrize("method", METHODS)
def test_all_flat_book_is_sized_off_view_direction(method: str) -> None:
    """With every asset flat the result is finite, keyed per asset, and the view-direction book."""
    assets = ["AAPL", "MSFT", "GOOGL"]
    views = _views({"AAPL": 0.6, "MSFT": -0.4, "GOOGL": 0.0})
    history = pd.DataFrame(np.zeros((120, len(assets))), columns=assets)

    result = PortfolioConstructor(method=method).construct(views, history, AS_OF)

    expected = PortfolioConstructor(method=method)._view_direction_book(assets, views)
    assert set(result) == set(assets)
    assert all(math.isfinite(weight) for weight in result.values())
    assert result == pytest.approx(expected)
    assert result["AAPL"] > 0.0
    assert result["MSFT"] < 0.0
