# test_cot_fx_sign.py

"""The CFTC currency future each FX pair maps to is long the pair's non-USD
currency against USD. For a USD-quoted pair (EURUSD, GBPUSD, AUDUSD) a long
future is long the pair; for a USD-base pair (USDJPY) it is short the pair,
so the COT analyst's direction there is the negation of the future's."""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from agentic_trading.agents.analysts.macro_rates_pod import cot_positioning_analyst
from agentic_trading.agents.analysts.macro_rates_pod.cot_positioning_analyst import (
    COTPositioningAnalyst,
)
from agentic_trading.data.loaders.lakehouse_loader import _COT_TFF_CODES


def _reports(newest: float) -> pd.DataFrame:
    """Thirty weekly reports oscillating around 0, the newest at `newest`."""
    report_dates = pd.date_range(end="2024-06-11", periods=30, freq="W-TUE")
    ratios = 0.05 * np.sin(np.arange(30.0))
    ratios[-1] = newest
    return pd.DataFrame(
        {
            "net_positioning_ratio": ratios,
            "published_date": report_dates + pd.Timedelta(days=3),
        },
        index=report_dates,
    )


@pytest.mark.parametrize(
    ("asset", "crowded_long_direction"),
    [
        ("EURUSD=X", "short"),
        ("GBPUSD=X", "short"),
        ("AUDUSD=X", "short"),
        ("USDJPY=X", "long"),
    ],
)
def test_crowded_long_currency_future_reads_through_the_pair_direction(
    monkeypatch: pytest.MonkeyPatch, asset: str, crowded_long_direction: str
) -> None:
    """A crowded long in the currency future is contrarian-bearish on that currency: short
    a USD-quoted pair, long USDJPY. A crowded short reverses both."""
    opposite = {"long": "short", "short": "long"}[crowded_long_direction]
    analyst = COTPositioningAnalyst()
    for newest, expected in ((0.5, crowded_long_direction), (-0.5, opposite)):
        frame = _reports(newest)

        def reports(*_args: Any, frame: pd.DataFrame = frame, **_kwargs: Any) -> pd.DataFrame:
            return frame

        monkeypatch.setattr(cot_positioning_analyst, "load_cot_reports_from_lakehouse", reports)

        signal = analyst.generate_signal(asset, pd.DataFrame(), pd.Timestamp("2024-06-17"))

        assert signal.status == "active"
        assert signal.direction == expected, (asset, newest)
        assert signal.confidence > 0.0


def test_every_fx_pair_with_a_cot_code_is_covered_above() -> None:
    """The pairs parametrized above are every FX pair the TFF mapping names."""
    fx_pairs = {ticker for ticker in _COT_TFF_CODES if ticker.endswith("=X")}
    assert fx_pairs == {"EURUSD=X", "GBPUSD=X", "AUDUSD=X", "USDJPY=X"}
