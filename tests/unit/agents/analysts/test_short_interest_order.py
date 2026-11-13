# test_short_interest_order.py

"""`ShortInterestSqueezeAnalyst` reads days-to-cover in the order
`load_short_interest_from_lakehouse` returns: by settlement date, a
settlement's unrevised row before its revised one."""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from agentic_trading.agents.analysts.event_credit_pod import short_interest_squeeze_analyst
from agentic_trading.agents.analysts.event_credit_pod.short_interest_squeeze_analyst import (
    ShortInterestSqueezeAnalyst,
)

_MODULE = "agentic_trading.agents.analysts.event_credit_pod.short_interest_squeeze_analyst"


def _history(settlements: int) -> pd.DataFrame:
    """`settlements` semimonthly settlements as the loader returns them, each but the last with
    an unrevised row and then a revised one carrying a different days-to-cover."""
    dates = pd.date_range("2015-01-15", periods=settlements, freq="SMS-15")
    rows: list[dict[str, Any]] = []
    for number, settlement in enumerate(dates):
        revisions = (0, 1) if number < settlements - 1 else (0,)
        rows.extend(
            {
                "settlement_date": settlement,
                "dissemination_date": settlement + pd.Timedelta(days=12),
                "short_interest_shares": 1_000_000.0 + number,
                "days_to_cover": 2.0 + number + 0.5 * revision,
                "avg_daily_volume": 500_000.0,
            }
            for revision in revisions
        )
    return pd.DataFrame(rows)


def test_days_to_cover_keep_the_loaders_order_within_a_settlement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With many settlements holding an unrevised and a revised row, the days-to-cover series
    the z-score reads is the loader's, unrevised before revised in every settlement."""
    history = _history(60)
    seen: list[np.ndarray] = []
    spread_zscore = short_interest_squeeze_analyst.spread_zscore

    def _recording_zscore(values: np.ndarray, window: int) -> np.ndarray:
        seen.append(np.array(values, copy=True))
        return spread_zscore(values, window)

    monkeypatch.setattr(f"{_MODULE}.load_short_interest_from_lakehouse", lambda *_: history)
    monkeypatch.setattr(f"{_MODULE}.spread_zscore", _recording_zscore)

    ShortInterestSqueezeAnalyst().generate_signal(
        "AAPL", pd.DataFrame(), pd.Timestamp("2017-06-30")
    )

    assert len(seen) == 1
    np.testing.assert_array_equal(seen[0], history["days_to_cover"].to_numpy(dtype=np.float64))
