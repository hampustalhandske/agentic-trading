# test_undefined_statistics.py

"""An analyst whose statistic is undefined on the day's data (a NaN z-score, a non-finite
half-life or p-value, a zero volatility) never trades on it: it reports `unavailable` when the
data cannot define the statistic and `error` when a fitted model's diagnostic is undefined.
Every loader is stubbed on the analyst's own module, so no test reads the lake."""

import pandas as pd
import pytest

from agentic_trading.agents.analysts.event_credit_pod.short_interest_squeeze_analyst import (
    ShortInterestSqueezeAnalyst,
)

_SHORT_INTEREST = "agentic_trading.agents.analysts.event_credit_pod.short_interest_squeeze_analyst"


def _constant_short_interest(settlements: int, days_to_cover: float) -> pd.DataFrame:
    """`settlements` semimonthly settlements, every one with the same days-to-cover."""
    dates = pd.date_range("2015-01-15", periods=settlements, freq="SMS-15")
    return pd.DataFrame(
        {
            "settlement_date": dates,
            "dissemination_date": dates + pd.Timedelta(days=12),
            "short_interest_shares": 1_000_000.0,
            "days_to_cover": days_to_cover,
            "avg_daily_volume": 500_000.0,
        }
    )


@pytest.mark.parametrize("days_to_cover", [0.0, 3.0])
def test_short_interest_constant_days_to_cover_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, days_to_cover: float
) -> None:
    """A days-to-cover that never moves over the z window has no z-score."""
    history = _constant_short_interest(8, days_to_cover)
    monkeypatch.setattr(f"{_SHORT_INTEREST}.load_short_interest_from_lakehouse", lambda *_: history)

    signal = ShortInterestSqueezeAnalyst().generate_signal(
        "AAPL", pd.DataFrame(), pd.Timestamp("2015-06-30")
    )

    assert signal.status == "unavailable"
    assert signal.direction == "flat"
    assert signal.confidence == 0.0
    assert signal.source_as_of == history["dissemination_date"].max().date()
    assert signal.sample_count == len(history)
