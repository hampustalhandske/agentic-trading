# test_undefined_statistics.py

"""An analyst whose statistic is undefined on the day's data (a NaN z-score, a non-finite
half-life or p-value, a zero volatility) never trades on it: it reports `unavailable` when the
data cannot define the statistic and `error` when a fitted model's diagnostic is undefined.
Every loader is stubbed on the analyst's own module, so no test reads the lake."""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from agentic_trading.agents.analysts.event_credit_pod.short_interest_squeeze_analyst import (
    ShortInterestSqueezeAnalyst,
)
from agentic_trading.agents.analysts.stat_arb_pod import ou_mean_reversion_analyst
from agentic_trading.agents.analysts.stat_arb_pod.ou_mean_reversion_analyst import (
    OUMeanReversionAnalyst,
)
from agentic_trading.agents.schemas import AgentSignal

_SHORT_INTEREST = "agentic_trading.agents.analysts.event_credit_pod.short_interest_squeeze_analyst"
AS_OF = pd.Timestamp("2024-06-14")


def _market(returns: np.ndarray) -> pd.DataFrame:
    """A single-asset frame of business-day `log_return`s ending AS_OF, with closes."""
    index = pd.bdate_range(end=AS_OF, periods=len(returns))
    frame = pd.DataFrame({"log_return": np.asarray(returns, dtype=np.float64)}, index=index)
    frame["close"] = 100.0 * np.exp(np.cumsum(frame["log_return"].to_numpy()))
    return frame


def _noise(size: int, seed: int = 7) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, 0.01, size)


def _flat(signal: AgentSignal) -> tuple[str, str, float]:
    return signal.status, signal.direction, signal.confidence


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


class UndefinedResultError(ValueError):
    """Stands in for quantcore's undefined-result error, a `ValueError` subclass."""


def _raising(error: BaseException) -> Any:
    def _fit(*_args: Any, **_kwargs: Any) -> Any:
        raise error

    return _fit


def test_ou_straight_line_spread_is_unavailable() -> None:
    """A log price rising by the same amount every day is a straight-line spread, whose
    half-life is undefined."""
    analyst = OUMeanReversionAnalyst()
    analyst._open_direction_by_asset["AAPL"] = "long"

    signal = analyst.generate_signal("AAPL", _market(np.full(120, 0.001)), AS_OF)

    assert _flat(signal) == ("unavailable", "flat", 0.0)
    assert "AAPL" not in analyst._open_direction_by_asset


@pytest.mark.parametrize(
    "fit",
    [
        lambda _spread: float("nan"),
        lambda _spread: float("inf"),
        _raising(UndefinedResultError("half-life undefined")),
    ],
    ids=["nan", "inf", "undefined_result"],
)
def test_ou_undefined_half_life_is_unavailable(monkeypatch: pytest.MonkeyPatch, fit: Any) -> None:
    monkeypatch.setattr(ou_mean_reversion_analyst, "ou_half_life", fit)
    analyst = OUMeanReversionAnalyst()
    analyst._open_direction_by_asset["AAPL"] = "long"

    signal = analyst.generate_signal("AAPL", _market(_noise(120)), AS_OF)

    assert _flat(signal) == ("unavailable", "flat", 0.0)
    assert "AAPL" not in analyst._open_direction_by_asset


def test_ou_not_mean_reverting_abstains(monkeypatch: pytest.MonkeyPatch) -> None:
    """quantcore's phi >= 0 refusal is a spread the model declines to trade."""
    error = ValueError("estimated series does not exhibit mean reversion (phi >= 0, diverging)")
    monkeypatch.setattr(ou_mean_reversion_analyst, "ou_half_life", _raising(error))

    signal = OUMeanReversionAnalyst().generate_signal("AAPL", _market(_noise(120)), AS_OF)

    assert _flat(signal) == ("active", "flat", 0.0)


def test_ou_any_other_fit_failure_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ou_mean_reversion_analyst, "ou_half_life", _raising(ValueError("singular regression"))
    )

    signal = OUMeanReversionAnalyst().generate_signal("AAPL", _market(_noise(120)), AS_OF)

    assert _flat(signal) == ("error", "flat", 0.0)
