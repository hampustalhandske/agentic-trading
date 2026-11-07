# test_config.py

"""AT-140: the firm group carries the history horizon a bounded-memory run keeps its track record
and decisions to."""

import pytest
from pydantic import ValidationError

from agentic_trading.config import apply_overrides, settings, temporary_overrides


def test_history_horizon_days() -> None:
    """Defaults to 300 days, covering the IC lookback; accepts 1 and refuses anything below it."""
    assert settings.firm.history_horizon_days == 300
    assert settings.firm.history_horizon_days >= settings.portfolio.ic_lookback_days

    with temporary_overrides({"firm.history_horizon_days": 1}):
        assert settings.firm.history_horizon_days == 1
    for value in (0, -5):
        with pytest.raises(ValidationError):
            apply_overrides({"firm.history_horizon_days": value})
    assert settings.firm.history_horizon_days == 300
