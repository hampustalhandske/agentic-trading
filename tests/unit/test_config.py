# test_config.py

"""AT-140: the firm group carries the history horizon a bounded-memory run keeps its track record
and decisions to. AT-167: no override moves stage_a.data_end past the locked holdout cut."""

from datetime import date, timedelta

import pytest
from pydantic import ValidationError

from agentic_trading.config import (
    HOLDOUT_LOCKED_DATA_END,
    StageAConfig,
    apply_overrides,
    settings,
    temporary_overrides,
)


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


def test_data_end_cannot_move_past_the_locked_holdout() -> None:
    """The cut defaults to the locked holdout cut; a later date is refused by temporary_overrides,
    apply_overrides, assignment and construction, and leaves the cut where it was."""
    assert HOLDOUT_LOCKED_DATA_END == date(2024, 12, 31)
    assert settings.stage_a.data_end == HOLDOUT_LOCKED_DATA_END
    later = HOLDOUT_LOCKED_DATA_END + timedelta(days=1)

    with pytest.raises(ValidationError, match="holdout"):
        with temporary_overrides({"stage_a.data_end": later}):
            pass
    with pytest.raises(ValidationError, match="holdout"):
        apply_overrides({"stage_a.data_end": "2025-06-30"})
    with pytest.raises(ValidationError, match="holdout"):
        settings.stage_a.data_end = later
    with pytest.raises(ValidationError, match="holdout"):
        StageAConfig(data_end=later)
    assert settings.stage_a.data_end == HOLDOUT_LOCKED_DATA_END


def test_data_end_can_move_earlier() -> None:
    """An earlier cut, or the locked cut itself, is accepted through every path."""
    earlier = date(2020, 12, 31)
    with temporary_overrides({"stage_a.data_end": earlier}):
        assert settings.stage_a.data_end == earlier
    with temporary_overrides({"stage_a.data_end": HOLDOUT_LOCKED_DATA_END.isoformat()}):
        assert settings.stage_a.data_end == HOLDOUT_LOCKED_DATA_END
    assert StageAConfig(data_end=earlier).data_end == earlier
    assert settings.stage_a.data_end == HOLDOUT_LOCKED_DATA_END
