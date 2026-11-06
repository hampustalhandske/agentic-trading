# test_submission.py

"""`firm_record.submission`: sources.json entries map onto edgelab's `DecisionSource` shape
with unique names, a dated source stating `can_publish_after_close` and an undated one its
`limitation`; a prior-exposure block keeps edgelab's `PriorExposure` checks."""

import pytest
from pydantic import ValidationError

from agentic_trading.firm_record.submission import (
    SubmissionPriorExposure,
    SubmissionSource,
    decision_sources,
)


def test_sources_map_dated_and_undated_entries() -> None:
    """A dated entry and an undated one map field for field and dump back to the same keys."""
    entries = [
        {"name": "fred", "can_publish_after_close": True},
        {"name": "sic", "undated": True, "limitation": "current SIC code screens candidates"},
    ]
    sources = decision_sources(entries)

    assert [source.model_dump(exclude_defaults=True) for source in sources] == entries


def test_sources_refuse_a_repeated_name() -> None:
    """Two entries under one name are refused."""
    entry = {"name": "fred", "can_publish_after_close": True}
    with pytest.raises(ValueError, match="repeat"):
        decision_sources([entry, entry])


@pytest.mark.parametrize(
    "entry",
    [
        {"name": "fred"},
        {"name": "fred", "can_publish_after_close": True, "limitation": "x"},
        {"name": "sic", "undated": True},
        {"name": "sic", "undated": True, "limitation": "x", "can_publish_after_close": False},
        {"name": "fred", "can_publish_after_close": True, "extra": 1},
    ],
    ids=["dated-no-flag", "dated-limitation", "undated-no-limitation", "undated-flag", "extra"],
)
def test_sources_refuse_entries_edgelab_refuses(entry: dict[str, object]) -> None:
    """An entry that is neither a dated nor an undated source, or carries an unknown key, is
    refused."""
    with pytest.raises(ValidationError):
        SubmissionSource.model_validate(entry)


def test_prior_exposure_round_trips_and_keeps_edgelab_checks() -> None:
    """A valid block dumps back unchanged; one without windows, with a reversed window or a
    negative count is refused."""
    block = {
        "note": "Firm Sharpe seen",
        "windows": [{"start": "2010-01-04", "end": "2017-12-29"}],
        "min_count": 3,
    }

    assert SubmissionPriorExposure.model_validate(block).model_dump(mode="json") == block
    for broken in (
        block | {"windows": []},
        block | {"windows": [{"start": "2017-12-29", "end": "2010-01-04"}]},
        block | {"min_count": -1},
    ):
        with pytest.raises(ValidationError):
            SubmissionPriorExposure.model_validate(broken)
