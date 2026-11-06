# test_availability.py

"""AT-130: `firm_record.availability` counts each analyst's signals by status per calendar year
and over the record, once per instrument decision dated its decision day (an option row and a
carried decision repeat signals already counted), and flags the analysts whose whole-record
available share is below a threshold, marking those the registration does not declare."""

from datetime import date
from pathlib import Path

import pytest

from agentic_trading.agents.schemas import (
    ManagerSignal,
    OptionContract,
    PortfolioDecision,
    SignalStatus,
)
from agentic_trading.firm_record.availability import (
    TOTAL_KEY,
    AvailabilityCounts,
    analyst_availability,
    low_availability,
)
from agentic_trading.trials.ledger import LedgerError
from agentic_trading.trials.rows import FIRM_HYPOTHESIS_ID, firm_expected_availability

_DAY_2015, _DAY_2016, _NEXT_2016 = date(2015, 12, 31), date(2016, 1, 4), date(2016, 1, 5)


def _signal(analyst: str, status: SignalStatus, day: date, asset: str = "SPY") -> ManagerSignal:
    return ManagerSignal(
        manager_id="pod",
        source_agent_id=analyst,
        asset=asset,
        as_of_date=day,
        direction="flat",
        confidence=0.0,
        raw_confidence=0.0,
        scale_applied=1.0,
        rationale="stub",
        status=status,
    )


def _decision(day: date, signals: list[ManagerSignal], asset: str = "SPY") -> PortfolioDecision:
    return PortfolioDecision(
        asset=asset, as_of_date=day, target_weight=0.0, contributing_signals=signals
    )


def test_counts_by_year_and_total() -> None:
    """Signals count per calendar year of their decision day and over the record; a flat
    signal at confidence 0 with status active counts as active; the share is active over all
    three statuses."""
    record = analyst_availability(
        {
            _DAY_2015: [
                _decision(_DAY_2015, [_signal("a", "active", _DAY_2015)]),
                _decision(_DAY_2015, [_signal("a", "error", _DAY_2015, "GLD")], "GLD"),
            ],
            _DAY_2016: [_decision(_DAY_2016, [_signal("a", "unavailable", _DAY_2016)])],
            _NEXT_2016: [_decision(_NEXT_2016, [_signal("a", "active", _NEXT_2016)])],
        }
    ).root

    assert list(record["a"]) == ["2015", "2016", TOTAL_KEY]
    assert record["a"]["2015"] == AvailabilityCounts(
        active=1, unavailable=0, error=1, available_share=0.5
    )
    assert record["a"]["2016"] == AvailabilityCounts(
        active=1, unavailable=1, error=0, available_share=0.5
    )
    assert record["a"][TOTAL_KEY] == AvailabilityCounts(
        active=2, unavailable=1, error=1, available_share=0.5
    )


def test_option_rows_and_carried_signals_are_not_counted_again() -> None:
    """An option row's signals, and a decision's signals dated an earlier day (an instrument
    carried over a day without its bar), add nothing."""
    put = OptionContract(
        underlying="SPY", expiry=date(2016, 3, 18), strike=180.0, option_type="put"
    )
    today = _signal("a", "unavailable", _NEXT_2016)
    record = analyst_availability(
        {
            _NEXT_2016: [
                _decision(_NEXT_2016, [today]),
                PortfolioDecision(
                    asset=put.contract_id,
                    as_of_date=_NEXT_2016,
                    target_weight=0.0,
                    contributing_signals=[today],
                    instrument=put,
                ),
                _decision(_NEXT_2016, [_signal("a", "active", _DAY_2016, "GAP")], "GAP"),
            ]
        }
    ).root

    assert record["a"][TOTAL_KEY] == AvailabilityCounts(
        active=0, unavailable=1, error=0, available_share=0.0
    )


def test_low_availability_marks_undeclared_analysts() -> None:
    """Only analysts strictly under the threshold are listed, in analyst order, each undeclared
    unless the expected availability names it."""
    record = analyst_availability(
        {
            _DAY_2016: [
                _decision(
                    _DAY_2016,
                    [
                        _signal("b", "unavailable", _DAY_2016),
                        _signal("a", "unavailable", _DAY_2016),
                        _signal("c", "active", _DAY_2016),
                    ],
                )
            ]
        }
    )

    flagged = low_availability(record, 0.5, {"b": "from 2017-01-03"})

    assert [(entry.analyst_id, entry.available_share, entry.undeclared) for entry in flagged] == [
        ("a", 0.0, True),
        ("b", 0.0, False),
    ]


def _registry(tmp_path: Path, body: str) -> Path:
    registry = tmp_path / "research" / "registry"
    registry.mkdir(parents=True)
    (registry / f"{FIRM_HYPOTHESIS_ID}.yaml").write_text(f"id: {FIRM_HYPOTHESIS_ID}\n{body}")
    return tmp_path


def test_expected_availability_is_read_beside_trials(tmp_path: Path) -> None:
    """The registry file's expected_availability is read in its order; a file without it, or
    no file, declares nothing."""
    repo = _registry(
        tmp_path / "declared",
        "trials:\n  declared_n: 1\nexpected_availability:\n"
        "  z_analyst: full\n  a_analyst: none in window\n  m_analyst: from 2016-01-04\n",
    )

    assert firm_expected_availability(repo) == {
        "z_analyst": "full",
        "a_analyst": "none in window",
        "m_analyst": "from 2016-01-04",
    }
    assert firm_expected_availability(_registry(tmp_path / "bare", "trials: {}\n")) == {}
    assert firm_expected_availability(tmp_path / "absent") == {}


@pytest.mark.parametrize(
    "entry",
    ["  a_analyst: sometimes\n", "  a_analyst: from 2016-13-01\n", "  a_analyst: 3\n"],
    ids=["unknown-form", "bad-date", "not-a-string"],
)
def test_malformed_expected_availability_is_refused(tmp_path: Path, entry: str) -> None:
    """An entry in none of the three forms is refused."""
    repo = _registry(tmp_path, f"expected_availability:\n{entry}")

    with pytest.raises(LedgerError, match="expected availability"):
        firm_expected_availability(repo)
