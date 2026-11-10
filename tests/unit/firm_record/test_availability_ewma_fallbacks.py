# test_availability_ewma_fallbacks.py

"""The availability report counts, per analyst and period, the counted signals whose
`volatility_estimator` says the RiskMetrics EWMA fallback gave their volatility
(`AvailabilityCounts.ewma_fallbacks`). An analyst that recorded an estimator on any counted
signal of the record has the count, zero included, in every period; one that never did has
none, and analyst_availability.json leaves the field out for it."""

import json
from datetime import date
from pathlib import Path

from agentic_trading.agents.schemas import (
    ManagerSignal,
    OptionContract,
    PortfolioDecision,
    VolatilityEstimator,
)
from agentic_trading.firm_record.availability import (
    TOTAL_KEY,
    AvailabilityCounts,
    analyst_availability,
    analyst_availability_of_payloads,
)
from agentic_trading.firm_record.submission import (
    ANALYST_AVAILABILITY_FILE,
    AgenticMeta,
    write_analyst_availability,
)
from agentic_trading.provenance import runtime_environment

_DAY_2015, _DAY_2016, _NEXT_2016 = date(2015, 12, 31), date(2016, 1, 4), date(2016, 1, 5)


def _signal(
    analyst: str, day: date, estimator: VolatilityEstimator | None, asset: str = "SPY"
) -> ManagerSignal:
    return ManagerSignal(
        manager_id="pod",
        source_agent_id=analyst,
        asset=asset,
        as_of_date=day,
        direction="long",
        confidence=0.2,
        raw_confidence=0.2,
        scale_applied=1.0,
        rationale="stub",
        volatility_estimator=estimator,
    )


def _decision(day: date, signals: list[ManagerSignal], asset: str = "SPY") -> PortfolioDecision:
    return PortfolioDecision(
        asset=asset, as_of_date=day, target_weight=0.0, contributing_signals=signals
    )


def _days() -> dict[date, list[PortfolioDecision]]:
    put = OptionContract(
        underlying="SPY", expiry=date(2016, 3, 18), strike=180.0, option_type="put"
    )
    ewma_today = _signal("vol", _NEXT_2016, "ewma")
    return {
        _DAY_2015: [
            _decision(
                _DAY_2015,
                [_signal("vol", _DAY_2015, "garch_11"), _signal("plain", _DAY_2015, None)],
            )
        ],
        _DAY_2016: [
            _decision(_DAY_2016, [_signal("vol", _DAY_2016, "ewma")]),
            _decision(_DAY_2016, [_signal("vol", _DAY_2016, "ewma", "GLD")], "GLD"),
        ],
        _NEXT_2016: [
            _decision(_NEXT_2016, [ewma_today]),
            # An option row repeats its underlying's signal, and a carried signal is dated an
            # earlier day: neither counts again.
            PortfolioDecision(
                asset=put.contract_id,
                as_of_date=_NEXT_2016,
                target_weight=0.0,
                contributing_signals=[ewma_today],
                instrument=put,
            ),
            _decision(_NEXT_2016, [_signal("vol", _DAY_2016, "ewma", "GAP")], "GAP"),
        ],
    }


def test_ewma_fallbacks_are_counted_per_year_and_over_the_record() -> None:
    record = analyst_availability(_days(), ["vol", "plain", "silent"]).root

    assert record["vol"]["2015"] == AvailabilityCounts(
        active=1, unavailable=0, error=0, available_share=1.0, ewma_fallbacks=0
    )
    assert record["vol"]["2016"].ewma_fallbacks == 3
    assert record["vol"][TOTAL_KEY].ewma_fallbacks == 3
    assert record["vol"][TOTAL_KEY].active == 4
    for analyst in ("plain", "silent"):
        assert all(counts.ewma_fallbacks is None for counts in record[analyst].values())


def test_the_payload_path_counts_the_same() -> None:
    days = _days()
    payloads = [
        (day, json.dumps([decision.model_dump(mode="json") for decision in decisions]))
        for day, decisions in days.items()
    ]

    assert analyst_availability_of_payloads(payloads, ["vol", "plain"]) == analyst_availability(
        days, ["vol", "plain"]
    )


def test_the_written_report_omits_the_count_only_where_it_is_none(tmp_path: Path) -> None:
    availability = analyst_availability(_days(), ["vol", "plain"])
    meta = AgenticMeta(
        counted=False,
        git_sha="a" * 40,
        src_tree_clean=True,
        declared_n_from="uncounted_default",
        environment=runtime_environment(),
    )

    write_analyst_availability(tmp_path, availability, meta)

    written = json.loads((tmp_path / ANALYST_AVAILABILITY_FILE).read_text())
    assert written["vol"][TOTAL_KEY]["ewma_fallbacks"] == 3
    assert written["vol"]["2015"]["ewma_fallbacks"] == 0
    assert written["plain"][TOTAL_KEY] == {
        "active": 1,
        "unavailable": 0,
        "error": 0,
        "available_share": 1.0,
    }
