"""How often each analyst of a firm record had a signal to give: its
signals counted by status per calendar year of the record and over the
whole record, and the analysts whose share of available signals falls
below `settings.firm.low_availability_threshold`.

The counts are taken from the Commander's decisions of each recorded day
(`PortfolioDecision.contributing_signals`), as the engine checkpoint holds
them. A signal counts once, on its decision day, for each decision of an
instrument (an equity, a future, an FX pair or a coin) it fed, when the
signal itself is dated that day: an option row repeats its underlying's
signals, and a decision on an instrument without its bar carries the
signals of an earlier day, so neither is counted again. A signal counts under the status
its pod carried it with (`ManagerSignal.status`); `active` includes a flat
signal at confidence 0. An analyst run by several pods counts once per pod
for each instrument."""

from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, RootModel

from agentic_trading.agents.schemas import PortfolioDecision, SignalStatus

# The key of an analyst's whole-record counts, beside its year keys.
TOTAL_KEY = "total"

_SPEC = ConfigDict(frozen=True, extra="forbid")


class AvailabilityCounts(BaseModel):
    """An analyst's signals over one period by status (`active`,
    `unavailable`, `error`) and `available_share`, active over all three."""

    model_config = _SPEC

    active: int = Field(ge=0)
    unavailable: int = Field(ge=0)
    error: int = Field(ge=0)
    available_share: float = Field(ge=0.0, le=1.0)

    @classmethod
    def of(cls, counts: Mapping[SignalStatus, int]) -> Self:
        """The counts of `counts` (a status missing from it counts 0) with
        their available share; a period without any signal has share 0."""
        active = counts.get("active", 0)
        unavailable = counts.get("unavailable", 0)
        error = counts.get("error", 0)
        total = active + unavailable + error
        return cls(
            active=active,
            unavailable=unavailable,
            error=error,
            available_share=active / total if total else 0.0,
        )


class AnalystAvailability(RootModel[dict[str, dict[str, AvailabilityCounts]]]):
    """What analyst_availability.json holds: each analyst id that signaled
    in the record, sorted, to its counts per calendar year (`"2016"`, ...,
    in order) and over the whole record (`TOTAL_KEY`)."""

    model_config = ConfigDict(frozen=True)


class LowAvailability(BaseModel):
    """An analyst whose whole-record `available_share` is below the firm's
    threshold, and whether the record's registration leaves its availability
    undeclared (`undeclared`)."""

    model_config = _SPEC

    analyst_id: str
    available_share: float = Field(ge=0.0, le=1.0)
    undeclared: bool


def analyst_availability(
    decisions_by_day: Mapping[date, Sequence[PortfolioDecision]],
) -> AnalystAvailability:
    """Each analyst's signals in `decisions_by_day` (decision day to the
    Commander's decisions of that day) counted by status per calendar year
    of the decision day and over every day (see the module docstring)."""
    by_year: dict[str, dict[str, Counter[SignalStatus]]] = {}
    for day, decisions in decisions_by_day.items():
        year = str(day.year)
        for decision in decisions:
            if decision.instrument is not None:
                continue
            for signal in decision.contributing_signals:
                if signal.as_of_date != day:
                    continue
                years = by_year.setdefault(signal.source_agent_id, {})
                years.setdefault(year, Counter())[signal.status] += 1
    record: dict[str, dict[str, AvailabilityCounts]] = {}
    for analyst_id in sorted(by_year):
        years = by_year[analyst_id]
        total: Counter[SignalStatus] = Counter()
        for counts in years.values():
            total.update(counts)
        record[analyst_id] = {
            **{year: AvailabilityCounts.of(years[year]) for year in sorted(years)},
            TOTAL_KEY: AvailabilityCounts.of(total),
        }
    return AnalystAvailability(record)


def low_availability(
    availability: AnalystAvailability,
    threshold: float,
    expected: Mapping[str, str],
) -> list[LowAvailability]:
    """The analysts of `availability` whose whole-record share is below
    `threshold`, in analyst order, each `undeclared` when `expected` (the
    registration's expected availability per analyst id) does not name it."""
    return [
        LowAvailability(
            analyst_id=analyst_id,
            available_share=periods[TOTAL_KEY].available_share,
            undeclared=analyst_id not in expected,
        )
        for analyst_id, periods in availability.root.items()
        if periods[TOTAL_KEY].available_share < threshold
    ]
