"""How often each analyst of a firm record had a signal to give: its
signals counted by status per calendar year of the record and over the
whole record, and the analysts whose share of available signals falls
below `settings.firm.low_availability_threshold`.

The counts are taken from the Commander's decisions of each recorded day
(`PortfolioDecision.contributing_signals`), as the engine checkpoint holds
them: `analyst_availability` counts decisions already built,
`analyst_availability_of_payloads` the checkpoint's stored JSON of each
day, reading only the fields counted. A signal counts once, on its
decision day, for each decision of an instrument (an equity, a future,
an FX pair or a coin) it fed, when the signal itself is dated that day:
an option row repeats its underlying's signals, and a decision on an
instrument without its bar carries the signals of an earlier day, so
neither is counted again. A signal counts under the status its pod
carried it with (`ManagerSignal.status`); `active` includes a flat
signal at confidence 0. An analyst run by several pods counts once per
pod for each instrument.

Every analyst of the record's roster is listed, with every year of the
recorded days, whether or not it signaled: one that emitted no signal
shows zero counts and an available share of 0, and is always flagged."""

from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import date
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, RootModel, TypeAdapter

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
    """What analyst_availability.json holds: each analyst id of the roster
    and each other one that signaled, sorted, to its counts per calendar
    year of the recorded days (`"2016"`, ..., in order) and over the whole
    record (`TOTAL_KEY`)."""

    model_config = ConfigDict(frozen=True)


class LowAvailability(BaseModel):
    """An analyst whose whole-record `available_share` is below the firm's
    threshold or that emitted no signal, its number of `signals` of every
    status over the record, and whether the record's registration leaves
    its availability undeclared (`undeclared`)."""

    model_config = _SPEC

    analyst_id: str
    available_share: float = Field(ge=0.0, le=1.0)
    signals: int = Field(ge=0)
    undeclared: bool


class _CountedSignal(BaseModel):
    """The fields of a stored `ManagerSignal` the counts read."""

    source_agent_id: str
    status: SignalStatus
    as_of_date: date


class _CountedDecision(BaseModel):
    """The fields of a stored `PortfolioDecision` the counts read;
    `instrument` only for whether it is set."""

    contributing_signals: list[_CountedSignal]
    instrument: dict[str, Any] | None = None


_COUNTED_DAY = TypeAdapter(list[_CountedDecision])


def analyst_availability(
    decisions_by_day: Mapping[date, Sequence[PortfolioDecision]],
    roster: Iterable[str],
) -> AnalystAvailability:
    """The signals in `decisions_by_day` (decision day to the Commander's
    decisions of that day) of each analyst id of `roster`, and of any other
    analyst that signaled, counted by status in every calendar year of the
    decision days and over every day (see the module docstring); an
    analyst without a signal in a period counts 0 in it."""
    return _tally(
        ((day, _counted(day, decisions)) for day, decisions in decisions_by_day.items()), roster
    )


def analyst_availability_of_payloads(
    days: Iterable[tuple[date, str]], roster: Iterable[str]
) -> AnalystAvailability:
    """`analyst_availability` of each `(decision day, decisions)` of
    `days`, the decisions given as the JSON text the engine checkpoint
    stores for the day (a list of `PortfolioDecision` payloads), each day
    read once. Only the fields counted are parsed."""
    return _tally(
        ((day, _counted(day, _COUNTED_DAY.validate_json(payload))) for day, payload in days),
        roster,
    )


def _counted(
    day: date, decisions: Iterable[PortfolioDecision] | Iterable[_CountedDecision]
) -> Iterator[tuple[str, SignalStatus]]:
    """The `(analyst id, status)` of each signal of `decisions`, the
    decisions of `day`, that counts on it: one dated `day` that fed a
    decision without an option `instrument`."""
    for decision in decisions:
        if decision.instrument is not None:
            continue
        for signal in decision.contributing_signals:
            if signal.as_of_date != day:
                continue
            yield signal.source_agent_id, signal.status


def _tally(
    days: Iterable[tuple[date, Iterable[tuple[str, SignalStatus]]]], roster: Iterable[str]
) -> AnalystAvailability:
    """The availability record of the counted `(analyst id, status)`
    signals of each decision day of `days`, every analyst id of `roster`
    listed, with every calendar year of the days."""
    years_seen: set[str] = set()
    by_year: dict[str, dict[str, Counter[SignalStatus]]] = {analyst_id: {} for analyst_id in roster}
    for day, signals in days:
        year = str(day.year)
        years_seen.add(year)
        for analyst_id, status in signals:
            years = by_year.setdefault(analyst_id, {})
            years.setdefault(year, Counter())[status] += 1
    years_recorded = sorted(years_seen)
    record: dict[str, dict[str, AvailabilityCounts]] = {}
    for analyst_id in sorted(by_year):
        years = by_year[analyst_id]
        total: Counter[SignalStatus] = Counter()
        for counts in years.values():
            total.update(counts)
        record[analyst_id] = {
            **{year: AvailabilityCounts.of(years.get(year, {})) for year in years_recorded},
            TOTAL_KEY: AvailabilityCounts.of(total),
        }
    return AnalystAvailability(record)


def low_availability(
    availability: AnalystAvailability,
    threshold: float,
    expected: Mapping[str, str],
) -> list[LowAvailability]:
    """The analysts of `availability` whose whole-record share is below
    `threshold` or that emitted no signal at all, in analyst order, each
    `undeclared` when `expected` (the registration's expected availability
    per analyst id) does not name it."""
    return [
        LowAvailability(
            analyst_id=analyst_id,
            available_share=periods[TOTAL_KEY].available_share,
            signals=_signals(periods[TOTAL_KEY]),
            undeclared=analyst_id not in expected,
        )
        for analyst_id, periods in availability.root.items()
        if periods[TOTAL_KEY].available_share < threshold or _signals(periods[TOTAL_KEY]) == 0
    ]


def _signals(counts: AvailabilityCounts) -> int:
    """The number of signals `counts` holds, of every status."""
    return counts.active + counts.unavailable + counts.error
