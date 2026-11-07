# rows.py

"""One firm-record day: the Commander's `PodTarget`s as `DecisionRow`s,
under the stage (a) look-ahead audit, and the sources a record declares.

A pod writes a row for an instrument when it ran an analyst on it (its
target carries a contributing signal) or holds a weight in it. Its
status is `active` when the weight is nonzero (the firm holds that
target) or the target is carried over a day without the instrument's
bar; otherwise `active` when any contributing signal is active, `error`
when one errored, else `unavailable`, and then its weight is 0.

Its `source` and `source_as_of` are those of the contributing input whose
date has the latest effective known time. The inputs are the contributing
signals, each dated by its `source_as_of` from its analyst's row source
(`decisions.row_source`), and for an option row the option-chain snapshot
its premium and delta rest on (`options_chain`). A date from a source that
can publish after the close (`decisions.SOURCES`) counts from the next
session of the decision calendar, any other from its own day; on an equal
effective day the after-close source wins, then the later date, then the
alphabetically first source. Every input passes the stage (a) audit on its
own. A target the pod holds rather than sets (`PodTarget.held`: carried
over a day without a bar at its carried weight, kept by the no-trade band,
a held put) keeps the source and source_as_of of the row that set it, the
pod's row on the previous day. A carried target a firm limit scaled is a
new decision resting on that day's inputs: its own and those of every
target of the day not carried, since the firm limits act on the whole
book.

An instrument held over a day without its bar is written at each pod's
carried target (see `Commander.decide`); one without a bar and without a
decided target gets one `unavailable` row at weight 0 per pod whose scope
holds it. A held put (`PodTarget.hold_contracts`) is written at the
premium weight of the contracts the book holds, at the decision day's
close mark (`PodTarget.mark_mid`), with its delta-equivalent weight scaled
to match.

Known limit: an instrument that stops trading (a delisting) has no bar
again, so it is carried at its last decided target to the end of the
window; edgelab settles delistings in replay."""

import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, timedelta

from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.schemas import ManagerSignal, OptionContract, PodTarget, SignalStatus
from agentic_trading.config import settings
from agentic_trading.data.instruments import futures_root
from agentic_trading.data.trading_calendar import sessions_between
from agentic_trading.stage_a.decisions import (
    SOURCES,
    LookAheadError,
    check_source_as_of,
    row_source,
    source_of,
    undated_declarations,
)
from agentic_trading.stage_a.store import DecisionKind, DecisionRow

# The source an option row's chain snapshot is dated by.
OPTIONS_CHAIN_SOURCE = "options_chain"

# A row's provenance: its source and source_as_of.
type Provenance = tuple[str | None, date | None]

# Calendar days searched for the session after a date: longer than any
# exchange closure on record.
_NEXT_SESSION_SEARCH_DAYS = 30


def firm_sources(
    analysts: Iterable[AnalystAgent], *, option_rows: bool = False
) -> list[dict[str, str | bool]]:
    """What a firm record's sources.json lists: every dated source any of
    `analysts` declares, and `options_chain` when the firm writes option
    rows (`option_rows`), by name, with whether it can publish after the
    close, then each undated source declared, with its look-ahead
    limitation (`decisions.undated_declarations`). Raises `LookAheadError`
    for a source neither list knows."""
    declared: list[str] = [
        requirement.source for analyst in analysts for requirement in analyst.data_requirements
    ]
    if option_rows:
        declared.append(OPTIONS_CHAIN_SOURCE)
    source_of(declared)
    dated: list[dict[str, str | bool]] = [
        {"name": name, "can_publish_after_close": SOURCES[name]}
        for name in sorted(set(declared))
        if name in SOURCES
    ]
    return dated + undated_declarations(declared)


def analyst_sources(analysts: Iterable[AnalystAgent]) -> dict[str, str | None]:
    """Each analyst's `agent_id` to the source its signals carry
    (`decisions.row_source`), `None` for a price-only analyst."""
    return {analyst.agent_id: row_source(analyst) for analyst in analysts}


# An OSI root: 1 to 6 uppercase letters.
_OSI_ROOT = re.compile(r"[A-Z]{1,6}")
# Tolerance on strike x 1000 being a whole number, far below a strike's
# 0.001 resolution and far above float error on strikes up to 1e5.
_OSI_STRIKE_TOLERANCE = 1e-6


def osi_symbol(contract: OptionContract) -> str:
    """`contract`'s unpadded OSI symbol, the key the lake's option chains and
    edgelab's replay use: the root, the expiry as yymmdd, `C` or `P`, and the
    strike x 1000 in 8 zero-padded digits (`SPY150320P00190000` for the SPY
    2015-03-20 190 put).

    Raises `ValueError` when OSI cannot carry the contract exactly: a root
    that is not 1-6 uppercase letters, an expiry outside 2000-2099, or a
    strike with more than 3 dp or whose x 1000 value needs more than 8
    digits. A strike is never rounded.

    The strike and root are those the firm's contract carries: its contract
    id prints the strike at 2 dp, and an option-chain strike is divided by
    the underlying's later splits. The symbol is the lake's as-traded one
    when the chain strike has at most 2 dp and the underlying did not split
    or change ticker after the contract, as holds for every SPY contract the
    2010-2017 chains list."""
    if not _OSI_ROOT.fullmatch(contract.underlying):
        raise ValueError(f"OSI root must be 1-6 uppercase letters, not {contract.underlying!r}")
    if not 2000 <= contract.expiry.year <= 2099:
        raise ValueError(f"OSI expiry must fall in 2000-2099, not {contract.expiry.isoformat()}")
    milli = contract.strike * 1000
    whole = round(milli)
    if abs(milli - whole) > _OSI_STRIKE_TOLERANCE or not 0 < whole < 10**8:
        raise ValueError(
            f"OSI strike must have at most 3 dp and fit 8 digits x 1000, not {contract.strike!r}"
        )
    letter = "C" if contract.option_type == "call" else "P"
    return f"{contract.underlying}{contract.expiry:%y%m%d}{letter}{whole:08d}"


def row_instrument_id(asset: str, instrument: OptionContract | None = None) -> str:
    """The `instrument_id` a row for `asset` is written under, the id
    edgelab's replay reads bars by: an option (`instrument`, the contract
    `asset` names) its OSI symbol (`osi_symbol`, which raises `ValueError`
    for a contract OSI cannot carry); a continuous future's root (`GC` for
    `GC=F`); an equity, ETF, FX pair or coin its lake ticker."""
    if instrument is not None:
        return osi_symbol(instrument)
    return futures_root(asset) or asset


def instrument_kind(ticker: str) -> DecisionKind:
    """The `decision_file.v1` kind of a lake ticker: `future` for a `=F`
    continuous future, `fx` for a `=X` pair, `crypto` for a `-USD` coin,
    `equity` for everything else (stocks and ETFs, bond ETFs included)."""
    if ticker.endswith("=F"):
        return "future"
    if ticker.endswith("=X"):
        return "fx"
    if ticker.endswith("-USD"):
        return "crypto"
    return "equity"


def _next_session(day: date, calendar: str) -> date:
    """The first session of `calendar` strictly after `day`."""
    following = sessions_between(
        day + timedelta(days=1), day + timedelta(days=_NEXT_SESSION_SEARCH_DAYS), calendar
    )
    if not following:
        raise ValueError(f"no {calendar} session within a month after {day.isoformat()}")
    return following[0]


def _row_status(signals: Sequence[ManagerSignal], weight: float, carried: bool) -> SignalStatus:
    """A row's status (see the module docstring)."""
    if weight != 0.0 or carried or any(signal.status == "active" for signal in signals):
        return "active"
    if any(signal.status == "error" for signal in signals):
        return "error"
    return "unavailable"


def _row_source(
    day: date,
    target: PodTarget,
    sources: Mapping[str, str | None],
    calendar: str,
) -> tuple[str | None, date | None]:
    """The source and source_as_of of `target`'s row from its own inputs,
    after auditing every one (see the module docstring). Raises
    `ValueError` naming an analyst outside `sources`, and `LookAheadError`
    when an active signal of an analyst with a dated source carries no
    `source_as_of`, or an input fails `decisions.check_source_as_of`."""
    candidates: list[tuple[date, bool, date, str]] = []
    if target.instrument is not None and target.mark_as_of is not None:
        check_source_as_of(target.mark_as_of, day, OPTIONS_CHAIN_SOURCE)
        after_close = SOURCES[OPTIONS_CHAIN_SOURCE]
        effective = _next_session(target.mark_as_of, calendar) if after_close else target.mark_as_of
        candidates.append((effective, after_close, target.mark_as_of, OPTIONS_CHAIN_SOURCE))
    for signal in target.contributing_signals:
        if signal.source_agent_id not in sources:
            raise ValueError(
                f"signal from {signal.source_agent_id!r} on {target.asset!r} comes from no "
                "analyst of the record's roster"
            )
        declared = sources[signal.source_agent_id]
        if declared is not None and signal.status == "active" and signal.source_as_of is None:
            raise LookAheadError(
                f"active signal of {signal.source_agent_id!r} for {target.asset!r} on "
                f"{day.isoformat()} has no source_as_of, but the analyst declares source "
                f"{declared!r}"
            )
        source = declared if signal.source_as_of is not None else None
        check_source_as_of(signal.source_as_of, day, source)
        if source is not None and signal.source_as_of is not None:
            after_close = SOURCES[source]
            effective = (
                _next_session(signal.source_as_of, calendar) if after_close else signal.source_as_of
            )
            candidates.append((effective, after_close, signal.source_as_of, source))
    if not candidates:
        return None, None
    _, _, as_of, source = min(
        candidates,
        key=lambda c: (-c[0].toordinal(), not c[1], -c[2].toordinal(), c[3]),
    )
    return source, as_of


def firm_day_rows(
    day: date,
    targets: Sequence[PodTarget],
    *,
    absent: Mapping[str, Sequence[str]],
    analyst_sources: Mapping[str, str | None],
    instrument_kinds: Mapping[str, DecisionKind],
    calendar: str,
    previous: Mapping[tuple[str, str], Provenance] | None = None,
    held_contracts: Mapping[str, float] | None = None,
    nav: float | None = None,
) -> list[DecisionRow]:
    """The decision rows of `day`: one per `targets` entry that carries a
    contributing signal or a nonzero weight, then, for each instrument of
    `absent` (no bar on `day`) without a target and each pod id it maps
    to, an unavailable weight-0 row. `analyst_sources` maps each roster
    analyst to its row source, `instrument_kinds` each ticker to its kind
    (an option row is `option`, keyed by its OSI symbol), and `calendar`
    is the decision calendar the after-close rule counts sessions on.
    `previous` maps (instrument id, pod id) to the provenance of the pod's
    row on the previous day, which a held target keeps; `held_contracts`
    maps each option the book holds to its signed contracts, which a held
    put is written at, priced at its `mark_mid` against `nav`, the book's
    value on `day`.

    Raises `ValueError` when a target is dated another day, a ticker has no
    kind, a signal comes from an analyst outside `analyst_sources`, or an
    instrument of `absent` has a target that is not carried, or an option's
    contract has no OSI symbol (`osi_symbol`); `LookAheadError` as
    `_row_source` documents, or when a kept provenance fails
    `decisions.check_source_as_of`."""
    previous = previous or {}
    held_contracts = held_contracts or {}
    multiplier = float(settings.options.contract_multiplier)
    day_signals = [
        signal for target in targets if not target.carried for signal in target.contributing_signals
    ]
    rows: list[DecisionRow] = []
    targeted: set[str] = set()
    for target in targets:
        if target.as_of_date != day:
            raise ValueError(
                f"target for {target.asset!r} is dated {target.as_of_date.isoformat()}, "
                f"not {day.isoformat()}"
            )
        if target.asset in absent and not target.carried:
            raise ValueError(f"{target.asset} has no bar on {day.isoformat()} but has a target")
        targeted.add(target.asset)
        weight, delta_equiv = target.target_weight, target.delta_equiv_weight
        if (
            target.hold_contracts
            and target.asset in held_contracts
            and target.mark_mid is not None
            and nav is not None
            and nav > 0.0
        ):
            weight = held_contracts[target.asset] * target.mark_mid * multiplier / nav
            delta_equiv = (
                None
                if delta_equiv is None or target.target_weight == 0.0
                else delta_equiv * weight / target.target_weight
            )
        if not target.contributing_signals and weight == 0.0:
            continue
        inputs = target
        if target.carried and not target.held:
            inputs = target.model_copy(
                update={"contributing_signals": [*target.contributing_signals, *day_signals]}
            )
        source, source_as_of = _row_source(day, inputs, analyst_sources, calendar)
        instrument_id = row_instrument_id(target.asset, target.instrument)
        kept = previous.get((instrument_id, target.pod_id))
        if target.held and kept is not None:
            source, source_as_of = kept
            check_source_as_of(source_as_of, day, source)
        rows.append(
            DecisionRow(
                decision_date=day,
                instrument_kind="option"
                if target.instrument is not None
                else _kind(target.asset, instrument_kinds),
                instrument_id=instrument_id,
                target_weight=weight,
                status=_row_status(target.contributing_signals, weight, target.carried),
                source=source,
                source_as_of=source_as_of,
                pod_id=target.pod_id,
                delta_equiv_weight=delta_equiv,
            )
        )
    for instrument, pods in absent.items():
        if instrument in targeted:
            continue
        kind = _kind(instrument, instrument_kinds)
        rows.extend(
            DecisionRow(
                decision_date=day,
                instrument_kind=kind,
                instrument_id=row_instrument_id(instrument),
                target_weight=0.0,
                status="unavailable",
                pod_id=pod,
            )
            for pod in pods
        )
    return rows


def row_provenance(rows: Iterable[DecisionRow]) -> dict[tuple[str, str], Provenance]:
    """(instrument id, pod id) to the source and source_as_of of each of
    `rows` with a pod: what the next day's held targets keep."""
    return {
        (row.instrument_id, row.pod_id): (row.source, row.source_as_of)
        for row in rows
        if row.pod_id is not None
    }


def _kind(ticker: str, instrument_kinds: Mapping[str, DecisionKind]) -> DecisionKind:
    """`ticker`'s kind from `instrument_kinds`; raises `ValueError` without one."""
    if ticker not in instrument_kinds:
        raise ValueError(f"{ticker} has no instrument kind in the record's universe")
    return instrument_kinds[ticker]
