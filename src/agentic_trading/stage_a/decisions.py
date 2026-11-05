# decisions.py

"""One stage (a) decision day: the weights an analyst's signals map to,
and the look-ahead audit of each row's `source` and `source_as_of`.

`decision_weights` applies the decision rule, `spec.DECISION_RULE`
(appended below), with `s_i` the signed confidence (long +1, short -1,
flat 0). An instrument whose status is `unavailable` or `error` weighs 0
and is outside the denominator, so an active day's book has gross 1.0.

`SOURCES` lists every dated data source an analyst can declare, with
whether it can publish after 16:00 ET; `UNDATED_SOURCES` lists the inputs
with no publication date, each with the look-ahead limitation a trial
declares for it. `row_source` names the one dated source a row from an
analyst carries, and `check_source_as_of` refuses a row whose
`source_as_of` the decision date's close cannot know: `source` and
`source_as_of` are both filled or both empty, the source is listed, and
`source_as_of` is on or before the decision date, strictly before it for
a source that can publish after the close (such a date counts from the
next session)."""

from collections.abc import Iterable, Sequence
from datetime import date

from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.stage_a.spec import DECISION_RULE

__doc__ = f"{__doc__}\n\nDecision rule: {DECISION_RULE}"

# Every dated source a `DataRequirement` can name, to whether it can publish
# after 16:00 ET (True) or always publishes before the close (False).
SOURCES: dict[str, bool] = {
    # CFTC Commitments of Traders: released Friday 15:30 ET.
    "cftc_cot": False,
    # Dividends: declared before the ex-date the row is dated by.
    "dividends": False,
    # Earnings: announced before the open or after the close.
    "earnings": True,
    # FINRA short interest: disseminated after the close.
    "finra_short_interest": True,
    # FRED: a vintage is dated by the day FRED publishes it; the H.15 daily rates post
    # around 16:15 ET and VIXCLS the next morning.
    "fred": True,
    # Ken French daily factors: posted after their month ends and dated by the end of
    # the following month, a conservative public date.
    "ken_french_factors": False,
    # Option chains: end-of-day snapshots taken at the 16:00 ET close of their session.
    "options_chain": False,
    # SEC EDGAR company facts: filings are accepted until 22:00 ET on their filed date.
    "sec_edgar_facts": True,
    # SEC Form 4 insider transactions: accepted until 22:00 ET on their filed date.
    "sec_insider_transactions": True,
    # S&P 500 membership: changes are announced days before their effective date.
    "sp500_membership": False,
}


# Every undated source a `DataRequirement` can name, to the look-ahead
# limitation it brings. An undated source has no publication date, so a
# decision row never names it (`source_of` skips it); each trial declares it
# once with its limitation (`undated_declarations`), and edgelab's grade
# carries it as a disclosed limitation.
UNDATED_SOURCES: dict[str, str] = {
    # SEC SIC codes: one current classification per ticker, no history.
    "sec_sic": (
        "current SIC code applied to past dates in the manager's pair screen to choose "
        "candidate partners; signals use prices only"
    ),
}


class LookAheadError(ValueError):
    """A decision row whose `source` and `source_as_of` the decision
    date's close cannot know, or which breaks the both-or-neither rule."""


def decision_weights(signals: Sequence[AgentSignal]) -> dict[str, float]:
    """Each signal's asset to its target weight under the decision rule:
    signed confidence over the sum of absolute signed confidences of the
    active signals; 0.0 for every non-active signal, and for every signal
    when that sum is 0. Takes one day's signals, one per asset; returns
    `{}` for none. Raises `ValueError` when two signals share an asset."""
    signed: dict[str, float] = {}
    for signal in signals:
        if signal.asset in signed:
            raise ValueError(f"two signals for asset {signal.asset!r}")
        signed[signal.asset] = _signed_confidence(signal) if signal.status == "active" else 0.0
    gross = sum(abs(value) for value in signed.values())
    if gross == 0.0:
        return dict.fromkeys(signed, 0.0)
    return {asset: value / gross for asset, value in signed.items()}


def _signed_confidence(signal: AgentSignal) -> float:
    """`signal`'s confidence signed by its direction: long +, short -, flat 0."""
    sign = {"long": 1.0, "short": -1.0, "flat": 0.0}[signal.direction]
    return sign * signal.confidence


def row_source(analyst: AnalystAgent) -> str | None:
    """The source `analyst`'s rows carry, from its data requirements (see
    `source_of`)."""
    return source_of(requirement.source for requirement in analyst.data_requirements)


def undated_declarations(sources: Iterable[str]) -> list[dict[str, str | bool]]:
    """One declaration per undated source among `sources`, in name order:
    `{"name": ..., "undated": True, "limitation": ...}` with the limitation
    `UNDATED_SOURCES` gives it. Dated sources are left out."""
    return [
        {"name": name, "undated": True, "limitation": UNDATED_SOURCES[name]}
        for name in sorted(set(sources))
        if name in UNDATED_SOURCES
    ]


def undated_sources_of(analyst: AnalystAgent) -> list[dict[str, str | bool]]:
    """`undated_declarations` of `analyst`'s data requirements."""
    return undated_declarations(requirement.source for requirement in analyst.data_requirements)


def source_of(sources: Iterable[str]) -> str | None:
    """The one source a row resting on `sources` carries: `None` when none
    of `sources` is dated, else the alphabetically first dated source that
    can publish after the close, else the alphabetically first dated
    source. Undated sources (`UNDATED_SOURCES`) never name a row. An
    after-close source is chosen whenever there is one because it is
    audited more strictly. Raises `LookAheadError` naming a source neither
    `SOURCES` nor `UNDATED_SOURCES` lists."""
    listed = sorted(set(sources))
    unlisted = [name for name in listed if name not in SOURCES and name not in UNDATED_SOURCES]
    if unlisted:
        raise LookAheadError(f"sources not in SOURCES: {unlisted}")
    names = [name for name in listed if name in SOURCES]
    if not names:
        return None
    after_close = [name for name in names if SOURCES[name]]
    return (after_close or names)[0]


def check_source_as_of(source_as_of: date | None, decision_date: date, source: str | None) -> None:
    """Return when a row decided at the close of `decision_date`, carrying
    `source` published on `source_as_of`, uses only what that close can
    know; both `None` is a price-only row and passes. Raises
    `LookAheadError` when exactly one of the two is `None`, when `source`
    is not in `SOURCES`, when `source_as_of` is after `decision_date`, or
    when it equals `decision_date` and `source` can publish after the
    close."""
    if source_as_of is None and source is None:
        return
    if source_as_of is None or source is None:
        raise LookAheadError(
            f"source {source!r} and source_as_of {source_as_of} must both be set or both be empty"
        )
    if source not in SOURCES:
        raise LookAheadError(f"source {source!r} is not in SOURCES")
    if source_as_of > decision_date:
        raise LookAheadError(
            f"source_as_of {source_as_of.isoformat()} of {source!r} is after the decision date "
            f"{decision_date.isoformat()}"
        )
    if SOURCES[source] and source_as_of == decision_date:
        raise LookAheadError(
            f"source_as_of {source_as_of.isoformat()} of after-close source {source!r} is not "
            f"before the decision date {decision_date.isoformat()}"
        )
