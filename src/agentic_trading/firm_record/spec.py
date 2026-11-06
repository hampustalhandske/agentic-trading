# spec.py

"""The frozen spec of one firm record (`stage_a.spec.FirmSpec`), built from
the Commander that runs it."""

import hashlib
import sys
from datetime import UTC, date, datetime
from pathlib import Path

from agentic_trading import provenance
from agentic_trading.agents._stable_value import qualified_name, stable_value
from agentic_trading.agents.commander import Commander
from agentic_trading.backtest.resume_fingerprint import roster_agents
from agentic_trading.stage_a.spec import (
    RUNTIME_SETTINGS_FIELDS,
    FirmSpec,
    _module_hashes,
    hashed_settings,
)

# How the Commander's targets become the record's decision rows.
FIRM_DECISION_RULE = (
    "one row per (decision_date, instrument_id, pod_id) on every session of "
    "stage_a.session_calendar: target_weight is the pod's contribution (allocation x pod weight) "
    "after firm limits, each core pod's contribution to an instrument scaled by the factor the "
    "limits applied to it, the tail hedge at face value, the leverage clamp pro rata; option rows "
    "are signed premium weights with delta_equiv_weight = weight x delta x spot / mid, a held put "
    "at its held contracts x mid x multiplier / NAV; a pod writes a row only for an instrument it "
    "ran an analyst on or holds; an instrument held over a session without its bar keeps each "
    "pod's last decided target, counted by the firm limits, and is filled at the next session "
    "with a bar; one without a bar and without a decided target gets an unavailable weight-0 row "
    "per pod in its scope; status is active when the weight is nonzero or the target is carried, "
    "else active when any contributing signal is active, else error when any errored, else "
    "unavailable; source and source_as_of are the contributing input's with the latest effective "
    "known time (signals, and an option row's chain snapshot), an after-close date counting from "
    "the next session; a target the pod holds rather than sets keeps the provenance of the row "
    "that set it"
)

# The modules, besides each agent's own, whose source decides the record
# even when it starts before they are imported: their hashes, and those of
# every agents/data module they import, enter the spec.
_HASHED_ROOTS = (
    "agentic_trading.agents.commander",
    "agentic_trading.backtest.engine",
    "agentic_trading.backtest.portfolio",
    "agentic_trading.firm_record.rows",
    "agentic_trading.firm_record.runner",
    "agentic_trading.stage_a.decisions",
    "agentic_trading.stage_a.store",
)
# Loaded packages whose source never moves a record: the dashboard and the
# command line around it.
_UNHASHED_PACKAGES = ("agentic_trading.gui", "agentic_trading.cli")


def _loaded_module_hashes() -> dict[str, str]:
    """Dotted name to the sha256 hex digest of the source file of every
    `agentic_trading` module in `sys.modules` with a `.py` file, the
    modules under `_UNHASHED_PACKAGES` aside."""
    hashes: dict[str, str] = {}
    for name, module in sorted(sys.modules.items()):
        if name != "agentic_trading" and not name.startswith("agentic_trading."):
            continue
        if any(name == root or name.startswith(f"{root}.") for root in _UNHASHED_PACKAGES):
            continue
        file = getattr(module, "__file__", None)
        if file is None or not file.endswith(".py"):
            continue
        hashes[name] = hashlib.sha256(Path(file).read_bytes()).hexdigest()
    return hashes


def build_firm_spec(
    commander: Commander,
    universe: list[str],
    window_start: date,
    window_end: date,
    lake_time: datetime,
) -> FirmSpec:
    """The `FirmSpec` of running `commander` over `universe` from
    `window_start` to `window_end` at `lake_time` (timezone-aware, recorded
    in UTC), with the current `settings` and `RUNTIME_SETTINGS_FIELDS` read
    at call time. The roster is each of `roster_agents(commander)`'s
    `built_parameters()`, in that order; the commander entry holds its
    class, weighting strategy, pod-return lookback and drawdown breaker,
    never its worker count. The module hashes cover every `agentic_trading`
    module loaded at the call (`_loaded_module_hashes`) and every module
    reachable from the agents' modules and `_HASHED_ROOTS`. Raises
    `ValueError` when the window is reversed, `lake_time` is naive, the
    installed quantcore's commit is unknown, or a hashed module has no
    source file."""
    quantcore_sha = provenance.quantcore_sha()
    if quantcore_sha is None:
        raise ValueError(
            "the installed quantcore records no commit, so the record's code cannot be pinned"
        )
    if lake_time.tzinfo is None:
        raise ValueError(f"lake_time {lake_time.isoformat()} has no timezone")
    agents = roster_agents(commander)
    modules = sorted({type(agent).__module__ for agent in agents} | set(_HASHED_ROOTS))
    module_hashes = _loaded_module_hashes()
    for module in modules:
        module_hashes.update(_module_hashes(module))
    return FirmSpec(
        commander={
            "class": qualified_name(type(commander)),
            "weighting_strategy": stable_value(commander.weighting_strategy),
            "pod_returns_lookback_days": commander.pod_returns_lookback_days,
            "drawdown_breaker": stable_value(commander.drawdown_breaker),
        },
        roster=[agent.built_parameters() for agent in agents],
        settings=hashed_settings(RUNTIME_SETTINGS_FIELDS),
        module_hashes=dict(sorted(module_hashes.items())),
        universe=list(universe),
        window_start=window_start,
        window_end=window_end,
        decision_rule=FIRM_DECISION_RULE,
        quantcore_sha=quantcore_sha,
        lake_time=lake_time.astimezone(UTC),
    )
