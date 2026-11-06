# test_spec.py

"""AT-125 acceptance 3: a firm record's spec holds the hashed settings, the roster (each
agent's built parameters in roster order), the module hashes, the universe, the window, the
decision rule and the quantcore commit, like a stage (a) trial; its hash is the trial's
config_hash."""

import json
import sys
from datetime import UTC, date, datetime, timedelta, timezone
from importlib import import_module

import pytest
from pydantic import ValidationError

from agentic_trading import provenance
from agentic_trading.agents.commander import Commander
from agentic_trading.agents.roster import build_default_analysts, build_default_managers
from agentic_trading.backtest.resume_fingerprint import roster_agents
from agentic_trading.config import temporary_overrides
from agentic_trading.firm_record.spec import FIRM_DECISION_RULE, build_firm_spec
from agentic_trading.scoring.weighting import Hedge, ShrinkageWeight
from agentic_trading.stage_a.spec import (
    RUNTIME_SETTINGS_FIELDS,
    FirmSpecRecord,
    canonical_firm_spec_json,
    firm_spec_hash,
    hashed_settings,
)
from agentic_trading.trials.registry import config_hash_of

_START, _END = date(2010, 1, 4), date(2010, 1, 29)
_UNIVERSE = ["SPY", "AAPL", "GC=F"]
_T = datetime(2026, 11, 6, 12, 0, tzinfo=UTC)


def _commander() -> Commander:
    analysts = build_default_analysts(excluded_analyst_ids={"pairs_cointegration_analyst"})
    return Commander(
        managers=build_default_managers(analysts), weighting_strategy=ShrinkageWeight(Hedge())
    )


def test_spec_records_the_firm() -> None:
    """Every field is filled from the commander, the settings and the arguments."""
    commander = _commander()
    spec = build_firm_spec(commander, _UNIVERSE, _START, _END, _T)

    assert spec.roster == [agent.built_parameters() for agent in roster_agents(commander)]
    assert spec.roster[0]["agent_id"] == commander.managers[0].agent_id
    assert spec.settings == hashed_settings(RUNTIME_SETTINGS_FIELDS)
    assert spec.commander["class"] == "agentic_trading.agents.commander.Commander"
    assert spec.commander["weighting_strategy"]["class"].endswith("ShrinkageWeight")
    assert "parallel_workers" not in json.dumps(spec.commander)
    for module in (
        "agentic_trading.agents.commander",
        "agentic_trading.agents.managers.risk_manager",
        "agentic_trading.backtest.engine",
        "agentic_trading.backtest.portfolio",
        "agentic_trading.firm_record.rows",
        "agentic_trading.stage_a.store",
    ):
        assert module in spec.module_hashes, module
    assert spec.universe == _UNIVERSE
    assert (spec.window_start, spec.window_end) == (_START, _END)
    assert spec.decision_rule == FIRM_DECISION_RULE
    assert spec.quantcore_sha == provenance.quantcore_sha()


def test_hash_is_the_ledger_config_hash() -> None:
    """The hash is config_hash_of the canonical JSON, which sorts the universe."""
    spec = build_firm_spec(_commander(), _UNIVERSE, _START, _END, _T)
    reordered = spec.model_copy(update={"universe": list(reversed(_UNIVERSE))})

    assert config_hash_of(json.loads(canonical_firm_spec_json(spec))) == firm_spec_hash(spec)
    assert firm_spec_hash(reordered) == firm_spec_hash(spec)


def test_hash_moves_with_what_the_firm_computes() -> None:
    """A roster parameter or a hashed setting changes the hash; a runtime setting does not."""
    base = build_firm_spec(_commander(), _UNIVERSE, _START, _END, _T)
    with temporary_overrides({"commander.drawdown_trip_threshold": -0.2}):
        overridden = build_firm_spec(_commander(), _UNIVERSE, _START, _END, _T)
    with temporary_overrides({"stage_a.check_every_days": 3}):
        runtime = build_firm_spec(_commander(), _UNIVERSE, _START, _END, _T)
    wider = build_firm_spec(
        Commander(
            managers=build_default_managers(
                build_default_analysts(excluded_analyst_ids=frozenset())
            ),
            weighting_strategy=ShrinkageWeight(Hedge()),
        ),
        _UNIVERSE,
        _START,
        _END,
        _T,
    )

    assert firm_spec_hash(overridden) != firm_spec_hash(base)
    assert firm_spec_hash(runtime) == firm_spec_hash(base)
    assert firm_spec_hash(wider) != firm_spec_hash(base)


def test_record_refuses_a_wrong_hash() -> None:
    """A record whose hash is not its spec's is refused."""
    spec = build_firm_spec(_commander(), _UNIVERSE, _START, _END, _T)
    with pytest.raises(ValidationError):
        FirmSpecRecord(
            spec=spec,
            spec_hash="0" * 64,
            lake_time=_T,
            lake_version="v",
            quantcore_version="0.3.1",
        )


def test_lake_time_enters_the_hash() -> None:
    """Another lake time is another trial; the same instant in another zone is not."""
    base = build_firm_spec(_commander(), _UNIVERSE, _START, _END, _T)
    later = build_firm_spec(_commander(), _UNIVERSE, _START, _END, _T + timedelta(hours=1))
    shifted = build_firm_spec(
        _commander(), _UNIVERSE, _START, _END, _T.astimezone(timezone(timedelta(hours=2)))
    )

    assert base.lake_time == _T
    assert firm_spec_hash(later) != firm_spec_hash(base)
    assert firm_spec_hash(shifted) == firm_spec_hash(base)


def test_record_lake_time_is_the_specs() -> None:
    """A record whose lake time is not its spec's is refused."""
    spec = build_firm_spec(_commander(), _UNIVERSE, _START, _END, _T)
    with pytest.raises(ValidationError, match="lake time"):
        FirmSpecRecord(
            spec=spec,
            spec_hash=firm_spec_hash(spec),
            lake_time=_T + timedelta(days=1),
            lake_version="v",
            quantcore_version="0.3.1",
        )


def test_unknown_quantcore_commit_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the installed quantcore's commit the record cannot be pinned."""
    monkeypatch.setattr(provenance, "quantcore_sha", lambda: None)
    with pytest.raises(ValueError, match="quantcore"):
        build_firm_spec(_commander(), _UNIVERSE, _START, _END, _T)


def test_every_loaded_module_is_hashed() -> None:
    """Every agentic_trading module loaded when the record starts is hashed (scoring,
    backtest and config included), the GUI and CLI aside."""
    import_module("agentic_trading.cli")
    spec = build_firm_spec(_commander(), _UNIVERSE, _START, _END, _T)
    loaded = {
        name
        for name, module in sys.modules.items()
        if name.startswith("agentic_trading.")
        and getattr(module, "__file__", None)
        and not name.startswith(("agentic_trading.gui", "agentic_trading.cli"))
    }

    assert loaded <= set(spec.module_hashes)
    assert {
        "agentic_trading.config",
        "agentic_trading.scoring.track_record",
        "agentic_trading.backtest.engine",
    } <= set(spec.module_hashes)
    assert not any(name.startswith("agentic_trading.cli") for name in spec.module_hashes)
