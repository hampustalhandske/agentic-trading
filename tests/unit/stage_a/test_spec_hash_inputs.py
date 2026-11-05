# test_spec_hash_inputs.py

"""Every constant that changes a stage (a) signal enters the spec hash: `stage_a.warmup_days`
is a hashed setting, and `module_hashes` covers every `agentic_trading.agents` and
`agentic_trading.data` module reachable through imports from the analyst's module."""

import hashlib
import importlib
from datetime import date
from pathlib import Path

import pytest

from agentic_trading.agents.analysts import (
    BlackScholesValuationAnalyst,
    CIRRateRegimeAnalyst,
    HestonVolSurfaceAnalyst,
)
from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.roster import build_default_analysts
from agentic_trading.config import StageAConfig, settings, temporary_overrides
from agentic_trading.stage_a import spec as spec_module
from agentic_trading.stage_a.spec import RUNTIME_SETTINGS_FIELDS, build_spec, spec_hash

_UNIVERSE = ["SPY"]
_START, _END = date(2010, 1, 4), date(2024, 12, 31)
_PACKAGES = ("agentic_trading.agents.", "agentic_trading.data.")
# Modules the options analysts reach only through `cvar_pod._options_data`.
_OPTIONS_CHAIN = {
    "agentic_trading.agents.analysts.cvar_pod._options_data",
    "agentic_trading.data.loaders.lakehouse_loader",
    "agentic_trading.data.loaders.option_marks",
}


def _sha256(module: str) -> str:
    source = Path(importlib.import_module(module).__file__ or "")
    return hashlib.sha256(source.read_bytes()).hexdigest()


def test_warmup_days_is_a_hashed_stage_a_setting() -> None:
    """stage_a.warmup_days defaults to 1100, is hashed, and changing it changes the spec hash."""
    analyst = CIRRateRegimeAnalyst()
    base = build_spec(analyst, _UNIVERSE, _START, _END)
    with temporary_overrides({"stage_a.warmup_days": settings.stage_a.warmup_days + 100}):
        overridden = build_spec(analyst, _UNIVERSE, _START, _END)

    assert StageAConfig().warmup_days == 1100
    assert "stage_a.warmup_days" not in RUNTIME_SETTINGS_FIELDS
    assert base.settings["stage_a"]["warmup_days"] == 1100
    assert overridden.settings["stage_a"]["warmup_days"] == 1200
    assert spec_hash(overridden) != spec_hash(base)


def test_warmup_days_must_be_positive() -> None:
    """A warm-up of zero days is refused."""
    with pytest.raises(ValueError), temporary_overrides({"stage_a.warmup_days": 0}):
        pass


@pytest.mark.parametrize("analyst_cls", [HestonVolSurfaceAnalyst, BlackScholesValuationAnalyst])
def test_options_analysts_hash_the_loaders_behind_their_options_data(
    analyst_cls: type[AnalystAgent],
) -> None:
    """The options analysts' module_hashes reach the lakehouse loader through _options_data."""
    hashes = build_spec(analyst_cls(), _UNIVERSE, _START, _END).module_hashes

    assert analyst_cls.__module__ in hashes
    assert _OPTIONS_CHAIN <= set(hashes)
    for name, digest in hashes.items():
        assert digest == _sha256(name), name


def test_roster_module_hashes_stay_inside_agents_and_data() -> None:
    """No roster analyst's module_hashes names a module outside agents and data, and each one
    holds its own module."""
    outside = {}
    for analyst in build_default_analysts():
        hashes = build_spec(analyst, _UNIVERSE, _START, _END).module_hashes
        assert type(analyst).__module__ in hashes, analyst.agent_id
        stray = sorted(name for name in hashes if not name.startswith(_PACKAGES))
        if stray:
            outside[analyst.agent_id] = stray

    assert outside == {}


def test_module_hashes_follow_from_package_import_submodule() -> None:
    """`from <package> import <submodule>` reaches the submodule and the package's own module."""
    hashes = spec_module._module_hashes("agentic_trading.agents.managers.cvar_manager")

    assert "agentic_trading.data.loaders.option_marks" in hashes
    assert "agentic_trading.data.loaders" in hashes


def test_package_imports_parses_relative_and_submodule_imports() -> None:
    """Relative imports resolve against the module's package; only agents and data submodules
    are kept, and a name imported from a module is not itself a module."""
    source = b"\n".join(
        [
            b"import numpy as np",
            b"from agentic_trading.config import settings",
            b"from agentic_trading.data.loaders import option_marks",
            b"from . import _options_data",
            b"from ..risk_pod.gbm_momentum_analyst import GBMMomentumAnalyst",
            b"def lazy():",
            b"    import agentic_trading.data.point_in_time",
        ]
    )
    found = spec_module._package_imports("agentic_trading.agents.analysts.cvar_pod.example", source)

    assert found == {
        "agentic_trading.data.loaders",
        "agentic_trading.data.loaders.option_marks",
        "agentic_trading.agents.analysts.cvar_pod",
        "agentic_trading.agents.analysts.cvar_pod._options_data",
        "agentic_trading.agents.analysts.risk_pod.gbm_momentum_analyst",
        "agentic_trading.data.point_in_time",
    }
