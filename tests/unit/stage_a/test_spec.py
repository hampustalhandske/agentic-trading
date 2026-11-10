# test_spec.py

"""AT-110 precompute spec item 1: `StageASpec` freezes what defines one stage (a) trial and
`spec_hash` is the sha256 of its canonical JSON. `SpecRecord` carries the lake time, lake version
and quantcore version and sha beside the hash, outside the hashed part, with no machine path."""

import hashlib
import importlib
import json
import re
from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest

from agentic_trading import provenance
from agentic_trading.agents.analysts import CIRRateRegimeAnalyst
from agentic_trading.config import settings, temporary_overrides
from agentic_trading.stage_a import spec as spec_module
from agentic_trading.stage_a.spec import SpecRecord, StageASpec, build_spec, spec_hash

_UNIVERSE = ("SHY", "IEF", "TLT")
_START, _END = date(2010, 1, 4), date(2024, 12, 31)
_DECISION_RULE = (
    "w_i = s_i / sum_j |s_j| over active instruments; s_i = sign(direction_i) * confidence_i; "
    "non-active and zero-sum days weight 0"
)
_RECORD_FIELDS = {"lake_time", "lake_version", "quantcore_version"}
# The agentic_trading.agents modules cir_rate_regime_analyst.py imports directly.
_CIR_AGENT_IMPORTS = {
    "agentic_trading.agents._market_math",
    "agentic_trading.agents.base",
    "agentic_trading.agents.schemas",
}


def _build(
    analyst: CIRRateRegimeAnalyst | None = None,
    universe: tuple[str, ...] = _UNIVERSE,
    start: date = _START,
    end: date = _END,
) -> StageASpec:
    return build_spec(analyst or CIRRateRegimeAnalyst(), list(universe), start, end)


def _hash(**kwargs: Any) -> str:
    return spec_hash(_build(**kwargs))


def _record(spec: StageASpec, **overrides: Any) -> SpecRecord:
    fields: dict[str, Any] = {
        "lake_time": datetime(2026, 10, 29, tzinfo=UTC),
        "lake_version": "f89aef2f428a",
        "quantcore_version": "0.4.0",
    }
    return SpecRecord(spec=spec, spec_hash=spec_hash(spec), **(fields | overrides))


def _reversed(value: Any) -> Any:
    """`value` with the key order of every dict in it reversed, recursively."""
    if isinstance(value, dict):
        return {key: _reversed(value[key]) for key in reversed(list(value))}
    return value


def _strings(value: Any) -> Iterator[str]:
    """Every string in a parsed JSON value, dict keys included."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def test_build_spec_records_the_trial() -> None:
    """A built spec holds the analyst, its parameters, settings, universe, window and rule."""
    spec = _build()
    cls = CIRRateRegimeAnalyst

    assert spec_module.RUNTIME_SETTINGS_FIELDS == (
        "stage_a.min_free_memory_pct",
        "stage_a.min_free_disk_gb",
        "stage_a.check_every_days",
        "trials.ledger_path",
        "trials.registry_dir",
        "trials.firm_family",
        "lake.request_quiet_seconds",
    )
    assert spec.agent_id == "cir_rate_regime_analyst"
    assert spec.analyst_class in {cls.__qualname__, f"{cls.__module__}.{cls.__qualname__}"}
    assert spec.analyst_params["rate_sensitivity"] == -1.0
    assert spec.settings["analysts"]["risk_window_long"] == settings.analysts.risk_window_long
    assert spec.universe == list(_UNIVERSE)
    assert (spec.window_start, spec.window_end) == (_START, _END)
    assert spec.decision_rule == _DECISION_RULE
    assert re.fullmatch(r"[0-9a-f]{64}", spec_hash(spec))


def test_spec_hash_is_stable_and_independent_of_dict_order() -> None:
    """Two builds of the same analyst hash alike, and reordering every dict leaves the hash."""
    spec = _build()
    reordered = StageASpec.model_validate(_reversed(spec.model_dump()))

    assert list(reordered.settings) != list(spec.settings)
    assert spec_hash(_build()) == spec_hash(spec) == spec_hash(reordered)


def test_spec_hash_changes_with_each_hashed_input() -> None:
    """A settings override, a constructor parameter, the universe or the window changes it."""
    analyst = CIRRateRegimeAnalyst()
    base = _hash(analyst=analyst)
    with temporary_overrides({"analysts.var_confidence_level": 0.99}):
        overridden = _build(analyst)

    changed = {
        "settings": spec_hash(overridden),
        "params": _hash(analyst=CIRRateRegimeAnalyst(rate_sensitivity=-0.5)),
        "universe": _hash(analyst=analyst, universe=("SHY", "IEF")),
        "window_start": _hash(analyst=analyst, start=date(2011, 1, 3)),
        "window_end": _hash(analyst=analyst, end=date(2023, 12, 29)),
    }

    assert overridden.settings["analysts"]["var_confidence_level"] == 0.99
    assert [label for label, digest in changed.items() if digest == base] == []
    assert len(set(changed.values())) == len(changed)


def test_record_fields_stay_outside_the_hash() -> None:
    """Lake time, lake version and quantcore version live on SpecRecord only and leave the hash."""
    spec = _build()
    records = [
        _record(spec),
        _record(
            spec,
            lake_time=datetime(2026, 11, 5, 12, tzinfo=UTC),
            lake_version="c69d5dd",
            quantcore_version="0.5.0",
        ),
    ]

    assert not _RECORD_FIELDS & set(StageASpec.model_fields)
    assert _RECORD_FIELDS <= set(SpecRecord.model_fields)
    for record in records:
        reread = SpecRecord.model_validate_json(record.model_dump_json())
        assert spec_hash(reread.spec) == record.spec_hash == spec_hash(spec)


def test_runtime_settings_fields_are_excluded(monkeypatch: pytest.MonkeyPatch) -> None:
    """A settings key hashed by default leaves the hash once named in RUNTIME_SETTINGS_FIELDS."""
    analyst = CIRRateRegimeAnalyst()
    key = "analysts.risk_window_long"
    override = {key: settings.analysts.risk_window_long + 50}
    hashed_base = _hash(analyst=analyst)
    with temporary_overrides(override):
        hashed_changed = _hash(analyst=analyst)

    monkeypatch.setattr(spec_module, "RUNTIME_SETTINGS_FIELDS", (key,))
    runtime_base = _hash(analyst=analyst)
    with temporary_overrides(override):
        runtime_changed = _hash(analyst=analyst)

    assert hashed_changed != hashed_base
    assert runtime_changed == runtime_base


def test_trials_settings_stay_outside_the_hash() -> None:
    """The trials group never enters the snapshot, so overriding it leaves the hash."""
    analyst = CIRRateRegimeAnalyst()
    base = _build(analyst)
    overrides = {"trials.firm_family": "other-firm", "trials.ledger_path": "state/trials.duckdb"}
    with temporary_overrides(overrides):
        overridden = _build(analyst)

    assert "trials" not in base.settings
    assert spec_hash(overridden) == spec_hash(base)


def test_module_hashes_cover_the_analyst_and_its_agents_imports() -> None:
    """module_hashes holds the analyst's module and its direct agents imports, each by sha256."""
    hashes = _build().module_hashes

    assert {CIRRateRegimeAnalyst.__module__, *_CIR_AGENT_IMPORTS} <= set(hashes)
    assert all(
        name.startswith(("agentic_trading.agents.", "agentic_trading.data.")) for name in hashes
    )
    for name, digest in hashes.items():
        source = Path(importlib.import_module(name).__file__ or "")
        assert digest == hashlib.sha256(source.read_bytes()).hexdigest(), name


def test_spec_record_json_has_no_machine_path() -> None:
    """No key or value in SpecRecord's JSON contains a user, temp or home directory path."""
    home = str(Path.home())
    forbidden = ["/Users", "/private", "/home", *([home] if home != "/" else [])]
    texts = list(_strings(json.loads(_record(_build()).model_dump_json())))

    assert texts
    assert [text for text in texts if any(part in text for part in forbidden)] == []


def test_spec_hashes_the_installed_quantcore_sha(monkeypatch: pytest.MonkeyPatch) -> None:
    """The spec carries the installed quantcore commit, and another commit is another hash."""
    installed = provenance.quantcore_sha()
    spec = _build()
    other = "f" * 40
    monkeypatch.setattr(provenance, "quantcore_sha", lambda: other)
    repinned = _build()

    assert spec.quantcore_sha == installed
    assert repinned.quantcore_sha == other
    assert spec_hash(repinned) != spec_hash(spec)
