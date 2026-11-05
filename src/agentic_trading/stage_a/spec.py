# spec.py

"""The frozen definition of one stage (a) trial and its hash.

`StageASpec` holds everything that defines a trial: the analyst (its
`agent_id`, class and constructor parameters), the `settings` snapshot
minus `RUNTIME_SETTINGS_FIELDS`, the sha256 of the analyst's own module
and of every `agentic_trading.agents` and `agentic_trading.data` module
reachable through imports from it, the universe, the window and the
decision rule. `spec_hash` is the sha256 of the spec's canonical JSON,
so two specs hash alike exactly when they define the same trial.

`SpecRecord` is what spec.json holds: the spec and its hash, plus the
lake time T, the lake version and the quantcore version and sha, which
are recorded beside the hash and never enter it. No field holds a
machine path: modules are named by their dotted import name, never by
file location."""

import ast
import hashlib
import importlib.util
import json
from datetime import date
from pathlib import Path
from typing import Any, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, model_validator

from agentic_trading.agents.base import AnalystAgent
from agentic_trading.config import settings

# Dotted `settings` paths (`<group>.<field>`, or `pod_sizing.<pod>.<field>`)
# that tune how a run executes or where it is recorded rather than what it
# computes, and so are left out of the hashed snapshot: the batch runner's
# resource guard, and the `trials` group (the trial ledger, the registry
# directory and the firm family), which is therefore absent from the
# snapshot. Every other `settings` field is a strategy constant.
RUNTIME_SETTINGS_FIELDS: tuple[str, ...] = (
    "stage_a.min_free_memory_pct",
    "stage_a.min_free_disk_gb",
    "stage_a.check_every_days",
    "trials.ledger_path",
    "trials.registry_dir",
    "trials.firm_family",
)

# The rule that turns one day's analyst signals into instrument weights.
DECISION_RULE = (
    "w_i = s_i / sum_j |s_j| over active instruments; s_i = sign(direction_i) * confidence_i; "
    "non-active and zero-sum days weight 0"
)

# The top-level package, and the packages within it whose modules
# `module_hashes` covers and follows imports through.
_ROOT_PACKAGE = "agentic_trading"
_HASHED_PACKAGES = ("agentic_trading.agents", "agentic_trading.data")


class StageASpec(BaseModel):
    """Everything that defines one stage (a) trial; every field is hashed
    by `spec_hash`.

    - `agent_id`, `analyst_class`: the analyst's id and its class as
      `<module>.<qualname>`;
    - `analyst_params`: the analyst's public fields as JSON values;
    - `settings`: `settings` as JSON values, minus
      `RUNTIME_SETTINGS_FIELDS` and any group they empty;
    - `module_hashes`: dotted module name to the sha256 hex digest of the
      module's source file, for the analyst's own module and every
      `agentic_trading.agents` and `agentic_trading.data` module reachable
      through imports from it (see `_module_hashes`);
    - `universe`: the instrument ids the analyst runs over;
    - `window_start`, `window_end`: the trial's first and last date,
      inclusive;
    - `decision_rule`: the rule turning signals into weights.

    Raises `ValueError` when `window_start` is after `window_end`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    agent_id: str
    analyst_class: str
    analyst_params: dict[str, Any]
    settings: dict[str, Any]
    module_hashes: dict[str, str]
    universe: list[str]
    window_start: date
    window_end: date
    decision_rule: str

    @model_validator(mode="after")
    def _window_is_ordered(self) -> Self:
        if self.window_start > self.window_end:
            raise ValueError(
                f"window_start {self.window_start.isoformat()} is after window_end "
                f"{self.window_end.isoformat()}"
            )
        return self


def canonical_spec_json(spec: StageASpec) -> str:
    """`spec`'s canonical JSON: its JSON-mode dump with `universe` sorted,
    serialized with sorted keys and no whitespace, non-JSON values written
    through `str`. Dict key order and instrument order never change it."""
    payload = spec.model_dump(mode="json")
    payload["universe"] = sorted(payload["universe"])
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def spec_hash(spec: StageASpec) -> str:
    """The sha256 hex digest of `canonical_spec_json(spec)`, UTF-8 encoded."""
    return hashlib.sha256(canonical_spec_json(spec).encode("utf-8")).hexdigest()


class SpecRecord(BaseModel):
    """The contents of one trial's spec.json: `spec` and its `spec_hash`,
    plus what the trial ran against, recorded outside the hash:
    `lake_time` (T, timezone-aware), `lake_version` (the lake service's
    git commit), `quantcore_version` and `quantcore_sha` (the pinned
    quantcore commit, `None` when unknown). Raises `ValueError` when
    `spec_hash` is not the hash of `spec`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    spec: StageASpec
    spec_hash: str
    lake_time: AwareDatetime
    lake_version: str
    quantcore_version: str
    quantcore_sha: str | None

    @model_validator(mode="after")
    def _hash_matches_spec(self) -> Self:
        expected = spec_hash(self.spec)
        if self.spec_hash != expected:
            raise ValueError(f"spec_hash {self.spec_hash} is not the spec's hash {expected}")
        return self


def build_spec(
    analyst: AnalystAgent, universe: list[str], window_start: date, window_end: date
) -> StageASpec:
    """The `StageASpec` of running `analyst` over `universe` from
    `window_start` to `window_end`, with the current `settings` and
    `RUNTIME_SETTINGS_FIELDS` read at call time. Raises `KeyError` when a
    `RUNTIME_SETTINGS_FIELDS` path names no setting, and `ValueError` when
    the window is reversed."""
    cls = type(analyst)
    return StageASpec(
        agent_id=analyst.agent_id,
        analyst_class=f"{cls.__module__}.{cls.__qualname__}",
        analyst_params=analyst.model_dump(mode="json"),
        settings=_hashed_settings(RUNTIME_SETTINGS_FIELDS),
        module_hashes=_module_hashes(cls.__module__),
        universe=list(universe),
        window_start=window_start,
        window_end=window_end,
        decision_rule=DECISION_RULE,
    )


def _hashed_settings(runtime_fields: tuple[str, ...]) -> dict[str, Any]:
    """`settings` as JSON values with every dotted path in `runtime_fields`
    removed, and every group (or pod) left empty by the removal dropped
    too, so a group made only of runtime fields leaves no trace in the
    snapshot. Raises `KeyError` naming a path that is not a setting."""
    snapshot: dict[str, Any] = settings.model_dump(mode="json")
    for path in runtime_fields:
        *parents, leaf = path.split(".")
        nodes: list[dict[str, Any]] = [snapshot]
        for part in parents:
            node = nodes[-1].get(part)
            if not isinstance(node, dict):
                raise KeyError(f"runtime settings field {path!r} names no setting")
            nodes.append(node)
        if leaf not in nodes[-1]:
            raise KeyError(f"runtime settings field {path!r} names no setting")
        del nodes[-1][leaf]
        for depth in range(len(parents), 0, -1):
            if nodes[depth]:
                break
            del nodes[depth - 1][parents[depth - 1]]
    return snapshot


def _module_file(module: str) -> Path:
    """The source file of importable `module`. Raises `ValueError` when it
    has none."""
    found = importlib.util.find_spec(module)
    if found is None or found.origin is None or not found.has_location:
        raise ValueError(f"module {module!r} has no source file")
    return Path(found.origin)


def _in_hashed_packages(name: str) -> bool:
    """Whether dotted `name` lies under one of `_HASHED_PACKAGES`."""
    return any(name.startswith(f"{package}.") for package in _HASHED_PACKAGES)


def _package_source(name: str) -> Path | None:
    """The source file of `agentic_trading` submodule `name`, located on
    disk under the package's directory without importing anything:
    `<parts>.py`, or `<parts>/__init__.py` for a package. `None` when
    neither exists, as for a name defined inside a module, or when `name`
    is not under `agentic_trading`."""
    top, _, rest = name.partition(".")
    if top != _ROOT_PACKAGE or not rest:
        return None
    root = _module_file(_ROOT_PACKAGE).parent
    parts = rest.split(".")
    for candidate in (
        root.joinpath(*parts[:-1], f"{parts[-1]}.py"),
        root.joinpath(*parts, "__init__.py"),
    ):
        if candidate.is_file():
            return candidate
    return None


def _package_imports(module: str, source: bytes, *, is_package: bool = False) -> set[str]:
    """The `_HASHED_PACKAGES` modules that `source`, the source of
    `module`, imports, found by parsing it without running it: every
    `import X` and `from X import ...` anywhere in the file, lazy imports
    included, whose module `X` is under the packages, and for
    `from X import name`, `X.name` when that is itself a module. Relative
    imports resolve against `module`'s package, which is `module` itself
    when `is_package`. Names are kept only when they have a source file
    under `agentic_trading`."""
    package = module if is_package else module.rpartition(".")[0]
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            candidates = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                base = importlib.util.resolve_name("." * node.level + base, package)
            candidates = [base]
            candidates += [f"{base}.{alias.name}" for alias in node.names if alias.name != "*"]
        else:
            continue
        for name in candidates:
            if _in_hashed_packages(name) and _package_source(name) is not None:
                found.add(name)
    return found


def _module_hashes(module: str) -> dict[str, str]:
    """Dotted module name to the sha256 hex digest of its source file's
    bytes, for `module` and every `agentic_trading.agents` and
    `agentic_trading.data` module reachable from it: the modules `module`
    imports (`_package_imports`), the modules those import, and so on,
    following imports only within those two packages. A module's parent
    packages, which Python runs on import, are covered only when a
    module imports them by name. Imports are found by parsing source, so
    no code runs. Raises `ValueError` when `module` has no source file."""
    files = {module: _module_file(module)}
    sources = {module: files[module].read_bytes()}
    pending = [module]
    while pending:
        name = pending.pop()
        is_package = files[name].name == "__init__.py"
        for imported in _package_imports(name, sources[name], is_package=is_package):
            path = _package_source(imported)
            if imported in files or path is None:
                continue
            files[imported] = path
            sources[imported] = path.read_bytes()
            pending.append(imported)
    return {name: hashlib.sha256(sources[name]).hexdigest() for name in sorted(sources)}
