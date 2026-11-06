"""The A/B reference `scripts/at126_legacy.py` stays auditable: every
`agentic_trading` symbol it reaches has a pinned source hash.

The reference imports live helpers from the loader module and its
neighbours, so it can change without an edit to its own file. The test walks
every symbol the reference names (its `from agentic_trading... import` names
and each `ll.<name>` of the loader module), then, through each function's
global names, every `agentic_trading` function, class and plain constant those
reach. It hashes each one's source: a function's or class's definition, and
a plain constant's module-level assignment (so a test that patches the value
at run time does not change its pin), and compares the result with
`at126_legacy_pins.json`. Modules,
loggers, the settings object and other runtime objects are not pinned;
`agentic_trading.logging_config` is not followed.

When a pinned symbol changes and the reference is meant to follow it, list
the change among the exceptions in the module docstring of
`scripts/at126_legacy.py` and in the A/B report, then rewrite the pins with
`uv run python tests/unit/data/loaders/test_at126_legacy_pins.py --write`."""

import ast
import hashlib
import importlib
import inspect
import json
import sys
import types
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[4]
_SCRIPTS = _ROOT / "scripts"
_LEGACY = _SCRIPTS / "at126_legacy.py"
_PINS = Path(__file__).with_name("at126_legacy_pins.json")
_LOADER_ALIAS = "ll"
_NOT_FOLLOWED = ("agentic_trading.logging_config",)
_PLAIN = (str, int, float, bool, bytes, type(None))


def _legacy_module() -> types.ModuleType:
    """`scripts/at126_legacy.py`, imported from the scripts directory."""
    sys.path.insert(0, str(_SCRIPTS))
    try:
        return importlib.import_module("at126_legacy")
    finally:
        sys.path.remove(str(_SCRIPTS))


def _is_plain(value: Any) -> bool:
    """Whether `value` is a constant with a stable `repr`: a str, number,
    bool, bytes or None, or a tuple, list, frozenset, set or dict of them."""
    if isinstance(value, _PLAIN):
        return True
    if isinstance(value, tuple | list | frozenset | set):
        return all(_is_plain(item) for item in value)
    if isinstance(value, dict):
        return all(_is_plain(key) and _is_plain(item) for key, item in value.items())
    return False


def _in_scope(module: str | None) -> bool:
    """Whether a symbol defined in `module` is pinned and followed."""
    return (
        module is not None
        and (module == "agentic_trading" or module.startswith("agentic_trading."))
        and not any(module == skip or module.startswith(f"{skip}.") for skip in _NOT_FOLLOWED)
    )


def _assignment_source(module_name: str, name: str) -> str:
    """The source of the module-level assignment of `name` in the module
    `module_name`; raises `LookupError` when the module assigns it nowhere
    at its top level."""
    module = importlib.import_module(module_name)
    source = inspect.getsource(module)
    for node in ast.parse(source).body:
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            segment = ast.get_source_segment(source, node)
            assert segment is not None
            return segment
    raise LookupError(f"{module_name} assigns no top-level {name}")


def _digest(name: str, value: Any) -> str:
    """The sha256 hex digest of the source of the symbol `name`: the
    definition of a function or class, or the module-level assignment of a
    plain constant."""
    if inspect.isfunction(value) or inspect.isclass(value):
        text = inspect.getsource(value)
    else:
        module_name, _, attribute = name.rpartition(".")
        text = _assignment_source(module_name, attribute)
    return hashlib.sha256(text.encode()).hexdigest()


def _code_names(code: types.CodeType) -> set[str]:
    """Every name `code` and the code objects nested in it look up."""
    names = set(code.co_names)
    for constant in code.co_consts:
        if isinstance(constant, types.CodeType):
            names |= _code_names(constant)
    return names


def _roots(legacy: types.ModuleType) -> dict[str, Any]:
    """Each `agentic_trading` symbol the reference's source names, by its
    dotted name: its from-imports that are not modules, and every
    `ll.<name>` of the loader module."""
    tree = ast.parse(_LEGACY.read_text())
    loader = getattr(legacy, _LOADER_ALIAS)
    roots: dict[str, Any] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and _in_scope(node.module):
            for alias in node.names:
                value = getattr(legacy, alias.asname or alias.name)
                if not inspect.ismodule(value):
                    roots[f"{node.module}.{alias.name}"] = value
        elif (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == _LOADER_ALIAS
        ):
            roots[f"{loader.__name__}.{node.attr}"] = getattr(loader, node.attr)
    return roots


def legacy_pins() -> dict[str, str]:
    """Dotted name to source digest of every `agentic_trading` function,
    class and plain constant the reference reaches (see the module
    docstring), sorted by name."""
    pins: dict[str, str] = {}
    stack = list(_roots(_legacy_module()).items())
    while stack:
        name, value = stack.pop()
        if name in pins:
            continue
        if inspect.isfunction(value) or inspect.isclass(value):
            if not _in_scope(value.__module__):
                continue
        elif not _is_plain(value):
            continue
        pins[name] = _digest(name, value)
        if not inspect.isfunction(value):
            continue
        function = inspect.unwrap(value)
        for global_name in sorted(_code_names(function.__code__)):
            if global_name not in function.__globals__:
                continue
            dependency = function.__globals__[global_name]
            if inspect.isfunction(dependency) or inspect.isclass(dependency):
                dependency_name = f"{dependency.__module__}.{dependency.__qualname__}"
            else:
                dependency_name = f"{function.__module__}.{global_name}"
            stack.append((dependency_name, dependency))
    return dict(sorted(pins.items()))


def test_every_symbol_the_legacy_reference_reaches_is_pinned() -> None:
    """A change to any `agentic_trading` source the reference reaches fails
    here until the pins are rewritten and the change is listed among the
    exceptions of `scripts/at126_legacy.py`."""
    pinned = json.loads(_PINS.read_text())
    current = legacy_pins()
    shared = current.keys() & pinned.keys()
    changed = sorted(name for name in shared if current[name] != pinned[name])
    added = sorted(current.keys() - pinned.keys())
    removed = sorted(pinned.keys() - current.keys())
    assert not (changed or added or removed), (
        "the A/B reference reaches changed live code; list it among the exceptions of "
        "scripts/at126_legacy.py and rewrite the pins: "
        f"changed {changed}, added {added}, removed {removed}"
    )


def test_the_pins_cover_the_shared_sec_availability_rule() -> None:
    """The walk reaches the shared helpers the reference calls and what they
    call in turn: `_sec_available_dates` and the `next_session` it uses."""
    pins = legacy_pins()
    assert "agentic_trading.data.loaders.lakehouse_loader._sec_available_dates" in pins
    assert "agentic_trading.data.trading_calendar.next_session" in pins
    assert "agentic_trading.data.loaders.lakehouse_loader._DECISION_TIMEZONE" in pins


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        raise SystemExit("usage: test_at126_legacy_pins.py --write")
    _PINS.write_text(json.dumps(legacy_pins(), indent=2) + "\n")
    print(f"wrote {len(json.loads(_PINS.read_text()))} pins to {_PINS}")
