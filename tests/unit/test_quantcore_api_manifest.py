# test_quantcore_api_manifest.py

"""Every quantcore name this project uses is part of the quantcore API it pins.

`tests/fixtures/quantcore_api_manifest_0.3.1-599b8bd.json` is quantcore's API manifest of
release 0.3.1 at commit 599b8bd, the commit `uv.lock` pins. It is a byte-for-byte copy of
`docs/api_manifests/quantcore-0.3.1-599b8bd.json` at quantcore commit c8b7218 (file sha256
9b7aabc448cddbe02445aaad86a3f10124144002790b73945d75c435ffa9610e); its own `sha256` field is
040a2f86c986b78717de5d109271693957027aee4d2d920ed18d9d3b5ccfb341. quantcore's
`docs/API_MANIFEST.md` at that commit defines the manifest, its hash and the reference check
ported here.

With no quantcore checkout present, the tests check that the vendored manifest's `sha256`
matches its canonical content, that every quantcore import and attribute use under `src/` and
`tests/` resolves in it, and that the installed quantcore and the `uv.lock` pin are the version
and commit the vendored file names, so a re-pin without re-vendoring fails. Once the installed
quantcore ships its own `api_manifest.json`, its `sha256` must equal the vendored one; until
then that test is skipped."""

import ast
import hashlib
import json
import re
import tomllib
from importlib import metadata, resources
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
_MANIFEST = _REPO / "tests" / "fixtures" / "quantcore_api_manifest_0.3.1-599b8bd.json"
_MANIFEST_SHA256 = "040a2f86c986b78717de5d109271693957027aee4d2d920ed18d9d3b5ccfb341"
# The version and commit the vendored file is named after.
_PINNED = re.fullmatch(
    r"quantcore_api_manifest_(?P<version>[^-]+)-(?P<commit>[0-9a-f]{7,40})\.json", _MANIFEST.name
)
assert _PINNED is not None


def _load(path: Path) -> dict[str, Any]:
    """The manifest JSON at `path`."""
    manifest: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return manifest


def manifest_sha256(manifest: dict[str, Any]) -> str:
    """The SHA-256 hex digest of `manifest` without its `sha256` field, as canonical JSON:
    sorted keys, separators `,` and `:`, UTF-8, `ensure_ascii=False`."""
    body = {k: v for k, v in manifest.items() if k != "sha256"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _resolves(modules: dict[str, Any], module: str, name: str) -> bool:
    """Whether `name` is a public name, an export or a submodule of `module` in `modules`."""
    entry = modules.get(module, {})
    if name in entry.get("names", {}):
        return True
    origin = entry.get("exports", {}).get(name)
    if origin is not None:
        return name in modules[origin]["names"]
    return f"{module}.{name}" in modules


def _dotted(node: ast.expr) -> str | None:
    """The dotted path an attribute chain on a bare name spells, else `None`."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    return ".".join([node.id, *reversed(parts)])


def unresolved_quantcore_uses(manifest: dict[str, Any], files: list[Path]) -> list[str]:
    """Every quantcore import or attribute use in `files` that `manifest` does not list, as
    sorted `<path>:<line> <module>.<name>` strings: `from quantcore.x import name`, and
    `name.attr` after `import quantcore.x` or `import quantcore.x as name`."""
    modules = manifest["modules"]
    found = []
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        aliases: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("quantcore"):
                for alias in node.names:
                    if not _resolves(modules, str(node.module), alias.name):
                        found.append(f"{path}:{node.lineno} {node.module}.{alias.name}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("quantcore"):
                        if alias.name not in modules:
                            found.append(f"{path}:{node.lineno} {alias.name}")
                        # `import quantcore.x` binds the dotted path; `as y` binds `y`.
                        aliases[alias.asname or alias.name] = alias.name
        for node in ast.walk(tree):
            dotted = _dotted(node) if isinstance(node, ast.Attribute) else None
            if dotted is None:
                continue
            prefixes = [a for a in aliases if dotted.startswith(a + ".")]
            if not prefixes:
                continue
            prefix = max(prefixes, key=len)
            name = dotted[len(prefix) + 1 :].split(".")[0]
            if not _resolves(modules, aliases[prefix], name):
                found.append(f"{path}:{node.lineno} {aliases[prefix]}.{name}")
    return sorted(set(found))


def _python_files() -> list[Path]:
    """Every `.py` file under this repository's `src/` and `tests/`."""
    return sorted([*(_REPO / "src").rglob("*.py"), *(_REPO / "tests").rglob("*.py")])


def _locked_quantcore_commit() -> str:
    """The commit `uv.lock` pins quantcore to: the fragment of its git source."""
    lock = tomllib.loads((_REPO / "uv.lock").read_text(encoding="utf-8"))
    [package] = [p for p in lock["package"] if p["name"] == "quantcore"]
    return str(package["source"]["git"]).rsplit("#", 1)[1]


def test_vendored_manifest_sha256_matches_its_content() -> None:
    """The vendored manifest's `sha256` field is the pinned one and the hash of its content."""
    manifest = _load(_MANIFEST)
    assert manifest["sha256"] == _MANIFEST_SHA256
    assert manifest_sha256(manifest) == manifest["sha256"]


def test_one_byte_change_breaks_the_manifest_sha256(tmp_path: Path) -> None:
    """Negative control: changing one byte of the manifest's content breaks its `sha256`."""
    text = _MANIFEST.read_text(encoding="utf-8")
    marker = '"value": "str"'
    assert marker in text
    tampered = tmp_path / _MANIFEST.name
    tampered.write_text(text.replace(marker, '"value": "stR"', 1), encoding="utf-8")
    manifest = _load(tampered)

    assert manifest["sha256"] == _MANIFEST_SHA256
    assert manifest_sha256(manifest) != manifest["sha256"]


def test_every_quantcore_name_used_resolves_in_the_manifest() -> None:
    """Every quantcore import and attribute use under src/ and tests/ is in the manifest."""
    files = _python_files()
    assert any("quantcore" in path.read_text(encoding="utf-8") for path in files)
    assert unresolved_quantcore_uses(_load(_MANIFEST), files) == []


def test_unknown_quantcore_names_are_reported(tmp_path: Path) -> None:
    """Negative control: a planted file's unknown names are reported, and only those."""
    planted = tmp_path / "planted.py"
    planted.write_text(
        "import quantcore.risk\n"
        "import quantcore.risk as qr\n"
        "from quantcore.x import does_not_exist\n"
        "from quantcore.risk import ConvergenceError, no_such_function\n"
        "quantcore.risk.variance_backcast\n"
        "quantcore.risk.not_public\n"
        "qr.also_missing\n",
        encoding="utf-8",
    )

    assert unresolved_quantcore_uses(_load(_MANIFEST), [planted]) == [
        f"{planted}:3 quantcore.x.does_not_exist",
        f"{planted}:4 quantcore.risk.no_such_function",
        f"{planted}:6 quantcore.risk.not_public",
        f"{planted}:7 quantcore.risk.also_missing",
    ]


def test_vendored_manifest_is_the_pinned_release() -> None:
    """The vendored manifest's version is the installed quantcore's, and the commit `uv.lock`
    pins is the one its file is named after, so a re-pin without re-vendoring fails."""
    assert _PINNED is not None
    manifest = _load(_MANIFEST)
    assert manifest["package"] == "quantcore"
    assert manifest["version"] == _PINNED["version"]
    assert metadata.version("quantcore") == manifest["version"]
    assert _locked_quantcore_commit().startswith(_PINNED["commit"])


# The quantcore releases built before the manifest shipped inside the package.
_RELEASES_WITHOUT_A_SHIPPED_MANIFEST = ("0.3.1", "0.3.2")


def test_installed_quantcore_ships_the_vendored_manifest() -> None:
    """The installed quantcore's own `api_manifest.json` has the vendored `sha256`; a release
    from before the manifest shipped (`_RELEASES_WITHOUT_A_SHIPPED_MANIFEST`) has none, and any
    later release without one fails."""
    shipped = resources.files("quantcore") / "api_manifest.json"
    version = metadata.version("quantcore")
    if not shipped.is_file():
        assert version in _RELEASES_WITHOUT_A_SHIPPED_MANIFEST, (
            f"quantcore {version} ships no api_manifest.json"
        )
        return
    installed = json.loads(shipped.read_text(encoding="utf-8"))
    assert installed["sha256"] == _load(_MANIFEST)["sha256"]
