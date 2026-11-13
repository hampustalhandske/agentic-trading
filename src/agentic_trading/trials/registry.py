# registry.py

"""Registry files and config hashes, in pitch-edge's format.

A registry file declares one hypothesis; its name is `<id>.yaml`, where `id` is
`<name>-v<version>` in lower-case letters, digits and hyphens. A registration is hashed on the
file's raw bytes, so any edit (comments and whitespace included) changes it."""

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from agentic_trading.trials.models import LedgerError, Registration

_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9-]*-v[0-9]+")


def config_hash_of(config: dict[str, Any]) -> str:
    """The hex sha256 of `config`'s JSON with sorted keys and no whitespace, non-JSON values
    written through `str`, so the same config hashes the same whatever its key order."""
    text = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def file_registration(
    path: str | Path,
    repo_root: str | Path,
    *,
    family: str,
    registered_at: datetime,
    metric_kind: str | None,
) -> Registration:
    """The `Registration` of the registry file at `path`, which must lie under `repo_root`.

    The id is the file's stem, `source_path` the resolved file's path relative to the resolved
    `repo_root`, and `source_sha256` the sha256 of the file's raw bytes; `family`,
    `registered_at` (timezone-aware) and `metric_kind` are kept as given. Raises `LedgerError`
    when the stem is not a valid id or the file is outside `repo_root`, and `OSError` when the
    file cannot be read."""
    file = Path(path)
    hypothesis_id = file.stem
    if _ID_PATTERN.fullmatch(hypothesis_id) is None:
        raise LedgerError(
            f"{file.name}: a registry file is named <id>.yaml with an id matching "
            f"{_ID_PATTERN.pattern}, got {hypothesis_id!r}"
        )
    resolved = file.resolve()
    root = Path(repo_root).resolve()
    if not resolved.is_relative_to(root):
        raise LedgerError(f"{resolved} is not under the repository root {root}")
    return Registration(
        id=hypothesis_id,
        family=family,
        registered_at=registered_at,
        metric_kind=metric_kind,
        source_path=resolved.relative_to(root),
        source_sha256=hashlib.sha256(resolved.read_bytes()).hexdigest(),
    )
