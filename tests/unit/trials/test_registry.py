# test_registry.py

"""AT-40 round 1: `agentic_trading.trials.registry` hashes a config exactly as pitch-edge's
`research.registry.config_hash` does, and builds a `Registration` from a registry file under the
repository root, hashing the file's raw bytes."""

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest

from agentic_trading.trials.ledger import LedgerError, Registration
from agentic_trading.trials.registry import config_hash_of, file_registration

_REGISTERED_AT = datetime(2026, 10, 28, 1, 2, 3, tzinfo=UTC)


def _pitch_edge_config_hash(config: dict[str, Any]) -> str:
    """pitch-edge's `config_hash`, spelled out."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _registry(root: Path, name: str, data: bytes | None = None) -> Path:
    """Writes `research/registry/<name>.yaml` under `root` (default text `id: <name>`)."""
    path = root / "research" / "registry" / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"id: {name}\n".encode() if data is None else data)
    return path


def test_config_hash_of() -> None:
    """config_hash_of is the sha256 of the compact sorted JSON, non-JSON values through str."""
    assert config_hash_of({"theta": 0.02}) == _pitch_edge_config_hash({"theta": 0.02})
    assert config_hash_of({"theta": 0.02}) == hashlib.sha256(b'{"theta":0.02}').hexdigest()
    assert config_hash_of({"b": 1, "a": 2}) == config_hash_of({"a": 2, "b": 1})
    assert config_hash_of({"b": 1, "a": 2}) == hashlib.sha256(b'{"a":2,"b":1}').hexdigest()
    nested = {"grid": {"z": [1, 2], "a": None}, "day": date(2026, 1, 2)}
    expected = b'{"day":"2026-01-02","grid":{"a":null,"z":[1,2]}}'
    assert config_hash_of(nested) == hashlib.sha256(expected).hexdigest()
    assert config_hash_of(nested) == _pitch_edge_config_hash(nested)
    assert config_hash_of({}) == hashlib.sha256(b"{}").hexdigest()


def test_file_registration(registry_file: Path, tmp_path: Path) -> None:
    """file_registration keeps the given fields and hashes the file's raw bytes."""
    registration = file_registration(
        registry_file, tmp_path, family="demo", registered_at=_REGISTERED_AT, metric_kind="return"
    )
    assert isinstance(registration, Registration)
    assert registration.id == "demo-v1"
    assert registration.family == "demo"
    assert registration.registered_at == _REGISTERED_AT
    assert registration.metric_kind == "return"
    assert registration.source_sha256 == hashlib.sha256(registry_file.read_bytes()).hexdigest()
    assert (tmp_path / registration.source_path).resolve() == registry_file.resolve()

    raw = b"id: crlf-v1\r\n# s\xc3\xa4tt\r\n"
    crlf = file_registration(
        _registry(tmp_path, "crlf-v1", raw),
        tmp_path,
        family="demo",
        registered_at=_REGISTERED_AT,
        metric_kind=None,
    )
    assert crlf.id == "crlf-v1"
    assert crlf.metric_kind is None
    assert crlf.source_sha256 == hashlib.sha256(raw).hexdigest()

    for name in ("a-v1", "x1-y2-v10", "0-v0"):
        accepted = file_registration(
            _registry(tmp_path, name),
            tmp_path,
            family="demo",
            registered_at=_REGISTERED_AT,
            metric_kind=None,
        )
        assert accepted.id == name


def test_file_registration_refuses(tmp_path: Path) -> None:
    """file_registration raises LedgerError for a malformed id or a file outside the root."""
    for name in ("Demo-v1", "demo_v1", "demo", "-demo-v1", "demo-v", "demo-v1a", "demo v1"):
        path = _registry(tmp_path, name)
        with pytest.raises(LedgerError):
            file_registration(
                path, tmp_path, family="demo", registered_at=_REGISTERED_AT, metric_kind=None
            )

    root = tmp_path / "repo"
    root.mkdir()
    outside = _registry(tmp_path / "elsewhere", "demo-v1")
    for path in (outside, root / ".." / "elsewhere" / "research" / "registry" / "demo-v1.yaml"):
        with pytest.raises(LedgerError):
            file_registration(
                path, root, family="demo", registered_at=_REGISTERED_AT, metric_kind=None
            )
