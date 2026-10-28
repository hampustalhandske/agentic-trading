# provenance.py

import json
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from agentic_trading.config import settings
from agentic_trading.data.loaders.lakehouse_loader import LakeProvenance, lake_provenance

# The lake's frozen data contract the loaders are written against
# (tests/fixtures/lakehouse_contract_v1.json checks it).
LAKE_CONTRACT = "lakehouse-contract-v1"

_GIT_TIMEOUT_SECONDS = 5.0


class RunProvenance(BaseModel):
    """Everything needed to identify what produced a run: the code revision
    (`git_sha`, `git_dirty`, `quantcore_sha`), the full strategy `settings`
    in effect, the resolved `cli_args`, the analyst and manager ids of the
    roster, the `lake` build that was read and the `lake_contract` name the
    loaders target. Git and quantcore fields are `None` when unknown."""

    git_sha: str | None
    git_dirty: bool | None
    quantcore_sha: str | None
    settings: dict[str, Any]
    cli_args: dict[str, Any]
    analyst_ids: list[str]
    manager_ids: list[str]
    lake: LakeProvenance
    lake_contract: str


def _git_output(*args: str) -> str | None:
    """Stdout of `git <args>` run in this package's source directory, or
    `None` when git is missing, times out, or the directory is not a checkout."""
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=False,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def _git_state() -> tuple[str | None, bool | None]:
    """The HEAD sha and whether the working tree is dirty. Dirty means
    `git status --porcelain` prints anything, so untracked files count."""
    sha = _git_output("rev-parse", "HEAD")
    status = _git_output("status", "--porcelain")
    if sha is None or status is None:
        return None, None
    return sha.strip(), bool(status.strip())


def _quantcore_sha() -> str | None:
    """The VCS commit quantcore was installed from, read from its
    `direct_url.json`; `None` when quantcore was not installed from a VCS URL."""
    try:
        raw = metadata.distribution("quantcore").read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return None
    if not raw:
        return None
    commit_id = json.loads(raw).get("vcs_info", {}).get("commit_id")
    return str(commit_id) if commit_id else None


def build_run_provenance(
    cli_args: dict[str, Any], analyst_ids: list[str], manager_ids: list[str]
) -> RunProvenance:
    """Snapshot the code revision, live settings, `cli_args`, roster ids and
    lake build for a run. The git working tree is dirty when untracked files
    exist as well as when tracked files changed."""
    git_sha, git_dirty = _git_state()
    return RunProvenance(
        git_sha=git_sha,
        git_dirty=git_dirty,
        quantcore_sha=_quantcore_sha(),
        settings=settings.model_dump(mode="json"),
        cli_args=cli_args,
        analyst_ids=analyst_ids,
        manager_ids=manager_ids,
        lake=lake_provenance(),
        lake_contract=LAKE_CONTRACT,
    )
