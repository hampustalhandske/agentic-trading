# provenance.py

import json
import platform
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from agentic_trading.config import settings
from agentic_trading.data.loaders.lakehouse_loader import LakeProvenance
from agentic_trading.data.loaders.lakehouse_loader import lake_provenance as lake_provenance

# `lake_provenance` is re-exported (`as lake_provenance`): the stage (a) CLI
# reads the lake version through this module.

# The lake's frozen data contract the loaders are written against
# (tests/fixtures/lakehouse_contract_v1.json checks it).
LAKE_CONTRACT = "lakehouse-contract-v1"

_GIT_TIMEOUT_SECONDS = 5.0


class LibraryVersions(BaseModel):
    """The installed version of each numerical library a run's numbers
    depend on, as `importlib.metadata` reports it; `None` for a library
    that is not installed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    numpy: str | None
    scipy: str | None
    numba: str | None
    arch: str | None
    statsmodels: str | None
    pandas: str | None
    pyarrow: str | None
    duckdb: str | None
    quantcore: str | None


class RuntimeEnvironment(BaseModel):
    """The machine a run computed on: the Python version, the operating
    system (`platform_system`, `platform_machine`, `platform_release`, as
    `platform` reports them), the BLAS numpy was built against
    (`blas_name`, `blas_version`, as numpy reports them; `None` when it
    reports none), the
    numerical `libraries` installed and the commit quantcore was installed
    from (`quantcore_sha`, `None` when unknown). It describes the machine,
    not the hypothesis, so no spec hash covers it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    python_version: str
    platform_system: str
    platform_machine: str
    platform_release: str
    blas_name: str | None
    blas_version: str | None
    libraries: LibraryVersions
    quantcore_sha: str | None


class RunProvenance(BaseModel):
    """Everything needed to identify what produced a run: the code revision
    (`git_sha`, `git_dirty`, `quantcore_sha`), the full strategy `settings`
    in effect, the resolved `cli_args`, the analyst and manager ids of the
    roster, the `lake` build that was read, the `lake_contract` name the
    loaders target and the runtime `environment`. Git and quantcore fields
    are `None` when unknown."""

    git_sha: str | None
    git_dirty: bool | None
    quantcore_sha: str | None
    settings: dict[str, Any]
    cli_args: dict[str, Any]
    analyst_ids: list[str]
    manager_ids: list[str]
    lake: LakeProvenance
    lake_contract: str
    environment: RuntimeEnvironment


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


def source_tree_state() -> tuple[str | None, bool | None]:
    """The HEAD sha and whether the source tree this package lives in (its
    parent directory, `src/` in a checkout) has changes, untracked files
    included; both `None` outside a git checkout."""
    sha = _git_output("rev-parse", "HEAD")
    status = _git_output("status", "--porcelain", "--", str(Path(__file__).resolve().parent.parent))
    if sha is None or status is None:
        return None, None
    return sha.strip(), bool(status.strip())


def quantcore_sha() -> str | None:
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


def installed_version(name: str) -> str | None:
    """The installed version of the distribution `name`, or `None` when it
    is not installed."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _numpy_blas() -> tuple[str | None, str | None]:
    """The name and version of the BLAS numpy was built against, from
    `numpy.show_config(mode="dicts")`; each `None` when numpy does not
    report it."""
    try:
        import numpy

        blas = numpy.show_config(mode="dicts")["Build Dependencies"]["blas"]
    except (ImportError, AttributeError, KeyError, TypeError, ValueError):
        return None, None
    name, version = blas.get("name"), blas.get("version")
    return (str(name) if name else None, str(version) if version else None)


def runtime_environment() -> RuntimeEnvironment:
    """The `RuntimeEnvironment` of this process: Python and platform, numpy's
    BLAS, each `LibraryVersions` library's installed version and the
    installed quantcore's commit."""
    blas_name, blas_version = _numpy_blas()
    return RuntimeEnvironment(
        python_version=platform.python_version(),
        platform_system=platform.system(),
        platform_machine=platform.machine(),
        platform_release=platform.release(),
        blas_name=blas_name,
        blas_version=blas_version,
        libraries=LibraryVersions(
            **{name: installed_version(name) for name in LibraryVersions.model_fields}
        ),
        quantcore_sha=quantcore_sha(),
    )


def build_run_provenance(
    cli_args: dict[str, Any], analyst_ids: list[str], manager_ids: list[str]
) -> RunProvenance:
    """Snapshot the code revision, live settings, `cli_args`, roster ids,
    lake build and runtime environment for a run. The git working tree is dirty when untracked files
    exist as well as when tracked files changed."""
    git_sha, git_dirty = _git_state()
    return RunProvenance(
        git_sha=git_sha,
        git_dirty=git_dirty,
        quantcore_sha=quantcore_sha(),
        settings=settings.model_dump(mode="json"),
        cli_args=cli_args,
        analyst_ids=analyst_ids,
        manager_ids=manager_ids,
        lake=lake_provenance(),
        lake_contract=LAKE_CONTRACT,
        environment=runtime_environment(),
    )
