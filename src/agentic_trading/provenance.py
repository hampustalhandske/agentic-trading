# provenance.py

import ctypes
import json
import os
import platform
import plistlib
import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from agentic_trading.config import settings
from agentic_trading.data.loaders.lakehouse_loader import LakeProvenance
from agentic_trading.data.loaders.lakehouse_loader import lake_provenance as lake_provenance

# `lake_provenance` is re-exported (`as lake_provenance`): the stage (a) CLI
# reads the lake version through this module.

# The lake's frozen data contract the loaders are written against
# (tests/fixtures/lakehouse_contract_v2.json checks it).
LAKE_CONTRACT = "lakehouse-contract-v2"

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


class BlasThreads(BaseModel):
    """The environment variables that set how many threads the BLAS and
    OpenMP libraries use, as this process saw them at start (`None` when
    unset). Multi-threaded OpenBLAS reduces in a thread-dependent order, so
    its results can differ in the last bits from a single-threaded run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    openblas_num_threads: str | None
    omp_num_threads: str | None
    mkl_num_threads: str | None
    veclib_maximum_threads: str | None


BlasVendor = Literal["accelerate", "openblas", "mkl", "blis"]


class LinkedBlas(BaseModel):
    """One BLAS library loaded into this process: its `vendor`, the
    `library` file it was loaded from (the file name only, so the record
    does not depend on where the environment is installed) and its
    `version` as the library reports it (`None` when it reports none)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    vendor: BlasVendor
    library: str
    version: str | None


class RuntimeEnvironment(BaseModel):
    """The machine a run computed on: the Python version, the operating
    system (`platform_system`, `platform_machine`, `platform_release`, as
    `platform` reports them), the BLAS numpy was built against
    (`blas_name`, `blas_version`, as numpy reports them; `None` when it
    reports none), the BLAS and OpenMP thread settings (`blas_threads`;
    `None` in an environment recorded without them), the BLAS libraries
    actually loaded once numpy and scipy.linalg are imported (`linked_blas`;
    `None` in an environment recorded without them or where the loaded
    libraries cannot be listed), the numerical `libraries` installed and the
    commit quantcore was installed from (`quantcore_sha`, `None` when
    unknown). It describes the machine, not the hypothesis, so no spec hash
    covers it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    python_version: str
    platform_system: str
    platform_machine: str
    platform_release: str
    blas_name: str | None
    blas_version: str | None
    libraries: LibraryVersions
    quantcore_sha: str | None
    blas_threads: BlasThreads | None = None
    linked_blas: tuple[LinkedBlas, ...] | None = None


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


def blas_threads() -> BlasThreads:
    """This process's `BlasThreads`, read from its environment."""
    return BlasThreads(
        openblas_num_threads=os.environ.get("OPENBLAS_NUM_THREADS"),
        omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
        mkl_num_threads=os.environ.get("MKL_NUM_THREADS"),
        veclib_maximum_threads=os.environ.get("VECLIB_MAXIMUM_THREADS"),
    )


def blas_vendor_of(path: str) -> BlasVendor | None:
    """The vendor of the BLAS library at `path`, judged from its file name
    (and, for Accelerate, its framework): `vecLib`'s `libBLAS.dylib` inside
    Accelerate.framework, any library whose name contains `openblas`
    (numpy's and scipy's bundled `scipy_openblas` builds included),
    `mkl_rt`, or `libblis`; `None` for any other library."""
    name = Path(path).name.lower()
    if "accelerate.framework" in path.lower() and name == "libblas.dylib":
        return "accelerate"
    if "openblas" in name:
        return "openblas"
    if "mkl_rt" in name:
        return "mkl"
    if name.startswith("libblis"):
        return "blis"
    return None


def openblas_version(config: str) -> str | None:
    """The version in an `openblas_get_config()` string ("OpenBLAS 0.3.29
    DYNAMIC_ARCH ..."), or `None` when it names none."""
    match = re.search(r"OpenBLAS\s+(\S+)", config)
    return match.group(1) if match else None


def _loaded_library_paths() -> list[str] | None:
    """The file of every shared library loaded in this process: the dyld
    image list on macOS, the mapped files of /proc/self/maps on Linux;
    `None` on any other platform or when the list cannot be read."""
    try:
        if sys.platform == "darwin":
            dyld = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
            dyld._dyld_image_count.restype = ctypes.c_uint32
            dyld._dyld_get_image_name.restype = ctypes.c_char_p
            dyld._dyld_get_image_name.argtypes = [ctypes.c_uint32]
            names = (dyld._dyld_get_image_name(i) for i in range(dyld._dyld_image_count()))
            return [name.decode() for name in names if name]
        if sys.platform.startswith("linux"):
            lines = Path("/proc/self/maps").read_text().splitlines()
            return [fields[5] for fields in map(str.split, lines) if len(fields) >= 6]
    except (OSError, AttributeError, UnicodeDecodeError):
        return None
    return None


def _c_string(library: ctypes.CDLL, symbols: tuple[str, ...]) -> str | None:
    """The string the first of `symbols` that `library` exports returns when
    called with no arguments, or `None` when it exports none of them."""
    for symbol in symbols:
        function = getattr(library, symbol, None)
        if function is not None:
            function.restype = ctypes.c_char_p
            value = function()
            return value.decode() if value else None
    return None


def _blas_version(vendor: BlasVendor, path: str) -> str | None:
    """The version the BLAS library at `path` reports: OpenBLAS's
    `openblas_get_config`, MKL's `MKL_Get_Version_String`, BLIS's
    `bli_info_get_version_str`, and for Accelerate the vecLib framework's
    bundle version. `None` when the library reports none."""
    try:
        if vendor == "accelerate":
            info = Path(path).parent / "Resources" / "Info.plist"
            version = plistlib.loads(info.read_bytes()).get("CFBundleShortVersionString")
            return str(version) if version else None
        library = ctypes.CDLL(path)
        if vendor == "openblas":
            config = _c_string(
                library,
                (
                    "openblas_get_config64_",
                    "openblas_get_config",
                    "scipy_openblas_get_config64_",
                    "scipy_openblas_get_config",
                ),
            )
            return None if config is None else openblas_version(config)
        if vendor == "mkl":
            buffer = ctypes.create_string_buffer(256)
            library.MKL_Get_Version_String(buffer, len(buffer))
            match = re.search(r"Version\s+(\S+)", buffer.value.decode())
            return match.group(1) if match else None
        return _c_string(library, ("bli_info_get_version_str",))
    except (OSError, AttributeError, UnicodeDecodeError, ValueError, plistlib.InvalidFileException):
        return None


def linked_blas() -> tuple[LinkedBlas, ...] | None:
    """The BLAS libraries loaded in this process once numpy and scipy.linalg
    are imported, one entry per library file, sorted by vendor and file
    name; `None` when the loaded libraries cannot be listed
    (`_loaded_library_paths`)."""
    try:
        import numpy  # noqa: F401
        import scipy.linalg  # noqa: F401
    except ImportError:
        pass
    paths = _loaded_library_paths()
    if paths is None:
        return None
    found: dict[tuple[str, str], LinkedBlas] = {}
    for path in paths:
        vendor = blas_vendor_of(path)
        if vendor is None or (vendor, Path(path).name) in found:
            continue
        found[(vendor, Path(path).name)] = LinkedBlas(
            vendor=vendor, library=Path(path).name, version=_blas_version(vendor, path)
        )
    return tuple(found[key] for key in sorted(found))


def runtime_environment() -> RuntimeEnvironment:
    """The `RuntimeEnvironment` of this process: Python and platform, numpy's
    BLAS and its thread settings, the BLAS libraries loaded, each
    `LibraryVersions` library's installed version and the installed
    quantcore's commit."""
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
        blas_threads=blas_threads(),
        linked_blas=linked_blas(),
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
