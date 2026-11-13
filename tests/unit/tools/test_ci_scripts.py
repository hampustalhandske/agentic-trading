# test_ci_scripts.py

"""The CI scripts under `tools/ci/` fail closed, checked on a scratch git repository with `uv`,
`docker`, `apt-get` and `pip` stubbed on PATH (no container, no real test run):

- `with_test_lock.sh` lets one holder run at a time and refuses a pytest command with no target;
- `local_ci.sh` refuses a sha that is not a commit, and runs its test step under the test lock;
- `container_ci.sh` refuses a sha that is not a commit before starting a container, and passes
  the container's exit status through;
- `container_check.sh` (the in-container part) exits non-zero without running a step when the
  checkout of the requested sha fails, and with the failing step's status otherwise."""

import subprocess
from pathlib import Path

from unit.tools.support import TOOLS, write_stub

_LOCK = TOOLS / "ci" / "with_test_lock.sh"
_LOCAL_CI = TOOLS / "ci" / "local_ci.sh"
_CONTAINER_CI = TOOLS / "ci" / "container_ci.sh"
_CONTAINER_CHECK = TOOLS / "ci" / "container_check.sh"
_MISSING_SHA = "0123456789abcdef0123456789abcdef01234567"

# A `uv` stub that appends its arguments to $UV_LOG, followed by " locked" when the directory
# $LOCK_DIR exists while it runs, else " unlocked"; it exits $UV_PYTEST_EXIT (default 0) when
# its arguments name pytest, else 0.
_UV_STUB = """\
state=unlocked; [ -d "${LOCK_DIR:-/nonexistent}" ] && state=locked
echo "$* $state" >> "$UV_LOG"
case " $* " in *" pytest "*) exit "${UV_PYTEST_EXIT:-0}" ;; esac
exit 0"""


def _run(
    args: list[str], cwd: Path, env: dict[str, str], timeout: float = 60
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(arg) for arg in args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def test_two_lock_holders_run_one_after_the_other(
    repo: tuple[Path, str], tool_env: dict[str, str], tmp_path: Path
) -> None:
    """Two commands started together under the lock do not overlap: the second starts only
    after the first ends, and both exit 0."""
    path, _ = repo
    log = tmp_path / "order.log"
    env = tool_env | {"TEST_LOCK_POLL": "0.1"}
    command = f'echo start >> "{log}"; sleep 1; echo end >> "{log}"'
    holders = [
        subprocess.Popen(
            [str(_LOCK), "bash", "-c", command],
            cwd=path,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for _ in range(2)
    ]
    assert [holder.wait(timeout=30) for holder in holders] == [0, 0]
    assert log.read_text().split() == ["start", "end", "start", "end"]
    assert not (path / ".git" / "agentic-test.lock").exists()


def test_lock_refuses_a_pytest_command_without_a_target(
    repo: tuple[Path, str], tool_env: dict[str, str]
) -> None:
    """A pytest command with options only exits 64 and never runs."""
    path, _ = repo
    done = _run([_LOCK, "uv", "run", "pytest", "-q"], path, tool_env)
    assert done.returncode == 64
    assert "needs at least one target" in done.stderr


def test_local_ci_runs_its_test_step_under_the_lock(
    repo: tuple[Path, str], tool_env: dict[str, str], stubs: Path, tmp_path: Path
) -> None:
    """Every step runs on the requested commit, the pytest step while the test lock is held,
    and the run passes."""
    path, sha = repo
    write_stub(stubs, "uv", _UV_STUB)
    uv_log = tmp_path / "uv.log"
    env = tool_env | {"UV_LOG": str(uv_log), "LOCK_DIR": str(path / ".git" / "agentic-test.lock")}
    done = _run([_LOCAL_CI, sha], path, env, timeout=120)
    assert done.returncode == 0, done.stdout + done.stderr
    assert f"overall PASS on {sha}" in done.stdout
    pytest_lines = [line for line in uv_log.read_text().splitlines() if " pytest " in line]
    assert len(pytest_lines) == 1
    assert pytest_lines[0].endswith(" locked")
    assert not (path / ".claude" / "worktrees" / f"ci-{sha}").exists()


def test_local_ci_fails_when_the_test_step_fails(
    repo: tuple[Path, str], tool_env: dict[str, str], stubs: Path, tmp_path: Path
) -> None:
    """A failing pytest step makes the run fail."""
    path, sha = repo
    write_stub(stubs, "uv", _UV_STUB)
    env = tool_env | {"UV_LOG": str(tmp_path / "uv.log"), "UV_PYTEST_EXIT": "1"}
    done = _run([_LOCAL_CI, sha], path, env, timeout=120)
    assert done.returncode == 1
    assert f"overall FAIL on {sha}" in done.stdout


def test_local_ci_refuses_a_sha_that_is_not_a_commit(
    repo: tuple[Path, str], tool_env: dict[str, str], stubs: Path, tmp_path: Path
) -> None:
    """An unknown sha exits non-zero before any step runs."""
    path, _ = repo
    write_stub(stubs, "uv", _UV_STUB)
    uv_log = tmp_path / "uv.log"
    done = _run([_LOCAL_CI, _MISSING_SHA], path, tool_env | {"UV_LOG": str(uv_log)})
    assert done.returncode != 0
    assert not uv_log.exists()


# A `docker` stub that appends its arguments to $DOCKER_LOG and exits $DOCKER_EXIT (default 0).
_DOCKER_STUB = 'echo "$*" >> "$DOCKER_LOG"\nexit "${DOCKER_EXIT:-0}"'


def test_container_ci_refuses_a_sha_that_is_not_a_commit(
    repo: tuple[Path, str], tool_env: dict[str, str], stubs: Path, tmp_path: Path
) -> None:
    """An unknown sha exits non-zero and never starts a container."""
    path, _ = repo
    write_stub(stubs, "docker", _DOCKER_STUB)
    docker_log = tmp_path / "docker.log"
    done = _run(
        [_CONTAINER_CI, path, _MISSING_SHA], tmp_path, tool_env | {"DOCKER_LOG": str(docker_log)}
    )
    assert done.returncode != 0
    assert not docker_log.exists()


def test_container_ci_passes_the_full_sha_and_the_container_status(
    repo: tuple[Path, str], tool_env: dict[str, str], stubs: Path, tmp_path: Path
) -> None:
    """A short sha is resolved to the full commit sha the container checks out, and the
    container's exit status is the script's."""
    path, sha = repo
    write_stub(stubs, "docker", _DOCKER_STUB)
    docker_log = tmp_path / "docker.log"
    env = tool_env | {"DOCKER_LOG": str(docker_log)}
    passed = _run([_CONTAINER_CI, path, sha[:7]], tmp_path, env)
    assert passed.returncode == 0, passed.stderr
    assert f"SHA={sha}" in docker_log.read_text()
    failed = _run([_CONTAINER_CI, path, sha], tmp_path, env | {"DOCKER_EXIT": "1"})
    assert failed.returncode == 1


def _container_check_env(
    tool_env: dict[str, str], stubs: Path, tmp_path: Path, source: Path, sha: str
) -> dict[str, str]:
    """The environment of `container_check.sh` on the host: `apt-get`, `pip` and `uv` stubbed,
    cloning `source` into a scratch directory and checking out `sha`."""
    write_stub(stubs, "apt-get", "exit 0")
    write_stub(stubs, "pip", "exit 0")
    write_stub(stubs, "uv", _UV_STUB)
    return tool_env | {
        "SOURCE": str(source),
        "WORK": str(tmp_path / "work"),
        "SHA": sha,
        "UV_LOG": str(tmp_path / "uv.log"),
    }


def test_container_check_stops_when_the_checkout_fails(
    repo: tuple[Path, str], tool_env: dict[str, str], stubs: Path, tmp_path: Path
) -> None:
    """A sha the clone does not have exits non-zero without running a step."""
    path, _ = repo
    env = _container_check_env(tool_env, stubs, tmp_path, path, _MISSING_SHA)
    done = _run(["bash", _CONTAINER_CHECK], tmp_path, env)
    assert done.returncode != 0
    assert not (tmp_path / "uv.log").exists()


def test_container_check_runs_every_step_on_the_requested_commit(
    repo: tuple[Path, str], tool_env: dict[str, str], stubs: Path, tmp_path: Path
) -> None:
    """On a sha the clone has, the five steps run on that commit and the run passes; a failing
    pytest step makes it fail."""
    path, sha = repo
    env = _container_check_env(tool_env, stubs, tmp_path, path, sha)
    done = _run(["bash", _CONTAINER_CHECK], tmp_path, env)
    assert done.returncode == 0, done.stdout + done.stderr
    assert f"checked out {sha[:7]}" in done.stdout
    assert len((tmp_path / "uv.log").read_text().splitlines()) == 5
    env["WORK"] = str(tmp_path / "work2")
    failed = _run(["bash", _CONTAINER_CHECK], tmp_path, env | {"UV_PYTEST_EXIT": "1"})
    assert failed.returncode == 1
