# test_watch_limits.py

"""`tools/ab/watch_limits.sh` stops the process it watches when free memory or disk runs low,
or when a probe gives no reading, and leaves a healthy process alone. Its probes
(`memory_pressure` and `df`, macOS commands) are replaced through `WATCH_MEMORY_CMD` and
`WATCH_DISK_CMD` by stubs printing fixed readings in the same format."""

import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

from unit.tools.support import TOOLS, write_stub

_WATCH = TOOLS / "ab" / "watch_limits.sh"


def _probe_env(stubs: Path, tool_env: dict[str, str], free_memory: str, free_disk: str) -> dict:
    """`tool_env` with stub probes reporting `free_memory` percent and `free_disk` GB, polled
    every 0.1 s."""
    memory = write_stub(
        stubs, "memory_pressure_stub", f'echo "System-wide memory free percentage: {free_memory}%"'
    )
    disk = write_stub(
        stubs,
        "df_stub",
        "echo 'Filesystem 1G-blocks Used Available Capacity Mounted on'\n"
        f"echo '/dev/disk1 500 100 {free_disk} 20% /'",
    )
    return tool_env | {
        "WATCH_MEMORY_CMD": str(memory),
        "WATCH_DISK_CMD": str(disk),
        "WATCH_INTERVAL": "0.1",
    }


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.mark.parametrize(("free_memory", "free_disk"), [("19", "200"), ("50", "9")])
def test_a_low_reading_stops_the_process_and_its_children(
    stubs: Path, tool_env: dict[str, str], tmp_path: Path, free_memory: str, free_disk: str
) -> None:
    """Below 20% free memory or 10 GB free disk, the watched process and its child get TERM and
    the watch exits 3."""
    child_file = tmp_path / "child.pid"
    watched = subprocess.Popen(["bash", "-c", f'sleep 60 & echo $! > "{child_file}"; wait'])
    deadline = time.monotonic() + 10
    while not child_file.exists() or not child_file.read_text().strip():
        assert time.monotonic() < deadline
        time.sleep(0.05)
    child = int(child_file.read_text())
    env = _probe_env(stubs, tool_env, free_memory, free_disk)
    watch = subprocess.run(
        [str(_WATCH), str(watched.pid)], env=env, capture_output=True, text=True, timeout=30
    )
    assert watch.returncode == 3, watch.stdout + watch.stderr
    assert "LIMIT" in watch.stdout
    assert watched.wait(timeout=10) == -signal.SIGTERM
    deadline = time.monotonic() + 10
    while _alive(child):
        assert time.monotonic() < deadline, "the watched process's child was not stopped"
        time.sleep(0.05)


def test_a_probe_without_a_reading_stops_the_process(stubs: Path, tool_env: dict[str, str]) -> None:
    """A memory probe whose output has no free percentage stops the process; the watch exits 2."""
    watched = subprocess.Popen(["sleep", "60"])
    env = _probe_env(stubs, tool_env, "50", "200")
    env["WATCH_MEMORY_CMD"] = str(write_stub(stubs, "broken_probe", "echo unexpected"))
    watch = subprocess.run(
        [str(_WATCH), str(watched.pid)], env=env, capture_output=True, text=True, timeout=30
    )
    assert watch.returncode == 2, watch.stdout + watch.stderr
    assert watched.wait(timeout=10) == -signal.SIGTERM


def test_a_healthy_process_is_left_alone(stubs: Path, tool_env: dict[str, str]) -> None:
    """With 50% free memory and 200 GB free disk the process runs to its end, and the watch
    then reports the lowest readings and exits 0."""
    watched = subprocess.Popen(["sleep", "1"])
    env = _probe_env(stubs, tool_env, "50", "200")
    watch = subprocess.Popen(
        [str(_WATCH), str(watched.pid)], env=env, stdout=subprocess.PIPE, text=True
    )
    assert watched.wait(timeout=10) == 0
    out, _ = watch.communicate(timeout=10)
    assert watch.returncode == 0, out
    assert "lowest free mem 50%, lowest free disk 200 GB" in out
