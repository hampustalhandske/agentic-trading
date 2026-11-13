# test_skip_guard.py

"""The skip guard of `tests/edgelab_live.py` fails a pytest run with any skip other than an
`edgelab_live` test skipped for `EDGELAB_ABSENT`, and reports how many live checks skipped.
Each case runs a scratch suite in a separate pytest process with the guard configured as
`tests/conftest.py` configures it."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from edgelab_live import CONTRACT_TEXT_ABSENT, EDGELAB_ABSENT

_TESTS = Path(__file__).resolve().parents[1]
_CONFTEST = """\
from edgelab_live import configure


def pytest_configure(config):
    configure(config)
"""
_LIVE_SKIP = f"""\
import pytest


@pytest.mark.edgelab_live
def test_live():
    pytest.skip("{EDGELAB_ABSENT}: no docs/schema/x.json; set EDGELAB_DIR to check drift")


def test_passes():
    pass
"""
_PRIVATE_SKIP = f"""\
import pytest


@pytest.mark.private_contract
def test_private():
    pytest.skip("{CONTRACT_TEXT_ABSENT}: no contract.text")
"""


def _run_suite(tmp_path: Path, source: str) -> subprocess.CompletedProcess[str]:
    """Runs the test module `source` under the guard in a fresh pytest process."""
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    (tmp_path / "conftest.py").write_text(_CONFTEST)
    (tmp_path / "test_case.py").write_text(source)
    env = os.environ | {"PYTHONPATH": str(_TESTS)}
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(tmp_path)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_a_live_check_skipped_for_an_absent_checkout_passes_and_is_counted(
    tmp_path: Path,
) -> None:
    done = _run_suite(tmp_path, _LIVE_SKIP)
    assert done.returncode == 0, done.stdout + done.stderr
    assert f"edgelab_live: 1 live edgelab check(s) skipped ({EDGELAB_ABSENT})" in done.stdout


def test_a_private_contract_check_skipped_for_an_absent_text_passes_and_is_counted(
    tmp_path: Path,
) -> None:
    done = _run_suite(tmp_path, _PRIVATE_SKIP)
    assert done.returncode == 0, done.stdout + done.stderr
    assert f"private_contract: 1 private contract check(s) skipped ({CONTRACT_TEXT_ABSENT})" in (
        done.stdout
    )


@pytest.mark.parametrize(
    "source",
    [
        # A plain skip.
        "import pytest\n\n\ndef test_x():\n    pytest.skip('not today')\n",
        # A skip marker.
        "import pytest\n\n\n@pytest.mark.skip(reason='later')\ndef test_x():\n    pass\n",
        # A live check skipped for another reason.
        "import pytest\n\n\n@pytest.mark.edgelab_live\ndef test_x():\n    pytest.skip('flaky')\n",
        # A private contract check skipped for another reason.
        "import pytest\n\n\n@pytest.mark.private_contract\ndef test_x():\n    pytest.skip('x')\n",
        # The declared reason on a test that is not a live check.
        f"import pytest\n\n\ndef test_x():\n    pytest.skip('{EDGELAB_ABSENT}: no x')\n",
        # A whole module skipped at collection.
        "import pytest\n\npytest.skip('no module', allow_module_level=True)\n",
    ],
)
def test_any_other_skip_fails_the_run(tmp_path: Path, source: str) -> None:
    done = _run_suite(tmp_path, source)
    assert done.returncode == pytest.ExitCode.TESTS_FAILED, done.stdout + done.stderr
    assert "skip guard: 1 skip(s) not allowed" in done.stdout


def test_a_failing_run_keeps_its_failure(tmp_path: Path) -> None:
    done = _run_suite(tmp_path, _LIVE_SKIP + "\n\ndef test_fails():\n    assert False\n")
    assert done.returncode == pytest.ExitCode.TESTS_FAILED
