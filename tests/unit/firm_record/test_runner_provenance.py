# test_runner_provenance.py

"""A firm record's trial folder carries the provenance of its numbers: the environment in its
agentic_meta.json names the installed numerical library versions, the commit quantcore was
installed from, the BLAS and OpenMP thread settings and the BLAS libraries actually loaded.
`_assert_records_provenance` checks a recorded environment for all four, and each negative
control shows it fails when one of them is missing."""

import copy
import json
import re
from collections.abc import Callable
from importlib import metadata
from typing import Any

import pytest

from agentic_trading.provenance import BlasThreads, LibraryVersions, LinkedBlas
from unit.firm_record.test_runner import _run

_META = "agentic_meta.json"


def _installed_quantcore_commit() -> str:
    raw = metadata.distribution("quantcore").read_text("direct_url.json")
    assert raw is not None
    return str(json.loads(raw)["vcs_info"]["commit_id"])


def _assert_records_provenance(environment: dict[str, Any]) -> None:
    """`environment` names every `LibraryVersions` library at its installed version, the
    installed quantcore's commit, every `BlasThreads` variable and at least one loaded BLAS
    library with its vendor."""
    libraries = environment.get("libraries")
    assert isinstance(libraries, dict), "no library versions"
    assert set(libraries) == set(LibraryVersions.model_fields), "library versions incomplete"
    for name in LibraryVersions.model_fields:
        assert libraries[name] == metadata.version(name), f"libraries.{name}"
        assert libraries[name], f"libraries.{name} is empty"

    commit = environment.get("quantcore_sha")
    assert isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit), "no quantcore commit"
    assert commit == _installed_quantcore_commit(), "quantcore commit is not the installed one"

    threads = environment.get("blas_threads")
    assert isinstance(threads, dict), "no BLAS thread settings"
    assert set(threads) == set(BlasThreads.model_fields), "BLAS thread settings incomplete"

    linked = environment.get("linked_blas")
    assert isinstance(linked, list) and linked, "no linked BLAS"
    for entry in linked:
        assert set(entry) == set(LinkedBlas.model_fields), "linked BLAS entry incomplete"
        assert entry["vendor"] and entry["library"], "linked BLAS entry has no vendor or library"


# The environment the firm record below records, kept after the first test that builds it.
_RECORDED: dict[str, dict[str, Any]] = {}


@pytest.fixture
def recorded_environment(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """The environment in the agentic_meta.json of a firm record built from fakes, recorded
    once per module (inside a test, so the suite's offline-lake fixtures apply)."""
    if "environment" not in _RECORDED:
        store_dir = tmp_path_factory.mktemp("store")
        result = _run(store_dir, max_days=2)
        meta: dict[str, Any] = json.loads((store_dir / result.spec_hash / _META).read_text())
        _RECORDED["environment"] = meta["environment"]
    return _RECORDED["environment"]


def test_trial_folder_records_the_provenance_of_its_numbers(
    recorded_environment: dict[str, Any],
) -> None:
    _assert_records_provenance(recorded_environment)


def _drop(key: str) -> Callable[[dict[str, Any]], None]:
    return lambda environment: environment.pop(key)


def _null(key: str) -> Callable[[dict[str, Any]], None]:
    return lambda environment: environment.update({key: None})


def _drop_library(environment: dict[str, Any]) -> None:
    environment["libraries"].pop("numpy")


def _drop_thread_setting(environment: dict[str, Any]) -> None:
    environment["blas_threads"].pop("openblas_num_threads")


def _empty_linked_blas(environment: dict[str, Any]) -> None:
    environment["linked_blas"] = []


def _other_quantcore_commit(environment: dict[str, Any]) -> None:
    environment["quantcore_sha"] = "0" * 40


@pytest.mark.parametrize(
    "remove",
    [
        _drop("libraries"),
        _drop_library,
        _drop("quantcore_sha"),
        _null("quantcore_sha"),
        _other_quantcore_commit,
        _drop("blas_threads"),
        _null("blas_threads"),
        _drop_thread_setting,
        _drop("linked_blas"),
        _null("linked_blas"),
        _empty_linked_blas,
    ],
    ids=[
        "no_libraries",
        "no_numpy_version",
        "no_quantcore_commit",
        "null_quantcore_commit",
        "other_quantcore_commit",
        "no_blas_threads",
        "null_blas_threads",
        "no_openblas_thread_setting",
        "no_linked_blas",
        "null_linked_blas",
        "empty_linked_blas",
    ],
)
def test_a_missing_provenance_part_fails_the_check(
    recorded_environment: dict[str, Any], remove: Callable[[dict[str, Any]], None]
) -> None:
    """Negative control: the recorded environment with one part removed fails the check."""
    environment = copy.deepcopy(recorded_environment)
    remove(environment)

    with pytest.raises(AssertionError):
        _assert_records_provenance(environment)
