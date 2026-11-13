from pathlib import Path

import pytest

from agentic_trading import config
from agentic_trading.data.loaders import lakehouse_loader

_URL_VARIABLE = "LAKEHOUSE_API_URL"
_TOKEN_VARIABLE = "LAKEHOUSE_API_TOKEN"


@pytest.fixture(autouse=True)
def _isolated_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
    real_lake_accessor: object,
) -> None:
    """No `.env` file and a fresh accessor cache, so only the variables a
    test sets are visible."""
    monkeypatch.setattr(config, "_ENV_FILE", tmp_path_factory.mktemp("env") / ".env")
    monkeypatch.delenv(_URL_VARIABLE, raising=False)
    monkeypatch.delenv(_TOKEN_VARIABLE, raising=False)
    monkeypatch.delenv(config.ENV_FILE_VARIABLE, raising=False)


def test_missing_token_raises_naming_the_variable_without_leaking_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_URL_VARIABLE, "http://lake-host-secret.invalid:8600")
    with pytest.raises(RuntimeError) as excinfo:
        lakehouse_loader._lake()
    assert _TOKEN_VARIABLE in str(excinfo.value)
    assert "lake-host-secret" not in str(excinfo.value)


def test_missing_url_raises_naming_the_variable_without_leaking_the_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_TOKEN_VARIABLE, "token-value-secret")
    with pytest.raises(RuntimeError) as excinfo:
        lakehouse_loader._lake()
    assert _URL_VARIABLE in str(excinfo.value)
    assert "token-value-secret" not in str(excinfo.value)


def test_both_variables_set_builds_one_cached_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_URL_VARIABLE, "http://localhost:8600")
    monkeypatch.setenv(_TOKEN_VARIABLE, "token-value-secret")
    first = lakehouse_loader._lake()
    assert lakehouse_loader._lake() is first
    first.close()


def test_no_variables_and_no_env_file_raise_before_any_request() -> None:
    """With no variable set and no env file, the client is never built: a named error, not an
    anonymous request."""
    with pytest.raises(RuntimeError, match=_URL_VARIABLE):
        lakehouse_loader._lake()


def test_the_named_env_file_supplies_both_variables(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`AGENTIC_ENV_FILE` points the loader at another checkout's env file."""
    env_file = tmp_path / "main-checkout.env"
    env_file.write_text(
        f"# lake\n{_URL_VARIABLE}=http://localhost:8600\n{_TOKEN_VARIABLE}='token-from-file'\n"
    )
    monkeypatch.setenv(config.ENV_FILE_VARIABLE, str(env_file))

    assert config._env_value(_URL_VARIABLE) == "http://localhost:8600"
    assert config._env_value(_TOKEN_VARIABLE) == "token-from-file"


def test_the_named_env_file_takes_the_place_of_the_repo_root_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When `AGENTIC_ENV_FILE` is set, the repo-root `.env` is not read."""
    root_env = tmp_path / ".env"
    root_env.write_text(f"{_TOKEN_VARIABLE}=root-token\n")
    monkeypatch.setattr(config, "_ENV_FILE", root_env)
    named = tmp_path / "named.env"
    named.write_text(f"{_URL_VARIABLE}=http://localhost:8600\n")
    monkeypatch.setenv(config.ENV_FILE_VARIABLE, str(named))

    assert config._env_value(_TOKEN_VARIABLE) is None


def test_a_named_env_file_that_does_not_exist_refuses(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A missing named file raises naming the variable; it never falls back to the repo-root
    `.env`, and no client is built."""
    root_env = tmp_path / ".env"
    root_env.write_text(f"{_URL_VARIABLE}=http://localhost:8600\n{_TOKEN_VARIABLE}=t\n")
    monkeypatch.setattr(config, "_ENV_FILE", root_env)
    monkeypatch.setenv(config.ENV_FILE_VARIABLE, str(tmp_path / "absent.env"))

    with pytest.raises(config.EnvFileMissingError, match=config.ENV_FILE_VARIABLE):
        lakehouse_loader._lake()
