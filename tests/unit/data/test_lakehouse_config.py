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
