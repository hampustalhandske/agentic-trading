from unittest.mock import patch

import pandas as pd
import pytest

from agentic_trading.data.loaders.factor_loader import load_fama_french_factors


def _factor_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Mkt-RF": [0.001, 0.002, -0.001],
            "SMB": [0.0, 0.001, 0.0],
            "HML": [0.0, 0.0, 0.001],
            "RF": [0.0, 0.0, 0.0],
        },
        index=pd.to_datetime(["2024-05-01", "2024-05-02", "2024-06-03"]),
    )


@patch("agentic_trading.data.loaders.factor_loader._download_factor_csv")
def test_truncates_to_rows_on_or_before_as_of_date(mock_download, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("agentic_trading.data.cache.CACHE_ROOT", tmp_path)
    mock_download.return_value = _factor_frame()

    result = load_fama_french_factors(pd.Timestamp("2024-05-02"))

    assert (result.index <= pd.Timestamp("2024-05-02")).all()
    assert len(result) == 2


def test_unsupported_model_raises_value_error() -> None:
    with pytest.raises(ValueError):
        load_fama_french_factors(pd.Timestamp("2024-05-02"), model="not_a_model")
