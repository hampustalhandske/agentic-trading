from unittest.mock import MagicMock, patch

import pandas as pd

from agentic_trading.data.loaders.cboe_loader import load_put_call_ratio


def _ratio_frame() -> pd.DataFrame:
    dates = pd.to_datetime(["2024-06-01", "2024-06-02", "2024-06-03"])
    return pd.DataFrame(
        {
            "calls": [1_000_000, 1_100_000, 900_000],
            "puts": [800_000, 950_000, 1_200_000],
            "total": [1_800_000, 2_050_000, 2_100_000],
            "put_call_ratio": [0.80, 0.86, 1.33],
        },
        index=dates,
    )


@patch("agentic_trading.data.loaders.cboe_loader._fetch_put_call_ratio")
def test_only_rows_on_or_before_as_of_date_are_visible(
    mock_fetch: MagicMock, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("agentic_trading.data.cache.CACHE_ROOT", tmp_path)
    mock_fetch.return_value = _ratio_frame()

    result = load_put_call_ratio(pd.Timestamp("2024-06-02"))

    assert list(result.index) == [pd.Timestamp("2024-06-01"), pd.Timestamp("2024-06-02")]


@patch("agentic_trading.data.loaders.cboe_loader._fetch_put_call_ratio")
def test_repeated_calls_the_same_day_do_not_refetch(
    mock_fetch: MagicMock, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("agentic_trading.data.cache.CACHE_ROOT", tmp_path)
    mock_fetch.return_value = _ratio_frame()

    load_put_call_ratio(pd.Timestamp("2024-06-02"))
    load_put_call_ratio(pd.Timestamp("2024-06-03"))

    mock_fetch.assert_called_once()
