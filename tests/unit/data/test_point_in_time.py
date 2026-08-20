"""Unit tests for agentic_trading.data.point_in_time.

No mocking needed here — `get_point_in_time_data` operates purely on
in-memory pandas DataFrames.
"""

import pandas as pd

from agentic_trading.data.point_in_time import get_point_in_time_data


def _sample_data() -> pd.DataFrame:
    index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-05", "2024-01-08"])
    return pd.DataFrame({"close": [10.0, 20.0, 30.0, 40.0]}, index=index)


def test_data_after_as_of_date_is_excluded() -> None:
    """No row dated after as_of_date may appear in the result — the no-look-ahead guarantee."""
    result = get_point_in_time_data(_sample_data(), pd.Timestamp("2024-01-05"))

    assert (result.index <= pd.Timestamp("2024-01-05")).all()
    assert pd.Timestamp("2024-01-08") not in result.index


def test_data_exactly_on_as_of_date_is_included() -> None:
    """The as_of_date's own row is visible that day, not withheld until the next simulated day."""
    result = get_point_in_time_data(_sample_data(), pd.Timestamp("2024-01-05"))

    assert pd.Timestamp("2024-01-05") in result.index
    assert result.loc[pd.Timestamp("2024-01-05"), "close"] == 30.0


def test_as_of_date_with_no_exact_row_returns_all_prior_rows() -> None:
    """A non-trading as_of_date with no row of its own still acts as an inclusive `<=` range filter.

    The docstring's only stated guarantee is that no row after as_of_date appears in the
    result — it does not require an exact index match to exist. The natural reading is
    therefore a `<=` range filter: 2024-01-04 has no row of its own, so the result should
    contain every row strictly before it (2024-01-02, 2024-01-03) and nothing from
    2024-01-05 onward.
    """
    result = get_point_in_time_data(_sample_data(), pd.Timestamp("2024-01-04"))

    assert list(result.index) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]


def test_as_of_date_before_all_data_returns_empty_result() -> None:
    """An as_of_date earlier than any available data yields an empty frame, not an error."""
    result = get_point_in_time_data(_sample_data(), pd.Timestamp("2020-01-01"))

    assert result.empty
    assert isinstance(result, pd.DataFrame)
