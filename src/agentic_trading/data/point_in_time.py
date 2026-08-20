# point_in_time.py

import pandas as pd


def get_point_in_time_data(data: pd.DataFrame, as_of_date: pd.Timestamp) -> pd.DataFrame:
    """Returns the subset of `data` visible as of `as_of_date` — no row
    with a date after `as_of_date` may appear in the result.

    This is the structural enforcement point for the project's hard
    no-look-ahead rule: on simulated day t, no agent may access data from
    day t+1 onward. Every caller that hands market data to an agent must
    route it through this function first; agents themselves must not be
    trusted to self-enforce this.
    """
    return data.loc[data.index <= as_of_date]
