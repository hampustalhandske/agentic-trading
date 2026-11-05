# test_runner_excluded.py

"""`run_batch` refuses the analysts in `EXCLUDED_ANALYSTS`, naming the analyst and the reason,
before it reads or writes anything."""

from datetime import UTC, date, datetime
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from agentic_trading.agents.analysts import KalmanHedgeRatioAnalyst, PairsCointegrationAnalyst
from agentic_trading.agents.base import AnalystAgent
from agentic_trading.stage_a.runner import EXCLUDED_ANALYSTS, run_batch

_REASON = (
    "trades seed pairs selected on 2017-04..2018-03 data, inside the grading window; "
    "returns with point-in-time pair selection"
)
_UNIVERSE = ["SHY", "IEF"]


def _market_data() -> pd.DataFrame:
    index = pd.bdate_range("2009-12-01", "2010-02-26", name="date")
    close = np.linspace(100.0, 110.0, len(index))
    return pd.DataFrame(
        {(asset, field): close for asset in _UNIVERSE for field in ("open", "close")}, index=index
    )


def test_excluded_analysts_hold_the_seed_pair_analysts() -> None:
    """The two seed-pair analysts are excluded, each with the same reason."""
    assert EXCLUDED_ANALYSTS == {
        "pairs_cointegration_analyst": _REASON,
        "kalman_hedge_ratio_analyst": _REASON,
    }


@pytest.mark.parametrize("excluded", [PairsCointegrationAnalyst, KalmanHedgeRatioAnalyst])
def test_run_batch_refuses_an_excluded_analyst(
    tmp_path: Path, excluded: type[AnalystAgent]
) -> None:
    """An excluded analyst among others raises ValueError naming it and its reason; nothing is
    computed, checked or written."""
    analyst = excluded()
    resources = MagicMock(return_value=True)
    store = tmp_path / "store"
    with pytest.raises(ValueError) as raised:
        run_batch(
            [analyst],
            _UNIVERSE,
            "etf",
            date(2010, 1, 14),
            date(2010, 1, 21),
            store,
            _market_data(),
            lake_time=datetime(2026, 10, 29, tzinfo=UTC),
            lake_version="f89aef2f428a",
            quantcore_version="0.4.0",
            quantcore_sha=None,
            resources_ok=resources,
        )

    assert analyst.agent_id in str(raised.value)
    assert _REASON in str(raised.value)
    resources.assert_not_called()
    assert not store.exists()
