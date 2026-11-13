# test_commander_hash_seed.py

"""The cross-pod allocation is bit-identical whatever the interpreter's string-hash seed, so two
processes sizing the same pod returns give the same weights."""

import subprocess
import sys

_SIZING_STEP = """
import numpy as np
import pandas as pd

from agentic_trading.agents.commander import _cross_pod_risk_parity_weights

pods = [
    "risk_manager", "cvar_manager", "stat_arb_manager", "macro_rates_manager", "rates_manager",
    "cross_asset_manager", "credit_manager", "factor_manager", "alt_data_manager",
    "tail_hedge_manager",
]
rng = np.random.default_rng(0)
index = pd.bdate_range("2016-01-04", periods=12)
history = {pod: pd.Series(rng.normal(0.0, 0.01, len(index)), index=index) for pod in pods}
weights = _cross_pod_risk_parity_weights(history, set(pods))
print(sorted((pod, weight.hex()) for pod, weight in weights.items()))
"""


def _weights_under_hash_seed(seed: int) -> str:
    completed = subprocess.run(
        [sys.executable, "-c", _SIZING_STEP],
        capture_output=True,
        text=True,
        check=True,
        env={"PYTHONHASHSEED": str(seed), "PATH": ""},
    )
    return completed.stdout


def test_cross_pod_risk_parity_weights_do_not_depend_on_the_hash_seed() -> None:
    """One sizing step over ten pods, run under several hash seeds, prints identical weight bits."""
    outputs = {_weights_under_hash_seed(seed) for seed in range(1, 7)}

    assert len(outputs) == 1, outputs
