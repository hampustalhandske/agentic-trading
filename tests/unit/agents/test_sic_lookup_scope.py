# test_sic_lookup_scope.py

"""The SEC SIC lookup is current-state (bounded only by the lake time, not by
any decision date), so only the pairs screen and the preflight's coverage
report may read it; no analyst signal may rest on it."""

from pathlib import Path

import agentic_trading

_SRC = Path(agentic_trading.__file__).resolve().parent

# Module (path under `agentic_trading/`) -> the current-state names it may use.
_ALLOWED = {
    "data/loaders/lakehouse_loader.py": {
        "load_sic_code_from_lakehouse",
        "same_sic_candidate_pairs",
    },
    "backtest/preflight.py": {"load_sic_code_from_lakehouse"},
    "agents/analysts/stat_arb_pod/_pairs_screening.py": {
        "load_sic_code_from_lakehouse",
        "same_sic_candidate_pairs",
        "_pairs_screening",
    },
    "agents/managers/stat_arb_manager.py": {"_pairs_screening"},
    "agents/roster.py": {"_pairs_screening"},
}
_NAMES = ("load_sic_code_from_lakehouse", "same_sic_candidate_pairs", "_pairs_screening")


def test_only_the_pairs_screen_and_preflight_read_the_sic_lookup() -> None:
    """Every module that names the SIC lookup or the pairs screen is on the allow-list, and
    names only what it is allowed."""
    found: dict[str, set[str]] = {}
    for path in sorted(_SRC.rglob("*.py")):
        text = path.read_text()
        used = {name for name in _NAMES if name in text}
        if used:
            found[path.relative_to(_SRC).as_posix()] = used

    for module, used in found.items():
        assert module in _ALLOWED, f"{module} uses {sorted(used)}"
        assert used <= _ALLOWED[module], f"{module} uses {sorted(used - _ALLOWED[module])}"
