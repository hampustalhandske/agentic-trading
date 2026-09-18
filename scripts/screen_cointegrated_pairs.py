# screen_cointegrated_pairs.py

"""One-off formation-period cointegration screen that prints the seed
tuple for `roster._COINTEGRATED_PAIRS`.

Tests same-sector candidate pairs from the 20-ticker research basket
(one economically-motivated candidate pair per sector bucket, not a
data-mined all-C(20,2) list) for Engle-Granger cointegration over a
formation window that ends strictly before the basket's own test period
starts (2018-04-02), so the seed pairs never see the data they will
later be evaluated on. Prices come from the lakehouse
(`load_price_history_from_lakehouse`), the same source every backtest
run reads.

The printed tuple is only the starting wiring: `StatArbManager`
re-screens the candidate set quarterly at runtime and replaces partners
and hedge ratios with whatever passes its admit/keep/drop hysteresis.
Re-run this by hand and paste the output into `agents/roster.py` only
when the seed itself should change (a different basket, a longer
formation window).

Usage: `uv run python scripts/screen_cointegrated_pairs.py`
"""

from datetime import date

import numpy as np
from quantcore.statistics.cointegration import adf_test, engle_granger_test

from agentic_trading.data.loaders.lakehouse_loader import load_price_history_from_lakehouse
from agentic_trading.logging_config import configure_logging, get_logger

logger = get_logger(__name__)

# One candidate pair per sector bucket. V, XOM, and DIS are swapped for
# JPM, COP, and NFLX respectively — all three had real SEC EDGAR
# fundamentals-data gaps that made sec_fundamentals_analyst unusable for
# them across some or all of the intended test window.
SECTOR_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("AAPL", "AMZN", "Big Tech/Platforms"),
    ("AVGO", "NVDA", "Semiconductors"),
    ("MSFT", "ADBE", "Enterprise Software"),
    ("MA", "JPM", "Financials/Payments"),
    ("JNJ", "PFE", "Healthcare"),
    ("COP", "CVX", "Energy"),
    ("SBUX", "WMT", "Consumer Discretionary"),
    ("VZ", "NFLX", "Communication/Media"),
    ("BA", "CAT", "Industrials/Materials"),
    ("PLD", "AMT", "Staples/Utilities/REITs"),
)

# 12 months, ending the day before the research basket's own test period
# starts (2018-04-02) — long enough for a meaningful test, and strictly
# non-overlapping with the calibration/test periods.
FORMATION_START = date(2017, 4, 1)
FORMATION_END = date(2018, 3, 30)
SIGNIFICANCE_LEVEL = 0.05


def _is_cointegrated(ticker_a: str, ticker_b: str) -> tuple[bool, str]:
    frame = load_price_history_from_lakehouse([ticker_a, ticker_b], FORMATION_START, FORMATION_END)
    if frame is None:
        return False, "no lakehouse price coverage"
    covered = set(frame.columns.get_level_values(0))
    if ticker_a not in covered or ticker_b not in covered:
        return False, "no lakehouse price coverage"
    combined = (
        frame[ticker_a][["log_return"]]
        .join(frame[ticker_b][["log_return"]], lsuffix="_a", rsuffix="_b", how="inner")
        .dropna()
    )

    log_prices_a = np.cumsum(combined["log_return_a"].to_numpy(dtype=np.float64))
    log_prices_b = np.cumsum(combined["log_return_b"].to_numpy(dtype=np.float64))

    _, p_a = adf_test(log_prices_a)
    _, p_b = adf_test(log_prices_b)
    if p_a < SIGNIFICANCE_LEVEL or p_b < SIGNIFICANCE_LEVEL:
        return False, f"ADF p={p_a:.3f}/{p_b:.3f} (one leg already stationary)"

    # engle_granger_test returns (beta, adf_statistic, p_value) — the
    # p-value is the third element.
    _, _, p_value = engle_granger_test(log_prices_a, log_prices_b)
    return p_value < SIGNIFICANCE_LEVEL, f"Engle-Granger p={p_value:.4f}"


def main() -> None:
    configure_logging(verbose=False, disable=True)
    passing: list[tuple[str, str]] = []
    print(f"{'sector':28s} {'pair':14s} {'detail':30s} verdict")
    for ticker_a, ticker_b, sector in SECTOR_PAIRS:
        cointegrated, detail = _is_cointegrated(ticker_a, ticker_b)
        verdict = "COINTEGRATED" if cointegrated else "not cointegrated"
        print(f"{sector:28s} {ticker_a}/{ticker_b:10s} {detail:30s} {verdict}")
        if cointegrated:
            passing.append((ticker_a, ticker_b))

    print("\n_COINTEGRATED_PAIRS = (")
    for ticker_a, ticker_b in passing:
        print(f'    ("{ticker_a}", "{ticker_b}"),')
    print(")")


if __name__ == "__main__":
    main()
