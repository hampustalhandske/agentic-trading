# test_firm_v1_universe.py

"""The registered firm's universe (agentic-firm-v1) is named once, in `roster.py`: the 30
equities `scripts/select_firm_v1_equities.py` selected, then the 17 fixed instruments."""

from agentic_trading.agents.roster import FIRM_V1_EQUITIES, FIRM_V1_UNIVERSE

_EQUITIES = (
    "BAC AAPL GS JPM WFC XOM C GOOGL GE MSFT INTC CSCO WMT FCX PFE IBM CVX MS AMZN QCOM JNJ T PG "
    "HPQ ORCL AIG COP MON V FSLR"
).split()
_FIXED = [
    "SPY",
    "SHY",
    "IEF",
    "TLT",
    "HYG",
    "LQD",
    "GC=F",
    "CL=F",
    "SI=F",
    "NG=F",
    "HG=F",
    "EURUSD=X",
    "GBPUSD=X",
    "USDJPY=X",
    "AUDUSD=X",
    "BTC-USD",
    "ETH-USD",
]


def test_firm_v1_universe_is_the_registered_one() -> None:
    """30 equities in selection order, then the fixed instruments; no instrument twice."""
    assert list(FIRM_V1_EQUITIES) == _EQUITIES
    assert list(FIRM_V1_UNIVERSE) == _EQUITIES + _FIXED
    assert len(set(FIRM_V1_UNIVERSE)) == len(FIRM_V1_UNIVERSE) == 47
