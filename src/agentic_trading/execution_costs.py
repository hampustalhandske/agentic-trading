# execution_costs.py

"""The linear part of the fill cost model as one number, shared by every
consumer that needs a per-dollar cost before an order exists."""

from agentic_trading.config import settings


def expected_fill_cost_fraction(
    commission_rate: float | None = None, slippage_bps: float | None = None
) -> float:
    """The commission plus slippage a linear fill pays per reference dollar.

    Args:
        commission_rate: commission as a fraction of traded notional; None
            reads `settings.firm.commission_rate` at call time.
        slippage_bps: slippage in basis points of the reference price; None
            reads `settings.firm.slippage_bps` at call time.

    Returns:
        `s + c x (1 + s)`, with `c` the commission rate and `s` the slippage
        as a fraction. This equals the fraction `Portfolio.apply_decision`
        charges a buy, because commission is charged on the slipped price.
        A sell's commission is charged on `1 - s`, so it pays `2 x c x s`
        less (5e-7 at the defaults); the value is therefore an upper bound
        on the linear cost of either side.

    Square-root market impact and option bid/ask half-spreads are left
    out. Both depend on an order's size or a quote, and the no-trade band
    that consumes this number works on weight fractions with neither, so
    the band clears only the linear cost (see KNOWN_LIMITATIONS §3.3).
    """
    commission = settings.firm.commission_rate if commission_rate is None else commission_rate
    slippage = (settings.firm.slippage_bps if slippage_bps is None else slippage_bps) / 10_000.0
    return slippage + commission * (1.0 + slippage)
