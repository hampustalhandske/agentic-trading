# schemas.py

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class AgentSignal(BaseModel):
    """A single analyst's directional view on a single asset, for one day."""

    agent_id: str
    asset: str
    as_of_date: date
    direction: Literal["long", "short", "flat"]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


class ManagerSignal(BaseModel):
    """An analyst's signal after being scaled by its owning manager.

    Same direction as the original AgentSignal — a manager dampens
    confidence, it does not override what an analyst believes. Traceable
    back to both the manager that scaled it and the analyst that
    originated it, which is what the Commander/scoring system will need
    to correctly attribute performance later.
    """

    manager_id: str
    source_agent_id: str
    asset: str
    as_of_date: date
    direction: Literal["long", "short", "flat"]
    confidence: float = Field(ge=0.0, le=1.0)
    scale_applied: float = Field(ge=0.0)
    rationale: str


class PortfolioDecision(BaseModel):
    """The Commander's final allocation decision for one asset, one day.

    Traceable back to every ManagerSignal that fed into it, so the
    decision is fully auditable — not just "why this weight" but "which
    signals, from which managers and analysts, produced it."
    """

    asset: str
    as_of_date: date
    target_weight: float = Field(ge=-1.0, le=1.0)
    contributing_signals: list[ManagerSignal]

    @property
    def agent_weights(self) -> dict[str, float]:
        """Convenience view deriving {source_agent_id: weight applied} from
        contributing_signals, for callers that only need per-agent
        attribution rather than the full ManagerSignal objects."""
        return {
            signal.source_agent_id: signal.scale_applied for signal in self.contributing_signals
        }


class TradeRecord(BaseModel):
    """One asset's realized position change for one day, produced by
    `Portfolio.apply_decision()`. The single source of truth for "what did
    the portfolio actually do" — both the CLI and the GUI's trade-blotter
    display are built from this rather than each re-deriving fill mechanics
    independently."""

    as_of_date: date
    asset: str
    action: Literal["buy", "sell", "hold"]
    target_weight: float = Field(ge=-1.0, le=1.0)
    previous_weight: float
    target_shares: float
    delta_shares: float
    price: float = Field(gt=0.0)
    notional: float = Field(ge=0.0)


class PerformanceRecord(BaseModel):
    """One day's realized, attributable performance for one analyst or
    manager, on one asset. attributable_return is computed as
    direction_sign * confidence * realized_return — this class only
    defines the data shape, not the computation."""

    source_id: str
    source_type: Literal["analyst", "manager"]
    asset: str
    as_of_date: date
    attributable_return: float


class AgentWeight(BaseModel):
    """One agent's portfolio-influence weight for one day, as produced by
    a WeightingStrategy.

    Not bounded above by 1.0: WeightingStrategy.get_weights() (see
    scoring/weighting.py) normalizes weights to average 1.0 across
    sources rather than sum to 1.0, so an outperforming source's weight
    legitimately exceeds 1.0.
    """

    agent_id: str
    as_of_date: date
    weight: float = Field(ge=0.0, le=1.0)


class BacktestResult(BaseModel):
    """Summary of a completed backtest run.

    References
    ----------
    Sharpe, W. F. (1966). Mutual Fund Performance. Journal of Business,
    39(1), 119-138.
    Sharpe, W. F. (1994). The Sharpe Ratio. Journal of Portfolio
    Management, 21(1), 49-58.
    See REFERENCES.md.
    """

    start_date: date
    end_date: date
    portfolio_value_history: list[tuple[date, float]]
    final_sharpe_ratio: float
    final_max_drawdown: float
    decisions_by_day: dict[date, list[PortfolioDecision]] = Field(default_factory=dict)
    trade_log: list[TradeRecord] = Field(default_factory=list)
