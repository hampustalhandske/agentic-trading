# agentic-trading

A multi-agent backtesting framework for portfolio risk management, built on
top of `quantcore` (option pricing, GARCH volatility, VaR/CVaR).

## Architecture

Three layers, following the standard structure used across published
multi-agent trading research (TradingAgents, FinCon, ai-hedge-fund):

- **Layer 1 — Analyst agents**: heterogeneous, run in parallel, no
  authority. Each proposes a directional view on an asset. Internally
  wildly different (quant/numeric vs. news/text-based); externally, all
  conform to the same `AgentSignal` schema.
- **Layer 2 — Risk Manager**: translates raw risk metrics (GARCH regime,
  VaR/CVaR from `quantcore`) into hard position-size constraints. Does not
  decide direction, only how much is allowed.
- **Layer 3 — Commander (Portfolio Manager)**: holds portfolio state,
  combines Layer 1 signals weighted by each agent's trailing track record,
  respects Layer 2's constraints, emits the final decision.

A separate, non-agent **scoring system** tracks each Layer 1 agent's
hypothetical attributable performance daily and converts trailing track
record into portfolio-allocation weight — this is what the Commander
actually uses to decide "who gets how much say."

The backtest execution engine (order fills, portfolio accounting) is
deterministic plain code, not an LLM agent — correctness there is
non-negotiable.

## Status

Early scaffolding — see CLAUDE.md for conventions and REFERENCES.md for
citations behind the scoring/weighting methodology.

## Setup

```bash
uv sync --extra dev
uv run pytest
```