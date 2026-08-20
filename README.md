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
- **Layer 2 — Managers**: translate raw risk metrics (GARCH regime, VaR/CVaR
  from `quantcore`) into hard position-size constraints. Do not decide
  direction, only how much is allowed. Currently two managers, each owning a
  distinct subset of the Layer 1 analysts (see `agents/roster.py`):
  `RiskManager` (VaR-based dampening) over the directional/sentiment
  analysts, `CVaRManager` (tail-risk dampening) over the valuation analysts.
  More managers and a richer ownership/communication structure will be
  layered on later.
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

Active development — see CLAUDE.md for current conventions and command
defaults.

## Setup

```bash
uv sync --locked --group dev
uv run ruff check .
uv run ruff format --check .
uv run mypy src/
uv run pytest
```

## Running it

Two console scripts are installed with the package:

```bash
uv run agentic-trading        # launches the Streamlit dashboard
uv run agentic-trading-cli <assets> --start-date <date> --end-date <date>  # headless backtest, prints a BacktestResult
```

Equivalently, without the installed scripts:

```bash
uv run src/agentic_trading/main.py                                              # dashboard
uv run src/agentic_trading/main.py cli <assets> --start-date <date> --end-date <date>  # headless backtest
```

The dashboard can also be launched directly with Streamlit:

```bash
uv run streamlit run src/agentic_trading/gui/dashboard.py
```

It uses a dark theme and defaults to a single-asset desk simulation so you
can inspect analysts, managers, trades, and the portfolio curve in one
place — with day-by-day playback (step forward/backward, a day-scrubber
slider, and autoplay) and a per-day signal breakdown showing every
contributing analyst/manager signal behind that day's decision, not just
the net trade.