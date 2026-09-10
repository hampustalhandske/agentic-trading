# agentic-trading

A multi-agent **backtesting** framework for portfolio risk management.
It is not a live trading system and does not place real orders — it
simulates, day by day, how a committee of heterogeneous quant "analysts"
would have argued about an asset, how risk-aware "managers" would have
capped their conviction, and how a final "commander" would have turned
that into one position size, then scores every agent's hypothetical
track record along the way.

All actual numerical finance — option pricing, GARCH-family volatility
models, VaR/CVaR, factor regressions, portfolio optimization — comes from
`quantcore`, a separate dependency. This repo does not reimplement
quantitative math; it **orchestrates** it through a layered multi-agent
architecture and turns the output into position decisions and a backtest
result.

## Why this exists

Real trading desks don't make decisions from a single model — they run
many analysts with different, sometimes contradictory views, a risk desk
that caps how much any one view is allowed to bet, and a portfolio
manager who allocates capital based on who has actually been right
lately, not who shouts loudest. This project simulates that structure
mechanically and reproducibly, so different quantitative strategies can
be compared, combined, and backtested under one deterministic execution
engine — with every intermediate signal, constraint, and decision
inspectable, not just the final trade.

## Tech stack

| Tool | Role |
|---|---|
| Python 3.12+ | Language |
| [uv](https://docs.astral.sh/uv/) | Dependency management, venv, build, running scripts |
| [Typer](https://typer.tiangolo.com/) | CLI (`agentic-trading-cli`, `agentic-trading-experiment`) |
| [Streamlit](https://streamlit.io/) | GUI dashboard (`agentic-trading`) |
| [Pydantic](https://docs.pydantic.dev/) v2 | All cross-component schemas (`AgentSignal`, `PortfolioDecision`, ...) |
| [pandas](https://pandas.pydata.org/) | Time-series data handling throughout |
| `quantcore` (private git dep) | All numerical finance — pricing, GARCH, VaR/CVaR, factor math |
| `yfinance`, `pandas-datareader` | Price history |
| `fredapi` | FRED macro/rates/VIX series |
| `pytrends` | Google Trends attention data |
| `requests` | Direct HTTP calls to Stooq/SEC/FINRA/Finnhub/Tiingo/etc. |
| `python-dotenv` | Loads `.env` for API keys |
| [pytest](https://docs.pytest.org/) | Tests |
| [ruff](https://docs.astral.sh/ruff/) | Lint + format |
| [mypy](https://mypy-lang.org/) (`strict`) | Type checking |

`quantcore` is the one dependency this project doesn't own the math of —
if a calculation is missing, that's a `quantcore` task, not something to
patch in here.

## Architecture — four layers, one deterministic engine

```
Layer 1: Analysts (29)  →  Layer 2: Managers (8 pods)  →  Layer 3: Commander  →  (optional) Layer 4: Portfolio construction
                                                                  ↓
                                                    Backtest engine (fills, accounting)
                                                                  ↓
                                                       Scoring / track record
                                                     (feeds back into Layer 3's weights)
```

- **Layer 1 — Analyst agents** (`agents/analysts/`): heterogeneous,
  run independently in parallel, have no authority over each other. Each
  proposes a directional view (`AgentSignal`: direction, confidence,
  rationale) on an asset for the current simulated day. Internally each
  analyst can be built on a totally different quantitative technique —
  option pricing, time-series econometrics, cointegration, factor
  regression, sentiment — but externally every analyst implements the
  same interface (`agents/base.py`) and emits the same schema, so the
  layers above never need to know which technique produced a signal.
- **Layer 2 — Managers** (`agents/managers/`): each manager owns a themed
  subset ("pod") of Layer 1 analysts and turns raw risk metrics (GARCH
  regime, VaR/CVaR, spread volatility, event severity, factor
  significance...) into a hard *dampening* of that analyst's confidence.
  Managers never flip direction and never invent conviction — they can
  only scale it down. See "The eight pods" below.
- **Layer 3 — Commander** (`agents/commander.py`): the final allocation
  authority. Holds portfolio state, combines every manager-adjusted
  signal weighted by that analyst's *trailing track record* (not its
  type, not a hardcoded prior — see Scoring below), and emits one
  `PortfolioDecision` per asset per day. Can fan out manager evaluation
  across a thread or process pool (`parallel_workers`/`parallel_backend`)
  for multi-asset runs.
- **Layer 4 — Portfolio construction** (`agents/portfolio_construction.py`,
  optional, off by default / no-op unless enabled): a post-Commander
  stage that takes the Commander's per-asset decisions as Black-Litterman
  *views*, blends them with a market-implied equilibrium prior, shrinks
  the covariance estimate (Ledoit-Wolf), runs a risk-budgeting optimizer
  (min-variance / mean-variance / risk-parity, selectable), scales the
  result by a Kelly fraction, and logs a component-VaR risk report.

Two non-agent, always-deterministic components sit underneath all of
this:

- **`backtest/engine.py`** — order fills and portfolio accounting. No
  LLM calls, no randomness beyond what's explicitly seeded, fully
  testable in isolation.
- **`data/point_in_time.py`** — a hard architectural boundary: on
  simulated day *t*, no agent can access data from day *t+1* onward.
  Enforced structurally by the data-loading layer, not just documented,
  because a look-ahead bug here would silently invalidate any backtest
  result without producing an error.

## The eight pods

`agents/roster.py` wires up the default 29-analyst / 8-manager roster;
it's the file to change when adding more analysts or managers. Each pod
is themed around one coherent family of quant strategy:

| Pod | Manager | Analysts | What ties them together |
|---|---|---|---|
| **Risk** | `RiskManager` | `risk_minimizing`, `risk_maximizing`, `gbm_momentum` | Directional views off volatility/return characteristics. Dampens confidence for elevated quantcore VaR. |
| **CVaR** | `CVaRManager` | `black_scholes_valuation`, `monte_carlo_valuation`, `greeks_delta_hedge`, `heston_vol_surface` | Options-valuation analysts pricing *theoretical/synthetic* options off model-estimated vol (no free source has historical chains). Dampens confidence for elevated tail-loss (CVaR). |
| **StatArb** | `StatArbManager` | `ou_mean_reversion`, `pairs_cointegration`, `kalman_hedge_ratio`, `regime_hmm` | Classic statistical arbitrage: OU mean-reversion, cointegration, a Kalman-filtered hedge ratio, a Gaussian HMM regime detector. Dampens confidence for elevated spread volatility. |
| **Macro/Rates** | `MacroRatesManager` | `egarch_asymmetric_vol`, `gjr_garch_crash_risk`, `arima_forecast`, `macro_var_spillover`, `cir_rate_regime` | Volatility-regime and macro-driven signals, including a VAR system against FRED series (VIX, yield curve) and a CIR short-rate regime read. |
| **Factor** | `FactorManager` | `fama_french_factor`, `fama_macbeth`, `newey_west_significance` | Cross-sectional/regression analysts: Fama-French tilt, a two-pass Fama-MacBeth estimate, a Newey-West-robust predictor significance test. |
| **Event/Credit** | `EventCreditManager` | `news_sentiment`, `sec_fundamentals`, `short_interest_squeeze` | Text/filings-driven: Finnhub headlines scored by a local keyword lexicon, an OLS trend over SEC EDGAR filings (FMP fallback), a FINRA short-interest squeeze z-score. |
| **Alt-Data/Crowd Attention** | `AltDataManager` | `insider_cluster_buying`, `retail_attention`, `wikipedia_pageviews`, `reddit_sentiment` | Alt-data/behavioral signals: SEC Form 4 insider clustering, Google Trends + Wikipedia attention spikes, Reddit sentiment. Dampens confidence for staleness/thin samples. |
| **Anomaly/Network** | `AnomalyNetworkManager` | `calendar_seasonality`, `lead_lag_network_centrality`, `options_skew_sentiment` | Documented calendar anomalies, a lead-lag cross-correlation network centrality read, and the CBOE put/call ratio. |

A `congressional_trading` analyst previously covered Senate/House Stock
Watcher disclosures; both chambers' public S3 buckets now permanently
return 403 (a provider-side bucket-policy change, not a code/key issue),
so it was removed rather than left silently dead.

## Scoring — how the Commander decides who to trust

`scoring/` is a separate, non-agent system: it tracks each Layer 1
analyst's daily hypothetical attributable performance
(`TrackRecord`/`PerformanceRecord`) and converts trailing track record
into an allocation weight (`WeightingStrategy`, e.g. `Hedge`) via
`PerformanceScoreboard`. This — not agent type, not a hardcoded trust
table — is what `Commander.decide()` and each `ManagerAgent.evaluate()`
use to decide how much say a given analyst gets today.

## Data sources

All external data funnels through `data/cache.py` — a local on-disk
parquet cache keyed by `(source, identifier, as_of_date)`, so re-running
a backtest never re-hits a rate-limited free API. Every cache-miss fetch
is also recorded, write-once, in a local SQLite database (`data/db.py`).
Point-in-time gating differs by source — see `data/point_in_time.py` and
`CLAUDE.md` for the exact rules.

| Source | Used for | Key required |
|---|---|---|
| `yfinance` → Stooq → Alpha Vantage → synthetic | Daily price history (fallback chain, in order) | Alpha Vantage only: `ALPHAVANTAGE_API_KEY` |
| `yfinance` option chains | Live-mode options snapshots only — not usable by backtest-driving analysts | No |
| FRED | Macro/rates/VIX series | `FRED_API_KEY` |
| Ken French Data Library | Fama-French factor returns | No |
| SEC EDGAR | Fundamentals (filed-date gated); falls back to Financial Modeling Prep | No (`SEC_EDGAR_USER_AGENT` header) |
| SEC EDGAR full-text search | Form 4 insider transactions, 10-K/10-Q metadata | No (same header) |
| FINRA | Biweekly short interest | No |
| Finnhub | News headlines, scored locally (no paid sentiment endpoint) | `FINNHUB_API_KEY` |
| Tiingo | EOD price cross-check / divergence signal, not a price fallback | `TIINGO_API_KEY` |
| Financial Modeling Prep | Fundamentals fallback | `FMP_API_KEY` |
| Google Trends (`pytrends`) | Search-interest attention | No |
| Wikimedia Pageviews | Wikipedia attention | No |
| Reddit | Mention-volume in finance subreddits | `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` |
| CBOE | Market-wide put/call ratio | No |

**Historical options data is a documented limitation, not solved.**
Databento and Polygon.io were evaluated; neither has a genuinely free
ongoing tier for historical options, so the CVaR pod still prices
theoretical/synthetic options off model-estimated vol
(`agents/analysts/cvar_pod/_theoretical_options.py`).

## Project structure

```
src/agentic_trading/
├── agents/
│   ├── analysts/          # Layer 1 — one subpackage per pod (risk_pod, cvar_pod, ...)
│   ├── managers/          # Layer 2 — one manager class per pod
│   ├── base.py            # shared AnalystAgent / ManagerAgent interfaces
│   ├── commander.py        # Layer 3
│   ├── portfolio_construction.py  # optional Layer 4
│   ├── roster.py           # default 29-analyst / 8-manager wiring
│   └── schemas.py          # Pydantic schemas shared across layers
├── backtest/
│   ├── engine.py           # deterministic fills + accounting
│   ├── portfolio.py        # holdings/positions
│   └── baselines.py        # equal-weight/buy-and-hold/momentum/vol-target benchmarks
├── data/
│   ├── loaders/             # one module per external source
│   ├── cache.py             # on-disk parquet cache
│   ├── db.py                 # SQLite fetch log
│   ├── loader.py             # price-loading fallback chain
│   ├── point_in_time.py      # no-look-ahead enforcement
│   └── ticker_catalog.py
├── scoring/                 # track record + weighting strategies
├── gui/dashboard.py         # Streamlit dashboard
├── cli.py                   # `agentic-trading-cli` / `agentic-trading-experiment`
├── main.py                  # `agentic-trading` (GUI) entry point
└── logging_config.py        # the project's only logging setup
tests/unit/                  # mirrors the src/ layout
```

## Setup

```bash
uv sync --locked --group dev     # install deps + dev tools
cp .env.example .env             # then fill in the keys you want to use
```

Every key in `.env.example` is optional — without it, that specific data
source falls back or is skipped (see the Data sources table above);
prices work with no keys at all via `yfinance`/Stooq.

## Running it

```bash
uv run agentic-trading                    # launch the Streamlit dashboard

# headless single-phase backtest
uv run agentic-trading-cli <assets> --start-date <date> --end-date <date>

# calibration phase + out-of-sample test phase, with trust weights carried over
uv run agentic-trading-experiment <assets> \
  --train-start <date> --train-end <date> \
  --test-start <date> --test-end <date>

uv run pytest                             # tests
uv run ruff check . && uv run ruff format --check .
uv run mypy src/
```

`agentic-trading-experiment` calls its first phase *calibration*, not
training — there's no persistent ML model, only measured analyst/manager
track records that carry into the test period. Useful flags:
`--include-baselines` (equal-weight/buy-and-hold/momentum/vol-target
benchmarks from `backtest/baselines.py`), `--include-isolated` (run every
analyst/manager as its own strategy, for diagnostics), `--exclude-analyst`
(repeatable, omit one analyst by id), and `--calibration-checkpoint`/
`--test-checkpoint` with `--resume` to checkpoint and resume a long run.

The dashboard (`gui/dashboard.py`) is a visualization layer, not part of
the backtest engine: a tabbed (Overview / Desks / Analysts) multi-asset
"trading desk" view with day-by-day playback. The Overview tab leads with
an **agent-flow view** — Analysts → Pod Managers → Commander → (optional)
Portfolio Construction rendered live for the current day/asset — and a
**data-source status strip** (live vs. cached vs. untouched). The Desks
and Analysts tabs each add a **trust panel**: trailing Sharpe/contribution
next to the scoring-derived weight the backtest actually applied.

## Conventions worth knowing before touching the code

- **Every cross-component data structure is a Pydantic `BaseModel`** —
  no raw dicts or dataclasses cross a module boundary.
- **Two date types, two layers, do not mix them.** Market-data code
  works in `pd.Timestamp`; `scoring/` works in plain `datetime.date`.
  `Commander.decide()` and `ManagerAgent.evaluate()` are the one place
  that converts between them (`.date()`).
- `logging_config.py` is the project's only logging setup —
  `get_logger(__name__)` at import time everywhere, `configure_logging()`
  called once by an entry point.
- `quantcore` is a dependency, never something to fork or patch from
  this repo.

See `CLAUDE.md` for the full command reference, coding conventions, and
current architectural rules — it takes precedence over this file for
anything that conflicts.

## Status

Active development. `agents/roster.py` is the place to extend the
analyst/manager roster; `agents/portfolio_construction.py` is the place
to extend Layer 4 once single-asset behavior is stable.
