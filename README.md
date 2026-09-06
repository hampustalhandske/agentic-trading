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
result. If a calculation this project needs is missing from `quantcore`,
that's a `quantcore` task, not something to patch in here.

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

## Architecture — four layers, one deterministic engine

```
Layer 1: Analysts (30)  →  Layer 2: Managers (8 pods)  →  Layer 3: Commander  →  (optional) Layer 4: Portfolio construction
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
  `PortfolioDecision` per asset per day.
- **Layer 4 — Portfolio construction** (`agents/portfolio_construction.py`,
  optional, off by default / no-op unless enabled): a post-Commander
  stage that takes the Commander's per-asset decisions as Black-Litterman
  *views*, blends them with a market-implied equilibrium prior, shrinks
  the covariance estimate (Ledoit-Wolf), runs a risk-budgeting optimizer
  (min-variance / mean-variance / risk-parity, selectable), scales the
  result by a Kelly fraction, and logs a component-VaR risk report. Exists
  for going from "30 independent per-asset opinions" to "one
  coherent, risk-budgeted multi-asset portfolio," without being required
  for the simpler single-asset desk-simulation use case.

Two non-agent, always-deterministic components sit underneath all of
this:

- **`backtest/engine.py`** — order fills and portfolio accounting. No
  LLM calls, no randomness beyond what's explicitly seeded, fully
  testable in isolation. This is the part where correctness is
  non-negotiable, because everything above it is only as trustworthy as
  the fills and P&L it's scored against.
- **`data/point_in_time.py`** — a hard architectural boundary: on
  simulated day *t*, no agent can access data from day *t+1* onward.
  Enforced structurally by the data-loading layer (agents physically
  cannot reach future rows), not just documented as a rule, because a
  look-ahead bug here would silently invalidate any backtest result
  without producing an error.

## The eight pods

`agents/roster.py` is the single place that wires up the default
30-analyst / 8-manager roster; it's the file to change when adding more
analysts or managers. Each pod is themed around one coherent family of
quant strategy, so the Commander gets eight genuinely different risk
overlays rather than eight relabelings of the same signal:

| Pod | Manager | Analysts | What ties them together |
|---|---|---|---|
| **Risk** | `RiskManager` | `risk_minimizing`, `risk_maximizing`, `gbm_momentum` | Directional views built straight off volatility/return characteristics — favor low-VaR assets, favor high-GARCH-vol assets, or simulate a GBM forward path to see if it lands above today's price. `RiskManager` dampens confidence for elevated quantcore VaR. |
| **CVaR** | `CVaRManager` | `black_scholes_valuation`, `monte_carlo_valuation`, `greeks_delta_hedge`, `heston_vol_surface` | Options-valuation analysts. Since no free source has historical options chains, these price *theoretical/synthetic* options off model-estimated vol rather than live chains (`_theoretical_options.py`) — Black-Scholes closed form, Monte Carlo real-world-drift projection, a gamma-exposure proxy from the Greeks, and a two-regime Heston stochastic-vol comparison. `CVaRManager` dampens confidence for elevated tail-loss (CVaR). |
| **StatArb** | `StatArbManager` | `ou_mean_reversion`, `pairs_cointegration`, `kalman_hedge_ratio`, `regime_hmm` | Classic statistical arbitrage: an Ornstein-Uhlenbeck mean-reversion spread, Engle-Granger/Johansen cointegration against candidate partner tickers, a Kalman-filtered *time-varying* hedge ratio version of the same idea, and a Gaussian HMM regime detector. `StatArbManager` dampens confidence for elevated spread volatility. |
| **Macro/Rates** | `MacroRatesManager` | `egarch_asymmetric_vol`, `gjr_garch_crash_risk`, `arima_forecast`, `macro_var_spillover`, `cir_rate_regime` | Volatility-regime and macro-driven signals: EGARCH/GJR-GARCH asymmetric-volatility ("leverage effect") reads, an ARIMA return forecast, a small VAR system relating the asset to FRED macro series (VIX, yield curve), and a CIR short-rate regime read. GARCH-family parameters here are documented, fixed, honest approximations, not MLE-fitted — `quantcore` has no EGARCH/GJR-GARCH fitter yet. |
| **Factor** | `FactorManager` | `fama_french_factor`, `fama_macbeth`, `newey_west_significance` | Cross-sectional/regression analysts: a Fama-French factor-tilt regression, a genuine Fama-MacBeth two-pass estimate (the one analyst that needs more than one ticker's data), and a general "does X actually predict this asset" Newey-West-robust significance test against a configurable lagged predictor. |
| **Event/Credit** | `EventCreditManager` | `news_sentiment`, `sec_fundamentals`, `short_interest_squeeze` | Event-driven, mostly text/filings-based rather than pure price math: Finnhub news sentiment, an OLS trend over trailing SEC EDGAR filing values, and a short-interest days-to-cover squeeze z-score from FINRA data. `EventCreditManager` dampens confidence as event/filing severity rises. |
| **Alt-Data/Crowd Attention** | `AltDataManager` | `congressional_trading`, `insider_cluster_buying`, `retail_attention`, `wikipedia_pageviews`, `reddit_sentiment` | Methodologically distinct from every pod above: alt-data/behavioral signals rather than price or filing math — Senate/House Stock Watcher disclosure imbalance, SEC Form 4 insider-filing clustering, Google Trends and Wikipedia pageview attention spikes (two independent attention proxies, kept both for cross-validation — see `wikipedia_pageviews_analyst`'s docstring), and Reddit community-score sentiment. `AltDataManager` dampens confidence for both staleness and thin sample sizes (`[source_as_of=...]` / `[sample_count=...]` rationale tags). |
| **Anomaly/Network** | `AnomalyNetworkManager` | `calendar_seasonality`, `lead_lag_network_centrality`, `options_skew_sentiment` | Documented-anomaly and graph/network-science analysts, distinct from the factor_pod regressions: day-of-week/turn-of-month/January/Santa-rally calendar effects computed straight from price history, a lead-lag cross-correlation network centrality read across a configurable ticker universe (the second analyst, after `fama_macbeth`, that needs more than one ticker's data), and the CBOE aggregate put/call ratio as a real market-wide positioning read. `AnomalyNetworkManager` dampens confidence for low sample counts outside each analyst's well-studied regime. |

## Scoring — how the Commander decides who to trust

`scoring/` is a separate, non-agent system: it tracks each Layer 1
analyst's daily hypothetical attributable performance
(`TrackRecord`/`PerformanceRecord`) and converts trailing track record
into an allocation weight (`WeightingStrategy`, e.g. `Hedge`) via
`PerformanceScoreboard`. This — not agent type, not a hardcoded trust
table — is what `Commander.decide()` and each `ManagerAgent.evaluate()`
actually use to decide how much say a given analyst gets today. An
analyst that's been wrong lately loses influence automatically; one
that's been right gains it.

## Data sources

All external data funnels through `data/cache.py` — a local on-disk
parquet cache keyed by `(source, identifier, as_of_date)`, so re-running
a backtest never re-hits a rate-limited free API. Every cache-miss fetch
is also recorded, write-once, in a local SQLite database (`data/db.py`)
alongside the parquet cache — a durable, queryable log of every fetch
this project has ever made, independent of the per-file cache layout.
Point-in-time gating differs by source (see `data/point_in_time.py` and
CLAUDE.md for the exact rules — e.g. SEC filings gate on filed date, not
fiscal period end; FRED series gate on release/vintage date, not the
observation period).

| Source | Used for | Key required |
|---|---|---|
| `yfinance` (primary), Stooq (fallback), synthetic (last resort) | Daily price history | No |
| `yfinance` option chains | Live-mode options snapshots only — **not usable by backtest-driving analysts**, since no free source has historical chains | No |
| FRED | Macro/rates/VIX series | `FRED_API_KEY` |
| Ken French Data Library | Fama-French factor returns | No |
| SEC EDGAR | Company fundamentals (filed-date gated) | No (needs `SEC_EDGAR_USER_AGENT` contact header) |
| SEC EDGAR full-text search | Form 4 insider-transaction and 10-K/10-Q filing metadata (filed-date gated) | No (same `SEC_EDGAR_USER_AGENT` header) |
| FINRA | Biweekly short-interest | No |
| Finnhub | News/sentiment | `FINNHUB_API_KEY` |
| Tiingo | IEX-sourced EOD price cross-check against the primary yfinance/Stooq series (divergence is a data-quality signal, not just a gap-filler) | `TIINGO_API_KEY` |
| Alpha Vantage | Low-volume alternative EOD price source (~25 req/day free tier) for gaps in the other sources | `ALPHAVANTAGE_API_KEY` |
| Financial Modeling Prep | Fundamentals alternative/supplement to SEC EDGAR (~250 req/day free tier) | `FMP_API_KEY` |
| Google Trends (`pytrends`) | Search-interest attention proxy (provisional `isPartial` points dropped) | No |
| Wikimedia Pageviews | Wikipedia page-view attention proxy (gated on a ~24-48h publication lag) | No |
| Reddit | Mention-volume in finance subreddits (recent-history window only — the free search API doesn't expose a full historical archive) | `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` |
| Senate/House Stock Watcher | Congressional trading disclosures, gated on disclosure date, never the trade date (30-45 day STOCK Act reporting lag) | No |
| CBOE | Total put/call ratio — aggregate market-wide options positioning, distinct from the CVaR pod's per-name theoretical pricing | No |

**Historical options data remains a documented limitation.** Databento and
Polygon.io were investigated as the highest-value potential addition (it
would let the CVaR pod price real historical options instead of a
theoretical/synthetic model). Neither has a genuinely free ongoing tier
for it: Polygon's free "Basic" plan excludes options data entirely (options
access starts on a paid plan); Databento has no standing free tier, only a
one-time signup credit applied against its usage-based billing. Since
neither amounts to a free, repeatable historical-options source, the
CVaR-pod analysts still price theoretical/synthetic options with
model-estimated volatility (`agents/analysts/cvar_pod/_theoretical_options.py`)
rather than pretending otherwise.

## Conventions worth knowing before touching the code

- **Every cross-component data structure is a Pydantic `BaseModel`** —
  no raw dicts or dataclasses cross a module boundary. This is what
  keeps 30 wildly different analyst implementations interchangeable from
  every layer above them.
- **Two date types, two layers, do not mix them.** Market-data code
  (`_market_math.py`, `point_in_time.py`, every analyst's
  `generate_signal`/`scale_factor`) works in `pd.Timestamp`. Everything
  performance-tracking (`scoring/`) works in plain `datetime.date`.
  `Commander.decide()` and `ManagerAgent.evaluate()` are the one place
  that converts between them (`.date()`), right before calling into
  `scoring/`.
- `logging_config.py` is the project's only logging setup —
  `get_logger(__name__)` at import time everywhere, `configure_logging()`
  called once by an entry point. See CLAUDE.md for the one Streamlit
  `__name__` trap already hit here.
- `quantcore` is a dependency, never something to fork or patch from
  this repo.

## Running it

```bash
uv sync --locked --group dev              # install deps

uv run agentic-trading                    # launch the Streamlit dashboard
uv run agentic-trading-cli <assets> --start-date <date> --end-date <date>   # headless backtest

uv run pytest                             # tests
uv run ruff check . && uv run ruff format --check .
uv run mypy src/
```

The dashboard (`gui/dashboard.py`) is a visualization layer, not part of
the backtest engine itself: a multi-asset "trading desk" view, tabbed
(Overview / Desks / Analysts) to fit on one page, with day-by-day playback
(step forward/backward, a day-scrubber slider, autoplay). The Overview tab
leads with an **agent-flow view** — the Analysts → Pod Managers →
Commander → (optional) Portfolio Construction pipeline rendered live for
the current playback day and selected asset, each node colored by whether
it agrees with that day's majority-vote direction, so the roster's
heterogeneous agents debating is visible rather than just asserted — and a
**data-source status strip** showing which external sources this run
actually hit live vs. served from cache vs. never touched. The Desks and
Analysts tabs each add a **trust panel**: trailing Sharpe/contribution
next to the same scoring-derived weight the backtest itself applied, so
it's visually obvious trust is earned, not hardcoded. All of the original
per-day signal breakdown, holdings/blotter views, and scoreboard are still
there.

See `CLAUDE.md` for the full command reference, coding conventions, and
current architectural rules — it takes precedence over this file for
anything that conflicts.

## Status

Active development. `agents/roster.py` is the place to extend the
analyst/manager roster; `agents/portfolio_construction.py` is the place
to extend Layer 4 once single-asset behavior is stable.
