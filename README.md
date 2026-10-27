# agentic-trading

A multi-agent **backtesting** framework for a multi-strategy fund. It is
not a live trading system and places no orders. Day by day it simulates
how a committee of heterogeneous quant "analysts" would argue about each
asset, how risk-aware pod "managers" size their own books, how a
"commander" allocates capital across pods and applies firm-level limits,
and scores every agent's track record along the way.

All numerical finance (option pricing, GARCH-family volatility, VaR/CVaR,
cointegration, factor regressions, portfolio optimization) comes from
`quantcore`, a separate dependency. This repo does not reimplement
quantitative math; it orchestrates it and turns the output into position
decisions and a backtest result. Every input a run reads comes from a
local lakehouse of previously scraped data (see "Data"); nothing calls a
live API during a run.

## Architecture

```
Layer 1  Analysts (38 instances, one AgentSignal per asset per day)
   ↓
Layer 2  Pod managers (11 pods): vol-regime overlay, then an independently
         sized pod book (Black-Litterman views, long/short mean-variance
         or view-tilted risk parity, vol targeting, Kelly, leverage caps)
   ↓
Layer 3  Commander: trailing-track-record capital allocation across pods,
         drawdown breaker, tail-hedge budget, firm-level limits
   ↓
         Backtest engine: next-open fills, commission, slippage, financing
   ↓
         Scoring: attribution and measured information coefficients,
         fed back into layers 2 and 3
```

- **Analysts** (`agents/analysts/`, one subpackage per pod) implement one
  interface (`agents/base.py::AnalystAgent`) and emit one schema
  (`AgentSignal`: direction, confidence, rationale, status, provenance),
  whatever technique produced it.
- **Managers** (`agents/managers/`) each own a themed pod of analysts.
  `scale_factor` dampens confidence with a logistic in the asset's
  vol-regime z-score (alt-data and event pods add staleness and
  sample-count overlays); `construct_portfolio` turns the pod's signals
  into a `PodView` per asset and hands it to the pod's own
  `PortfolioConstructor`.
- **PortfolioConstructor** (`agents/portfolio_construction.py`): Grinold
  alpha (`IC × σ × z`, with each analyst's measured, shrunk IC) as a
  Black-Litterman view against a zero prior with Idzorek uncertainty;
  unconstrained long/short mean-variance (or risk parity tilted by
  conviction for the commodity/FX/crypto and credit pods); vol targeting
  to the pod's own target; a bounded Kelly overlay; gross and net
  leverage caps; a cost-aware no-trade band. A pod net short in its views
  is net short in its book.
- **Commander** (`agents/commander.py`) allocates capital across pods by
  trailing track record, cuts a pod in drawdown, carves out a fixed
  tail-hedge budget, and applies firm limits (single-name, gross,
  one-day VaR) to the combined book. It never re-sizes inside a pod.
- **Engine** (`backtest/engine.py`) is deterministic: decide on the
  close, fill at the next open with commission and slippage, charge
  stock borrow and margin financing, mark to market, record every trade
  and every contributing signal, checkpoint and resume.
- **Point-in-time** (`data/point_in_time.py`): on simulated day *t* no
  agent sees data from *t+1*; lakehouse loaders gate each source on its
  own availability date (filing date, publication lag, dissemination
  date).

## The eleven pods

`agents/roster.py` wires the default roster. Equity pods share the run's
universe; the others are scoped to their asset class.

| Pod | Manager | Analysts | Strategy |
|---|---|---|---|
| Risk | `RiskManager` | risk_minimizing, gbm_momentum, dividend_payout | Vol-regime long/short, momentum, a real cash-flow distress signal |
| CVaR *(11 names with a real option chain)* | `CVaRManager` | black_scholes_valuation, monte_carlo_valuation, greeks_delta_hedge, heston_vol_surface, risk_maximizing | Model vol vs real ATM implied vol, normalised gamma regime, variance risk premium and term slope, capitulation buying |
| StatArb | `StatArbManager` | ou_mean_reversion, pairs_cointegration, kalman_hedge_ratio, regime_hmm | Quarterly walk-forward Engle-Granger pair screen with hysteresis, 2σ/0.5σ spread trading on the cointegrating hedge ratio, two-state Hamilton regimes |
| Macro/Rates | `MacroRatesManager` | egarch_asymmetric_vol, gjr_garch_crash_risk, arima_forecast, macro_var_spillover | Fitted asymmetric-GARCH regimes, Box-Jenkins forecasts, Granger spillover from FRED series |
| Rates *(SHY/IEF/TLT)* | `RatesManager` | cir_rate_regime, cot_positioning_rates, arima_forecast_rates | CIR short-rate regime on a Treasury ladder plus CFTC positioning |
| Cross-Asset *(gold, crude, silver, nat-gas, copper futures; four FX pairs; BTC, ETH)* | `CrossAssetManager` | gjr_garch, egarch, regime_hmm, arima, cot_commodities, cot_fx | Generic vol/regime models plus CFTC positioning, risk-parity sized |
| Credit *(HYG/LQD)* | `CreditManager` | credit_spread_signal, egarch_credit, regime_hmm_credit | ICE BofA HY spread as a risk-appetite overlay |
| Factor | `FactorManager` | fama_french_factor, fama_macbeth, newey_west_significance, earnings_surprise | Significance-gated factor loadings times factor premia, post-earnings drift |
| Event | `EventCreditManager` | sec_fundamentals, short_interest_squeeze | SEC EDGAR filing trends, FINRA short-interest squeezes |
| Alt-Data | `AltDataManager` | calendar_seasonality, lead_lag_network_centrality, insider_cluster_buying | Documented calendar anomalies, lead-lag network centrality, SEC Form 4 open-market purchase clusters |
| Tail-Hedge *(SPY)* | `TailHedgeManager` | vix_tail_hedge | A standing short-SPY sleeve that scales with VIX elevation, funded by a fixed budget |

## Data

Every run reads only the lakehouse (`data/loaders/lakehouse_loader.py`,
a lake maintained by a separate scraping project, read through its
HTTP API with `LAKEHOUSE_API_URL` and `LAKEHOUSE_API_TOKEN`): prices (dividend-adjusted, with the open scaled to the same
basis), option chains (11 underlyings, three expiries per snapshot),
dividends, CFTC COT positioning, earnings surprises, SIC codes, S&P 500
membership, FRED series, Ken French factors, FINRA short interest, SEC
company facts and SEC Form 4 insider transactions. Each loader gates on the source's own availability date.
`backtest/preflight.py` checks every analyst's declared data requirement
against the run's tickers and dates before anything is simulated and
drops analysts that cannot run.

## Sizing and risk, in one paragraph

A pod's per-asset view is `sum(direction × confidence)` over its active
signals, scaled by `sqrt(count)` and capped at 2σ. It becomes an
expected return through `IC × σ_asset × z`, where the IC is each
contributing analyst's trailing correlation between signed forecast and
realized return, shrunk toward a 0.05 prior. The zero-prior
Black-Litterman posterior feeds an unconstrained mean-variance book; vol
targeting sets the gross; Kelly nudges it by the pod's own trailing
Sharpe; caps bound it. The Commander allocates across pods by trailing
Sharpe, cuts a pod 75% at a 10% drawdown until it recovers to 3%, gives
the tail hedge a fixed 2%, and clips the combined book to a 25%
single-name cap, 4× gross and a 2% one-day 95% VaR. The engine charges
5 bps commission, 5 bps slippage, 50 bps annual stock borrow on shorts
and risk-free plus 150 bps on borrowed cash. `KNOWN_LIMITATIONS.md`
lists what still departs from a production desk.

## Constants and per-run overrides

Every tunable number lives in `src/agentic_trading/config.py` as one
validated Pydantic `settings` object, each value with a one-line reason.
Code reads it at use time, so a run can override anything:

```bash
uv run agentic-trading-experiment AAPL MSFT ... \
  --set commander.drawdown_trip_threshold=-0.15 \
  --set pod_sizing.stat_arb_manager.leverage_cap=4.0 \
  --set analysts.z_entry_threshold=1.5
```

`config.temporary_overrides({...})` does the same inside a sweep script.
Overrides are recorded in the experiment JSON.

## Running it

```bash
uv sync --locked --group dev
uv run agentic-trading                    # Streamlit dashboard
uv run agentic-trading-cli <assets> --start-date <date> --end-date <date>
uv run agentic-trading-experiment <assets> \
  --train-start <date> --train-end <date> \
  --test-start <date> --test-end <date> --yes
uv run pytest && uv run ruff check . && uv run mypy src/
```

The experiment's first phase is *calibration*: no model is trained, only
analyst and manager track records accumulate and carry into the test
phase. Useful flags: `--include-baselines`, `--include-isolated`,
`--exclude-analyst <id>` (repeatable), `--set group.field=value`
(repeatable), `--calibration-checkpoint`/`--test-checkpoint` with
`--resume`. The first simulated day is the slowest (every model fits from
scratch; the default warm-up loads three years of history so every
window is full); later days reuse cached fits.

The dashboard (`gui/dashboard.py`) is a visualization layer over a saved
run: day-by-day playback, the analyst → pod → commander flow for the
current day and asset, holdings and blotter, and trust panels showing
each analyst's and pod's trailing performance next to the weight the
backtest applied.

## Project structure

```
src/agentic_trading/
├── config.py                  # every strategy constant, per-run overridable
├── agents/
│   ├── analysts/              # Layer 1, one subpackage per pod
│   ├── managers/              # Layer 2, one manager per pod
│   ├── base.py                # AnalystAgent / ManagerAgent interfaces
│   ├── portfolio_construction.py  # pod-level sizing pipeline
│   ├── commander.py           # Layer 3 allocation, breaker, firm limits
│   ├── roster.py              # default analyst and pod wiring
│   └── schemas.py             # Pydantic schemas shared across layers
├── backtest/                  # engine, portfolio ledger, baselines, preflight, checkpoints
├── data/
│   ├── loaders/lakehouse_loader.py  # every data read
│   ├── loaders/macro_loader.py      # FRED series ids, risk-free series
│   ├── loader.py              # price load with warm-up
│   ├── point_in_time.py       # no-look-ahead slicing
│   └── ticker_catalog.py      # static asset-class catalog for the UIs
├── scoring/                   # track record, IC, weighting strategies
├── gui/dashboard.py           # Streamlit dashboard
├── cli.py                     # agentic-trading-cli / agentic-trading-experiment
└── logging_config.py
scripts/                       # pair screen seed, run summaries, attribution series
tests/unit/                    # mirrors src/
```

## Conventions

- Every cross-component structure is a Pydantic `BaseModel`; no raw
  dicts across module boundaries.
- `quantcore` owns the math. A missing primitive is a quantcore task;
  `KNOWN_LIMITATIONS.md` §5 lists the ones currently bridged with a
  minimal local fallback.
- Point-in-time discipline is structural: loaders gate on availability
  dates, the engine slices with `get_point_in_time_data`.
- Market-data code works in `pd.Timestamp`; scoring code in
  `datetime.date`; convert only at the manager and commander boundary.
- Docstrings and comments describe current behaviour only.
