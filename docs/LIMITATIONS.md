# Limitations

Where the code falls short of point-in-time data or of standard practice,
and what other projects still have to supply. Every item names its issue:
`AT-` for this repo, `LK-` for the lakehouse, `EL-` for edgelab, `QC-` for
quantcore. Paths are relative to `src/agentic_trading/`; a dotted name such
as `firm.slippage_bps` is a field of `settings` in `config.py`.

## 1. Data and point-in-time limits

Every read goes through the lakehouse SQL API
(`data/loaders/lakehouse_loader.py`): no live API, no synthetic fallback.
Each loader filters on its source's availability date in SQL and then
checks its own result with `data/point_in_time.py::assert_point_in_time`;
prices reach agents through `get_point_in_time_data`, which slices on the
bar date.

Lake state and holdout:

- Inside a `lake_session` every read sees the lake at one lake time T.
  Tables the lake rebuilds whole keep only their current build and are
  read unpinned, so a run cannot be reproduced at its T once one of them
  is rebuilt; its record is the saved result and the build times in its
  provenance (AT-106).
- `lake_session(data_end=...)` cuts reads on knowledge time and raises
  `HoldoutReadError` past it. Three reads are outside the cut: the option
  split factor (`split_adjustment_factor` reads split ratios with ex-dates
  after a chain row's date, up to T, only to put that row on the close's
  basis), SIC codes (`load_sic_code_from_lakehouse`, undated) and preflight
  coverage (`source_coverage`) (AT-110).

Per source, as coded:

- **Prices by ticker** (`instrument_prices`): gated on the bar date.
  `close` is the lake's current `adj_close`, so price levels carry splits
  and dividends after the bar date; returns do not (AT-57). The open is
  scaled by `adj_close / close`; an open outside its bar's raw low-high
  range loads as NaN and is not filled (AT-66). A ticker that passed from
  one company to another reads as one series; in the lake only MRK does so,
  before 2009-11-04 (AT-108). The static equity universe is today's large
  caps and so carries survivorship bias (AT-111).
- **Crypto** (`crypto_klines_1m`, Binance USDT spot): one bar per NYSE
  session, open at the 09:30 ET minute, close at the 15:59 ET minute; USDT
  stands in for USD (AT-15).
- **S&P 500 cross-section** (Fama-MacBeth, network centrality): members by
  `security_id` from `sp500_price_coverage` pieces
  (`piece_start <= day < piece_end`); prices from `equity_daily_from_bars`
  (`official_close`, else `close`) times the `cum_factor` of the next
  `equity_adjustment_factors` row; a return exists only between
  consecutive NYSE sessions. A member with fewer returns than the
  analyst's threshold drops out of the day's cross-section, and the
  exclusion share has to be stated with any result (AT-80). A split-sized
  one-day move with no factor row is logged and counted, not corrected
  (AT-92). Members without prices and missing delisting returns are AT-13.
- **Option chains** (`dolthub_option_chain`; `option_chain_history` for SPY,
  QQQ and IWM before their first DoltHub date): the freshest snapshot on or
  before the day within 10 days (`options.max_snapshot_age_days`). Agents
  read the chain as of the day minus `options.publication_lag_days` (1
  calendar day); marks and fills read it as of the mark date (AT-59,
  AT-20). Strikes and premiums are rescaled onto the underlying close's
  basis with `yahoo_split_close_basis` (AT-1). No open interest or volume
  is read, so liquidity is assumed.
- **CFTC COT**: gated on `cftc_cot_publication`. A timed release (15:30 ET)
  is usable on its published date, an untimed one the next day, a report
  with no published date (every report before 2000) never (AT-6).
  `USDJPY=X` reads the JPY futures code without negating it, so its
  positioning sign is inverted (AT-23).
- **Dividends** (`equity_distributions`, regular cash, split-adjusted):
  gated on the ex-date, which is safe but weeks after the declaration; the
  lake has no usable declaration date (AT-68).
- **Earnings** (`dolthub_earnings_calendar` joined to `dolthub_eps_history`
  within 75 days after the period end): every announcement is usable from
  the next NYSE session, so a before-open report loses a day (AT-118).
- **SEC company facts and Form 4 insider transactions**: usable from the
  next NYSE session after the later of the filed date and the ET date of
  EDGAR's acceptance time (after the filed date when there is no
  acceptance time). A filing usable at the close of the session it
  became public in therefore costs one session: one accepted before the
  close loses its acceptance day, and a 10-Q or 10-K accepted after
  17:30 ET, which EDGAR dates to the next business day, loses that day
  (AT-9, AT-133). Both read
  the lake's `_log` views with a local de-duplication (AT-10, AT-11). A
  fiscal Q4 is derived as the annual value minus three discrete quarters,
  which is approximate for per-share concepts (AT-10).
- **FINRA short interest**: usable on the session after the 8th NYSE
  session past settlement, the loader's own estimate in place of the
  lake's dissemination date. A revised row takes the next settlement's
  availability; the pre-revision value is not in the lake, and a revised
  row with no later settlement is dropped (AT-7).
- **FRED** (`fred_vintages`): a version is known at the later of its
  `realtime_start` and its observation date, and used only when that is
  strictly before the decision date (AT-2, AT-118). The risk-free series
  the engine uses for cash interest and excess returns
  (`data/loaders/macro_loader.py::load_risk_free_rate_series`, DGS3MO) is
  read as known at the run's end date, not day by day (AT-82).
- **Fama-French** (`famafrench_returns`): a day's factors count as public
  at the end of the following month; the values are today's revised
  history, because the source keeps no daily vintages (AT-5).
- **SEC SIC** (`sec_sic_classification`): one current code per ticker with
  no history, so the pair screen applies today's industry to past dates.
  It is the one undated input in `stage_a/decisions.py::UNDATED_SOURCES`:
  rows of `pairs_cointegration` and `kalman_hedge_ratio` are price-only,
  and each of their trials declares `sec_sic` with this limitation (AT-62,
  AT-118). Both analysts are refused in stage (a) for now (section 3).
- **Licence**: prices from Yahoo-sourced lake tables are used for
  personal, non-commercial research; no market data is in this repo
  (AT-113).

Data windows:

- Survivor-free equity prices run 1992-01-02 to 2026-03-31; there is no
  survivor-free source after 2026-03 (AT-13).
- FRED: DGS3MO and DGS10 vintages start 2005-06-28, and a run or option
  mark before the first DGS3MO observation raises (AT-2, AT-65). VIXCLS
  starts 2010-11-22. BAMLH0A0HYM2 starts 2023-09-29 (FRED serves the ICE
  spread indices for a rolling three years), so the credit-spread analyst
  is unavailable before then (AT-112).
- Options: `option_chain_history` holds SPY, QQQ and IWM daily from 2008 to
  2025-12-12; SPY continues on DoltHub from 2019-02-09, while QQQ and IWM
  have no chain from 2025-12-15 to 2026-09-27 (AT-79). Other underlyings
  start in 2019 at the earliest, most in 2023 (AT-112).
- FINRA short interest is usable from 2018-01-12 (settlements from
  2017-12); the earnings calendar starts 2020-01-22; SEC company facts
  start between 2009-05 and 2015-10 depending on the name (AT-112).
- Stage (a) decides from 2010-01-04 (`stage_a.window_start`) to 2024-12-31
  (`stage_a.data_end`); 2025-01-02 onward is a locked holdout (AT-110).

## 2. Departures from standard practice

- **Options on thin chains** (AT-50, AT-107). The CVaR pod expresses views
  on 11 underlyings (`OPTIONS_ASSETS` in `agents/managers/cvar_manager.py`)
  as listed options: a long view buys an ATM call near 30 days
  (`options.target_tenor_days`), a short view an ATM put, sized so the
  premium carries the stock weight's delta, capped at the stock weight. A
  variance-risk-premium view at confidence 0.5 or more
  (`options.short_straddle_min_confidence`) also sells a delta-hedged ATM
  straddle. A leg changes side only after 3 days of the opposite view
  (`options.expression_confirmation_days`) and rolls inside 7 days to
  expiry (`options.roll_min_dte`). With no tradeable contract the pod
  keeps the stock weight.
  - Fills (`backtest/portfolio.py::apply_option_decision`): whole
    as-traded contracts at the ask or the bid, $0.65 a contract, at least
    $0.05 of premium a share, no trade for a change under 50% of the held
    count (`options.rebalance_band`), at most 1.5x NAV of underlying
    notional per contract, short premium at most 5% of NAV, short margin
    of premium plus 20% of underlying notional charged at the margin rate.
  - Marks (`data/loaders/option_marks.py`): a same-day quote as quoted; an
    older one re-marked by Black-Scholes with no dividend yield at its
    implied vol and today's spot; a contract the latest snapshot no longer
    lists marked from its own last quote for up to 45 days at the surface's
    interpolated vol, untradeable once that quote is over 10 days old;
    settlement at intrinsic against the last close on or before expiry.
  - Not modelled: early exercise and assignment (AT-50), spreads and
    calendars, exit liquidity. A core pod's option row is sized by premium
    weight, so a decaying option is bought again (AT-107).
- **Tail hedge** (AT-93, AT-102, AT-46). A long SPY put at 95% of spot
  (`options.tail_hedge_strike_moneyness`) near a 60-day tenor
  (`options.tail_hedge_tenor_days`), rolled inside 7 days to expiry. Its
  premium weight is the Commander's 2% carve-out
  (`commander.tail_hedge_budget_fraction`) times the VIX analyst's
  confidence (at least 0.15, full at a 3-sigma VIX z-score), capped at the
  premium for 100% of NAV in underlying notional. New puts commit at most
  2% of NAV over any trailing 365 days
  (`options.tail_hedge_annual_premium_cap`), and a held put keeps its
  contract count until it rolls or expires. With no headroom the sleeve
  opens nothing and holds its put to expiry; with no usable chain it falls
  back to a short SPY weight. The sleeve sits outside the drawdown
  breaker, the risk-budget allocation and the firm limits. One strike
  rule, one tenor, no structures.
- **Costs and impact** (AT-24, AT-28, AT-44, AT-45, AT-105).
  - An equity fills at the reference price times 1 plus or minus
    (slippage + impact), with `firm.slippage_bps` 5, and pays commission
    `firm.commission_rate` 0.0005 on the executed notional.
  - Impact is the square-root law (quantcore `square_root_market_impact`):
    `firm.market_impact_coefficient` (1.0) x daily vol x
    sqrt(shares / ADV), in basis points. ADV and vol come from the trailing
    20 bars (`firm.adv_window_days`), the fill day included (AT-28). A fill
    is capped at 5% of ADV (`firm.max_adv_participation`) and the rest is
    left to the next decision. An asset with no volume (spot FX, indices)
    pays neither; crypto volume is in USD and futures volume in contracts,
    so both are misread (AT-24).
  - Cash: a negative balance pays the risk-free rate plus 1.5%
    (`firm.margin_spread_annual`); a positive balance earns the risk-free
    rate, actual/365, at the rate on the previous valuation date, a
    negative rate counted as 0 (AT-82). Short-option margin is charged at
    the margin rate even when cash is positive (AT-30).
  - Not modelled: an equity bid-ask spread, per-name borrow or a locate
    (flat `firm.short_borrow_rate_annual` 0.5%) (AT-44), margin or buying
    power (AT-45).
  - A pod's no-trade band clears 5x (`portfolio.cost_clearance_multiplier`)
    the linear cost `slippage + commission x (1 + slippage)`
    (`execution_costs.py`) plus a 0.005 weight buffer; impact and option
    spreads are not in it (AT-105).
  - Stage (a) books are costed by edgelab's replay, not by this model
    (AT-110).
- **Futures and FX** (AT-22, AT-23): a future trades as one continuous cash
  security with no contract, multiplier, margin or costed roll, and pays
  equity borrow when short. FX earns no carry, and `USDJPY=X`'s JPY price
  is booked as USD.
- **SIC, not GICS** (AT-62, AT-111, AT-56). Pair candidates are SEC SIC
  matches over the traded universe plus ten hand-picked pairs
  (`CANDIDATE_PAIRS` in `agents/analysts/stat_arb_pod/_pairs_screening.py`);
  the roster seeds four constant pairs chosen with knowledge of the sample
  (`_COINTEGRATED_PAIRS` in `agents/roster.py`). The screen runs two-year
  Engle-Granger, admits at p < 0.05 with a 5-60-day half-life, keeps while
  p < 0.10, holds at most 5 pairs and re-screens every 63 days. Legs are
  sized per asset, not at the hedge ratio (AT-56).
- **Sizing on short history** (AT-54, AT-48, AT-83). Views become Grinold
  alphas `IC x sigma x z` (z capped at 2) in a zero-prior Black-Litterman
  with Idzorek confidence in [0.1, 0.9]; there is no market-cap
  equilibrium prior. IC is the Pearson correlation of signal score and
  realized return pooled over asset-days in the trailing 250 days (AT-54):
  the 0.05 prior below 20 records, otherwise shrunk toward 0.05 with 60
  pseudo-observations and capped at 0.15 in magnitude. A Kelly overlay
  ramps in over 60 days. Vol targeting, Kelly and the drawdown breaker
  read each pod's paper book (target weights times close-to-close
  returns), not its fills and costs (AT-48). Each pod scales down, in one
  pass, any asset above 40% of its component VaR, which can leave the
  asset above the cap (AT-83).
- **Rebalancing** (AT-104): a pod re-trades every 5 trading days, or off
  cycle for an asset more than 10% (relative) from its held weight.
  Between those days it keeps its book even when its risk budget sizes
  zero, and it counts as holding.
- **Firm risk as limits, not a risk model** (AT-46, AT-47, AT-81). On the
  delta-equivalent target book, in order: 25% single name, 4x gross, 2% of
  NAV one-day 95% parametric VaR from a Ledoit-Wolf covariance of 250
  days, beta to SPY of 1.0 (skipped without SPY), and a stress test with
  correlations pulled toward 0.90 and vol doubled, one-day loss capped at
  15%. The covariance-based limits are skipped below 5 + n observations.
  Options count as delta only and limits check targets, not fills (AT-46);
  the stress test can lower a hedged book's loss (AT-47). There is no
  factor, liquidity or margin limit. Pods are sized by cross-pod risk
  parity once there are more overlapping days than pods, by inverse vol
  before; an idle pod keeps its allocation as cash (AT-81). The drawdown
  breaker trips at -10%, resets at -3% and keeps 25%.

## 3. Signal conventions

- **Status** (AT-119). `unavailable`: the data is not there (no coverage,
  too little history, no bar). `active` with confidence 0: the model
  declines on data it has (no serial dependence, a degenerate or
  misspecified fit, a signal under its threshold, a zero rate outside
  CIR's domain). `error`: the model or a loader failed, a non-converged fit
  included. `AnalystAgent.unavailable_signal`, `abstain_signal` and
  `error_signal` build the three; each weighs 0. The round-1 analysts use
  them. `vix_tail_hedge_analyst`, the round-2 option, Monte Carlo and
  dividend analysts, and `ManagerAgent.collect_signals`'s error fallback
  still set status directly, and `_market_math.conditional_daily_volatility`
  falls back to EWMA on a GARCH failure with no status at all (AT-119).
- **Confidence 0 is flat**: `AgentSignal` sets `direction` to flat whenever
  `confidence` is 0, whatever the analyst passed (AT-119).
- **`source_as_of`** (AT-118): the publication date of the newest data the
  signal rests on, not the day it became usable; `None` for a price-only
  signal. `sample_count` is how many observations the read rests on;
  managers dampen on both. Option analysts carry the chain's snapshot
  date, `dividend_payout` the ex-date. The stage (a) store refuses an
  active row without `source_as_of` from an analyst that declares a dated
  source, and `check_source_as_of` refuses a row dated after the decision
  date, or on it for a source that can publish after the close.
  `stage_a/decisions.py::SOURCES` marks earnings, FINRA, FRED, SEC facts
  and Form 4 as after-close; COT, dividends, Ken French factors, option
  chains and S&P 500 membership as before the close.
- **Stage (a) calendar** (AT-14): decision days are the sessions of
  `stage_a.session_calendar` (XNYS by default; part of the spec hash).
  Prices load with `align="sessions"`: rows are the calendar's sessions on
  which at least one asset has a close, bars on other days are dropped,
  and each asset's log return is recomputed over its own consecutive
  closes, so a dropped bar's move lands on the next kept bar. An asset
  with no close on a session is stored `unavailable` without asking the
  analyst, and the analyst sees only the rows on which its asset has a
  close. A calendar exchange-calendars cannot build from 1990 raises on
  first use.
- **Excluded from stage (a)** (AT-123): `pairs_cointegration_analyst` and
  `kalman_hedge_ratio_analyst` trade four seed pairs (`_COINTEGRATED_PAIRS`
  in `agents/roster.py`) selected on 2017-04 to 2018-03 data, inside the
  grading window. `stage_a.runner.EXCLUDED_ANALYSTS` refuses them until
  they select pairs point-in-time.
- **Spec hash inputs** (AT-120): the hash covers `settings` (including
  `stage_a.warmup_days`, the history loaded before the window) and every
  `agentic_trading.agents` and `agentic_trading.data` module an analyst
  reaches through its imports, so an edit to any of them is a new trial.
- **Decision rule** (AT-110): `w_i = s_i / sum_j |s_j|` over active
  instruments, `s_i` the signed confidence; a non-active instrument weighs
  0 and is outside the denominator; every weight is 0 when the sum is 0.

## 4. Engine path

The execution engine (`backtest/engine.py`, `backtest/portfolio.py`,
`backtest/checkpoint_store.py`), the experiment CLI and the dashboard's
performance pages are to be deleted (AT-88) once the firm runs as one
edgelab strategy (AT-85) on edgelab's engine, which records and replays it
(AT-77). Until then these hold and are not worked on:

- AT-14: the engine's callers align prices on the intersection of all
  assets.
- AT-20: a next-open option fill prices off the fill day's end-of-day
  chain.
- AT-22, AT-23, AT-24, AT-44, AT-45: futures, FX, volume units, spread and
  borrow, margin (section 2).
- AT-25: a next-open fill is sized with that day's closing NAV. AT-26: at
  lag 0 a signal is credited with the return it already saw. AT-27: the
  recorded NAV is taken before the day's fills. AT-28: the ADV and vol
  window includes the fill day.
- AT-30: short-option margin is charged even with positive cash.
- AT-32: `experiment` benchmarks against the ^GSPC price index. AT-33:
  per-source Sharpe pools asset-days. AT-43: `total_transaction_cost` is
  commission only (`total_execution_cost` holds all four kinds).
- AT-34: one Commander serves calibration and test. AT-35: a resume checks
  only dates and assets and does not restore CVaR and stat-arb held
  positions.
- AT-42: the engine re-trades any non-zero delta against holdings.
- AT-48, AT-49: pod returns from a paper book; next-open attribution is
  close to close. AT-52: a fill is stamped with its decision date. AT-53:
  the isolated-analyst baseline's gross grows with the number of assets.
- AT-50: no early exercise or assignment; re-marks carry no dividend
  yield. AT-51: 252 is hard-coded in `backtest/engine.py`,
  `scoring/scoreboard.py` and `backtest/baselines.py` instead of
  `firm.trading_days_per_year`.
- AT-46, AT-47, AT-54, AT-55, AT-56 outlive the engine and wait on the
  same decision.

## 5. Gaps in other projects

quantcore:

- QC-16: `cir_rate_regime_analyst` tells a no-mean-reversion CIR fit by
  its `kappa <= 0` error message; quantcore v0.3.0 has no typed exception
  for it, which comes with its next tag (AT-101).
- QC-14 deletes the active-set QP solver; nothing here calls it (AT-117).
- There is no implied-volatility surface primitive.
  `option_marks._surface_volatility` is the local fallback: linear in
  strike within an expiry, linear in total variance across expiries, flat
  outside the quoted range (AT-60).

Lakehouse:

- LK-13, LK-14: company facts and insider transactions are read from
  `_log` views and de-duplicated locally (AT-10, AT-11).
- LK-20, LK-44: delisting classes and S&P 500 leavers without prices
  (AT-13).
- LK-37: single-name option chains before 2019.
- LK-47: FINRA's dissemination date in the lake counts weekdays, not
  sessions; the loader estimates its own (AT-7).
- LK-49: point-in-time dividend declaration dates (AT-68).
- LK-50, LK-52: source gaps in `equity_daily_from_bars` (partly filled) and
  predecessor tickers of renamed members (AT-80).
- LK-85: point-in-time SIC; the pairs analysts move to it as a new trial
  (AT-118).

edgelab:

- EL-7: record then replay, mixed books, P&L by pod tag, futures margin and
  a crypto calendar. The counted stage (a) grade waits on it (AT-110), and
  AT-14, AT-22, AT-23, AT-24 and AT-48 close there.
- EL-25, EL-26: a view that carries the book's history, and a strategy's
  own pinned dependencies; AT-85 waits on both.
- EL-27, EL-69: the delisting convention and the S&P 500 survivorship
  acceptance that AT-13 closes against.
- EL-38: futures start with equity cost rates.
- EL-41: the metrics-table Sharpe ignores the risk-free rate, and the
  headline Sortino is on total return.

## 6. Results invalidated

- Results from option analysts and the CVaR pod on names with a split,
  produced from 2026-09-18 until the AT-1 split factor landed, are invalid
  (AT-1, AT-41).
