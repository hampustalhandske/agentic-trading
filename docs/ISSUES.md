# Issues

Open work on agentic-trading, one entry per issue, ids `AT-1` to `AT-78`. The list comes from the
agentic-trading audit 2026-10-27 (data layer D*, execution layer X*, reproducibility and docs R*).
Paths are relative to `src/agentic_trading/`.

## Conventions

- **Id prefix:** `AT-`. Other projects use their own prefixes: LK (lakehouse), EL (edgelab),
  PE (pitch-edge).
- **Status:** open, in-progress, done, wontfix.
- **Priority:** must, should, nice.
  - Each priority carries the requirement id it serves (M1-M8 for musts, S1-S5 for shoulds) where
    the issue list names one. Every must blocks the first submission (M6). Where the list names no
    other id, the entry shows `must (M6)` or plain `should`.
  - "Should" means after the musts. It is not optional: everything wrong or off industry practice
    gets fixed.
  - No issue is "nice" today.
- **Gate:**
  - **data** = needed under any design; approved by the owner on 2026-10-28.
  - **held** = engine or CLI item; not started until the owner decides AT-77 (one engine in edgelab).
  - **delete** = feeds no decision (sizing, pod weights, breaker, Kelly); edgelab grades, so the
    code is removed rather than fixed.
- **Evidence tags:** [V] verified by the orchestrator, [M] measured against the lake or a probe,
  [C] code-read, [I] inferred.
- **Raised by:** every entry is from the agentic-trading audit 2026-10-27 unless stated.
- **Blocked-by / Blocks:** ids in other projects, named where the issue list or the design note
  names them.
- Entries that AT-77 would delete or move carry a Note line. They stay open and held until the
  owner decides.

## Clean window for a first grade

- Survivor-free equity prices: 1992-01-02 to 2026-03-31.
- Burned: 2018-04 to 2026-03 (9 prior runs).
- After 2026-03: about 7 months, no survivor-free source.
- Baselines A1 use train before 2010-01-01 and test 2010-01-01 to 2017-12-29.
- Pre-2018 per-analyst coverage is counted after the must fixes. Options start 2019 (most names
  2023), FINRA 2017-12, earnings calendar 2020.

## Entries

### AT-1 Option split rescale multiplies every raw write of `dolthub_stock_splits`
- Status: done
- Priority: must (M6)
- Gate: data
- Raised by: agentic-trading-c5 audit 2026-10-27 (D-E1)
- Evidence: [V] `data/loaders/lakehouse_loader.py:130-159` has no dedup on `scraped_at`. Factors: AAPL 1.2e24 (true 4), NVDA 1e40 (10), TSLA 1.2e19 (3), AMZN/GOOGL 1.2e104 (20). CVaR results since 2026-09-18 are invalid.
- Acceptance: `_split_adjustment_factor` returns exactly 4.0 (AAPL 2020-08-28), 10.0 (NVDA 2024-06-07), 3.0 (TSLA 2022-08-01), 20.0 (AMZN, GOOGL the day before their 2022 ex-dates). Source is `equity_splits` with `applied`. A unit test writes one split row three times.
- Blocked-by / Blocks: none. Done first and alone.
- Commits: 6f56fa2 (factor magnitude), 99edc18 (factor from `yahoo_split_close_basis`, LK-41: product of 1/ratio over later splits with `in_close_basis IS NOT FALSE`; AVB 2026-08-17 and SSP 2025-12-08 excluded; verified live).

### AT-2 FRED read as latest value, not `fred_vintages`
- Status: done
- Priority: must (M6)
- Gate: data
- Raised by: agentic-trading-c5 audit 2026-10-27 (D-E2)
- Evidence: [V] `lakehouse_loader.py:575-615` gates on `date <= as_of`. [M] CPIAUCSL first published a median 41 days after its date; DGS10 and DGS3MO lag at least 1 day on 1566 of 1566 observations.
- Acceptance: the loader takes the latest `realtime_start <= as_of` per date from `fred_vintages`. DGS10 dated 2024-03-14 is absent as of 2024-03-14 and present as of 2024-03-15. GDP 2024-01-01 as of 2024-05-01 equals 28284.498.
- Blocked-by / Blocks: none
- Commits: dadea7a. Verified live: DGS10 2024-03-14 absent as of 2024-03-14, present as of 2024-03-15; GDP 2024-01-01 as of 2024-05-01 is 28284.498. DGS3MO vintages start 2005-06-28, so a run ending before then has no risk-free series.

### AT-3 COT and Fama-French read frozen retired directories
- Status: done
- Priority: must (M6)
- Gate: data
- Raised by: agentic-trading-c5 audit 2026-10-27 (D-E3)
- Evidence: [V] `lakehouse_loader.py:271-276` and `:618-621`. COT stops at report_date 2026-09-22, Fama-French at 2026-08-31.
- Acceptance: loaders read `cftc_cot_*` with the 12-code map and `famafrench_returns` (or `famafrench_vintages`, see AT-5). COT max report_date is 2026-10-20 or later.
- Blocked-by / Blocks: none
- Commits: 3ff8eff

### AT-4 Every loader reads lake Parquet by path; no lake T or version recorded
- Status: done
- Priority: must (M2, M6)
- Gate: data
- Raised by: agentic-trading-c5 audit 2026-10-27 (D-E4)
- Evidence: [V] every `read_parquet` in `lakehouse_loader.py` (lines 100, 144, 208, 213, 288, 331, 387, 394, 438, 515, 594, 660, 712, 758, 808, 858). Path reads see raw append logs, including flagged ranges, superseded keys and private tables.
- Acceptance: no `read_parquet` or `glob` against `LAKE_DIR` remains (a grep test). Every query goes through `/v1/sql` against the typed views, and the lake T and version are recorded.
- Blocked-by / Blocks: none
- Commits: 3ff8eff (every read on /v1/sql), 9ed37ed (each run records the lake version as reported, read time and contract `lakehouse-contract-v1`; a test binds every loader's SQL to the contract's views and types).

### AT-5 Fama-French gated on factor date, not publication or vintage
- Status: done
- Priority: must (M6)
- Gate: data
- Raised by: agentic-trading-c5 audit 2026-10-27 (D-M16)
- Evidence: [C] `lakehouse_loader.py:663` uses `date <= as_of_date`. Ken French publishes monthly with a lag of about a month and restates history. `famafrench_vintages` exists.
- Acceptance: factors come from the latest `famafrench_vintages.vintage_date <= as_of`. As of 2024-03-15 the latest factor date is no later than that vintage's last date (about 2024-01-31).
- Blocked-by / Blocks: none
- Commits: dadea7a. Gated on a one-month publication lag (a day's value is public at the end of the following month), not `famafrench_vintages`: the lake keeps no daily vintages, so daily values are today's revised history, stated in the loader. First-seen dates exist in `famafrench_returns_log` from 2026-09-27.

### AT-6 COT availability ignores holiday weeks
- Status: open
- Priority: must (M6)
- Gate: data
- Raised by: agentic-trading-c5 audit 2026-10-27 (D-M22)
- Evidence: [M] `lakehouse_loader.py:293` uses `report_date + 3 days`. 79 of 1,059 GC=F reports are exposed 1-4 days early. The exact date is in `etl.publication.cftc_cot_published_at`.
- Acceptance: the code follows `cftc_cot_published_at`, shifting to the next business day per holiday. The 2018-12-31 report is absent as of 2019-01-04 and present as of 2019-01-07.
- Blocked-by / Blocks: with LK-15
- Commits: none

### AT-7 FINRA short interest: no holiday calendar; revisions overwrite undated
- Status: done
- Priority: must (M6)
- Gate: data
- Raised by: agentic-trading-c5 audit 2026-10-27 (D-M23)
- Evidence: [M] `lakehouse_loader.py:756-762`. 80 of 212 AAPL settlements (38%) are early under a holiday-aware estimate. 2,589 rows carry `revision_flag='R'` and overwrite the original from the first dissemination.
- Acceptance: dissemination uses the FINRA calendar, or settlement + 8 NYSE business days + 1 day. Revised rows are available no earlier than the next dissemination date after the original. A test uses a settlement date with Thanksgiving in its window.
- Blocked-by / Blocks: none
- Commits: fd8e1ca. Availability = the session after the 8th NYSE session past settlement; the lake's weekday-only date is ignored (LK-47). Revised rows take the next settlement's availability; originals are not recoverable. Verified live: AAPL 2023-11-15 absent 2023-11-28, present 2023-11-29.

### AT-8 After-close earnings used in that day's close decision
- Status: done
- Priority: must (M6)
- Gate: data
- Raised by: agentic-trading-c5 audit 2026-10-27 (D-M17)
- Evidence: [V] the join ignores `when` (`lakehouse_loader.py:382-401`). [M] 5,453 of 13,319 rows are "After market close". AAPL as of 2024-08-01 returns the 2024-08-01 report.
- Acceptance: availability is `date + 1 trading day` when `when` is "After market close" or null. AAPL as of 2024-08-01 returns the 2024-05-02 quarter.
- Blocked-by / Blocks: none
- Commits: fd8e1ca. Verified live: AAPL as of 2024-08-01 returns the 2024-05-02 report; as of 2024-08-02 the 2024-08-01 report.

### AT-9 SEC filings accepted after 16:00 ET used the same day
- Status: done
- Priority: must (M6)
- Gate: data
- Raised by: agentic-trading-c5 audit 2026-10-27 (D-M18)
- Evidence: [M] `lakehouse_loader.py:715` and `:812` gate on `filed_date`. Since 2022, 79.0% of Form 4s (172,531 of 218,446), 47.8% of 10-Qs and 50.8% of 10-Ks were accepted after 16:00 ET.
- Acceptance: the gate is `accepted_at <= decision timestamp` (join `sec_filings` on `accession_number`), or `filed_date + 1 trading day`. A Form 4 accepted at 21:30 UTC on day t is absent at close t.
- Blocked-by / Blocks: none
- Commits: fd8e1ca. Acceptance from sec_filings, one row per accession (max accepted_at; a filing repeats once per issuer ticker). Live: 99.7% of NVDA insider rows and 54.9% of AAPL company-fact rows move to the next session; NVDA files Form 4s in the evening.

### AT-10 Company facts drop period columns and mix 3-month, YTD, annual and restated values
- Status: done
- Priority: must (M6)
- Gate: data
- Raised by: agentic-trading-c5 audit 2026-10-27 (D-M19)
- Evidence: [M] `lakehouse_loader.py:706` returns only `concept, value, fiscal_period, filed`. AAPL diluted EPS as of 2024-06-30: 308 rows; the last 12 read 4.82, 4.67, 1.26, 5.61, ... (9-month, 3-month and annual mixed). Lake key collision (read `_log`).
- Acceptance: the loader returns the period columns and form and keeps the latest `filed_date <= as_of` per (concept, unit, period_start, period_end). The analyst uses discrete quarters. AAPL's trailing 8 quarterly diluted EPS as of 2024-06-30 equal the reported 3-month figures.
- Blocked-by / Blocks: with LK-13
- Commits: 45daf84. Reads sec_company_facts_log de-duplicated until LK-13 lands; latest available filing per (concept, unit, period); the analyst uses discrete 80-100-day quarters with Q4 derived as annual minus Q1-Q3 (approximate for EPS). Verified live: AAPL's trailing 8 quarters as of 2024-06-30 are 1.20, 1.29, 1.88, 1.52, 1.26, 1.47, 2.18, 1.53.

### AT-11 Insider loader collapses distinct lines of one filing
- Status: done
- Priority: must (M6)
- Gate: data
- Raised by: agentic-trading-c5 audit 2026-10-27 (D-M20)
- Evidence: [M] `lakehouse_loader.py:803-806` partitions on the lake's defective key. NVDA: 1,867 loader rows vs 3,793 distinct, 121M vs 277M shares. AAPL 1,346 vs 1,874. JPM 2,106 vs 2,465.
- Acceptance: all distinct lines are kept (dedup on the full row or the line number). NVDA's row count equals the distinct-economic-row count.
- Blocked-by / Blocks: none
- Commits: fd8e1ca. Reads sec_insider_transactions_log de-duplicated over every column except scraped_at, dt, table until LK-14 lands. Verified live: NVDA 3,796 rows, 276.7M shares.

### AT-12 Dividend signal on mixed bases: splits and specials read as cuts
- Status: done
- Priority: must (M6)
- Gate: data
- Raised by: agentic-trading-c5 audit 2026-10-27 (D-M21)
- Evidence: [M] `lakehouse_loader.py:312-352` reads `dolthub_stock_dividends`. AAPL 0.82 to 0.205 (-75%) from the 4:1 split. COST special 15.00 makes the next 1.16 read as a -92% cut. XOM 2024-05 is missing.
- Acceptance: the source is `equity_distributions`, regular cash only, on one split-consistent basis. AAPL's change on 2020-08-07 is 0% and COST's special is excluded.
- Blocked-by / Blocks: none
- Commits: 45daf84. Verified live: AAPL 2020-08-07 to 2020-11-06 unchanged at 0.205; COST's 2023-12-27 special excluded; XOM 2024-05-14 present.

### AT-13 Survivorship: members without prices vanish; no delisting returns
- Status: open
- Priority: must (M6)
- Gate: data (moves to edgelab if AT-77 is approved; see Note)
- Raised by: agentic-trading-c5 audit 2026-10-27 (D-M14)
- Evidence: [M] 334 of 499 members of 2010-06-30 have a price over the next year (bars table: 465). FB is missing 2013-22. `equity_delistings` has a return on 32 of 620 rows (0 of 33 bankruptcies). 10 members sit inside a flagged price range on 2010-06-30.
- Acceptance: the 2010-06-30 universe has at least 465 priced members and no member inside a flagged range. A bankrupt 2010 member books its delisting return. Prices come from `equity_daily_from_bars` x factors, spliced to `instrument_prices` after the last bar.
- Blocked-by / Blocks: LK-20 (`delisting_class` from Form 25 / 8-K items, terminal dates, SWN). Interim on the 170 classified rows.
- Note: moves to edgelab (universe and delisting convention); filed as an EL issue if edgelab lacks it. Convention in config (Shumway 1997): -30% for performance or unknown delistings with no return; report 0 / -30% / -100%, with and without the unknown bucket (299 of 458 member delistings, none with a return). A name is held only while a member and is closed at the next open after removal, only where `instrument_prices` has a regular-session bar after the effective date. `reason` acquired or merged is a merger, bankrupt is performance, everything else is unknown. A name halted at or before removal (SIVB 2023-03-09) is a delisting while a member. The exit keys on the effective date.
- Commits: none

### AT-14 Run calendar is the intersection of all assets
- Status: open
- Priority: must (M6)
- Gate: data (moves to edgelab if AT-77 is approved; see Note)
- Raised by: agentic-trading-c5 audit 2026-10-27 (D-M9, X-M7)
- Evidence: [M] `data/loader.py:59-60` drops rows where any ticker lacks a close. BTC 2023 log return sums to 0.619 vs true 0.926; BTC 2024 0.5957 vs 0.7315. Adding COIN cuts a 2020 run to start 2021-04-14 with no warning.
- Acceptance: outer join on a master calendar (the lake calendar if it has one). BTC 2023 aligned `log_return` sums to 0.926 +/- 1e-9. [AAPL, COIN] keeps AAPL rows from 2016-12-27.
- Blocked-by / Blocks: EL-7 (mixed-book test: 24/7 crypto with the XNYS calendar)
- Note: moves to edgelab (calendar).
- Commits: none

### AT-15 Crypto daily close is about 4 hours after the US close
- Status: open
- Priority: must (M6)
- Gate: data
- Raised by: agentic-trading-c5 audit 2026-10-27 (D-M10)
- Evidence: [M] Yahoo BTC close is at 23:59 UTC. Median error 3.4 bp against 23:59 UTC and 39 bp against the US close. BTC moved 2.9% between the two on 2023-06-14. FX closes are believed to be London evening [I].
- Acceptance: every decision price is sampled at the decision timestamp (BTC at 16:00 America/New_York from `crypto_klines_1m`, or crypto excluded). BTC's 2023-06-14 decision price is within 5 bp of 25,874.6.
- Blocked-by / Blocks: none
- Commits: none

### AT-16 Cross-sectional and factor regressions align series by position, not date
- Status: open
- Priority: must (M6)
- Gate: data
- Raised by: agentic-trading-c5 audit 2026-10-27 (D-M15)
- Evidence: [C] `agents/analysts/factor_pod/fama_macbeth_analyst.py:118-128,180-183`; `fama_french_factor_analyst.py:87-101`; `network_centrality_analyst.py:91-92,173-181`. A frozen factor table ends weeks before as_of and shifts every pairing.
- Acceptance: every regression uses a date-indexed inner join. A test with a 1-day hole in one ticker gives loadings equal to the date-joined computation.
- Blocked-by / Blocks: none
- Commits: none

### AT-17 Option spot is spin-off/dividend adjusted while strikes are as quoted
- Status: done
- Priority: must (M6)
- Gate: data
- Raised by: agentic-trading-c5 audit 2026-10-27 (D-M6)
- Evidence: [M] `lakehouse_loader.py:509,556-558` and `_market_math.py:289-301`. GE 2023-01-03: option spot 52.94 vs as-traded 84.97 (-38%). T 2022-04-08: 18.23 vs 24.14.
- Acceptance: option spot is the as-traded `equity_daily_from_bars` `official_close` (else `close`); strikes stay as quoted. GE's spot on 2023-01-03 is within 0.1% of 84.98.
- Blocked-by / Blocks: none
- Commits: 99edc18 (AT-1). Resolved on the opposite basis to the acceptance: strikes are rescaled onto Yahoo's close basis, which counts spin-offs Yahoo writes as fractional splits, and `raw_spot` reads that same Yahoo close. Verified live: GE 2023-01-03 median strike 52.33 against spot 52.94; T 2022-04-08 18.13 against 18.23. Both sides scale by one factor, so moneyness and implied volatility are unchanged.

### AT-18 CVaR analysts choose ATM against the dividend-adjusted close
- Status: open
- Priority: must (M6)
- Gate: data
- Raised by: agentic-trading-c5 audit 2026-10-27 (D-M7)
- Evidence: [M] `black_scholes_valuation_analyst.py:75`, `greeks_delta_hedge_analyst.py:97`, `heston_vol_surface_analyst.py:115` pass `adj_close`; `_options_data.py:74` picks the nearest strike. `adj_close/close` on 2020-01-02: XOM 0.742, JPM 0.836, AAPL 0.962.
- Acceptance: analysts use the same spot basis as the strikes (AT-17). XOM's ATM strike on 2020-01-02 is within one strike step of the as-traded close 70.90.
- Blocked-by / Blocks: depends on AT-17
- Commits: none

### AT-19 A held option never matches its post-split strike after a non-integer split
- Status: open
- Priority: must (M6)
- Gate: data
- Raised by: agentic-trading-c5 audit 2026-10-27 (D-M5)
- Evidence: [M] exact strike match at `lakehouse_loader.py:876` and `option_marks.py:162`. TSLA 3-for-1: listed strike 208.33, rescaled 208.333333, so the held contract falls back to a stale IV.
- Acceptance: contract identity carries across a split (as-quoted OSI symbol plus an adjustment record, or a half-cent match). A TSLA call opened on 2022-08-24 gets a quoted, non-model mark on 2022-08-26. Superseded by the close/reopen pair rule.
- Blocked-by / Blocks: none
- Commits: none

### AT-20 Next-open option fills use that day's end-of-day chain quote
- Status: open
- Priority: must (M6)
- Gate: data (moves to edgelab if AT-77 is approved; see Note)
- Raised by: agentic-trading-c5 audit 2026-10-27 (D-M8, X-M9)
- Evidence: [C] `backtest/engine.py:589-592` calls `load_option_mark(contract, t+1, open_spot)`; `option_marks.py:256-279` uses the t+1 snapshot as quoted. DoltHub chain rows are end-of-day [I].
- Acceptance: an open-time mark uses only snapshots dated before the fill day, re-marked at the open spot. With snapshots on t and t+1 at different mids, the t+1 open fill prices off t.
- Blocked-by / Blocks: none
- Note: moves to edgelab (option fill quotes). The quote-timing part is data under any engine.
- Commits: none

### AT-21 Option fills ignore `tradeable` and fill on quotes up to 45 days old
- Status: open
- Priority: must (M6)
- Gate: data (moves to edgelab if AT-77 is approved; see Note)
- Raised by: agentic-trading-c5 audit 2026-10-27 (X-M10)
- Evidence: [C] `backtest/portfolio.py:164-185` checks expiry, mid, bid, ask and minimum premium only. `option_marks.py:256-262` marks quotes older than 10 days `tradeable=False`, but model bid/ask stay positive.
- Acceptance: `apply_option_decision` holds when `not mark.tradeable`. A 20-day-old quote returns action "hold" and leaves cash unchanged.
- Blocked-by / Blocks: none
- Note: the design note lists AT-21 under both "survive" and "moves to edgelab". It is filed as a move (option fill quotes).
- Commits: none

### AT-22 Futures traded as one cash security
- Status: open
- Priority: must (M6)
- Gate: held
- Raised by: agentic-trading-c5 audit 2026-10-27 (X-M5, D-M11)
- Evidence: [M] `backtest/portfolio.py:52-113,304-324`; `data/ticker_catalog.py:169-176`. No dated contracts, multiplier, margin or costed roll; equity borrow charged on shorts. CL=F 2015-19 log return +0.147 vs roll-adjusted -0.442.
- Acceptance: futures come from `futures_continuous_daily.ratio_adjusted_close` (or contracts with an explicit roll). P&L is excess return plus the collateral rate, with no borrow. CL 2015-2019 log return equals -0.442 +/- 0.005. A constant-price future shows zero P&L and zero financing.
- Blocked-by / Blocks: EL-7 (futures margin unverified in edgelab)
- Note: moves to edgelab (futures).
- Commits: none

### AT-23 FX: no carry; USDJPY quoted in JPY used as USD; COT sign inverted for JPY
- Status: open
- Priority: must (M6)
- Gate: held (COT sign: data)
- Raised by: agentic-trading-c5 audit 2026-10-27 (D-M12)
- Evidence: [M] `instruments.currency` is JPY for `USDJPY=X`; the engine books the price as USD (`backtest/engine.py:490-493`). `agents/analysts/.../cot_positioning_analyst.py:58` maps to CFTC 097741 (JPY futures), so the signal is inverted. USD/JPY carry was about 5%/yr in 2023.
- Acceptance: FX returns are spot plus point-in-time rate differential, P&L converted to USD, the COT series for `USDJPY=X` negated. Long USDJPY over 2023 earns spot plus carry within 50 bp of a forward-points benchmark.
- Blocked-by / Blocks: EL-7 (mixed-book test for FX and crypto)
- Note: FX carry and conversion move to edgelab. The COT JPY sign stays here.
- Commits: none

### AT-24 ADV cap and impact use the wrong volume units
- Status: open
- Priority: must (M6)
- Gate: held
- Raised by: agentic-trading-c5 audit 2026-10-27 (X-M6, D-M13)
- Evidence: [M] `backtest/engine.py:804-824`, `portfolio.py:72-86`. For a $1M CL trade: 34.2 bps vs correct 1.1; cap $1.7M vs $1.7B. BTC volume is in USD (about 27,000x off). GC=F volume is in contracts (100x). `adj_close/close` is 0.235 for T in 2006.
- Acceptance: ADV is dollar volume on one basis (as-traded volume x as-traded price; futures x multiplier). Unit tests reproduce the correct figures for CL, GC, ES and BTC. A $1M T order on 2006-01-03 shows the same participation from raw and adjusted data.
- Blocked-by / Blocks: EL-7 (futures volume cap)
- Note: moves to edgelab (ADV).
- Commits: none

### AT-25 lag=1: open fill sized with that day's closing NAV
- Status: open
- Priority: must (M6)
- Gate: held
- Raised by: agentic-trading-c5 audit 2026-10-27 (X-M1)
- Evidence: [V] `backtest/engine.py:556` feeds close-based NAV into `apply_decision` (`portfolio.py:73`). Probe P1: final NAV 29,999,589 vs 20,000,000.
- Acceptance: probe P1 ends at NAV 20,000,000 with 100,000 shares. The NAV passed to `apply_decision` under lag=1 is open-marked.
- Blocked-by / Blocks: none
- Note: Pending owner approval of AT-77: deleted with the engine if approved.
- Commits: none

### AT-26 lag=0: signal credited with the return it already saw
- Status: open
- Priority: must (M6)
- Gate: held
- Raised by: agentic-trading-c5 audit 2026-10-27 (X-M2)
- Evidence: [M] probe P4. `backtest/engine.py:635` and `:510-511`: the record dated t carries the close t-1 to close t return; the fill is at the signal's own close. Feeds IC, Black-Litterman views and Hedge weights.
- Acceptance: with lag=0 the record dated t carries the return from close t to close t+1 (or lag=0 is removed). A signal equal to sign(today's return) shows IC near 0 on i.i.d. returns, not near 0.8.
- Blocked-by / Blocks: none
- Note: Pending owner approval of AT-77: deleted with the engine if approved.
- Commits: none

### AT-27 Recorded NAV is pre-fill; last day's fills never reach NAV
- Status: open
- Priority: must (M6)
- Gate: held
- Raised by: agentic-trading-c5 audit 2026-10-27 (X-M11)
- Evidence: [V] `backtest/engine.py:556-575` records NAV before fills at `:586-634`. Probe P2: 10.0M recorded vs 11.0M true.
- Acceptance: the NAV for day t is cash + positions at close t + option marks, after day-t fills and costs. The P2 test records 11,000,000.
- Blocked-by / Blocks: none
- Note: Pending owner approval of AT-77: deleted with the engine if approved.
- Commits: none

### AT-28 lag=1 ADV/vol window includes the fill day's volume and close
- Status: open
- Priority: must (M6)
- Gate: held
- Raised by: agentic-trading-c5 audit 2026-10-27 (X-M12)
- Evidence: [C] `backtest/engine.py:623-625,816` slices `loc[:as_of_date]` with the fill day.
- Acceptance: the window ends at t-1 for an open fill on t. A volume spike on the fill day leaves the cap unchanged.
- Blocked-by / Blocks: none
- Note: Pending owner approval of AT-77: deleted with the engine if approved.
- Commits: none

### AT-29 A missing open aborts every lag=1 run spanning it
- Status: open
- Priority: must (M6)
- Gate: held
- Raised by: agentic-trading-c5 audit 2026-10-27 (X-M8)
- Evidence: [V] probe P3: a NaN open raises a pydantic `ValidationError` from `TradeRecord`. The lake has CL=F 2020-04-21 open -14 and WBD 2026-10-05 open 0.
- Acceptance: an invalid open leaves the order unfilled (or fills at a documented fallback) and logs it. P3 completes with a finite NAV history.
- Blocked-by / Blocks: none
- Note: Pending owner approval of AT-77: deleted with the engine if approved.
- Commits: none

### AT-30 No interest on cash or short proceeds; short-option margin charged as borrowing
- Status: open
- Priority: must (M6)
- Gate: held
- Raised by: agentic-trading-c5 audit 2026-10-27 (X-M3, X-M4, X-M16)
- Evidence: [M] `backtest/portfolio.py:302-331`. P5: 10M cash, rf 5%, 365 days gives 0 interest. P6: 100 short SPY puts are charged $71,500/yr on margin. rf is 0 in the single-phase CLI and dashboard (`cli.py:365-375`, `gui/dashboard.py:546-556`).
- Acceptance: a flat book with cash C ends at C x (1 + r x d/365); short proceeds earn rf minus borrow. Short-option margin with positive cash costs 0. CLI, dashboard and experiment pass the DGS3MO series.
- Blocked-by / Blocks: none
- Note: moves to edgelab (cash interest); filed as an EL issue if edgelab lacks it. The financing part of AT-31 is here.
- Commits: none

### AT-31 Sharpe with rf=0 in the single-phase CLI and dashboard
- Status: open
- Priority: must (M6)
- Gate: delete
- Raised by: agentic-trading-c5 audit 2026-10-27 (X-M16)
- Evidence: [C] `cli.py:365-375` and `gui/dashboard.py:546-556` build `BacktestEngine` without `risk_free_rate_series`. Report only.
- Acceptance: the metric is deleted. The financing part is tracked in AT-30.
- Blocked-by / Blocks: none
- Note: Pending owner approval of AT-77: deleted with the engine if approved.
- Commits: none

### AT-32 ^GSPC price-index benchmark and information ratio
- Status: open
- Priority: must (M6)
- Gate: delete
- Raised by: agentic-trading-c5 audit 2026-10-27 (X-M17)
- Evidence: [C] `cli.py:46-48,516-548`; `backtest/engine.py:759-765`. ^GSPC excludes dividends (about 1.3-1.5%/yr), so IR is biased up. Report only.
- Acceptance: the benchmark and information ratio are deleted from the report.
- Blocked-by / Blocks: none
- Note: Pending owner approval of AT-77: deleted with the engine if approved.
- Commits: none

### AT-33 Per-source Sharpe and returns pool asset-days
- Status: open
- Priority: must (M6)
- Gate: delete
- Raised by: agentic-trading-c5 audit 2026-10-27 (X-M18)
- Evidence: [V] `backtest/engine.py:290-344` writes `source_performance` with several records per date, annualised as one observation per day. No decision reads it. `PerformanceScoreboard.sharpe_ratio` and `sortino_ratio` have no caller.
- Acceptance: the per-source Sharpe, Sortino and annualised-return code is deleted.
- Blocked-by / Blocks: none
- Note: Pending owner approval of AT-77: deleted with the engine if approved.
- Commits: none

### AT-34 Calibration state leaks into the test block; pod history trimmed to 90 days
- Status: open
- Priority: must (M6)
- Gate: held
- Raised by: agentic-trading-c5 audit 2026-10-27 (X-M14)
- Evidence: [C] one `Commander` for both phases (`cli.py:493-529`). Breaker trips and pod weights carry over; `pod_returns_lookback_days` = 90 calendar days (`commander.py:141,157`, `config.py:247`).
- Acceptance: `test.pod_return_history` covers exactly test_start to test_end for every pod. Day-1 test pod returns are not booked from calibration weights.
- Blocked-by / Blocks: none
- Note: Pending owner approval of AT-77: deleted with the engine if approved.
- Commits: none

### AT-35 Resume fingerprint is dates+assets only; manager state not checkpointed
- Status: open
- Priority: must (M6)
- Gate: held
- Raised by: agentic-trading-c5 audit 2026-10-27 (X-M13, R-M6)
- Evidence: [C] `backtest/engine.py:215-221,442-444,684`. Lag, commission, `--set` overrides and roster are not checked. Only Commander state is restored. A resume replaces the track record.
- Acceptance: an interrupted-and-resumed run gives a byte-identical `BacktestResult` to an uninterrupted one. A resume with any changed setting raises.
- Blocked-by / Blocks: none
- Note: Pending owner approval of AT-77: deleted with the engine if approved.
- Commits: none

### AT-36 `--parallel-backend process` selectable; silently drops manager state
- Status: open
- Priority: must (M6)
- Gate: held
- Raised by: agentic-trading-c5 audit 2026-10-27 (X-M15, R-M5)
- Evidence: [C] `cli.py:249-252,404-406`; `agents/commander.py:641` builds a `ProcessPoolExecutor`. Managers are pickled per call, so held legs, pairs and refit caches never persist.
- Acceptance: the CLI rejects `process` and the help text no longer advertises it. A CLI test asserts rejection.
- Blocked-by / Blocks: none
- Note: Pending owner approval of AT-77: deleted with the engine if approved.
- Commits: none

### AT-37 No run provenance
- Status: done
- Priority: must (M2, M6)
- Gate: data
- Raised by: agentic-trading-c5 audit 2026-10-27 (R-M1)
- Evidence: [C] the `experiment` JSON at `cli.py:639-660` holds only assets, dates, warm-up, the user's exclusions and `--set` deltas. No git sha, quantcore sha, full settings, roster, seeds, lake T or version. `--yes` auto-drops analysts that fail preflight.
- Acceptance: `run` and `experiment` output a `provenance` block: git sha and dirty flag, quantcore sha, `settings.model_dump()`, all CLI args, final analyst and manager ids, seeds, and a lake fingerprint. A test asserts the keys.
- Blocked-by / Blocks: none
- Note: the design note replaces this with the registry and ledger (section 8) if AT-77 is approved.
- Commits: 9ed37ed. Seeds are constants in analyst code (42), so the git sha pins them; no separate seeds field.

### AT-38 `experiment` ignores `--set firm.commission_rate/slippage_bps/execution_lag_days` but records them
- Status: open
- Priority: must (M6)
- Gate: held (the experiment CLI is deleted with the engine if AT-77 is approved)
- Raised by: agentic-trading-c5 audit 2026-10-27 (R-M2)
- Evidence: [V] Typer defaults bind to `settings.firm.*` at import (`cli.py:407-422`); overrides apply at `:478`. `firm.max_leverage` is read at schema import before overrides.
- Acceptance: option defaults are `None`, resolved from `settings` after overrides. `experiment --set firm.commission_rate=0.003` runs the portfolio at 0.003. `--set firm.max_leverage` fails loudly or is documented unsupported.
- Blocked-by / Blocks: none
- Note: Pending owner approval of AT-77: deleted with the engine if approved.
- Commits: none

### AT-39 Monte Carlo seeds hard-coded 42 in four analysts
- Status: open
- Priority: must (M6)
- Gate: data
- Raised by: agentic-trading-c5 audit 2026-10-27 (R-M3)
- Evidence: [C] `seed=42` in `monte_carlo_valuation_analyst.py:80`, `gbm_momentum_analyst.py:63`, `cir_rate_regime_analyst.py:112`, `ou_mean_reversion_analyst.py:122`. Same seed for every asset and day, not in `config.py`, not recorded.
- Acceptance: `settings.analysts.mc_seed`, derived per (asset, date, analyst) from a base seed. The base seed is in the provenance block. `KNOWN_LIMITATIONS.md` lists all four analysts.
- Blocked-by / Blocks: none
- Commits: none

### AT-40 No trial ledger; `experiment` disables logging
- Status: open
- Priority: must (M6)
- Gate: data
- Raised by: agentic-trading-c5 audit 2026-10-27 (R-M4, EL-4); ledger format agreed with pitch-edge-8b 2026-10-28
- Evidence: [C] `cli.py:477` calls `configure_logging(disable=True)`. Nothing counts `--set`, `--exclude-analyst` or re-runs. 9 prior run directories reuse test windows after reading results; calibration is 3 months against an 8-year iterated test window.
- Acceptance: implement the shared ledger format (hash chain from GENESIS, canonical row JSON, row keys, registry file hash). The two conformance vectors hash byte-exact: registrations `13c98f3cdd15afc3fd9573cd933721ee65a81746ae35992fe81d064f361d3894`, then trials `1887978452d3a9f6fde6560cf0fafbd63acf36bf81a3b20ae06dd1eb2ad1d623`. The test-writer writes them before the implementer starts. Also: variant hash, grid registered before a family, `prior_exposure`.
- Blocked-by / Blocks: paired with PE-21 (shared ledger format)
- Commits: none

### AT-41 Docs state false things about numbers; CVaR results since 2026-09-18 not flagged invalid
- Status: open
- Priority: must (M6)
- Gate: data
- Raised by: agentic-trading-c5 audit 2026-10-27 (R-M7, R-M8, R-M9, R-S6, R-S7, X-S1)
- Evidence: [C] `KNOWN_LIMITATIONS.md` calls the split fix "fixed" (AT-1). Starting cash `10_000` vs 10,000,000 in config. "No market-impact model" vs square-root impact in code. Hedge tenor 30 vs 60 days. `CLAUDE.md` and `README.md` say FRED gates on vintage and name `LAKE_DIR`; the code reads `LAKEHOUSE_DIR`.
- Acceptance: each statement matches `config.py` and `backtest/portfolio.py` value for value. Results from options on splitting names are invalidated. `KNOWN_LIMITATIONS.md` and `CHANGELOG.md` are replaced by a current limitations document derived from this list.
- Blocked-by / Blocks: the untracked `CHANGELOG.md` and `KNOWN_LIMITATIONS.md` stay uncommitted until this is done
- Commits: none

### AT-42 Sizing assumes 5 bps; engine charges at least 10 bps plus impact
- Status: open
- Priority: should
- Gate: held
- Raised by: agentic-trading-c5 audit 2026-10-27 (X-S2)
- Evidence: [C] `config.py:288-291` vs `backtest/portfolio.py:87-92`. The pod band compares to the previous target, not holdings (`portfolio_construction.py:574-594`). The engine re-trades any non-zero delta; probe P1 shows a 50,002-share trade at an unchanged weight.
- Acceptance: one cost model shared by sizing and fills. The engine skips orders below a band measured against current holdings. Turnover on a constant-target, drifting-price test is 0.
- Blocked-by / Blocks: none
- Note: Pending owner approval of AT-77: deleted with the engine if approved.
- Commits: none

### AT-43 `total_transaction_cost` reports commission only
- Status: open
- Priority: should
- Gate: delete
- Raised by: agentic-trading-c5 audit 2026-10-27 (X-S3)
- Evidence: [C] `backtest/portfolio.py:90` stores commission only; slippage and impact sit in the fill price; `engine.py:797` reports the sum as total cost. Report only.
- Acceptance: the field is deleted.
- Blocked-by / Blocks: none
- Note: Pending owner approval of AT-77: deleted with the engine if approved.
- Commits: none

### AT-44 No spread model; flat 50 bps borrow; no locate
- Status: open
- Priority: should
- Gate: held
- Raised by: agentic-trading-c5 audit 2026-10-27 (X-S4)
- Evidence: [C] `config.py:101,106`. One borrow rate for TSLA/COIN shorts and SPY alike; no hard-to-borrow table.
- Acceptance: a per-name borrow rate and availability input (a tiered default is enough), and a spread term scaled by price and liquidity.
- Blocked-by / Blocks: none
- Note: Pending owner approval of AT-77: deleted with the engine if approved.
- Commits: none

### AT-45 No margin or buying-power constraint
- Status: open
- Priority: should
- Gate: held
- Raised by: agentic-trading-c5 audit 2026-10-27 (X-S5)
- Evidence: [C] `backtest/portfolio.py:91-92` lets cash go arbitrarily negative. Short options post no initial margin; no maintenance call.
- Acceptance: orders that breach a Reg-T or portfolio-margin requirement are cut. A target of 10x gross is reduced to the limit.
- Blocked-by / Blocks: none
- Note: Pending owner approval of AT-77: deleted with the engine if approved.
- Commits: none

### AT-46 Firm limits skip the hedge book and option gamma/vega; checked on targets only
- Status: open
- Priority: should
- Gate: held
- Raised by: agentic-trading-c5 audit 2026-10-27 (X-S6)
- Evidence: [C] `agents/commander.py:756-802` lets the tail-hedge book bypass gross, VaR, beta and stress. Options enter limits as delta only (`:525-557`), so a delta-hedged short straddle shows zero risk.
- Acceptance: firm beta and VaR include the hedge. Option rows contribute gamma/vega stress. A short straddle registers a non-zero stress loss.
- Blocked-by / Blocks: none
- Note: survives AT-77 (design note section 7); the gate stays held until the owner decides.
- Commits: none

### AT-47 Stress test can lower a hedged book's loss
- Status: open
- Priority: should
- Gate: held
- Raised by: agentic-trading-c5 audit 2026-10-27 (X-S7)
- Evidence: [C] `agents/commander.py:486-494` pulls correlations to +/-0.9 in their own sign. Long A / short B at rho 0.5, sigma 1%: base sigma_p 1.00%, stressed (sigma 2%, rho 0.9) 0.89%.
- Acceptance: stress scenarios include hedge failure (rho to 0 or a sign flip for opposite legs). The two-asset example gives a stressed loss above the base loss.
- Blocked-by / Blocks: none
- Note: survives AT-77 (design note section 7); the gate stays held until the owner decides.
- Commits: none

### AT-48 Pod returns, breaker and Kelly run on a paper book, not actual P&L
- Status: open
- Priority: should
- Gate: held
- Raised by: agentic-trading-c5 audit 2026-10-27 (X-S8)
- Evidence: [C] `agents/commander.py:143-157` uses target weights x close-to-close log return and ignores scaling, partial fills, costs and financing. Same in `portfolio_construction.py:248-265` (Kelly).
- Acceptance: per-pod P&L from actual fills and marks. Pod P&L plus unallocated financing equals the daily NAV change to 1e-6.
- Blocked-by / Blocks: EL-7 (tag = pod_id, P&L by tag)
- Note: the design note solves this with per-tag P&L in edgelab (section 6, item 2).
- Commits: none

### AT-49 lag=1 attribution is close-to-close, not the P&L from the open fill
- Status: open
- Priority: should
- Gate: held
- Raised by: agentic-trading-c5 audit 2026-10-27 (X-S9)
- Evidence: [C] `backtest/engine.py:635` records close t-1 to close t; `attributable_return` is reported as source performance (`:290-344`).
- Acceptance: attribution uses the return actually earned (open to close on the fill day, then close to close), or the field is relabelled as a signal score.
- Blocked-by / Blocks: none
- Note: Pending owner approval of AT-77: deleted with the engine if approved.
- Commits: none

### AT-50 No early exercise or assignment; q=0 European re-marks; stale marks jump to intrinsic
- Status: open
- Priority: should
- Gate: held
- Raised by: agentic-trading-c5 audit 2026-10-27 (X-S10)
- Evidence: [C] `backtest/portfolio.py:251-284` cash-settles at expiry only. `option_marks.py:167-209` uses European Black-Scholes with q=0. Marks with no quote for 45+ days hold the last mid (`portfolio.py:344-348`).
- Acceptance: the re-mark includes dividend yield (American adjustment for single names). A short ITM call held over an ex-date is assigned in a test.
- Blocked-by / Blocks: none
- Note: moves to edgelab (exercise).
- Commits: none

### AT-51 252 hard-coded instead of `trading_days_per_year`
- Status: open
- Priority: should
- Gate: held
- Raised by: agentic-trading-c5 audit 2026-10-27 (X-S11)
- Evidence: [C] `backtest/engine.py:317,321,733-741`, `scoring/scoreboard.py:70`, `backtest/baselines.py:19` ignore `settings.firm.trading_days_per_year` (`config.py:88`).
- Acceptance: every annualisation reads the setting. `--set firm.trading_days_per_year=365` changes reported Sharpe by sqrt(365/252).
- Blocked-by / Blocks: none
- Note: survives AT-77 (design note section 7); the gate stays held until the owner decides.
- Commits: none

### AT-52 Blotter stamps a fill with its decision date
- Status: open
- Priority: should
- Gate: held
- Raised by: agentic-trading-c5 audit 2026-10-27 (X-S12)
- Evidence: [V] `backtest/portfolio.py:102,152,236` use `decision.as_of_date`. Under lag=1 a fill on t is stamped t-1 (probe P1).
- Acceptance: `TradeRecord` carries both the decision date and the fill date.
- Blocked-by / Blocks: none
- Note: Pending owner approval of AT-77: deleted with the engine if approved.
- Commits: none

### AT-53 Isolated-analyst baseline has unbounded gross
- Status: open
- Priority: should
- Gate: held
- Raised by: agentic-trading-c5 audit 2026-10-27 (X-S13)
- Evidence: [C] `backtest/baselines.py:116-140` sets each asset to +/-confidence with no gross normalisation; 20 assets give up to 20x gross.
- Acceptance: isolated runs are gross-normalised or vol-targeted to the main book's level.
- Blocked-by / Blocks: none
- Note: Pending owner approval of AT-77: deleted with the engine if approved.
- Commits: none

### AT-54 IC is pooled Pearson with n = asset-days
- Status: open
- Priority: should
- Gate: held
- Raised by: agentic-trading-c5 audit 2026-10-27 (X-S14)
- Evidence: [C] `scoring/scoreboard.py:139-148,31-34`. About 5,000 heavily cross-correlated "observations" for 20 assets x 250 days. IC feeds Grinold sizing (`agents/base.py:305`, `portfolio_construction.py:443`), so it is kept.
- Acceptance: IC is the mean of daily cross-sectional rank ICs; shrinkage uses the number of dates.
- Blocked-by / Blocks: none
- Note: survives AT-77 (design note section 7); the gate stays held until the owner decides.
- Commits: none

### AT-55 Hedge weight exponent grows with universe size
- Status: open
- Priority: should
- Gate: held
- Raised by: agentic-trading-c5 audit 2026-10-27 (X-S15)
- Evidence: [C] `scoring/weighting.py:57-66`: exp(eta x sum of attributable return over all asset-days, 30 days). A 20-asset source has about 20x the exponent of a single-asset source.
- Acceptance: cumulative return per source is aggregated per date and normalised by gross signal exposure.
- Blocked-by / Blocks: none
- Note: survives AT-77 (design note section 7); the gate stays held until the owner decides.
- Commits: none

### AT-56 Stat-arb legs not sized to the hedge ratio
- Status: open
- Priority: should
- Gate: held
- Raised by: agentic-trading-c5 audit 2026-10-27 (X-S16)
- Evidence: [C] `agents/managers/stat_arb_manager.py:206-213` passes per-asset views to the generic constructor; the cointegrating ratio is not used for leg sizing (net cap 0.5, `config.py:466-469`).
- Acceptance: pair legs are sized at beta x dollar ratio. Pod net beta to each pair is near 0 in a test.
- Blocked-by / Blocks: none
- Note: survives AT-77 (design note section 7); the gate stays held until the owner decides.
- Commits: none

### AT-57 Price levels shown to agents include later splits and dividends
- Status: open
- Priority: must (M6)
- Gate: data
- Raised by: agentic-trading-c5 audit 2026-10-27 (D-M24)
- Evidence: [M] `lakehouse_loader.py:505-523` takes the latest-scrape `adj_close`. 349 AAPL dates differ across scrapes. Levels such as ATM strikes, share counts and ADV embed corporate actions after as_of. Returns are unaffected.
- Acceptance: levels shown to agents are as-traded or as-of-adjusted. Only returns use the total-return series.
- Blocked-by / Blocks: none
- Commits: none

### AT-58 Membership counts the removal day; renames read as new names
- Status: open
- Priority: must (M6)
- Gate: data
- Raised by: agentic-trading-c5 audit 2026-10-27 (D-M25)
- Evidence: [M] `lakehouse_loader.py:105-106` uses `end_date >= as_of`. On 2022-06-09 both FB and META are returned. No `security_id` join, so a rename is an exit plus an entry.
- Acceptance: membership is `start_date <= as_of < end_date`, keyed by `security_id`, with the lake's rename rule. 2022-06-09 returns one Meta entry.
- Blocked-by / Blocks: none
- Commits: none

### AT-59 Same-day DoltHub chain despite a 1-4 day publication lag
- Status: open
- Priority: must (M6)
- Gate: data
- Raised by: agentic-trading-c5 audit 2026-10-27 (D-S1)
- Evidence: [I] `lakehouse_loader.py:210-217` and `option_marks.py:256` use a snapshot dated t at close t. The lake says DoltHub publishes 1 to 4 days later.
- Acceptance: a configurable `option_publication_lag_days`, default 1.
- Blocked-by / Blocks: none
- Commits: none

### AT-60 Stale-IV marks up to 45 days; silent 0.0 rate
- Status: open
- Priority: should
- Gate: data
- Raised by: agentic-trading-c5 audit 2026-10-27 (D-S2)
- Evidence: [I] `option_marks.py:256-262`, `config.py:149,154`: positions are tradeable up to 10 days old and marked up to 45. `option_marks.py:85-90` returns a 0.0 rate when DGS3MO is missing, without a warning.
- Acceptance: IV is interpolated from a surface or the mark is flagged unusable after a few sessions. A missing rate raises or marks the signal `unavailable`.
- Blocked-by / Blocks: none
- Commits: none

### AT-61 Loader errors silently become "no coverage"
- Status: done
- Priority: must (M6)
- Gate: data
- Raised by: agentic-trading-c5 audit 2026-10-27 (D-S3)
- Evidence: [C] every loader wraps its query in `except Exception: ... return None` (`lakehouse_loader.py:108-113,147-152,220-225,298-303`). A schema or SQL error becomes `status="unavailable"` instead of `error`.
- Acceptance: query errors propagate, or return a typed error result that maps to `status="error"`.
- Blocked-by / Blocks: none
- Commits: 3ff8eff

### AT-62 SIC is a current mapping used historically
- Status: open
- Priority: should
- Gate: data
- Raised by: agentic-trading-c5 audit 2026-10-27 (D-S4)
- Evidence: [I] `lakehouse_loader.py:419-478` and `_pairs_screening.py:30`. The lake has no point-in-time rule for SIC, so reclassified names are paired with later knowledge.
- Acceptance: use a point-in-time classification, or document the limit.
- Blocked-by / Blocks: none
- Commits: none

### AT-63 Catalog offers untradeable indices and `JPY=X`
- Status: open
- Priority: should
- Gate: data
- Raised by: agentic-trading-c5 audit 2026-10-27 (D-S5, D-S6)
- Evidence: [M] `data/ticker_catalog.py:166-167,278-279` list ^GSPC and ^RUT; ^GSPC "volume" 3.95e9 becomes ADV. `:180,:286` use `JPY=X`, which the lake lacks (only `USDJPY=X`), so the "Diversified Multi-Asset 20" preset raises at load.
- Acceptance: tradeable proxies (SPY, IWM, ES=F) replace the indices. Every preset loads.
- Blocked-by / Blocks: none
- Commits: none

### AT-64 Preflight checks five sources and ignores the warm-up
- Status: open
- Priority: should
- Gate: data
- Raised by: agentic-trading-c5 audit 2026-10-27 (D-S7)
- Evidence: [I] `backtest/preflight.py:93-138` checks fred, ken_french, SEC facts, FINRA and insider against `start_date`, not the warm-up start. Options, COT, earnings, dividends, membership and SIC are never checked.
- Acceptance: preflight covers every source an analyst reads, from the warm-up start.
- Blocked-by / Blocks: none
- Commits: none

### AT-65 Risk-free series backfilled before its first value
- Status: done
- Priority: must (M6)
- Gate: data
- Raised by: agentic-trading-c5 audit 2026-10-27 (D-S8)
- Evidence: [C] `data/loaders/macro_loader.py:58` uses `.ffill().bfill()` and divides by 252 on a calendar-day index. Days before the first observation take a later value.
- Acceptance: no backfill; days before the first observation are missing, and the annualisation is consistent.
- Blocked-by / Blocks: none
- Commits: dadea7a. No backfill; a start before the first DGS3MO observation raises ValueError naming that date, rather than returning missing days.

### AT-66 Open used for fills with no OHLC sanity check
- Status: open
- Priority: should
- Gate: data
- Raised by: agentic-trading-c5 audit 2026-10-27 (D-S9)
- Evidence: [I] `lakehouse_loader.py:507`. The lake reports about 4,000 rows with inconsistent OHLC; `open` is used with no `low <= open <= high` check.
- Acceptance: rows failing the check are masked, and the count is logged.
- Blocked-by / Blocks: none
- Commits: none

### AT-67 Option expiry can settle at a later day's spot
- Status: open
- Priority: must (M6)
- Gate: data
- Raised by: agentic-trading-c5 audit 2026-10-27 (D-S10)
- Evidence: [C] `option_marks.py:244` settles at the first run day on or after expiry. If the expiry day is missing from the run calendar or is a holiday, the spot is from a later day.
- Acceptance: settlement uses the last close on or before expiry.
- Blocked-by / Blocks: none
- Commits: none

### AT-68 Dividend signal fires on ex-date, weeks after declaration
- Status: open
- Priority: should
- Gate: data
- Raised by: agentic-trading-c5 audit 2026-10-27 (D-S11)
- Evidence: [I] the gate is `ex_date`: safe, but late. Dividends are announced at declaration.
- Acceptance: the signal gates on the declaration date.
- Blocked-by / Blocks: none
- Commits: none

### AT-69 Dead scripts, symbols and test references
- Status: open
- Priority: should
- Gate: data
- Raised by: agentic-trading-c5 audit 2026-10-27 (R-S1, R-S2)
- Evidence: [V] `scripts/build_performance_time_series.py` and `build_raw_performance_time_series.py` read JSON checkpoints; checkpoints are SQLite. `extract_run_summary.py` needs a file the CLI never writes. Unused: `DEFAULT_BASKET`, `SUPPORTED_SERIES`. `tests/unit/gui/test_dashboard.py:928` imports a missing `_synthetic_price_history`. Comments cite files that do not exist (`data/db.py`, `data/cache.py`).
- Acceptance: unused symbols are deleted or used; the dead scripts and the `ijson` dependency are deleted or fixed (an `--output` option); no comment or doc names a non-existent path.
- Blocked-by / Blocks: none
- Commits: none

### AT-70 Unknown `--exclude-analyst` ids silently ignored; run scripts pass removed ids
- Status: open
- Priority: should
- Gate: data
- Raised by: agentic-trading-c5 audit 2026-10-27 (R-S3)
- Evidence: [V] `agents/roster.py:381` filters silently; a nonexistent id returned all 38 analysts. `cli.py:479-485` echoes the ids as applied. The `results/` run scripts exclude `news_sentiment_analyst`, `reddit_sentiment_analyst` and `retail_attention_analyst`, which do not exist. Two scripts compute TEST_END as "yesterday".
- Acceptance: unknown ids raise `BadParameter` listing the valid ids. Run scripts pin TEST_END and use only existing ids.
- Blocked-by / Blocks: none
- Commits: none

### AT-71 History narration in src docstrings and comments
- Status: open
- Priority: should
- Gate: data
- Raised by: agentic-trading-c5 audit 2026-10-27 (R-S4)
- Evidence: [V] `lakehouse_loader.py:37-42,185-189,424,545-548`; `fama_macbeth_analyst.py:180`; `credit_spread_signal_analyst.py:32-34`; `cvar_manager.py:28`; `baselines.py:35`; `engine.py:138`; `portfolio_construction.py:101`; `roster.py:533`; `checkpoint_store.py:92-95`; `track_record.py:13`.
- Acceptance: each is rewritten to current behaviour only. A grep for `confirmed|previously|used to|now (calls|reads)` over `src/` returns nothing.
- Blocked-by / Blocks: none
- Commits: none

### AT-72 quantcore unpinned in pyproject; numpy/scipy undeclared
- Status: open
- Priority: should
- Gate: data
- Raised by: agentic-trading-c5 audit 2026-10-27 (R-S5)
- Evidence: [V] `pyproject.toml:12,26` carry no rev. Only `uv.lock:1614` pins 450a5a2. numpy and scipy are imported but come in only through quantcore. CI clones quantcore from GitHub.
- Acceptance: `rev = "<sha>"` in `[tool.uv.sources]`; numpy and scipy declared directly; the quantcore sha appears in run output (AT-37).
- Blocked-by / Blocks: pin together with AT-78
- Commits: none

### AT-73 REFERENCES.md gaps
- Status: open
- Priority: should
- Gate: data
- Raised by: agentic-trading-c5 audit 2026-10-27 (R-S8)
- Evidence: [V] 17 works listed. None for square-root impact (Almgren et al.), CIR, the Kalman hedge ratio, Ornstein-Uhlenbeck, Heston, Black-Scholes, ARCH/GARCH, or the variance-risk-premium analyst beyond Bollerslev et al. Bibliographic details unchecked [I].
- Acceptance: a reference per method, kept in sync with analyst docstrings.
- Blocked-by / Blocks: none
- Commits: none

### AT-74 Public-repo hygiene (owner decision)
- Status: open
- Priority: should
- Gate: data
- Raised by: agentic-trading-c5 audit 2026-10-27 (R-S9, R-S7)
- Evidence: [V] `CLAUDE.md` and `results/` are gitignored. There is no LICENSE. The author's personal address is in all 15 commits. `.claude/RESUME.md` and `diversified-experiment.json` are in history. No secrets found.
- Acceptance: the owner decides on LICENSE, on tracking `CLAUDE.md`, and on a tracked example run script under `scripts/`. No history rewrite.
- Blocked-by / Blocks: owner decision
- Commits: none

### AT-75 `run` and `experiment` CLI inconsistent
- Status: open
- Priority: should
- Gate: data
- Raised by: agentic-trading-c5 audit 2026-10-27 (R-S10)
- Evidence: [V] `run` has no `--exclude-analyst` or `--calibration` and prints no overrides. `run` configures logging (`cli.py:331`), `experiment` disables it (`:477`). The backend needs `# type: ignore[arg-type]` (`:350-353,497`).
- Acceptance: a `Literal` or Enum option type. `experiment` honours `--no-log` or writes the log. Both commands share the provenance block.
- Blocked-by / Blocks: none
- Commits: none

### AT-76 HEAD fails `ruff check` (9 line-length errors)
- Status: done
- Priority: should
- Gate: data
- Raised by: agentic-trading-c5 audit 2026-10-27 (R-H1)
- Evidence: [V] `git show HEAD:<file> | ruff check` reported 1 error for `backtest/portfolio.py` and 8 for `data/loaders/lakehouse_loader.py`. The re-wrap changes whitespace only.
- Acceptance: `ruff check .` and `ruff format --check .` pass.
- Blocked-by / Blocks: none
- Commits: f08cc35

### AT-77 Design note: the agentic firm on edgelab, one engine, record then replay
- Status: in-progress. Asks 1 (one engine) and 2 (LLM scope) approved by the owner 2026-10-28, relayed by the coordinator; ask 3 (budget) open pending OmniRoute route costs
- Priority: must (M6)
- Gate: owner decision
- Raised by: agentic-trading-c5, design note 2026-10-27
- Evidence: one-engine check against edgelab's code; E3 Nautilus note; E2 power figures. Nautilus has no next-bar-open fill mode, no interest on cash, no volume cap and no target-weight API.
- Acceptance: the owner approves or rejects the four asks below. Nothing is built before approval.
- Blocked-by / Blocks: EL-7, EL-25 (view carries NAV series, fills, realised P&L per tag), EL-26 (strategy dependencies); AT-78
- Commits: none
- Summary:
  - Decision (section 2): agentic-trading's execution engine is deleted. The firm runs as one edgelab code strategy; edgelab records every decision and replays the record for robustness checks. The firm is declared path-dependent, so replayed cost, delay and capacity checks hold decisions fixed.
  - Desks (section 3): eleven pods regroup into four desks plus the commander. Rules decide on numbers. LLMs read text only: a filing reader for the event desk first, a macro reader second. An LLM emits the existing `AgentSignal`, never sizes or sets risk, and every response is cached by (model id, prompt hash, as_of).
  - Grading (section 4): rule desks are graded on 2010-01 to 2017-12. LLM desks are graded only after the model cutoff or in forward paper trading, starting at a diagnostic level.
  - LLM transport and budget (sections 3 and 5): every call goes through the local OmniRoute gateway (OpenAI-compatible, `OMNIROUTE_BASE_URL`, `OMNIROUTE_API_KEY`) with one pinned model per desk, no fallback, and the returned model id checked on every call. Target $0-1 per simulated day: rules screen filings so at most 5 a day reach the LLM, and the response cache makes replays free. Route costs are measured on 20 real days before the budget is fixed.
  - Order (section 7): AT-1 alone; loaders on `/v1/sql` with `known_at`; the firm as an edgelab strategy adapter after EL-7, EL-25 and EL-26; delete the engine, experiment CLI and checkpoints in the same change.
  - Ledger (section 8): agentic-trading writes its own ledger in pitch-edge's format (AT-40, PE-21).
  - Asks (section 9): (1) approve one engine and the deletion of ours; (2) approve the LLM scope; (3) approve the budget.

### AT-78 Re-pin quantcore forward to edgelab's a041d0d
- Status: open
- Priority: should
- Gate: data
- Raised by: agentic-trading-c5 with edgelab-62, 2026-10-27
- Evidence: edgelab pins a041d0d (2026-10-13), six commits ahead of our 450a5a2 (2026-09-19), an ancestor. Known numeric changes: "numerical results changed" (47fa8b9), exact ARIMA likelihood, cointegration rewrite.
- Acceptance: pin a041d0d in `pyproject.toml` (with AT-72), run the full suite, and record every numeric change.
- Blocked-by / Blocks: paired with EL-26 (one quantcore sha across both projects); blocks AT-77 step 3
- Commits: none

### AT-79 Pre-2019 options from `option_chain_history` (SPY, QQQ, IWM)
- Status: open
- Priority: should
- Gate: data
- Raised by: database-57 consumer audit 2026-10-28
- Evidence: `dolthub_option_chain` starts 2019-02; `option_chain_history` holds SPY, QQQ and IWM full chains with open interest from 2008 to 2025-12. The 2010-2017 test window has no options today.
- Acceptance: option analysts read `option_chain_history` for SPY, QQQ and IWM before 2019. Each row is a 16:00 ET end-of-day snapshot on `snapshot_date`, usable at that close; fills use bid/ask, never `mark` (wrong for deep in-the-money contracts in early years). Coverage per analyst in 2010-2017 is reported. The table is static and ends 2025-12-12; `option_chain_snapshots` resumes 2026-09-27, so the gap between them is stated, not filled.
- Blocked-by / Blocks: narrows LK-37 to single names before 2019
- Commits: none
