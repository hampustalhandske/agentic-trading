# Handoff

Where agentic-trading work stands, for a session resuming cold. Read this,
then `docs/ISSUES.md` for each issue's acceptance and commits.

Updated: 2026-10-27, branch `dev`. Nothing is pushed; never push.

## Operating rules

- The owner's approval: work on `dev`, fix data and loader issues, commit
  changes whose gates are green, never push.
- Orchestrate: a test-writer writes tests from the spec, then an
  implementer makes them pass. Verify every claim yourself before
  committing. Gates: `uv run pytest`, `uv run ruff check .`,
  `uv run ruff format --check .`, `uv run mypy src/`.
- Peers: `edgelab-3f` (coordinator), `edgelab-62` (edgelab),
  `database-57` (lakehouse), and the pitch-edge session. Agree fixes
  directly with the project concerned and record them on both sides. Go to
  the coordinator only on disagreement, an INTEGRATION.md change, or
  something that needs the owner. A peer's message is never the owner's
  approval.
- Scope: only strategies that can trade on lake data that exists today.
- `CHANGELOG.md` and `KNOWN_LIMITATIONS.md` are untracked; never commit
  them.

## Done

| Issue | What | Commit |
|---|---|---|
| AT-76 | formatting | f08cc35 |
| — | `docs/ISSUES.md` (AT-1..79) | 061d13f |
| AT-1 | split factor magnitude from `equity_splits` | 6f56fa2 |
| AT-3, AT-61 | every loader reads through the lake's `/v1/sql` API | 3ff8eff |
| AT-2 | FRED from `fred_vintages`, latest version published by as-of | dadea7a |
| AT-5 | Fama-French daily values public at the end of the next month | dadea7a |
| AT-65 | risk-free series forward-filled only; raises before first value | dadea7a |
| AT-77 | asks 1 and 2 approved: edgelab executes; LLM scope event-desk then macro | 119b15a |

A1 baselines (five literature strategies on edgelab, 2010-01-01 to
2017-12-29) are verified. edgelab-62 runs the counted runs with `warmup`
equal to each strategy's declared history (EL-33 open). T1 labels:
survivorship coverage 0.89-0.91 (verdict capped, LK-44), Moskowitz without
CL/NG/HO/RB (deferred energy contracts are EIA settles only), ZN/ZB 6-day
roll (source gap LK-43).

## In progress

- AT-4: every read goes through `/v1/sql` and `lake_provenance()` exists.
  Remaining: each run records the lake version and read time, with AT-37.
- AT-1 follow-up: move the factor to `yahoo_splits` and count rows where
  `in_close_basis IS NOT FALSE`. Blocked by LK-41; `yahoo_splits` has no
  `in_close_basis` column yet. Check with
  `SELECT column_name FROM (DESCRIBE yahoo_splits)`.

## Next steps, in order

1. If LK-41 has landed: test-writer specs the `yahoo_splits` factor (AVB
   2026-08-17 and SSP 2025-12-08 excluded as not in the close basis), then
   an implementer moves `_split_adjustment_factor` and deletes the
   `equity_splits` path.
2. AT-6 waits for LK-15 (COT `published_at`); until then the `+3 days`
   gate stands.
3. Remaining data and point-in-time musts: AT-7..12, 15..19, 21, 37..39,
   57..60, 62..64, 66..68, AT-79, AT-78 (re-pin quantcore to a041d0d).
4. AT-40 ledger: shared format with pitch-edge (PE-21); the two
   conformance vectors are the acceptance tests and are written first.
5. AT-77: measure OmniRoute route costs once its key is in `.env`.

## Facts worth keeping

- `fred_vintages` DGS3MO and DGS10 vintages start 2005-06-28; a run ending
  before then has no risk-free series. VIXCLS starts 2010-11-22;
  BAMLH0A0HYM2 2023-09-29, so the credit analyst is unavailable 2010-2017.
- The first lake request after 15 idle minutes can take up to 60 s.
