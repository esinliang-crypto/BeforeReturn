# BeforeReturn Status

Last updated: 2026-08-18

## Current Project Completion

Estimated true completion: 68%.

This estimate is based on the read-only audit against `AGENTS.md`: real ASOS GraphReturns data, local model artifacts, API endpoints, and frontend views exist, but several delivery-critical items remain unresolved or not fully verified.

## Current Unique Milestone

P0 model and evaluation integrity hardening.

The next milestone is not UI polish or deployment. The project must first fix and rebuild the strict evaluation chain so all reported metrics, model artifacts, and policy outputs are trustworthy.

## Recent Verification Results

- Read `AGENTS.md`, `SPEC.md`, and `STATUS.md` on 2026-08-18 before creating the audit backlog.
- Metrics audit and Overview artifact fixes were committed separately in `86ad947`.
- Current audit-planning changes are limited to `STATUS.md` and `docs/audit-backlog.md`.
- Previous validation evidence showed `conda run -n before-return ruff check .`, `conda run -n before-return pytest -q`, `cd web && npm run typecheck`, and `cd web && npm run build` passing.

## Known Blockers

- P0-01: `src/training/train.py` uses official test data as CatBoost `eval_set` with `use_best_model=True`.
- P0-02: strict Train/Validation/Calibration/Test chain must be explicitly rebuilt and documented after P0-01.
- P0-03: model, calibration, metric, and Overview artifacts must be regenerated after the evaluation-chain fix.
- P0-04: Policy Console currently simulates policy over three demo scenarios instead of the full strict test set.
- LIMIT-01: raw event data lacks usable timestamp/sequence for strict per-order historical relationship features.
- LIMIT-02: alternatives are same-brand/product-type peer variants, not proven same-item adjacent sizes.
- LIMIT-03: GraphReturns represents customers with at least one historical return, not the ASOS-wide population.

## Next Task ID

P0-01

Full task definitions and dependency order are recorded in `docs/audit-backlog.md`.
