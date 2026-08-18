# BeforeReturn Status

Last updated: 2026-08-18

## Current Project Completion

Estimated true completion: 70%.

This estimate is based on the read-only audit against `AGENTS.md`: real ASOS GraphReturns data, local model artifacts, API endpoints, and frontend views exist, but several delivery-critical items remain unresolved or not fully verified.

## Current Unique Milestone

P0 model and evaluation integrity hardening.

The next milestone is not UI polish or deployment. The project must first fix and rebuild the strict evaluation chain so all reported metrics, model artifacts, and policy outputs are trustworthy.

## Recent Verification Results

- Read `AGENTS.md`, `SPEC.md`, and `STATUS.md` on 2026-08-18 before creating the audit backlog.
- Metrics audit and Overview artifact fixes were committed separately in `86ad947`.
- Current P0-01 working-tree changes are limited to training split logic, dataset-role isolation tests, and this status update.
- Previous validation evidence showed `conda run -n before-return ruff check .`, `conda run -n before-return pytest -q`, `cd web && npm run typecheck`, and `cd web && npm run build` passing.
- P0-01 implemented locally: ordinary CatBoost now uses an internal validation split from official training for `eval_set`, and calibrated CatBoost now uses mutually exclusive train-fit, validation, and calibration splits.
- P0-01 validation passed: `conda run -n before-return ruff check .`.
- P0-01 validation passed: `conda run -n before-return pytest -q` with 28 tests.

## Known Blockers

- P0-02: strict Train/Validation/Calibration/Test chain must be explicitly rebuilt and documented after P0-01.
- P0-03: model, calibration, metric, and Overview artifacts must be regenerated after the evaluation-chain fix.
- P0-04: Policy Console currently simulates policy over three demo scenarios instead of the full strict test set.
- LIMIT-01: raw event data lacks usable timestamp/sequence for strict per-order historical relationship features.
- LIMIT-02: alternatives are same-brand/product-type peer variants, not proven same-item adjacent sizes.
- LIMIT-03: GraphReturns represents customers with at least one historical return, not the ASOS-wide population.

## Next Task ID

P0-02

Full task definitions and dependency order are recorded in `docs/audit-backlog.md`.
