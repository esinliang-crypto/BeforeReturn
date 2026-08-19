# BeforeReturn Status

Last updated: 2026-08-19

## Current Project Completion

Estimated true completion: 70%.

This estimate is based on the read-only audit against `AGENTS.md`: real ASOS GraphReturns data, local model artifacts, API endpoints, and frontend views exist, but several delivery-critical items remain unresolved or not fully verified.

## Current Unique Milestone

P1 product-contract hardening.

The current milestone is tightening user-facing claims, API contracts, and tests around the already rebuilt strict model, explanation, and policy artifacts.

## Recent Verification Results

- Read `AGENTS.md`, `SPEC.md`, and `STATUS.md` on 2026-08-18 before creating the audit backlog.
- Metrics audit and Overview artifact fixes were committed separately in `86ad947`.
- Current P0-01 working-tree changes are limited to training split logic, dataset-role isolation tests, and this status update.
- Previous validation evidence showed `conda run -n before-return ruff check .`, `conda run -n before-return pytest -q`, `cd web && npm run typecheck`, and `cd web && npm run build` passing.
- P0-01 implemented locally: ordinary CatBoost now uses an internal validation split from official training for `eval_set`, and calibrated CatBoost now uses mutually exclusive train-fit, validation, and calibration splits.
- P0-01 validation passed: `conda run -n before-return ruff check .`.
- P0-01 validation passed: `conda run -n before-return pytest -q` with 28 tests.
- P0-02 source-level chain documentation completed: Train/Validation/Calibration/Test responsibilities are recorded in `docs/audit-backlog.md`; model regeneration remains deferred to P0-03.
- P0-03 regenerated strict Logistic Regression, CatBoost, calibrated CatBoost, metrics, Overview metrics, SHAP summary, and demo scenario artifacts after the split-chain fixes.
- P0-04 implemented full strict-test offline policy simulation: `/simulate-policy` reads a cached ignored artifact with 1,460,366 rows instead of the three demo scenarios.
- P1-01 replaced Overview hardcoded SHAP data with `GET /model-explanations`, backed by `reports/explanations/strict_no_leak_catboost_shap_summary.json`; Overview now displays artifact/model metadata and `Unavailable` when the explanation summary is missing.
- P1-01 validation passed: `conda run -n before-return pytest -q` with 36 tests.
- P1-01 validation passed: `cd web && npm run typecheck`.
- P1-01 validation passed: `cd web && npm run build`.
- P1-02 normalized recommendation semantics to lower-risk same-brand, same-product-type historical peer options. API payloads, UI copy, Policy Console controls, README, SPEC, model card, leakage audit, and case study now state that candidate risk is model-estimated under the current user's checkout-available fields, inventory is not verified, and estimated risk differences are non-causal.
- P1-02 validation passed: `conda run -n before-return ruff check .`.
- P1-02 validation passed: `conda run -n before-return pytest -q` with 39 tests.
- P1-02 validation passed: `cd web && npm run typecheck`.
- P1-02 validation passed: `cd web && npm run build`.
- P1-03 added FastAPI TestClient API contract coverage for `/health`, `/model-metrics`, `/demo-scenarios`, `/predict-return-risk`, `/recommend-alternatives`, and `/simulate-policy` with small fixtures. The tests cover required fields, status codes, model version, explicit missing-artifact failures, invalid 422 payloads, full policy artifact row usage, and peer recommendation scope/disclaimer fields.
- P1-03 added optional live-service smoke coverage gated by `BEFORE_RETURN_LIVE_API_URL`; ordinary pytest skips it and does not require a running API.
- P1-03 validation passed: `conda run -n before-return ruff check .`.
- P1-03 validation passed: `conda run -n before-return pytest -q` with 54 passed and 1 skipped.
- P1-03 validation passed: `cd web && npm run typecheck`.
- P1-03 validation passed: `cd web && npm run build`.

## Known Blockers

- LIMIT-01: raw event data lacks usable timestamp/sequence for strict per-order historical relationship features.
- LIMIT-02: alternatives are same-brand/product-type peer variants, not proven same-item adjacent sizes.
- LIMIT-03: GraphReturns represents customers with at least one historical return, not the ASOS-wide population.

## Next Task ID

P1-05

Full task definitions and dependency order are recorded in `docs/audit-backlog.md`.
