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
- P1-04 local artifact work generated deterministic `reports/runtime/strict_no_leak_demo_runtime.json.gz` from existing strict demo scenarios, added a SHA256 artifact manifest and fetch verifier, and changed prediction/recommendation lookup to use the small runtime artifact instead of `data/processed/strict_no_leak_testing.pkl`.
- P1-04 local smoke passed with `data/processed/strict_no_leak_testing.pkl` temporarily moved away: `/health`, `/demo-scenarios`, `/predict-return-risk`, `/recommend-alternatives`, and `/simulate-policy` all returned 200, with `/simulate-policy` still evaluating 1,460,366 artifact rows.
- P1-04 final rehearsal passed from a fresh clone with no raw, processed, or local model artifacts. `scripts/fetch_demo_artifacts.py` downloaded the calibrated model from the GitHub Release asset, verified SHA256 `fd37f157cf755f3af62ebaa6900b6371f93811c918f99e48220ae50716acc21a`, and the core API smoke returned 200 for `/health`, `/demo-scenarios`, `/predict-return-risk`, `/recommend-alternatives`, and `/simulate-policy`.
- P1-05 expanded Checkout Simulator from 3 to 7 deterministic official strict-test scenarios covering true positive, false positive, false negative, true negative, high-risk without peer, low prediction margin, and incomplete metadata behaviors. Scenario payloads document fixed selection rules and hide observed outcomes by default; observed outcomes are used only for revealable error analysis and are excluded from the demo runtime artifact used for prediction/recommendation.
- P1-05 validation passed: `conda run -n before-return python scripts/fetch_demo_artifacts.py --manifest artifacts/demo-artifacts.json`; `conda run -n before-return ruff check .`; `conda run -n before-return pytest -q` with 61 passed and 1 skipped; `cd web && npm run typecheck`; `cd web && npm run build`. TestClient smoke confirmed all 7 scenarios return 200 for prediction and recommendation, and `/simulate-policy` still evaluates 1,460,366 rows.
- P1-05 manual local demo acceptance passed after minimal UI/provenance fixes: scenario labels no longer expose TP/FP/FN/TN or observed outcome before `Reveal outcome`, scenario switching resets reveal/prediction state, recommendation/no-intervention paths work across all 7 scenarios, and Policy Console displays the full 1,460,366-row artifact count.
- P1-05 has no remaining local verification gaps. Public hosted-demo validation is not covered by the local acceptance run and remains part of the later deployment/readiness track.

## Known Blockers

- LIMIT-01: raw event data lacks usable timestamp/sequence for strict per-order historical relationship features.
- LIMIT-02: alternatives are same-brand/product-type peer variants, not proven same-item adjacent sizes.
- LIMIT-03: GraphReturns represents customers with at least one historical return, not the ASOS-wide population.

## Next Task ID

P2-01

Full task definitions and dependency order are recorded in `docs/audit-backlog.md`.
