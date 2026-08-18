# BeforeReturn Audit Backlog

Last updated: 2026-08-18

This backlog freezes the read-only completion audit into stable task IDs. It is a planning artifact only: no code fixes, retraining, deployment, commit, or push were performed while creating it.

Allowed statuses: `BLOCKED`, `TODO`, `IN_PROGRESS`, `DONE`, `ACCEPTED_LIMITATION`.

## P0-01: Fix CatBoost Official Test Eval-Set Contamination

- ID: P0-01
- Priority: P0
- Status: TODO
- Problem: `train_catboost()` uses the official testing split as `eval_set` with `use_best_model=True`, so the uncalibrated CatBoost training path performs model selection against the test set.
- Risk: Test-set pollution can overstate model quality and weaken the credibility of every metric that depends on this artifact.
- Scope: Update the CatBoost training workflow so official testing data is used only once for final evaluation. Create an internal validation split from training data for early stopping/model selection.
- Acceptance Criteria: CatBoost fitting never passes official testing rows to `eval_set`; final metrics are computed after training on a held-out official test split only; prior polluted artifacts are clearly superseded or removed from reported results.
- Verification Commands: `conda run -n before-return pytest -q`; `conda run -n before-return ruff check .`; inspect `src/training/train.py` to confirm no `eval_set=test_pool` in CatBoost training.
- Evidence / Relevant Files: `src/training/train.py:97` `train_catboost()`; `src/training/train.py:117` `model.fit(train_pool, eval_set=test_pool, use_best_model=True)`; `reports/metrics/strict_no_leak_catboost.json`.
- Dependencies: None.

## P0-02: Confirm And Rebuild Strict Train/Validation/Calibration/Test Chain

- ID: P0-02
- Priority: P0
- Status: DONE
- Problem: The current calibrated path uses an internal calibration split, but the complete evaluation chain still needs to be explicitly confirmed and rebuilt after P0-01.
- Risk: If train, validation, calibration, and test roles are ambiguous, metric comparisons and Overview artifacts may mix incompatible model versions or probabilities.
- Scope: Define and document a single strict chain: train fit split, validation split for model selection, calibration split for isotonic calibration, and official test split for final evaluation.
- Acceptance Criteria: One manifest records row counts, split roles, random seed, feature set, model version, data processing version, and artifact paths; official testing rows are excluded from fitting, validation, and calibration.
- Verification Commands: `conda run -n before-return python scripts/build_datasets.py`; calibration/training command selected by implementation; `conda run -n before-return pytest -q`.
- Evidence / Relevant Files: `src/training/calibration.py:64` `calibrate_catboost()`; `src/training/calibration.py:70` `train_test_split(...)`; `data/processed/dataset_manifest.json`; `docs/leakage-audit.md`.
- Dependencies: P0-01.

### P0-02 Chain Responsibilities

- Official training split: the only source for internal train-fit, validation, and calibration rows.
- Train-fit rows: fit the Logistic Regression baseline and CatBoost base learner parameters.
- Validation rows: selected from the official training split only; used as CatBoost `eval_set` for early stopping/model selection with `use_best_model=True`.
- Calibration rows: selected from the official training split only; used only to fit the isotonic calibrator after the CatBoost base learner is fitted.
- Official test rows: held out from fitting, validation, calibration, and model selection; used only for final metrics and post-training inference artifacts.

### P0-02 Acceptance Evidence

- `src/training/train.py` now builds CatBoost `eval_set` from `split_train_validation(train, ...)`, not from the official test split.
- `src/training/calibration.py` now uses `split_train_validation_calibration(...)` so train-fit, validation, calibration, and official test roles are mutually exclusive.
- `src/training/splits.py` records row counts, positive rates, and random seed through `split_metadata(...)`.
- `tests/test_training_splits.py` verifies internal splits are disjoint, reproducible, label-stratified, and complete.
- `tests/test_training_splits.py` verifies official test rows are not used as CatBoost fit or eval rows.
- `tests/test_training_splits.py` verifies calibration labels go only to the calibrator and are disjoint from train-fit and validation rows.
- Model, calibration, metrics, and Overview artifacts still need regeneration under P0-03; no retraining was performed while closing P0-02.

## P0-03: Regenerate Models, Calibration, Metrics, And Overview Artifact

- ID: P0-03
- Priority: P0
- Status: DONE
- Problem: Existing model and metric artifacts may include outputs from the polluted uncalibrated CatBoost path or from a pre-fix evaluation chain.
- Risk: The app may display metrics that are technically calculated correctly but tied to obsolete or contaminated artifacts.
- Scope: Regenerate Logistic baseline, CatBoost model, calibrated CatBoost bundle, model reports, and `reports/metrics/overview_model_metrics.json` after P0-01 and P0-02.
- Acceptance Criteria: Overview metrics, model-card metrics, and API `/model-metrics` all reference the same rebuilt artifact set and model version; stale metric JSON files are either replaced or clearly marked obsolete.
- Verification Commands: `conda run -n before-return python scripts/train_models.py --feature-set strict_no_leak --model logistic_regression`; `conda run -n before-return python scripts/calibrate_models.py`; `conda run -n before-return python scripts/audit_overview_metrics.py`; `conda run -n before-return pytest -q`.
- Evidence / Relevant Files: `models/strict_no_leak_catboost_calibrated.joblib`; `reports/metrics/overview_model_metrics.json`; `scripts/audit_overview_metrics.py`; `api/main.py:82` `model_metrics()`.
- Dependencies: P0-01, P0-02.
- Completion Evidence: Rebuilt strict Logistic Regression, strict CatBoost, calibrated strict CatBoost, strict metrics JSON files, Overview metrics artifact, SHAP summary, and demo scenarios after the P0-01/P0-02 split-chain fixes. Model-card strict metrics were updated to match the rebuilt Overview artifact. No commit or push was performed during P0-03 execution.

## P0-04: Replace Policy Console With Full Strict-Test Offline Simulation Artifact/API

- ID: P0-04
- Priority: P0
- Status: TODO
- Problem: `simulate_policy()` currently computes policy metrics over only the three demo scenarios.
- Risk: Policy Console metrics can look responsive while representing three hand-selected cases, not the strict offline test distribution.
- Scope: Build a full strict-test policy simulation artifact/API that evaluates thresholds, confidence, prompt caps, alternatives, recall, precision, false positives, coverage, and disturbance rate across the full strict test set.
- Acceptance Criteria: `/simulate-policy` reports `evaluated_checkouts` equal to the strict test-set count or a documented full offline artifact count; changing policy controls recomputes metrics from the full artifact; UI disclaimers remain non-causal.
- Verification Commands: `conda run -n before-return pytest -q`; HTTP smoke test for `POST /simulate-policy`; compare response `evaluated_checkouts` against strict test artifact count.
- Evidence / Relevant Files: `api/service.py:187` `simulate_policy()`; `api/service.py:188` `scenarios = self.demo_scenarios()`; `web/app/page.tsx`.
- Dependencies: P0-03.

## P1-01: Replace Overview Hardcoded SHAP With Real Explanation Artifact

- ID: P1-01
- Priority: P1
- Status: TODO
- Problem: Overview uses a hardcoded `shapData` array instead of loading the generated explanation artifact.
- Risk: The explanation chart may drift away from the model actually served by the API.
- Scope: Expose or bundle a real explanation summary artifact and make Overview read it with clear unavailable behavior.
- Acceptance Criteria: No static feature-impact values remain in Overview; displayed features include artifact/model version; missing artifact shows `Unavailable` rather than invented values.
- Verification Commands: `npm run typecheck`; `npm run build`; API or artifact smoke test selected by implementation.
- Evidence / Relevant Files: `web/app/page.tsx:109` `shapData`; `web/app/page.tsx:342` `BarChart data={shapData}`; SHAP generation referenced in `STATUS.md`.
- Dependencies: P0-03.

## P1-02: Normalize Alternative Recommendation Product Meaning And Candidate Rules

- ID: P1-02
- Priority: P1
- Status: TODO
- Problem: Alternatives are peer variants selected by same `brandDesc` and `productType`, not proven same-product, same-size, adjacent-size, or in-stock substitutions.
- Risk: Product copy can overclaim the recommendation semantics and mislead users or reviewers.
- Scope: Rename and document the recommendation as same-brand/product-type peer variant unless stronger product relationship data becomes available; enforce candidate filters and reasons consistently.
- Acceptance Criteria: API response, UI labels, model-card text, and case study never claim same-item size/color alternatives; candidate rules and fallbacks are visible and tested.
- Verification Commands: `conda run -n before-return pytest -q`; `npm run typecheck`; targeted test for candidate rule labels.
- Evidence / Relevant Files: `src/inference/scenarios.py:101` `find_alternative()`; `src/inference/scenarios.py:117` `probabilities = score_frame(...)`; `AGENTS.md` section 4.3 and 7.
- Dependencies: LIMIT-02.

## P1-03: Add API Contract Tests And End-To-End Tests

- ID: P1-03
- Priority: P1
- Status: TODO
- Problem: Existing tests cover metric utilities and selected service logic, but critical API contracts and full demo flows are not fully automated.
- Risk: Endpoint schemas, large integer IDs, artifact loading, policy simulation, and frontend-to-API flows can regress without test failures.
- Scope: Add FastAPI contract tests, demo scenario prediction tests, recommendation tests, policy simulation tests, and at least one frontend or Playwright smoke path.
- Acceptance Criteria: Tests fail if `/health`, `/demo-scenarios`, `/predict-return-risk`, `/recommend-alternatives`, `/simulate-policy`, or `/model-metrics` returns missing required fields or fallback/random predictions.
- Verification Commands: `conda run -n before-return pytest -q`; `cd web && npm run typecheck && npm run build`; Playwright command if introduced.
- Evidence / Relevant Files: `tests/test_metrics.py`; `tests/test_api_service.py`; `api/main.py`; `web/app/page.tsx`.
- Dependencies: P0-03, P0-04.

## P1-04: Fix Fresh-Clone Demo Artifact Retrieval And One-Command Startup

- ID: P1-04
- Priority: P1
- Status: TODO
- Problem: `scripts/run_local_demo.sh` assumes dependencies and large local artifacts already exist.
- Risk: A fresh reviewer cannot launch the demo from a clean clone with one command, violating the MVP completion standard.
- Scope: Define artifact download/build flow, document prerequisites, and make a single command either fetch/build required artifacts or fail with precise instructions.
- Acceptance Criteria: From a fresh clone, documented commands reproduce processed data, models, metrics, and local API/web startup without manual guessing; raw data remains out of git.
- Verification Commands: Fresh-clone rehearsal in a temporary directory; `bash scripts/run_local_demo.sh`; `conda run -n before-return pytest -q`.
- Evidence / Relevant Files: `README.md`; `scripts/run_local_demo.sh`; `data/README.md`; `git status --short --branch` shows raw/model artifacts are local/untracked or gitignored.
- Dependencies: P0-03.

## P2-01: Deploy Public Demo

- ID: P2-01
- Priority: P2
- Status: TODO
- Problem: The app is currently verified locally only; no public demo URL has been established.
- Risk: Portfolio review may fail if the evaluator cannot access a running demo.
- Scope: Choose a deployment route for the web app and API with required artifacts available without committing raw data.
- Acceptance Criteria: A public URL loads Overview, Checkout Simulator, Safer Alternative, and Policy Console; API health is reachable; deployment steps are documented.
- Verification Commands: Browser smoke test against public URL; API `/health` check against deployed API; frontend build command.
- Evidence / Relevant Files: `web`; `api`; `README.md`.
- Dependencies: P1-04.

## P2-02: Polish README, Model Card, Leakage Audit, And Case Study

- ID: P2-02
- Priority: P2
- Status: TODO
- Problem: Documentation exists but still mixes implementation progress, limitations, and delivery narrative; case study is not yet final portfolio quality.
- Risk: The project may be technically credible but hard for reviewers to understand quickly.
- Scope: Finalize README, model-card, leakage audit, and case study around verified artifacts only.
- Acceptance Criteria: Documentation states data source, selection bias, leakage decisions, final metrics, model comparison, policy simulation, limitations, setup, and demo script; no causal or ASOS-wide claims.
- Verification Commands: Manual doc review against `AGENTS.md`; link/path check; rerun tests referenced in docs.
- Evidence / Relevant Files: `README.md`; `docs/model-card.md`; `docs/leakage-audit.md`; `docs/case-study.md`; `AGENTS.md`.
- Dependencies: P0-03, P0-04, P1-02.

## P2-03: Complete Three-Minute Demo Rehearsal And Recording

- ID: P2-03
- Priority: P2
- Status: TODO
- Problem: The required three-minute demo flow has not been rehearsed or recorded end to end.
- Risk: A locally working product can still fail during presentation due to timing, confusing copy, missing artifacts, or unstable startup.
- Scope: Rehearse and record the Overview -> Checkout -> Alternative -> Policy Console flow using the accepted artifacts and limitations.
- Acceptance Criteria: A three-minute recording exists; no live retraining or manual repair is needed; the script explicitly mentions non-causal estimates and selection bias.
- Verification Commands: Local or deployed demo startup command; manual recording review; optional smoke tests immediately before recording.
- Evidence / Relevant Files: `AGENTS.md` sections 3 and 15; `docs/case-study.md`; `scripts/run_local_demo.sh`.
- Dependencies: P2-01, P2-02.

## LIMIT-01: No Usable Timestamp/Sequence For Strict Per-Order History Features

- ID: LIMIT-01
- Priority: LIMIT
- Status: ACCEPTED_LIMITATION
- Problem: Raw event tables do not provide a usable timestamp or event sequence for reconstructing pre-checkout customer/product history.
- Risk: The strict MVP cannot prove historical aggregate features are computed only from events before each checkout.
- Scope: Do not fabricate temporal order. Keep strict model features limited to fields judged available at prediction time.
- Acceptance Criteria: Product and model documentation clearly state that strict per-order historical relationship features are excluded; no UI or case-study claim depends on reconstructed order history.
- Verification Commands: Review `docs/leakage-audit.md`; inspect `data/processed/dataset_manifest.json` strict feature list.
- Evidence / Relevant Files: `docs/leakage-audit.md`; `data/processed/dataset_manifest.json`; `AGENTS.md` section 5.3.
- Dependencies: None.
- Product/Model Impact: Model explanations should not claim user-specific historical return-rate behavior. Future richer data with timestamps or event sequence could support leakage-safe historical aggregates.

## LIMIT-02: Alternative Is Same Brand/Product-Type Peer Variant Only

- ID: LIMIT-02
- Priority: LIMIT
- Status: ACCEPTED_LIMITATION
- Problem: Current data does not prove candidate variants are same-item adjacent sizes or colors.
- Risk: Recommending a peer variant as a size/color substitute would overstate product relationship quality.
- Scope: Do not invent size, color, inventory, or same-product relationships.
- Acceptance Criteria: UI, API text, README, model-card, and case study describe recommendations as same-brand/product-type peer variants unless better metadata is added.
- Verification Commands: Search docs and UI for claims like `size`, `color`, `same item`, `adjacent`; run relevant frontend/API tests.
- Evidence / Relevant Files: `src/inference/scenarios.py:101` `find_alternative()`; `AGENTS.md` sections 4.3 and 7.
- Dependencies: None.
- Product/Model Impact: Alternative recommendation remains useful as a risk-aware peer suggestion, not a true fit/size substitution. Future catalog metadata with item group, size, color, and inventory would allow stronger candidate rules.

## LIMIT-03: GraphReturns Selection Bias

- ID: LIMIT-03
- Priority: LIMIT
- Status: ACCEPTED_LIMITATION
- Problem: ASOS GraphReturns includes customers with at least one historical return.
- Risk: Reported probabilities and positive rates describe this evaluation population and must not be interpreted as the ASOS-wide return rate.
- Scope: Keep the dataset limitation visible in Overview, model-card, leakage audit, and case study.
- Acceptance Criteria: All user-facing and portfolio-facing materials state the selection bias plainly; metric cards avoid ASOS-wide claims.
- Verification Commands: Search docs and UI for `selection bias`, `ASOS-wide`, and dataset limitation text; `npm run build`.
- Evidence / Relevant Files: `AGENTS.md` section 5.2; `docs/model-card.md`; `web/app/page.tsx`.
- Dependencies: None.
- Product/Model Impact: Probabilities are calibrated for the GraphReturns evaluation population only. Future access to a representative ASOS checkout sample would be required for site-wide calibration.

## Dependency Order

1. LIMIT-01, LIMIT-02, LIMIT-03 are accepted boundaries and must constrain all later wording and evaluation.
2. P0-01 fixes test-set contamination in the CatBoost training path.
3. P0-02 defines the strict train/validation/calibration/test chain.
4. P0-03 regenerates model, calibration, metrics, and Overview artifacts.
5. P0-04 rebuilds Policy Console simulation on the full strict test set.
6. P1-01, P1-02, and P1-03 harden explanations, recommendation semantics, and tests.
7. P1-04 makes fresh-clone startup reproducible.
8. P2-01, P2-02, and P2-03 finish public delivery, documentation, and demo recording.
