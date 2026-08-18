# BeforeReturn

BeforeReturn is a portfolio-grade AI product MVP for checkout-time fashion return risk prediction and policy-controlled intervention.

It uses the public ASOS GraphReturns dataset described in:

- Paper: https://arxiv.org/abs/2302.14096
- Official data entry: https://osf.io/c793h/

The product goal is to predict return risk for a selected anonymous user and product variant, explain why the scenario was flagged, and recommend lower-risk alternatives only when policy constraints are satisfied.

## Delivery Standard

This project follows `AGENTS.md` strictly. The MVP is not complete until data processing, leakage audit, baseline and CatBoost models, calibration, explainability, API, frontend, tests, model card, case study, and a stable 3-minute demo flow are all present.

## Data Policy

Raw ASOS GraphReturns files must not be committed. Use:

```bash
python scripts/download_osf_data.py
```

Downloaded files are stored in `data/raw/`, which is ignored by git.

The dataset contains users with at least one return, so all reporting and UI copy must state this selection bias. Model predictions are historical-data estimates, not causal effects.

## Planned Local Setup

Python:

```bash
conda env create -f environment.yml
conda activate before-return
```

Frontend setup will be added under `web/` after the Next.js app is scaffolded.

## Build Modeling Datasets

```bash
conda run -n before-return python scripts/build_datasets.py
```

This creates ignored full datasets under `data/processed/` and tiny committed samples under `data/samples/`.

Two feature sets are built:

- `strict_no_leak`: excludes return-derived node aggregates and uses only lower-risk checkout-available profile/product fields.
- `paper_feature_baseline`: uses official node features as a comparison baseline and records leakage-risk columns in the manifest.

## Train Models

```bash
conda run -n before-return python scripts/train_models.py
```

Model artifacts are stored under ignored `models/`. Small metrics JSON files are written to `reports/metrics/`.

## Calibrate And Explain

```bash
conda run -n before-return python scripts/calibrate_models.py --feature-set strict_no_leak
conda run -n before-return python scripts/generate_explanations.py
```

## Build Policy Simulation

```bash
conda run -n before-return python scripts/build_policy_simulation.py --feature-set strict_no_leak
```

The runtime API uses the committed pre-generated policy artifact at
`reports/policy/strict_no_leak_policy_simulation.pkl.gz`, so a fresh clone can
run the policy simulation demo without rebuilding the full artifact.

`scripts/build_policy_simulation.py` is only needed when local processed data
and the calibrated model are available and you want to regenerate the policy
artifact. The candidate pool is built from the official training catalog only,
without labels, and peer variants are rescored under each current test
checkout's user features. Recommendations must be described only as same-brand,
same-product-type historical peers.

The policy request field `min_prediction_margin` controls
`abs(p - 0.5) * 2`. This is only a probability-margin proxy, not an uncertainty
estimate, confidence interval, or guarantee of correctness.

## Run API

```bash
conda run -n before-return uvicorn api.main:app --reload
```

## Run Web

```bash
cd web
npm install
npm run dev
```

Open http://localhost:3000. The frontend expects the API at `http://127.0.0.1:8000` unless `NEXT_PUBLIC_API_BASE_URL` is set.

Or run both local services:

```bash
bash scripts/run_local_demo.sh
```

The committed metrics, demo scenarios, explanation summary, and policy
simulation artifact allow the demo API to serve evaluated outputs from a fresh
clone. Fresh clones do not guarantee raw-data-to-model retraining unless the
ignored raw data, processed data, and model artifacts are rebuilt or supplied
locally.
