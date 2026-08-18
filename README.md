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
