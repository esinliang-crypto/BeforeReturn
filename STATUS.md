# BeforeReturn Status

Last updated: 2026-08-18

## Current Milestone

Day 1: repository skeleton, official data access, data dictionary draft, and leakage audit draft.

## Completed

- Confirmed project location: `/Users/liangyingxin/Desktop/BeforeReturn`.
- Confirmed raw data was initially absent.
- Confirmed main model choice: CatBoost.
- Confirmed frontend stack: Next.js, TypeScript, Tailwind CSS, shadcn/ui.
- Initialized git repository.
- Created initial repository structure.
- Downloaded official OSF raw files into `data/raw/`.
- Inspected raw pickle schemas.
- Created Python 3.12 Conda environment: `before-return`.
- Installed Python project dependencies in editable mode.
- User selected dual-track modeling strategy: `strict_no_leak` and `paper_feature_baseline`.
- User selected missing metadata strategy C: keep all events for modeling, but restrict recommendation/demo candidates to complete metadata rows.
- Trained full-data Logistic Regression and CatBoost models for both modeling tracks.

## Verification

- OSF API endpoint for `https://osf.io/c793h/` is reachable.
- Raw data directory is git-ignored.
- Downloaded files passed SHA-256 checks in `scripts/download_osf_data.py`.
- Observed raw data size is approximately 742 MB.
- `conda run -n before-return python --version` reports Python 3.12.13.
- `conda run -n before-return ruff check .` passed.
- `conda run -n before-return pytest -q` passed with 3 tests.
- `conda run -n before-return python scripts/build_datasets.py` built both feature sets.
- `strict_no_leak` rows: 1,369,133 training and 1,460,366 testing.
- `paper_feature_baseline` rows: 1,369,133 training and 1,460,366 testing.
- Complete metadata rows for recommendation/demo filtering: 848,454 training and 960,769 testing.
- Full-data `strict_no_leak` CatBoost metrics: PR-AUC 0.6839, ROC-AUC 0.6587, Recall@Top 10% 0.1408, Brier 0.2285.
- Full-data `paper_feature_baseline` CatBoost metrics: PR-AUC 0.8612, ROC-AUC 0.8356, Recall@Top 10% 0.1811, Brier 0.1644.

## Known Issues

- Raw event tables have no timestamp or sequence field.
- Node aggregate fields include return counts/rates and return-code distributions that may leak target labels if used directly.
- System default Python remains 3.13.5; use Conda environment `before-return` for all project commands.

## Next Step

Add probability calibration artifacts, SHAP explanations, and stable demo scenario selection.
