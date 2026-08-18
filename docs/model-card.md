# Model Card

Status: placeholder, pending model training.

## Intended Use

Estimate checkout-time return risk for anonymous user-product variant combinations and support policy-controlled interventions.

## Not Intended For

- Causal claims about reduced returns.
- Automated denial of purchase.
- Identifying real individuals.
- ASOS-wide return-rate estimation.

## Dataset

ASOS GraphReturns from OSF: https://osf.io/c793h/

Known limitation: the dataset includes users with at least one return, creating selection bias.

## Models

- Baseline: Logistic Regression.
- Main model: CatBoost.

## Metrics To Report

- PR-AUC
- ROC-AUC
- F1
- Precision
- Recall
- Recall@Top 10%
- Brier Score
- Calibration Curve
- Confusion Matrix
- Country or product-type slices

## Calibration

Pending.

## Explainability

Pending SHAP or equivalent feature attribution.

