# Model Card

Status: initial full-data training completed on 2026-08-18.

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
- Primary product-demo track: `strict_no_leak`.
- Comparison-only track: `paper_feature_baseline`.

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

## Full Test Metrics

| Feature set | Model | PR-AUC | ROC-AUC | F1 | Precision | Recall | Recall@Top 10% | Brier |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `strict_no_leak` | Logistic Regression | 0.6725 | 0.6476 | 0.5994 | 0.6639 | 0.5464 | 0.1382 | 0.2337 |
| `strict_no_leak` | CatBoost | 0.6839 | 0.6587 | 0.6773 | 0.6239 | 0.7407 | 0.1408 | 0.2285 |
| `paper_feature_baseline` | Logistic Regression | 0.8546 | 0.8280 | 0.7580 | 0.7772 | 0.7397 | 0.1798 | 0.1702 |
| `paper_feature_baseline` | CatBoost | 0.8612 | 0.8356 | 0.7785 | 0.7553 | 0.8031 | 0.1811 | 0.1644 |

The `paper_feature_baseline` results are stronger, but they use official node aggregate features that may contain target leakage for checkout-time prediction. They are retained for comparison and must not be presented as strictly leakage-safe.

## Calibration

Initial metrics include Brier Score and calibration curve values in `reports/metrics/*.json`.

The primary `strict_no_leak` CatBoost model has an isotonic calibration artifact trained with an 80/20 split of the official training data:

- Fit rows: 1,095,306
- Calibration rows: 273,827
- Official test rows: 1,460,366
- Calibrated Brier Score: 0.2286
- Raw Brier Score from the same base model: 0.2285

The calibrated artifact did not materially improve Brier Score on the official test split. It is retained because it provides an explicit calibration workflow and a calibration curve, but the product must not claim calibration improved performance.

## Explainability

CatBoost SHAP summary was generated on 5,000 complete-metadata testing rows.

Top global factors for the primary strict model:

1. `productType`
2. `shippingCountry`
3. `avgGbpPrice`
4. `brandDesc`
5. `yearOfBirth`

Per-scenario top factors are included in `data/samples/demo_scenarios.json`.
