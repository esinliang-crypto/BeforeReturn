# Model Card

Status: model metrics audited on 2026-08-18.

## Intended Use

Estimate checkout-time return risk for anonymous user-product variant combinations and support policy-controlled interventions.

## Not Intended For

- Causal claims about reduced returns.
- Automated denial of purchase.
- Identifying real individuals.
- ASOS-wide return-rate estimation.

## Dataset

ASOS GraphReturns from OSF: https://osf.io/c793h/

License and attribution: ASOS GraphReturns is published on OSF under CC BY 4.0
International. This model is trained from that dataset with project-specific
processing and feature restrictions; no endorsement by ASOS or the original
authors is implied.

Known limitation: GraphReturns includes customers with at least one historical return. Reported probabilities describe this evaluation population and must not be interpreted as the ASOS-wide return rate.

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

All Overview metrics are recomputed in `reports/metrics/overview_model_metrics.json` from:

- Feature set: `strict_no_leak`
- Test split path: `data/processed/strict_no_leak_testing.pkl`
- Test sample count: 1,460,366
- Target column: `isReturned`
- Probability source: calibrated CatBoost probability
- Model version: `strict_no_leak_catboost_calibrated_v1`
- Data processing version: `dataset_manifest_sha256:89af5c192d24c418e7807d63f975783694d2b44542d1d32152c91fead9664ed3`

| Feature set | Model | PR-AUC | ROC-AUC | F1 | Precision | Recall | Recall@Top 10% | Precision@Top 10% | Lift@Top 10% | Brier |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `strict_no_leak` | Logistic Regression | 0.6725 | 0.6476 | 0.5994 | 0.6639 | 0.5464 | 0.1382 | unavailable | unavailable | 0.2337 |
| `strict_no_leak` | Calibrated CatBoost | 0.6802 | 0.6584 | 0.6753 | 0.6249 | 0.7346 | 0.1409 | 0.7673 | 1.4089 | 0.2286 |
| `paper_feature_baseline` | Logistic Regression | 0.8546 | 0.8280 | 0.7580 | 0.7772 | 0.7397 | 0.1798 | unavailable | unavailable | 0.1702 |
| `paper_feature_baseline` | CatBoost | 0.8612 | 0.8356 | 0.7785 | 0.7553 | 0.8031 | 0.1811 | unavailable | unavailable | 0.1644 |

The `paper_feature_baseline` results are stronger, but they use official node aggregate features that may contain target leakage for checkout-time prediction. They are retained for comparison and must not be presented as strictly leakage-safe.

## Baselines And Top 10% Definition

- Test positive rate: 0.5446. This is the no-skill PR-AUC baseline.
- CatBoost PR-AUC absolute gain over positive rate: +0.1356.
- CatBoost PR-AUC relative gain over positive rate: 0.2489.
- Constant-probability Brier baseline: 0.2480.
- Calibrated CatBoost Brier Skill Score: 0.0783.
- ECE: 0.0087.

Top 10% metrics use the highest predicted probabilities on the same official test split:

```python
n_top = max(1, int(len(y_true) * 0.10))
top_idx = np.argsort(y_prob)[::-1][:n_top]
top_y_true = y_true[top_idx]
recall_at_10 = top_y_true.sum() / y_true.sum()
precision_at_10 = top_y_true.mean()
lift_at_10 = precision_at_10 / y_true.mean()
```

## Calibration

Initial metrics include Brier Score and calibration curve values in `reports/metrics/*.json`.

The primary `strict_no_leak` CatBoost model has an isotonic calibration artifact trained with internal train-fit, validation, and calibration splits from the official training data:

- Fit rows: 876,244
- Validation rows: 219,062
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

## Policy Simulation

Policy Console uses prediction margin, defined as `abs(p - 0.5) * 2`, for the
`min_prediction_margin` control. This is only a probability-margin proxy and is
not an uncertainty estimate, confidence interval, or guarantee of correctness.
For high-risk thresholds above 0.5, the margin gate partly overlaps with the
risk threshold because `prediction_margin = 2p - 1`.

Offline policy simulation should read the full strict-test artifact from
`reports/policy/`. Same-brand, same-product-type historical peer candidates are
selected from the official training catalog without labels, then rescored under
the current test checkout's user features. The artifact stores compact
numeric/bool fields only; IDs, brand, product type, and recommendation text are
excluded from the full artifact.

## Recommendation Boundary

The MVP exposes a lower-risk peer option, not a same-item replacement. Candidate
options are limited to same-brand, same-product-type historical peers from the
available catalog metadata. Candidate risk is a model estimate rescored under
the current user's checkout-available fields. Same item identity, adjacent size,
color match, and real-time inventory are not verified. Estimated risk
differences are not randomized causal evidence of reduced returns.
