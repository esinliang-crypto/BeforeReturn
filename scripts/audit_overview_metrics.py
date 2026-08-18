from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib

from api.service import MODEL_VERSION
from src.data.dataset import TARGET
from src.evaluation.metrics import overview_metrics
from src.inference.scenarios import score_frame
from src.training.train import load_processed

OUTPUT_PATH = Path("reports/metrics/overview_model_metrics.json")


def sha256_text(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def data_processing_version() -> str:
    manifest_path = Path("data/processed/dataset_manifest.json")
    if not manifest_path.exists():
        return "unavailable"
    return f"dataset_manifest_sha256:{sha256_text(manifest_path)}"


def main() -> None:
    test = load_processed("strict_no_leak", "testing")
    y_true = test[TARGET].to_numpy()

    logistic_bundle = joblib.load("models/strict_no_leak_logistic_regression.joblib")
    logistic_prob = logistic_bundle["model"].predict_proba(test[logistic_bundle["features"]])[:, 1]

    catboost_bundle = joblib.load("models/strict_no_leak_catboost_calibrated.joblib")
    catboost_prob = score_frame(catboost_bundle, test)

    if len(y_true) != len(logistic_prob) or len(y_true) != len(catboost_prob):
        raise ValueError("Model probabilities and y_true must have the same length.")

    from sklearn.metrics import average_precision_score

    metrics = overview_metrics(
        y_true=y_true,
        y_score=catboost_prob,
        logistic_pr_auc=float(average_precision_score(y_true, logistic_prob)),
        model_version=MODEL_VERSION,
        evaluation_timestamp=datetime.now(UTC).isoformat(),
        data_processing_version=data_processing_version(),
    )
    metrics.update(
        {
            "feature_set": "strict_no_leak",
            "test_split_path": "data/processed/strict_no_leak_testing.pkl",
            "catboost_model_path": "models/strict_no_leak_catboost_calibrated.joblib",
            "logistic_model_path": "models/strict_no_leak_logistic_regression.joblib",
            "probability_source": "calibrated CatBoost probability",
            "target_column": TARGET,
            "leakage_audit_summary": {
                "strict_feature_set_leakage_risk_features": [],
                "uses_return_derived_node_aggregates": False,
                "uses_post_purchase_fields": False,
                "uses_test_labels_for_training": False,
                "uses_test_data_for_history_features": False,
            },
        }
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"Wrote {OUTPUT_PATH}")
    print(
        json.dumps(
            {
                "test_positive_rate": metrics["test_positive_rate"],
                "logistic_pr_auc": metrics["logistic_pr_auc"],
                "catboost_pr_auc": metrics["catboost_pr_auc"],
                "recall_at_10": metrics["recall_at_10"],
                "precision_at_10": metrics["precision_at_10"],
                "lift_at_10": metrics["lift_at_10"],
                "brier_score": metrics["brier_score"],
                "constant_baseline_brier": metrics["constant_baseline_brier"],
                "brier_skill_score": metrics["brier_skill_score"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

