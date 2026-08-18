from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.isotonic import IsotonicRegression

from src.data.dataset import TARGET
from src.evaluation.metrics import binary_metrics, slice_metrics
from src.training.splits import split_metadata, split_train_validation_calibration
from src.training.train import categorical_columns, feature_columns, load_processed

MODEL_DIR = Path("models")
REPORT_DIR = Path("reports/metrics")


@dataclass(frozen=True)
class CalibrationConfig:
    feature_set: str = "strict_no_leak"
    random_seed: int = 42
    calibration_size: float = 0.2
    validation_size: float = 0.16


def fit_base_catboost(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    categorical: list[str],
    random_seed: int,
) -> CatBoostClassifier:
    cat_indices = [features.index(column) for column in categorical]
    train_pool = Pool(train[features], label=train[TARGET], cat_features=cat_indices)
    validation_pool = Pool(validation[features], label=validation[TARGET], cat_features=cat_indices)
    model = CatBoostClassifier(
        iterations=350,
        learning_rate=0.08,
        depth=6,
        loss_function="Logloss",
        eval_metric="PRAUC",
        random_seed=random_seed,
        verbose=50,
        allow_writing_files=False,
    )
    model.fit(train_pool, eval_set=validation_pool, use_best_model=True)
    return model


def catboost_scores(
    model: CatBoostClassifier,
    frame: pd.DataFrame,
    features: list[str],
    categorical: list[str],
) -> Any:
    cat_indices = [features.index(column) for column in categorical]
    pool = Pool(frame[features], cat_features=cat_indices)
    return model.predict_proba(pool)[:, 1]


def calibrate_catboost(config: CalibrationConfig) -> dict[str, Any]:
    full_train = load_processed(config.feature_set, "training")
    test = load_processed(config.feature_set, "testing")
    features = feature_columns(full_train)
    categorical = categorical_columns(full_train, features)

    internal_split = split_train_validation_calibration(
        full_train,
        validation_size=config.validation_size,
        calibration_size=config.calibration_size,
        random_seed=config.random_seed,
    )
    if internal_split.calibration is None:
        raise ValueError("Calibration split was not created.")

    model = fit_base_catboost(
        internal_split.train_fit,
        internal_split.validation,
        features,
        categorical,
        config.random_seed,
    )
    calibration_scores = catboost_scores(model, internal_split.calibration, features, categorical)
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(calibration_scores, internal_split.calibration[TARGET])

    raw_test_scores = catboost_scores(model, test, features, categorical)
    calibrated_test_scores = calibrator.transform(raw_test_scores)
    metrics = binary_metrics(test[TARGET], calibrated_test_scores)
    metrics.update(
        {
            "feature_set": config.feature_set,
            "model_name": "catboost_calibrated_isotonic",
            "feature_count": len(features),
            "categorical_features": categorical,
            "base_model": "CatBoostClassifier",
            "calibrator": "IsotonicRegression",
            "raw_brier_score": binary_metrics(test[TARGET], raw_test_scores)["brier_score"],
            "slice_metrics": [
                *slice_metrics(test, calibrated_test_scores, "shippingCountry"),
                *slice_metrics(test, calibrated_test_scores, "productType"),
            ],
        }
    )
    metrics.update(
        split_metadata(
            random_seed=config.random_seed,
            train_fit=internal_split.train_fit,
            validation=internal_split.validation,
            calibration=internal_split.calibration,
            test=test,
        )
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"{config.feature_set}_catboost_calibrated.joblib"
    metrics_path = REPORT_DIR / f"{config.feature_set}_catboost_calibrated.json"
    joblib.dump(
        {
            "model": model,
            "calibrator": calibrator,
            "features": features,
            "categorical_features": categorical,
            "feature_set": config.feature_set,
        },
        model_path,
    )
    metrics["model_path"] = str(model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    return metrics
