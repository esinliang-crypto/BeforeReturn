from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.dataset import TARGET
from src.evaluation.metrics import binary_metrics, slice_metrics
from src.training.splits import split_metadata, split_train_validation

PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports/metrics")


@dataclass(frozen=True)
class TrainingConfig:
    feature_set: str
    model_name: str
    train_limit: int | None = None
    test_limit: int | None = None
    random_seed: int = 42


def load_processed(feature_set: str, split: str, limit: int | None = None) -> pd.DataFrame:
    frame = pd.read_pickle(PROCESSED_DIR / f"{feature_set}_{split}.pkl")
    if limit is not None:
        frame = frame.sample(n=min(limit, len(frame)), random_state=42)
    return frame


def feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {TARGET, "hash(customerId)", "hash(variantID)", "has_complete_metadata"}
    return [column for column in frame.columns if column not in excluded]


def categorical_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    return [
        column
        for column in columns
        if pd.api.types.is_object_dtype(frame[column])
        or pd.api.types.is_string_dtype(frame[column])
    ]


def build_logistic_pipeline(categorical: list[str], numeric: list[str]) -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", min_frequency=50),
                categorical,
            ),
            ("numeric", StandardScaler(), numeric),
        ],
        sparse_threshold=0.3,
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "model",
                LogisticRegression(
                    solver="saga",
                    max_iter=120,
                    tol=1e-3,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def train_logistic(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    categorical: list[str],
) -> tuple[Pipeline, dict[str, Any]]:
    numeric = [column for column in features if column not in categorical]
    pipeline = build_logistic_pipeline(categorical, numeric)
    pipeline.fit(train[features], train[TARGET])
    scores = pipeline.predict_proba(test[features])[:, 1]
    metrics = binary_metrics(test[TARGET], scores)
    return pipeline, metrics


def train_catboost(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    categorical: list[str],
    random_seed: int,
) -> tuple[CatBoostClassifier, dict[str, Any]]:
    internal_split = split_train_validation(train, validation_size=0.2, random_seed=random_seed)
    cat_indices = [features.index(column) for column in categorical]
    train_pool = Pool(
        internal_split.train_fit[features],
        label=internal_split.train_fit[TARGET],
        cat_features=cat_indices,
    )
    validation_pool = Pool(
        internal_split.validation[features],
        label=internal_split.validation[TARGET],
        cat_features=cat_indices,
    )
    test_pool = Pool(test[features], label=test[TARGET], cat_features=cat_indices)
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
    scores = model.predict_proba(test_pool)[:, 1]
    metrics = binary_metrics(test[TARGET], scores)
    metrics.update(
        split_metadata(
            random_seed=random_seed,
            train_fit=internal_split.train_fit,
            validation=internal_split.validation,
            test=test,
        )
    )
    return model, metrics


def train_model(config: TrainingConfig) -> dict[str, Any]:
    train = load_processed(config.feature_set, "training", config.train_limit)
    test = load_processed(config.feature_set, "testing", config.test_limit)
    features = feature_columns(train)
    categorical = categorical_columns(train, features)

    if config.model_name == "logistic_regression":
        model, metrics = train_logistic(train, test, features, categorical)
    elif config.model_name == "catboost":
        model, metrics = train_catboost(train, test, features, categorical, config.random_seed)
    else:
        raise ValueError(f"Unknown model: {config.model_name}")

    scores = metrics_to_scores(model, test, features, categorical)
    metrics.update(
        {
            "feature_set": config.feature_set,
            "model_name": config.model_name,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "feature_count": len(features),
            "categorical_features": categorical,
            "slice_metrics": [
                *slice_metrics(test, scores, "shippingCountry"),
                *slice_metrics(test, scores, "productType"),
            ],
        }
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"{config.feature_set}_{config.model_name}.joblib"
    metrics_path = REPORT_DIR / f"{config.feature_set}_{config.model_name}.json"
    joblib.dump(
        {"model": model, "features": features, "categorical_features": categorical},
        model_path,
    )
    metrics["model_path"] = str(model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    return metrics


def metrics_to_scores(
    model: Pipeline | CatBoostClassifier,
    test: pd.DataFrame,
    features: list[str],
    categorical: list[str],
) -> Any:
    if isinstance(model, CatBoostClassifier):
        cat_indices = [features.index(column) for column in categorical]
        return model.predict_proba(Pool(test[features], cat_features=cat_indices))[:, 1]
    return model.predict_proba(test[features])[:, 1]
