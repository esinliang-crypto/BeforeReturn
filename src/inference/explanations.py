from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import Pool

from src.training.train import feature_columns, load_processed

REPORT_DIR = Path("reports/explanations")


def load_model_bundle(path: Path) -> dict[str, Any]:
    return joblib.load(path)


def sample_complete_metadata(
    feature_set: str,
    split: str = "testing",
    sample_size: int = 5_000,
    random_seed: int = 42,
) -> pd.DataFrame:
    frame = load_processed(feature_set, split)
    frame = frame[frame["has_complete_metadata"]].copy()
    return frame.sample(n=min(sample_size, len(frame)), random_state=random_seed)


def catboost_shap_summary(
    model_path: Path,
    feature_set: str = "strict_no_leak",
    sample_size: int = 5_000,
) -> dict[str, Any]:
    bundle = load_model_bundle(model_path)
    model = bundle["model"]
    features = bundle.get("features") or feature_columns(load_processed(feature_set, "training"))
    categorical = bundle.get("categorical_features", [])
    sample = sample_complete_metadata(feature_set, sample_size=sample_size)
    cat_indices = [features.index(column) for column in categorical]
    pool = Pool(sample[features], cat_features=cat_indices)
    shap_values = model.get_feature_importance(pool, type="ShapValues")
    feature_shap = shap_values[:, :-1]
    mean_abs = np.abs(feature_shap).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    top_features = [
        {
            "feature": features[index],
            "mean_abs_shap": float(mean_abs[index]),
        }
        for index in order[:20]
    ]
    return {
        "feature_set": feature_set,
        "model_path": str(model_path),
        "sample_rows": int(len(sample)),
        "complete_metadata_only": True,
        "top_features": top_features,
    }


def write_shap_summary(summary: dict[str, Any]) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORT_DIR / f"{summary['feature_set']}_catboost_shap_summary.json"
    output_path.write_text(json.dumps(summary, indent=2))
    return output_path

