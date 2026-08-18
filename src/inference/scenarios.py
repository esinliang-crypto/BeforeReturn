from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import Pool

from src.data.dataset import CUSTOMER_KEY, TARGET, VARIANT_KEY
from src.training.train import load_processed

OUTPUT_PATH = Path("data/samples/demo_scenarios.json")

CUSTOMER_FEATURES = ["yearOfBirth", "isMale", "shippingCountry", "premier"]
PRODUCT_FEATURES = [
    "productType",
    "brandDesc",
    "avgGbpPrice",
    "avgDiscountValue",
    "productType__missing",
    "brandDesc__missing",
    "avgGbpPrice__missing",
    "avgDiscountValue__missing",
]


def confidence_from_probability(probability: float) -> float:
    return float(abs(probability - 0.5) * 2)


def risk_level(probability: float) -> str:
    if probability >= 0.6:
        return "high"
    if probability >= 0.4:
        return "medium"
    return "low"


def load_calibrated_bundle(
    path: Path = Path("models/strict_no_leak_catboost_calibrated.joblib"),
) -> dict[str, Any]:
    return joblib.load(path)


def score_frame(bundle: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    features = bundle["features"]
    categorical = bundle["categorical_features"]
    cat_indices = [features.index(column) for column in categorical]
    raw_scores = bundle["model"].predict_proba(
        Pool(frame[features], cat_features=cat_indices)
    )[:, 1]
    return bundle["calibrator"].transform(raw_scores)


def shap_top_factors(
    bundle: dict[str, Any],
    row: pd.DataFrame,
    limit: int = 3,
) -> list[dict[str, Any]]:
    features = bundle["features"]
    categorical = bundle["categorical_features"]
    cat_indices = [features.index(column) for column in categorical]
    values = bundle["model"].get_feature_importance(
        Pool(row[features], cat_features=cat_indices),
        type="ShapValues",
    )[0][:-1]
    order = np.argsort(np.abs(values))[::-1][:limit]
    return [
        {
            "feature": features[index],
            "impact": float(values[index]),
            "direction": "raises risk" if values[index] > 0 else "lowers risk",
        }
        for index in order
    ]


def product_catalog(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [VARIANT_KEY, *PRODUCT_FEATURES]
    catalog = frame[frame["has_complete_metadata"]][columns].drop_duplicates(subset=[VARIANT_KEY])
    return catalog.reset_index(drop=True)


def candidate_rows_for_user(
    current: pd.Series,
    candidates: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    rows = candidates.copy()
    for column in CUSTOMER_FEATURES:
        rows[column] = current[column]
    for column in features:
        if column not in rows.columns:
            rows[column] = 0
    return rows


def find_alternative(
    bundle: dict[str, Any],
    current: pd.Series,
    catalog: pd.DataFrame,
    current_probability: float,
    min_delta: float = 0.1,
) -> dict[str, Any] | None:
    candidates = catalog[
        (catalog["brandDesc"] == current["brandDesc"])
        & (catalog["productType"] == current["productType"])
        & (catalog[VARIANT_KEY] != current[VARIANT_KEY])
    ].copy()
    if candidates.empty:
        return None

    rows = candidate_rows_for_user(current, candidates, bundle["features"])
    probabilities = score_frame(bundle, rows)
    rows["risk_probability"] = probabilities
    rows = rows.sort_values("risk_probability")
    best = rows.iloc[0]
    delta = current_probability - float(best["risk_probability"])
    if delta < min_delta:
        return None
    return {
        "variant_id": int(best[VARIANT_KEY]),
        "product_type": str(best["productType"]),
        "brand": str(best["brandDesc"]),
        "risk_probability": float(best["risk_probability"]),
        "relative_risk_change": float(delta),
        "reason": "Same brand and product type with lower predicted return risk.",
    }


def scenario_payload(
    name: str,
    row: pd.Series,
    probability: float,
    bundle: dict[str, Any],
    alternative: dict[str, Any] | None,
) -> dict[str, Any]:
    row_frame = pd.DataFrame([row])
    confidence = confidence_from_probability(probability)
    return {
        "id": name,
        "user_id": int(row[CUSTOMER_KEY]),
        "variant_id": int(row[VARIANT_KEY]),
        "country": str(row["shippingCountry"]),
        "product_type": str(row["productType"]),
        "brand": str(row["brandDesc"]),
        "risk_probability": float(probability),
        "risk_level": risk_level(probability),
        "confidence": confidence,
        "actual_return_label": int(row[TARGET]),
        "top_factors": shap_top_factors(bundle, row_frame),
        "alternative": alternative,
    }


def generate_demo_scenarios() -> list[dict[str, Any]]:
    bundle = load_calibrated_bundle()
    frame = load_processed("strict_no_leak", "testing")
    complete = frame[frame["has_complete_metadata"]].copy()
    complete["risk_probability"] = score_frame(bundle, complete)
    complete["confidence"] = complete["risk_probability"].map(confidence_from_probability)
    catalog = product_catalog(complete)

    scenarios: list[dict[str, Any]] = []

    high_pool = complete[
        (complete["risk_probability"] >= 0.75)
        & (complete["risk_probability"] <= 0.9)
        & (complete["confidence"] >= 0.4)
        & (complete[TARGET] == 1)
    ].sort_values("risk_probability", ascending=False)
    for _, row in high_pool.head(2_000).iterrows():
        alternative = find_alternative(bundle, row, catalog, float(row["risk_probability"]))
        if alternative is not None:
            scenarios.append(
                scenario_payload(
                    "high_risk_high_confidence_with_alternative",
                    row,
                    float(row["risk_probability"]),
                    bundle,
                    alternative,
                )
            )
            break

    low_conf = complete[
        (complete["risk_probability"] >= 0.6)
        & (complete["risk_probability"] <= 0.62)
        & (complete["confidence"] <= 0.25)
        & (complete[TARGET] == 1)
    ].sort_values("risk_probability")
    if low_conf.empty:
        raise RuntimeError("Could not find high-risk low-confidence scenario.")
    row = low_conf.iloc[0]
    scenarios.append(
        scenario_payload(
            "high_risk_low_confidence_no_intervention",
            row,
            float(row["risk_probability"]),
            bundle,
            None,
        )
    )

    low_risk = complete[
        (complete["risk_probability"] >= 0.15)
        & (complete["risk_probability"] <= 0.35)
        & (complete[TARGET] == 0)
    ].sort_values("risk_probability")
    if low_risk.empty:
        raise RuntimeError("Could not find low-risk scenario.")
    row = low_risk.iloc[0]
    scenarios.append(
        scenario_payload(
            "low_risk_no_intervention",
            row,
            float(row["risk_probability"]),
            bundle,
            None,
        )
    )

    if len(scenarios) != 3:
        raise RuntimeError(f"Expected 3 scenarios, got {len(scenarios)}.")
    return scenarios


def write_demo_scenarios(scenarios: list[dict[str, Any]]) -> Path:
    OUTPUT_PATH.write_text(json.dumps(scenarios, indent=2))
    return OUTPUT_PATH
