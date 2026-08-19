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
PEER_OPTION_LABEL = "same-brand, same-product-type historical peer"
PEER_RISK_BASIS = (
    "Candidate risk is a model estimate rescored under the current checkout user's "
    "available profile fields."
)
INVENTORY_NOT_VERIFIED = "Inventory not verified."
NON_CAUSAL_RECOMMENDATION_DISCLAIMER = (
    "Estimated risk change is not randomized causal evidence of reduced returns."
)

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
        "variant_id": str(best[VARIANT_KEY]),
        "product_type": str(best["productType"]),
        "brand": str(best["brandDesc"]),
        "risk_probability": float(best["risk_probability"]),
        "relative_risk_change": float(delta),
        "reason": (
            "Lower-risk peer option from the same-brand, "
            "same-product-type historical peer pool."
        ),
        "candidate_type": PEER_OPTION_LABEL,
        "risk_basis": PEER_RISK_BASIS,
        "inventory_status": INVENTORY_NOT_VERIFIED,
        "disclaimer": NON_CAUSAL_RECOMMENDATION_DISCLAIMER,
    }


def scenario_payload(
    name: str,
    label: str,
    behavior: str,
    case_type: str,
    selection_rule: str,
    row: pd.Series,
    probability: float,
    bundle: dict[str, Any],
    alternative: dict[str, Any] | None,
) -> dict[str, Any]:
    row_frame = pd.DataFrame([row])
    prediction_margin = confidence_from_probability(probability)
    observed_outcome = "returned" if int(row[TARGET]) == 1 else "not_returned"
    return {
        "id": name,
        "label": label,
        "behavior": behavior,
        "case_type": case_type,
        "selection_rule": selection_rule,
        "user_id": str(row[CUSTOMER_KEY]),
        "variant_id": str(row[VARIANT_KEY]),
        "country": str(row["shippingCountry"]),
        "product_type": str(row["productType"]),
        "brand": str(row["brandDesc"]),
        "risk_probability": float(probability),
        "risk_level": risk_level(probability),
        "prediction_margin": prediction_margin,
        "confidence": prediction_margin,
        "observed_outcome": observed_outcome,
        "observed_outcome_hidden_by_default": True,
        "observed_outcome_note": (
            "Observed strict-test outcome; used only for error analysis and hidden "
            "until revealed in the UI."
        ),
        "top_factors": shap_top_factors(bundle, row_frame),
        "alternative": alternative,
    }


def sorted_pool(
    frame: pd.DataFrame,
    condition: pd.Series,
    *,
    risk_descending: bool,
) -> pd.DataFrame:
    return frame[condition].sort_values(
        ["risk_probability", CUSTOMER_KEY, VARIANT_KEY],
        ascending=[not risk_descending, True, True],
    )


def select_scenario(
    *,
    pool: pd.DataFrame,
    selected_pairs: set[tuple[int, int]],
    bundle: dict[str, Any],
    catalog: pd.DataFrame,
    require_alternative: bool | None,
    max_scan: int = 5_000,
) -> tuple[pd.Series, dict[str, Any] | None]:
    for _, row in pool.head(max_scan).iterrows():
        pair = (int(row[CUSTOMER_KEY]), int(row[VARIANT_KEY]))
        if pair in selected_pairs:
            continue
        alternative = None
        if row["has_complete_metadata"]:
            alternative = find_alternative(
                bundle,
                row,
                catalog,
                float(row["risk_probability"]),
            )
        if require_alternative is not None and (alternative is not None) != require_alternative:
            continue
        selected_pairs.add(pair)
        return row, alternative
    raise RuntimeError("Could not find deterministic demo scenario for rule.")


def generate_demo_scenarios() -> list[dict[str, Any]]:
    bundle = load_calibrated_bundle()
    frame = load_processed("strict_no_leak", "testing")
    frame = frame.copy()
    frame["risk_probability"] = score_frame(bundle, frame)
    frame["prediction_margin"] = frame["risk_probability"].map(confidence_from_probability)
    complete = frame[frame["has_complete_metadata"]].copy()
    catalog = product_catalog(frame)

    scenarios: list[dict[str, Any]] = []
    selected_pairs: set[tuple[int, int]] = set()
    scenario_rules = [
        {
            "name": "true_positive_high_risk_with_peer",
            "label": "High risk with peer",
            "behavior": "High model risk with a lower-risk peer option available.",
            "case_type": "true_positive",
            "selection_rule": (
                "Official strict test row; complete metadata; observed return; "
                "risk 0.75-0.90; prediction margin >= 0.40; first stable row with "
                "same-brand, same-product-type lower-risk peer."
            ),
            "frame": complete,
            "condition": (
                (complete["risk_probability"] >= 0.75)
                & (complete["risk_probability"] <= 0.90)
                & (complete["prediction_margin"] >= 0.40)
                & (complete[TARGET] == 1)
            ),
            "risk_descending": True,
            "require_alternative": True,
        },
        {
            "name": "false_positive_high_risk_with_peer",
            "label": "High risk peer error analysis",
            "behavior": (
                "High model risk with a peer option; useful for error analysis after reveal."
            ),
            "case_type": "false_positive",
            "selection_rule": (
                "Official strict test row; complete metadata; observed non-return; "
                "risk >= 0.70; first stable row with lower-risk peer."
            ),
            "frame": complete,
            "condition": (complete["risk_probability"] >= 0.70) & (complete[TARGET] == 0),
            "risk_descending": True,
            "require_alternative": True,
        },
        {
            "name": "true_positive_high_risk_without_peer",
            "label": "High risk without peer",
            "behavior": (
                "Risk is above threshold, but no eligible lower-risk peer option is available."
            ),
            "case_type": "true_positive",
            "selection_rule": (
                "Official strict test row; complete metadata; observed return; "
                "risk 0.60-0.70; first stable row without an eligible lower-risk peer."
            ),
            "frame": complete,
            "condition": (
                (complete["risk_probability"] >= 0.60)
                & (complete["risk_probability"] <= 0.70)
                & (complete[TARGET] == 1)
            ),
            "risk_descending": False,
            "require_alternative": False,
        },
        {
            "name": "false_negative_low_risk_no_prompt",
            "label": "Low risk error analysis",
            "behavior": (
                "Low model risk means the policy does not prompt; reveal outcome for analysis."
            ),
            "case_type": "false_negative",
            "selection_rule": (
                "Official strict test row; complete metadata; observed return; "
                "risk < 0.40; first stable row without an eligible lower-risk peer."
            ),
            "frame": complete,
            "condition": (complete["risk_probability"] < 0.40) & (complete[TARGET] == 1),
            "risk_descending": True,
            "require_alternative": False,
        },
        {
            "name": "true_negative_low_risk_no_prompt",
            "label": "Low risk no prompt",
            "behavior": "Low model risk; no intervention is expected under the current policy.",
            "case_type": "true_negative",
            "selection_rule": (
                "Official strict test row; complete metadata; observed non-return; "
                "risk < 0.40; first stable row without an eligible lower-risk peer."
            ),
            "frame": complete,
            "condition": (complete["risk_probability"] < 0.40) & (complete[TARGET] == 0),
            "risk_descending": True,
            "require_alternative": False,
        },
        {
            "name": "low_margin_borderline_no_prompt",
            "label": "Low margin borderline",
            "behavior": "Borderline prediction near threshold; policy margin blocks intervention.",
            "case_type": "low_margin",
            "selection_rule": (
                "Official strict test row; complete metadata; observed return; "
                "risk 0.58-0.60; prediction margin <= 0.20; first stable row without "
                "an eligible lower-risk peer."
            ),
            "frame": complete,
            "condition": (
                (complete["risk_probability"] >= 0.58)
                & (complete["risk_probability"] < 0.60)
                & (complete["prediction_margin"] <= 0.20)
                & (complete[TARGET] == 1)
            ),
            "risk_descending": False,
            "require_alternative": False,
        },
        {
            "name": "incomplete_metadata_high_risk_no_peer",
            "label": "Incomplete metadata",
            "behavior": "High model risk, but incomplete product metadata prevents peer matching.",
            "case_type": "metadata_incomplete",
            "selection_rule": (
                "Official strict test row; incomplete metadata; risk >= 0.75; first "
                "stable row. Observed outcome is used only for revealable error analysis."
            ),
            "frame": frame,
            "condition": (frame["risk_probability"] >= 0.75) & (~frame["has_complete_metadata"]),
            "risk_descending": True,
            "require_alternative": None,
        },
    ]

    for rule in scenario_rules:
        pool = sorted_pool(
            rule["frame"],
            rule["condition"],
            risk_descending=rule["risk_descending"],
        )
        row, alternative = select_scenario(
            pool=pool,
            selected_pairs=selected_pairs,
            bundle=bundle,
            catalog=catalog,
            require_alternative=rule["require_alternative"],
        )
        scenarios.append(
            scenario_payload(
                rule["name"],
                rule["label"],
                rule["behavior"],
                rule["case_type"],
                rule["selection_rule"],
                row,
                float(row["risk_probability"]),
                bundle,
                alternative,
            )
        )

    if len(scenarios) != 7:
        raise RuntimeError(f"Expected 7 scenarios, got {len(scenarios)}.")
    return scenarios


def write_demo_scenarios(scenarios: list[dict[str, Any]]) -> Path:
    OUTPUT_PATH.write_text(json.dumps(scenarios, indent=2))
    return OUTPUT_PATH
