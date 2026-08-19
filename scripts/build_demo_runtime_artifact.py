from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.dataset import CUSTOMER_KEY, VARIANT_KEY
from src.inference.runtime_artifact import (
    RUNTIME_ARTIFACT_PATH,
    RUNTIME_SCHEMA_VERSION,
    write_runtime_artifact,
)
from src.inference.scenarios import PRODUCT_FEATURES, load_calibrated_bundle, product_catalog
from src.training.train import load_processed

MODEL_VERSION = "strict_no_leak_catboost_calibrated_v1"
DEFAULT_SCENARIOS_PATH = Path("data/samples/demo_scenarios.json")


def json_safe(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if hasattr(value, "item"):
        return value.item()
    return value


def records_for_columns(rows, columns: list[str]) -> list[dict[str, Any]]:
    return [
        {column: json_safe(row[column]) for column in columns}
        for _, row in rows.iterrows()
    ]


def build_runtime_payload(
    *,
    model_path: Path,
    scenarios_path: Path,
    feature_set: str,
) -> dict[str, Any]:
    bundle = load_calibrated_bundle(model_path)
    testing = load_processed(feature_set, "testing")
    scenarios = json.loads(scenarios_path.read_text())
    features = bundle["features"]
    checkout_columns = [
        CUSTOMER_KEY,
        VARIANT_KEY,
        "shippingCountry",
        "productType",
        "brandDesc",
        "has_complete_metadata",
        *features,
    ]
    checkout_columns = list(dict.fromkeys(checkout_columns))

    checkout_rows = []
    for scenario in scenarios:
        user_id = int(scenario["user_id"])
        variant_id = int(scenario["variant_id"])
        match = testing[
            (testing[CUSTOMER_KEY] == user_id)
            & (testing[VARIANT_KEY] == variant_id)
        ]
        if match.empty:
            raise RuntimeError(f"Missing checkout row for scenario {scenario['id']}.")
        row = match.iloc[[0]].copy()
        row["scenario_id"] = scenario["id"]
        checkout_rows.append(row[["scenario_id", *checkout_columns]])

    checkout_frame = pd.concat(checkout_rows, ignore_index=True)

    catalog = product_catalog(testing)
    candidate_ids = {
        int(scenario["alternative"]["variant_id"])
        for scenario in scenarios
        if scenario.get("alternative")
    }
    peer_candidates = catalog[catalog[VARIANT_KEY].isin(candidate_ids)].copy()
    missing_candidates = candidate_ids - set(peer_candidates[VARIANT_KEY].astype(int))
    if missing_candidates:
        raise RuntimeError(f"Missing peer candidate variants: {sorted(missing_candidates)}")

    return {
        "artifact_schema_version": RUNTIME_SCHEMA_VERSION,
        "feature_set": feature_set,
        "model_version": MODEL_VERSION,
        "model_path": str(model_path),
        "source_scenarios_path": str(scenarios_path),
        "source_testing_path": f"data/processed/{feature_set}_testing.pkl",
        "generation_strategy": (
            "Deterministically selected from data/samples/demo_scenarios.json and "
            "bounded to the matching checkout rows plus referenced peer candidates."
        ),
        "checkout_row_count": int(len(checkout_frame)),
        "peer_candidate_count": int(len(peer_candidates)),
        "features": features,
        "categorical_features": bundle["categorical_features"],
        "notes": [
            "Runtime artifact contains only demo checkout rows and bounded peer candidates.",
            (
                "It is generated locally from processed data but fresh-clone demo "
                "startup must not download processed data."
            ),
            "Target labels are excluded from runtime checkout rows and peer candidates.",
        ],
        "checkout_rows": records_for_columns(
            checkout_frame,
            ["scenario_id", *checkout_columns],
        ),
        "peer_candidates": records_for_columns(
            peer_candidates,
            [VARIANT_KEY, *PRODUCT_FEATURES],
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the small demo runtime artifact.")
    parser.add_argument("--feature-set", default="strict_no_leak")
    parser.add_argument(
        "--model-path",
        default="models/strict_no_leak_catboost_calibrated.joblib",
    )
    parser.add_argument("--scenarios-path", default=str(DEFAULT_SCENARIOS_PATH))
    parser.add_argument("--output-path", default=str(RUNTIME_ARTIFACT_PATH))
    args = parser.parse_args()

    payload = build_runtime_payload(
        model_path=Path(args.model_path),
        scenarios_path=Path(args.scenarios_path),
        feature_set=args.feature_set,
    )
    output_path = write_runtime_artifact(payload, Path(args.output_path))
    print(f"Wrote {output_path}")
    print(
        json.dumps(
            {
                "checkout_row_count": payload["checkout_row_count"],
                "peer_candidate_count": payload["peer_candidate_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
