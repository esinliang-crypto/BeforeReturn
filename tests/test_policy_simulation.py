from __future__ import annotations

import numpy as np
import pandas as pd

import src.inference.policy_simulation as policy_module
from api.schemas import PolicySettings
from src.data.dataset import TARGET, VARIANT_KEY
from src.inference.policy_simulation import (
    PolicyArtifactConfig,
    compute_policy_artifact,
    prediction_margin,
    simulate_from_artifact,
)


def test_prediction_margin_is_probability_distance_proxy() -> None:
    assert prediction_margin(0.5) == 0
    assert prediction_margin(0.9) == 0.8
    assert prediction_margin(0.1) == 0.8


def test_simulate_from_artifact_applies_gates_and_prompt_cap() -> None:
    artifact = pd.DataFrame(
        {
            "risk_probability": [0.95, 0.9, 0.8, 0.7, 0.4],
            "prediction_margin": [0.9, 0.8, 0.6, 0.4, 0.2],
            "is_returned": [True, False, True, False, True],
            "complete_metadata": [True, True, True, True, True],
            "has_peer_candidate": [True, True, True, True, True],
            "best_peer_risk_probability": [0.4, 0.5, 0.75, 0.2, 0.1],
            "best_peer_risk_reduction": [0.55, 0.4, 0.05, 0.5, 0.3],
            "rescored_candidate_count": [2, 2, 2, 2, 2],
        }
    )

    metrics = simulate_from_artifact(
        artifact,
        PolicySettings(
            high_risk_threshold=0.6,
            min_prediction_margin=0.3,
            max_prompts_per_1000=400,
            min_risk_reduction=0.1,
        ),
    )

    assert metrics["evaluated_checkouts"] == 5
    assert metrics["prompt_budget"] == 2
    assert metrics["eligible_checkouts"] == 3
    assert metrics["estimated_prompts"] == 2
    assert metrics["false_positives"] == 1
    assert metrics["precision_at_policy"] == 0.5
    assert metrics["recall_at_policy"] == 1 / 3
    assert metrics["prompt_coverage"] == 0.4
    assert metrics["user_disturbance_rate"] == 0.4


def test_simulate_from_artifact_uses_request_min_risk_reduction_dynamically() -> None:
    artifact = pd.DataFrame(
        {
            "risk_probability": [0.8],
            "prediction_margin": [0.6],
            "is_returned": [True],
            "complete_metadata": [True],
            "has_peer_candidate": [True],
            "best_peer_risk_probability": [0.65],
            "best_peer_risk_reduction": [0.15],
            "rescored_candidate_count": [1],
        }
    )

    loose = simulate_from_artifact(
        artifact,
        PolicySettings(max_prompts_per_1000=1000, min_risk_reduction=0.1),
    )
    strict = simulate_from_artifact(
        artifact,
        PolicySettings(max_prompts_per_1000=1000, min_risk_reduction=0.2),
    )

    assert loose["estimated_prompts"] == 1
    assert strict["estimated_prompts"] == 0


def test_compute_policy_artifact_rescores_training_peers_under_current_user(monkeypatch) -> None:
    def fake_score_frame(bundle, frame):
        return frame["avgGbpPrice"].to_numpy(dtype=float) / 100

    monkeypatch.setattr(policy_module, "score_frame", fake_score_frame)
    bundle = {
        "features": [
            "yearOfBirth",
            "isMale",
            "shippingCountry",
            "premier",
            "productType",
            "brandDesc",
            "avgGbpPrice",
            "avgDiscountValue",
            "productType__missing",
            "brandDesc__missing",
            "avgGbpPrice__missing",
            "avgDiscountValue__missing",
        ]
    }
    training = pd.DataFrame(
        {
            VARIANT_KEY: [1, 1, 2, 3],
            TARGET: [0, 1, 1, 0],
            "has_complete_metadata": [True, True, True, True],
            "yearOfBirth": [1980, 1980, 1980, 1980],
            "isMale": [1, 1, 1, 1],
            "shippingCountry": ["Country_A"] * 4,
            "premier": [0, 0, 0, 0],
            "productType": ["Dress"] * 4,
            "brandDesc": ["Brand_A"] * 4,
            "avgGbpPrice": [20, 20, 40, 60],
            "avgDiscountValue": [0, 0, 0, 0],
            "productType__missing": [0, 0, 0, 0],
            "brandDesc__missing": [0, 0, 0, 0],
            "avgGbpPrice__missing": [0, 0, 0, 0],
            "avgDiscountValue__missing": [0, 0, 0, 0],
        }
    )
    testing = pd.DataFrame(
        {
            VARIANT_KEY: [9],
            TARGET: [1],
            "has_complete_metadata": [True],
            "yearOfBirth": [1990],
            "isMale": [0],
            "shippingCountry": ["Country_B"],
            "premier": [1],
            "productType": ["Dress"],
            "brandDesc": ["Brand_A"],
            "avgGbpPrice": [80],
            "avgDiscountValue": [0],
            "productType__missing": [0],
            "brandDesc__missing": [0],
            "avgGbpPrice__missing": [0],
            "avgDiscountValue__missing": [0],
        }
    )

    artifact, manifest = compute_policy_artifact(
        bundle=bundle,
        training=training,
        testing=testing,
        config=PolicyArtifactConfig(candidate_k=2),
    )

    assert list(artifact.columns) == policy_module.POLICY_COLUMNS
    assert artifact.loc[0, "risk_probability"] == np.float32(0.8)
    assert artifact.loc[0, "has_peer_candidate"]
    assert artifact.loc[0, "rescored_candidate_count"] == 2
    assert artifact.loc[0, "best_peer_risk_probability"] == np.float32(0.2)
    assert artifact.loc[0, "best_peer_risk_reduction"] == np.float32(0.6)
    assert manifest["candidate_source"] == "official_training_catalog_only"
    assert manifest["uses_labels_for_candidate_pool"] is False
