from __future__ import annotations

from fastapi.testclient import TestClient

import api.main as api_main
from api.service import InferenceService

FULL_ARTIFACT_ROWS = 1_460_366


def test_every_demo_scenario_predicts_and_recommends(monkeypatch) -> None:
    monkeypatch.setattr(api_main, "service", InferenceService())
    client = TestClient(api_main.app)

    scenarios_response = client.get("/demo-scenarios")
    assert scenarios_response.status_code == 200
    scenarios = scenarios_response.json()

    assert 6 <= len(scenarios) <= 8
    assert {scenario["case_type"] for scenario in scenarios} >= {
        "true_positive",
        "false_positive",
        "false_negative",
        "true_negative",
        "low_margin",
        "metadata_incomplete",
    }

    for scenario in scenarios:
        assert scenario["observed_outcome_hidden_by_default"] is True
        assert scenario["observed_outcome"] in {"returned", "not_returned"}
        assert "actual_return_label" not in scenario
        payload = {
            "user_id": scenario["user_id"],
            "variant_id": scenario["variant_id"],
            "policy": {},
        }
        prediction = client.post("/predict-return-risk", json=payload)
        recommendation = client.post("/recommend-alternatives", json=payload)

        assert prediction.status_code == 200
        assert recommendation.status_code == 200
        prediction_payload = prediction.json()
        recommendation_payload = recommendation.json()
        assert prediction_payload["model_version"]
        assert recommendation_payload["model_version"] == prediction_payload["model_version"]
        if recommendation_payload["alternative"] is not None:
            assert recommendation_payload["alternative"]["candidate_type"] == (
                "same-brand, same-product-type historical peer"
            )
            assert recommendation_payload["alternative"]["inventory_status"] == (
                "Inventory not verified."
            )

    policy = client.post("/simulate-policy", json={"policy": {}})
    assert policy.status_code == 200
    assert policy.json()["evaluated_checkouts"] == FULL_ARTIFACT_ROWS
