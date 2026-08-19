from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import api.main as api_main
from api.schemas import PolicySettings, PolicySimulationRequest
from api.service import ArtifactMissingError, InferenceService

MODEL_VERSION = "strict_no_leak_catboost_calibrated_v1"
FULL_ARTIFACT_ROWS = 1_460_366


def alternative_payload() -> dict[str, Any]:
    return {
        "variant_id": "peer-variant-1",
        "product_type": "productType_B",
        "brand": "Brand_B",
        "risk_probability": 0.42,
        "relative_risk_change": 0.28,
        "reason": (
            "Lower-risk peer option from the same-brand, "
            "same-product-type historical peer pool."
        ),
        "candidate_type": "same-brand, same-product-type historical peer",
        "risk_basis": (
            "Candidate risk is a model estimate rescored under the current checkout user's "
            "available profile fields."
        ),
        "inventory_status": "Inventory not verified.",
        "disclaimer": "Estimated risk change is not randomized causal evidence of reduced returns.",
    }


def prediction_payload(*, include_alternative: bool = True) -> dict[str, Any]:
    return {
        "user_id": "user-1",
        "variant_id": "variant-1",
        "country": "Country_E",
        "product_type": "productType_B",
        "brand": "Brand_B",
        "risk_probability": 0.7,
        "risk_level": "high",
        "confidence": 0.4,
        "should_intervene": include_alternative,
        "top_factors": [
            {"feature": "productType", "impact": 0.3, "direction": "raises risk"}
        ],
        "policy_reasons": ["pass: a lower-risk peer option is available"],
        "alternative": alternative_payload() if include_alternative else None,
        "model_version": MODEL_VERSION,
    }


class ContractService:
    def __init__(self, metrics_path: Path) -> None:
        self.metrics_path = metrics_path
        self.demo_scenario_count = 3
        self.simulate_policy_called = False

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "model_available": True,
            "processed_data_available": True,
            "model_version": MODEL_VERSION,
        }

    def model_metrics_path(self) -> Path:
        return self.metrics_path

    def demo_scenarios(self) -> list[dict[str, Any]]:
        return [
            {
                "id": "high_risk_high_confidence_with_alternative",
                "user_id": "user-1",
                "variant_id": "variant-1",
                "country": "Country_E",
                "product_type": "productType_B",
                "brand": "Brand_B",
                "risk_probability": 0.7,
                "risk_level": "high",
                "confidence": 0.4,
                "alternative": alternative_payload(),
            },
            {
                "id": "high_risk_low_confidence_no_intervention",
                "user_id": "user-2",
                "variant_id": "variant-2",
                "country": "Country_E",
                "product_type": "productType_H",
                "brand": "Brand_B",
                "risk_probability": 0.61,
                "risk_level": "high",
                "confidence": 0.22,
                "alternative": None,
            },
            {
                "id": "low_risk_no_intervention",
                "user_id": "user-3",
                "variant_id": "variant-3",
                "country": "Country_G",
                "product_type": "productType_J",
                "brand": "Brand_K",
                "risk_probability": 0.16,
                "risk_level": "low",
                "confidence": 0.68,
                "alternative": None,
            },
        ]

    def predict(
        self,
        user_id: str,
        variant_id: str,
        policy: PolicySettings,
    ) -> dict[str, Any]:
        if user_id == "missing":
            raise KeyError("No checkout row found for the requested user and variant.")
        return prediction_payload(include_alternative=policy.peer_recommendations_allowed())

    def simulate_policy(self, policy: PolicySettings) -> dict[str, Any]:
        self.simulate_policy_called = True
        return {
            "evaluated_checkouts": FULL_ARTIFACT_ROWS,
            "estimated_prompts": 120,
            "eligible_checkouts": 300,
            "prompt_budget": 219_054,
            "artifact_rows": FULL_ARTIFACT_ROWS,
            "prompt_coverage": 0.00008,
            "recall_at_policy": 0.01,
            "precision_at_policy": 0.6,
            "false_positives": 48,
            "user_disturbance_rate": 0.00008,
            "disclaimer": (
                "Policy metrics are offline model simulations, not verified causal reductions "
                "in returns. Recommendations are limited to same-brand, same-product-type "
                "historical peers; inventory is not verified."
            ),
        }


class MissingArtifactService:
    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "model_available": False,
            "processed_data_available": False,
            "model_version": MODEL_VERSION,
        }

    def model_metrics_path(self) -> Path:
        raise ArtifactMissingError("Overview metrics are missing.")

    def demo_scenarios(self) -> list[dict[str, Any]]:
        raise ArtifactMissingError("Demo scenarios are missing.")

    def predict(
        self,
        user_id: str,
        variant_id: str,
        policy: PolicySettings,
    ) -> dict[str, Any]:
        raise ArtifactMissingError("Model artifact is missing.")

    def simulate_policy(self, policy: PolicySettings) -> dict[str, Any]:
        raise ArtifactMissingError("Policy simulation artifact is missing.")


@pytest.fixture
def contract_service(tmp_path, monkeypatch) -> ContractService:
    metrics_path = tmp_path / "overview_model_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "test_positive_rate": 0.54,
                "catboost_pr_auc": 0.68,
                "test_sample_count": FULL_ARTIFACT_ROWS,
                "model_version": MODEL_VERSION,
                "evaluation_timestamp": "2026-08-19T00:00:00+00:00",
                "data_processing_version": "dataset_manifest_sha256:test",
            }
        )
    )
    service = ContractService(metrics_path)
    monkeypatch.setattr(api_main, "service", service)
    return service


@pytest.fixture
def client(contract_service) -> TestClient:
    return TestClient(api_main.app)


def test_simulate_policy_contract_uses_artifact_rows(monkeypatch) -> None:
    service = InferenceService()
    service.__dict__["policy_simulation_frame"] = pd.DataFrame(
        {
            "risk_probability": [0.9, 0.8, 0.7, 0.1],
            "prediction_margin": [0.8, 0.6, 0.4, 0.8],
            "is_returned": [True, False, True, False],
            "complete_metadata": [True, True, True, True],
            "has_peer_candidate": [True, True, False, True],
            "best_peer_risk_probability": [0.5, 0.6, float("nan"), 0.0],
            "best_peer_risk_reduction": [0.4, 0.2, float("nan"), 0.1],
            "rescored_candidate_count": [1, 1, 0, 1],
        }
    )
    monkeypatch.setattr(api_main, "service", service)

    payload = api_main.simulate_policy(
        PolicySimulationRequest(
            policy={"high_risk_threshold": 0.6, "max_prompts_per_1000": 1000}
        )
    )

    payload_dict = payload.model_dump()
    assert payload.evaluated_checkouts == 4
    assert payload.artifact_rows == 4
    assert payload.estimated_prompts == 2
    assert payload.eligible_checkouts == 2
    assert payload.prompt_budget == 4
    assert set(payload_dict) >= {
        "evaluated_checkouts",
        "estimated_prompts",
        "prompt_coverage",
        "recall_at_policy",
        "precision_at_policy",
        "false_positives",
        "user_disturbance_rate",
        "disclaimer",
    }


def test_health_endpoint_contract(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "ok": True,
        "model_available": True,
        "processed_data_available": True,
        "model_version": MODEL_VERSION,
    }


def test_model_metrics_endpoint_contract(client: TestClient) -> None:
    response = client.get("/model-metrics")

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_version"] == MODEL_VERSION
    assert payload["test_sample_count"] == FULL_ARTIFACT_ROWS
    assert payload["data_processing_version"] == "dataset_manifest_sha256:test"


def test_demo_scenarios_endpoint_contract(client: TestClient) -> None:
    response = client.get("/demo-scenarios")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 3
    required = {
        "id",
        "user_id",
        "variant_id",
        "country",
        "product_type",
        "brand",
        "risk_probability",
        "risk_level",
        "confidence",
        "alternative",
    }
    assert required <= set(payload[0])
    assert payload[0]["alternative"]["candidate_type"] == (
        "same-brand, same-product-type historical peer"
    )
    assert payload[0]["alternative"]["inventory_status"] == "Inventory not verified."
    assert "not randomized causal evidence" in payload[0]["alternative"]["disclaimer"]


@pytest.mark.parametrize("path", ["/predict-return-risk", "/recommend-alternatives"])
def test_prediction_endpoint_contracts(client: TestClient, path: str) -> None:
    response = client.post(
        path,
        json={
            "user_id": "user-1",
            "variant_id": "variant-1",
            "policy": {"high_risk_threshold": 0.6},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_version"] == MODEL_VERSION
    assert payload["risk_level"] == "high"
    assert payload["should_intervene"] is True
    assert set(payload) >= {
        "risk_probability",
        "risk_level",
        "confidence",
        "should_intervene",
        "top_factors",
        "policy_reasons",
        "model_version",
        "alternative",
    }
    assert payload["alternative"]["candidate_type"] == (
        "same-brand, same-product-type historical peer"
    )
    assert "current checkout user's available profile fields" in payload["alternative"][
        "risk_basis"
    ]
    assert payload["alternative"]["inventory_status"] == "Inventory not verified."
    assert "not randomized causal evidence" in payload["alternative"]["disclaimer"]


def test_predict_missing_checkout_returns_404(client: TestClient) -> None:
    response = client.post(
        "/predict-return-risk",
        json={"user_id": "missing", "variant_id": "variant-1", "policy": {}},
    )

    assert response.status_code == 404
    assert "No checkout row found" in response.json()["detail"]


def test_simulate_policy_endpoint_contract_uses_full_artifact(
    client: TestClient,
    contract_service: ContractService,
) -> None:
    response = client.post(
        "/simulate-policy",
        json={"policy": {"max_prompts_per_1000": 150}},
    )

    assert response.status_code == 200
    payload = response.json()
    assert contract_service.simulate_policy_called is True
    assert contract_service.demo_scenario_count == 3
    assert payload["evaluated_checkouts"] == FULL_ARTIFACT_ROWS
    assert payload["artifact_rows"] == FULL_ARTIFACT_ROWS
    assert payload["evaluated_checkouts"] != contract_service.demo_scenario_count
    assert "offline model simulations" in payload["disclaimer"]
    assert "same-brand, same-product-type historical peers" in payload["disclaimer"]


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/predict-return-risk", {"user_id": "user-1", "policy": {}}),
        (
            "/recommend-alternatives",
            {"user_id": "user-1", "variant_id": "variant-1", "policy": {"min_risk_reduction": -1}},
        ),
        ("/simulate-policy", {"policy": {"high_risk_threshold": 1.5}}),
    ],
)
def test_invalid_payloads_return_422(client: TestClient, path: str, body: dict[str, Any]) -> None:
    response = client.post(path, json=body)

    assert response.status_code == 422


@pytest.mark.parametrize(
    ("method", "path", "body", "expected_detail"),
    [
        ("GET", "/model-metrics", None, "Overview metrics are missing."),
        ("GET", "/demo-scenarios", None, "Demo scenarios are missing."),
        (
            "POST",
            "/predict-return-risk",
            {"user_id": "user-1", "variant_id": "variant-1", "policy": {}},
            "Model artifact is missing.",
        ),
        (
            "POST",
            "/recommend-alternatives",
            {"user_id": "user-1", "variant_id": "variant-1", "policy": {}},
            "Model artifact is missing.",
        ),
        (
            "POST",
            "/simulate-policy",
            {"policy": {}},
            "Policy simulation artifact is missing.",
        ),
    ],
)
def test_missing_artifacts_return_explicit_503(
    monkeypatch,
    method: str,
    path: str,
    body: dict[str, Any] | None,
    expected_detail: str,
) -> None:
    monkeypatch.setattr(api_main, "service", MissingArtifactService())
    missing_client = TestClient(api_main.app)

    response = missing_client.request(method, path, json=body)

    assert response.status_code == 503
    assert response.json()["detail"] == expected_detail


def test_policy_request_rejects_old_min_confidence_field() -> None:
    with pytest.raises(ValidationError):
        PolicySettings(min_confidence=0.3)


def test_model_explanations_contract_reads_artifact(tmp_path, monkeypatch) -> None:
    artifact = tmp_path / "reports/explanations/strict_no_leak_catboost_shap_summary.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(
        json.dumps(
            {
                "feature_set": "strict_no_leak",
                "model_path": "models/strict_no_leak_catboost_calibrated.joblib",
                "sample_rows": 10,
                "complete_metadata_only": True,
                "top_features": [{"feature": "productType", "mean_abs_shap": 0.3}],
            }
        )
    )
    monkeypatch.chdir(tmp_path)
    service = InferenceService()
    monkeypatch.setattr(api_main, "service", service)

    payload = api_main.model_explanations()

    assert payload["model_version"] == "strict_no_leak_catboost_calibrated_v1"
    assert (
        payload["artifact_path"]
        == "reports/explanations/strict_no_leak_catboost_shap_summary.json"
    )
    assert payload["top_features"] == [{"feature": "productType", "mean_abs_shap": 0.3}]
