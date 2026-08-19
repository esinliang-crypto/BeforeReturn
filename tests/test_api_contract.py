from __future__ import annotations

import json

import pandas as pd
import pytest
from pydantic import ValidationError

import api.main as api_main
from api.schemas import PolicySettings, PolicySimulationRequest
from api.service import InferenceService


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
