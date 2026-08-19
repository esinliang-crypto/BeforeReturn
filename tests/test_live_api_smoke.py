from __future__ import annotations

import os

import pytest
import requests

pytestmark = pytest.mark.skipif(
    not os.getenv("BEFORE_RETURN_LIVE_API_URL"),
    reason="Set BEFORE_RETURN_LIVE_API_URL to run live API smoke tests.",
)


def live_api_url() -> str:
    return os.environ["BEFORE_RETURN_LIVE_API_URL"].rstrip("/")


def test_live_api_core_flow_smoke() -> None:
    base_url = live_api_url()

    health = requests.get(f"{base_url}/health", timeout=10)
    assert health.status_code == 200
    assert health.json()["model_version"]

    metrics = requests.get(f"{base_url}/model-metrics", timeout=10)
    assert metrics.status_code == 200
    assert metrics.json()["model_version"] == health.json()["model_version"]

    scenarios = requests.get(f"{base_url}/demo-scenarios", timeout=10)
    assert scenarios.status_code == 200
    first_scenario = scenarios.json()[0]

    request_payload = {
        "user_id": first_scenario["user_id"],
        "variant_id": first_scenario["variant_id"],
        "policy": {},
    }
    prediction = requests.post(
        f"{base_url}/predict-return-risk",
        json=request_payload,
        timeout=10,
    )
    assert prediction.status_code == 200
    assert prediction.json()["model_version"] == health.json()["model_version"]

    recommendation = requests.post(
        f"{base_url}/recommend-alternatives",
        json=request_payload,
        timeout=10,
    )
    assert recommendation.status_code == 200
    if recommendation.json()["alternative"]:
        assert recommendation.json()["alternative"]["candidate_type"] == (
            "same-brand, same-product-type historical peer"
        )

    policy = requests.post(f"{base_url}/simulate-policy", json={"policy": {}}, timeout=10)
    assert policy.status_code == 200
    assert policy.json()["evaluated_checkouts"] > len(scenarios.json())
