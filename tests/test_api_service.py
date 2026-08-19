import pandas as pd

from api.schemas import PolicySettings
from api.service import InferenceService
from src.data.dataset import CUSTOMER_KEY, VARIANT_KEY
from src.inference.runtime_artifact import RUNTIME_ARTIFACT_PATH, write_runtime_artifact


def test_policy_reasons_blocks_low_prediction_margin_without_alternative() -> None:
    service = InferenceService()
    reasons = service.policy_reasons(
        probability=0.61,
        confidence=0.2,
        alternative=None,
        policy=PolicySettings(min_prediction_margin=0.3),
    )
    assert "fail: prediction margin is below the policy minimum" in reasons
    assert "fail: no eligible lower-risk peer option is available" in reasons


def test_policy_reasons_labels_available_peer_option() -> None:
    service = InferenceService()
    reasons = service.policy_reasons(
        probability=0.8,
        confidence=0.6,
        alternative=object(),
        policy=PolicySettings(),
    )

    assert "pass: a lower-risk peer option is available" in reasons


def test_parse_id_preserves_large_integer_string() -> None:
    service = InferenceService()
    assert service.parse_id("7867366894295744909") == 7867366894295744909


def test_simulate_policy_uses_cached_full_artifact_shape() -> None:
    service = InferenceService()
    service.__dict__["policy_simulation_frame"] = pd.DataFrame(
        {
            "risk_probability": [0.9, 0.8, 0.2, 0.7],
            "prediction_margin": [0.8, 0.6, 0.6, 0.4],
            "is_returned": [True, False, True, True],
            "complete_metadata": [True, True, True, True],
            "has_peer_candidate": [True, True, True, False],
            "best_peer_risk_probability": [0.5, 0.55, 0.1, float("nan")],
            "best_peer_risk_reduction": [0.4, 0.25, 0.1, float("nan")],
            "rescored_candidate_count": [2, 2, 1, 0],
        }
    )

    response = service.simulate_policy(PolicySettings(max_prompts_per_1000=1000))

    assert response.evaluated_checkouts == 4
    assert response.artifact_rows == 4
    assert response.eligible_checkouts == 2
    assert response.estimated_prompts == 2
    assert "prediction margin" in response.disclaimer
    assert "same-brand, same-product-type historical peers" in response.disclaimer
    assert "inventory is not verified" in response.disclaimer


def test_prediction_uses_demo_runtime_artifact_without_processed_test(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    payload = {
        "artifact_schema_version": 1,
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
        ],
        "categorical_features": ["shippingCountry", "productType", "brandDesc"],
        "checkout_rows": [
            {
                "scenario_id": "demo",
                CUSTOMER_KEY: 101,
                VARIANT_KEY: 201,
                "yearOfBirth": 1988,
                "isMale": 0,
                "shippingCountry": "Country_E",
                "premier": 1,
                "productType": "productType_B",
                "brandDesc": "Brand_B",
                "avgGbpPrice": 50.0,
                "avgDiscountValue": 5.0,
                "productType__missing": 0,
                "brandDesc__missing": 0,
                "avgGbpPrice__missing": 0,
                "avgDiscountValue__missing": 0,
                "has_complete_metadata": True,
            }
        ],
        "peer_candidates": [
            {
                VARIANT_KEY: 202,
                "productType": "productType_B",
                "brandDesc": "Brand_B",
                "avgGbpPrice": 42.0,
                "avgDiscountValue": 4.0,
                "productType__missing": 0,
                "brandDesc__missing": 0,
                "avgGbpPrice__missing": 0,
                "avgDiscountValue__missing": 0,
            }
        ],
    }
    write_runtime_artifact(payload, RUNTIME_ARTIFACT_PATH)

    service = InferenceService()
    service.__dict__["bundle"] = {
        "features": payload["features"],
        "categorical_features": payload["categorical_features"],
    }

    def fake_score_frame(bundle, frame):
        return [
            0.24 if int(row[VARIANT_KEY]) == 202 else 0.78
            for _, row in frame.iterrows()
        ]

    monkeypatch.setattr("api.service.score_frame", fake_score_frame)
    monkeypatch.setattr("src.inference.scenarios.score_frame", fake_score_frame)
    monkeypatch.setattr(
        "api.service.shap_top_factors",
        lambda bundle, row_frame: [
            {"feature": "productType", "impact": 0.3, "direction": "raises risk"}
        ],
    )

    assert not (tmp_path / "data/processed/strict_no_leak_testing.pkl").exists()

    response = service.predict("101", "201", PolicySettings())

    assert response.model_version == "strict_no_leak_catboost_calibrated_v1"
    assert response.risk_probability == 0.78
    assert response.alternative is not None
    assert response.alternative.variant_id == "202"
    assert response.alternative.candidate_type == (
        "same-brand, same-product-type historical peer"
    )
    assert service.eligible_products("101") == [
        {
            "variant_id": "201",
            "product_type": "productType_B",
            "brand": "Brand_B",
            "country": "Country_E",
            "has_complete_metadata": True,
        }
    ]
