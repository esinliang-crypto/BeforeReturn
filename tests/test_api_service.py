import pandas as pd

from api.schemas import PolicySettings
from api.service import InferenceService


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
