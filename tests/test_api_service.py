from api.schemas import PolicySettings
from api.service import InferenceService


def test_policy_reasons_blocks_low_confidence_without_alternative() -> None:
    service = InferenceService()
    reasons = service.policy_reasons(
        probability=0.61,
        confidence=0.2,
        alternative=None,
        policy=PolicySettings(min_confidence=0.3),
    )
    assert "fail: model confidence is below the policy minimum" in reasons
    assert "fail: no eligible lower-risk alternative is available" in reasons


def test_parse_id_preserves_large_integer_string() -> None:
    service = InferenceService()
    assert service.parse_id("7867366894295744909") == 7867366894295744909
