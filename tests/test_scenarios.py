from src.inference.scenarios import confidence_from_probability, risk_level


def test_confidence_from_probability() -> None:
    assert confidence_from_probability(0.5) == 0
    assert confidence_from_probability(0.8) == 0.6000000000000001


def test_risk_level() -> None:
    assert risk_level(0.7) == "high"
    assert risk_level(0.5) == "medium"
    assert risk_level(0.2) == "low"

