from src.training.calibration import CalibrationConfig


def test_calibration_config_defaults_to_strict_track() -> None:
    config = CalibrationConfig()
    assert config.feature_set == "strict_no_leak"
    assert config.calibration_size == 0.2
    assert config.validation_size == 0.16
