from src.inference.explanations import sample_complete_metadata


def test_sample_complete_metadata_function_exists() -> None:
    assert callable(sample_complete_metadata)

