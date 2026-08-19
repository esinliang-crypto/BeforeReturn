from pathlib import Path


def test_peer_recommendation_ui_copy_matches_supported_capability() -> None:
    source = Path("web/app/page.tsx").read_text()

    assert "Lower-risk peer option" in source
    assert "same-brand, same-product-type historical peers only" in source
    assert "Candidate risk is model-estimated under the current user's checkout fields." in source
    assert "Inventory is not verified." in source
    assert "Keep original choice" in source
    assert "Allow lower-risk peer options" in source
    assert "Allow variant recommendations" not in source
    assert "Allow product recommendations" not in source
    assert "Recommended alternative" not in source
