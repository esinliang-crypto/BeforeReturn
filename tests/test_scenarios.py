import pandas as pd

import src.inference.scenarios as scenarios_module
from src.data.dataset import VARIANT_KEY
from src.inference.scenarios import confidence_from_probability, find_alternative, risk_level


def test_confidence_from_probability() -> None:
    assert confidence_from_probability(0.5) == 0
    assert confidence_from_probability(0.8) == 0.6000000000000001


def test_risk_level() -> None:
    assert risk_level(0.7) == "high"
    assert risk_level(0.5) == "medium"
    assert risk_level(0.2) == "low"


def test_find_alternative_labels_same_brand_product_type_peer(monkeypatch) -> None:
    def fake_score_frame(bundle, frame):
        return pd.Series([0.35, 0.7]).to_numpy()

    monkeypatch.setattr(scenarios_module, "score_frame", fake_score_frame)
    current = pd.Series(
        {
            VARIANT_KEY: 1,
            "yearOfBirth": 1990,
            "isMale": 0,
            "shippingCountry": "Country_A",
            "premier": 1,
            "productType": "Dress",
            "brandDesc": "Brand_A",
        }
    )
    catalog = pd.DataFrame(
        {
            VARIANT_KEY: [2, 3, 4],
            "productType": ["Dress", "Dress", "Shirt"],
            "brandDesc": ["Brand_A", "Brand_A", "Brand_A"],
            "avgGbpPrice": [30, 40, 20],
            "avgDiscountValue": [0, 0, 0],
            "productType__missing": [0, 0, 0],
            "brandDesc__missing": [0, 0, 0],
            "avgGbpPrice__missing": [0, 0, 0],
            "avgDiscountValue__missing": [0, 0, 0],
        }
    )
    bundle = {
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
        ]
    }

    alternative = find_alternative(bundle, current, catalog, current_probability=0.7)

    assert alternative is not None
    assert alternative["candidate_type"] == "same-brand, same-product-type historical peer"
    assert "current checkout user's available profile fields" in alternative["risk_basis"]
    assert alternative["inventory_status"] == "Inventory not verified."
    assert "not randomized causal evidence" in alternative["disclaimer"]
    assert "Lower-risk peer option" in alternative["reason"]
