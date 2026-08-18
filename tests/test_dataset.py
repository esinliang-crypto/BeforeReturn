from src.data.dataset import leakage_risk_columns, make_unique_columns


def test_make_unique_columns_keeps_first_occurrence() -> None:
    assert make_unique_columns(["a", "b", "a", "a"]) == ["a", "b", "a__dup2", "a__dup3"]


def test_leakage_risk_columns_flags_return_derived_features() -> None:
    columns = ["shippingCountry", "customerReturnRate", "variantID_level_return_code_A"]
    assert leakage_risk_columns(columns) == [
        "customerReturnRate",
        "variantID_level_return_code_A",
    ]

