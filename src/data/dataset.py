from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
SAMPLES_DIR = Path("data/samples")

CUSTOMER_KEY = "hash(customerId)"
VARIANT_KEY = "hash(variantID)"
TARGET = "isReturned"

SPLITS = ("training", "testing")
FEATURE_SETS = ("strict_no_leak", "paper_feature_baseline")

STRICT_CUSTOMER_COLUMNS = [
    CUSTOMER_KEY,
    "yearOfBirth",
    "isMale",
    "shippingCountry",
    "premier",
]

STRICT_PRODUCT_COLUMNS = [
    VARIANT_KEY,
    "productType",
    "brandDesc",
    "avgGbpPrice",
    "avgDiscountValue",
]

LABEL_DERIVED_PATTERNS = (
    "return",
    "Return",
    "return_code",
)


@dataclass(frozen=True)
class DatasetBundle:
    feature_set: str
    split: str
    frame: pd.DataFrame
    feature_columns: list[str]
    categorical_columns: list[str]


def make_unique_columns(columns: list[str]) -> list[str]:
    """Return unique column names while keeping the first occurrence unchanged."""
    counts: dict[str, int] = {}
    unique_columns: list[str] = []
    for column in columns:
        count = counts.get(column, 0)
        counts[column] = count + 1
        unique_columns.append(column if count == 0 else f"{column}__dup{count + 1}")
    return unique_columns


def read_pickle_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_pickle(path)
    frame = frame.copy()
    frame.columns = make_unique_columns(list(frame.columns))
    return frame


def raw_path(kind: str, split: str) -> Path:
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    return RAW_DIR / f"{kind}_{split}.p"


def load_raw_split(split: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    events = read_pickle_frame(raw_path("event_table", split))
    customers = read_pickle_frame(raw_path("customer_nodes", split))
    products = read_pickle_frame(raw_path("product_nodes", split))
    return events, customers, products


def select_customer_features(customers: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    if feature_set == "strict_no_leak":
        return customers[STRICT_CUSTOMER_COLUMNS]
    if feature_set == "paper_feature_baseline":
        return customers
    raise ValueError(f"Unknown feature set: {feature_set}")


def select_product_features(products: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    if feature_set == "strict_no_leak":
        return products[STRICT_PRODUCT_COLUMNS]
    if feature_set == "paper_feature_baseline":
        return products
    raise ValueError(f"Unknown feature set: {feature_set}")


def build_feature_frame(feature_set: str, split: str) -> DatasetBundle:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {feature_set}")

    events, customers, products = load_raw_split(split)
    customer_features = select_customer_features(customers, feature_set)
    product_features = select_product_features(products, feature_set)

    frame = events.merge(customer_features, on=CUSTOMER_KEY, how="left", validate="many_to_one")
    frame = frame.merge(product_features, on=VARIANT_KEY, how="left", validate="many_to_one")

    feature_columns = [
        column
        for column in frame.columns
        if column not in {TARGET, CUSTOMER_KEY, VARIANT_KEY}
    ]

    customer_metadata_columns = [
        column for column in customer_features.columns if column != CUSTOMER_KEY
    ]
    product_metadata_columns = [
        column for column in product_features.columns if column != VARIANT_KEY
    ]
    frame["has_complete_metadata"] = (
        frame[customer_metadata_columns].notna().all(axis=1)
        & frame[product_metadata_columns].notna().all(axis=1)
    )

    missing_feature_columns = [
        column for column in feature_columns if frame[column].isna().any()
    ]
    for column in missing_feature_columns:
        frame[f"{column}__missing"] = frame[column].isna().astype("int8")

    categorical_columns = [
        column
        for column in feature_columns
        if pd.api.types.is_object_dtype(frame[column])
        or pd.api.types.is_string_dtype(frame[column])
    ]
    numeric_columns = [
        column for column in feature_columns if column not in categorical_columns
    ]

    for column in categorical_columns:
        frame[column] = frame[column].fillna("Unknown")
    for column in numeric_columns:
        frame[column] = frame[column].fillna(0)

    feature_columns = [
        column
        for column in frame.columns
        if column not in {TARGET, CUSTOMER_KEY, VARIANT_KEY, "has_complete_metadata"}
    ]
    categorical_columns = [
        column
        for column in feature_columns
        if pd.api.types.is_object_dtype(frame[column])
        or pd.api.types.is_string_dtype(frame[column])
    ]
    return DatasetBundle(feature_set, split, frame, feature_columns, categorical_columns)


def leakage_risk_columns(columns: list[str]) -> list[str]:
    return [
        column
        for column in columns
        if any(pattern in column for pattern in LABEL_DERIVED_PATTERNS)
    ]


def write_bundle(bundle: DatasetBundle) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / f"{bundle.feature_set}_{bundle.split}.pkl"
    bundle.frame.to_pickle(output_path)
    return output_path


def write_demo_sample(bundle: DatasetBundle, rows: int = 200) -> Path:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = SAMPLES_DIR / f"{bundle.feature_set}_{bundle.split}_sample.csv"
    sample = bundle.frame.head(rows).copy()
    sample.to_csv(output_path, index=False)
    return output_path
