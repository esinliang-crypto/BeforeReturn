from __future__ import annotations

import gzip
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from api.schemas import PolicySettings
from src.data.dataset import TARGET, VARIANT_KEY
from src.inference.scenarios import PRODUCT_FEATURES, score_frame
from src.training.train import load_processed

ARTIFACT_SCHEMA_VERSION = 1
POLICY_DIR = Path("reports/policy")
POLICY_ARTIFACT_PATH = POLICY_DIR / "strict_no_leak_policy_simulation.pkl.gz"
POLICY_MANIFEST_PATH = POLICY_DIR / "strict_no_leak_policy_simulation_manifest.json"
CUSTOMER_FEATURES = ["yearOfBirth", "isMale", "shippingCountry", "premier"]
GROUP_COLUMNS = ["brandDesc", "productType"]
POLICY_COLUMNS = [
    "risk_probability",
    "prediction_margin",
    "is_returned",
    "complete_metadata",
    "has_peer_candidate",
    "best_peer_risk_probability",
    "best_peer_risk_reduction",
    "rescored_candidate_count",
]


@dataclass(frozen=True)
class PolicyArtifactConfig:
    feature_set: str = "strict_no_leak"
    candidate_k: int = 25
    test_limit: int | None = None
    batch_size: int = 250_000
    model_version: str = "strict_no_leak_catboost_calibrated_v1"
    artifact_path: Path = POLICY_ARTIFACT_PATH
    manifest_path: Path = POLICY_MANIFEST_PATH


def prediction_margin(probability: float | np.ndarray) -> float | np.ndarray:
    """Probability-margin proxy: distance from 50/50, not an uncertainty estimate."""
    return np.abs(probability - 0.5) * 2


def build_bounded_peer_catalog(training: pd.DataFrame, candidate_k: int) -> pd.DataFrame:
    if candidate_k < 1:
        raise ValueError("candidate_k must be at least 1.")
    complete = training[training["has_complete_metadata"]].copy()
    catalog_columns = [VARIANT_KEY, *PRODUCT_FEATURES]
    counts = (
        complete.groupby([*GROUP_COLUMNS, VARIANT_KEY], dropna=False)
        .size()
        .rename("training_event_count")
        .reset_index()
    )
    product_features = complete[catalog_columns].drop_duplicates(subset=[VARIANT_KEY])
    catalog = counts.merge(product_features, on=[*GROUP_COLUMNS, VARIANT_KEY], how="left")
    catalog = catalog.sort_values(
        [*GROUP_COLUMNS, "training_event_count", VARIANT_KEY],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    return catalog.groupby(GROUP_COLUMNS, dropna=False, group_keys=False).head(candidate_k)


def _candidate_batch(
    checkout: pd.Series,
    candidates: pd.DataFrame,
    features: list[str],
    row_position: int,
) -> pd.DataFrame:
    rows = candidates[[VARIANT_KEY, *PRODUCT_FEATURES]].copy()
    for column in CUSTOMER_FEATURES:
        rows[column] = checkout[column]
    for column in features:
        if column not in rows.columns:
            rows[column] = 0
    rows = rows[rows[VARIANT_KEY] != checkout[VARIANT_KEY]]
    rows["__policy_row_position"] = row_position
    return rows


def best_peer_for_checkout(
    bundle: dict[str, Any],
    checkout: pd.Series,
    peer_catalog: pd.DataFrame,
) -> tuple[float, float, int]:
    if not bool(checkout["has_complete_metadata"]):
        return np.nan, np.nan, 0
    candidates = peer_catalog[
        (peer_catalog["brandDesc"] == checkout["brandDesc"])
        & (peer_catalog["productType"] == checkout["productType"])
    ]
    if candidates.empty:
        return np.nan, np.nan, 0
    candidate_rows = _candidate_batch(checkout, candidates, bundle["features"], row_position=0)
    if candidate_rows.empty:
        return np.nan, np.nan, 0
    probabilities = score_frame(bundle, candidate_rows)
    best_probability = float(np.min(probabilities))
    return (
        best_probability,
        float(checkout["risk_probability"] - best_probability),
        len(candidate_rows),
    )


def compute_policy_artifact(
    *,
    bundle: dict[str, Any],
    training: pd.DataFrame,
    testing: pd.DataFrame,
    config: PolicyArtifactConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    started = time.monotonic()
    if config.test_limit is not None:
        testing = testing.head(config.test_limit).reset_index(drop=True).copy()
    else:
        testing = testing.reset_index(drop=True).copy()

    risk_probabilities = score_frame(bundle, testing).astype("float32")
    testing["risk_probability"] = risk_probabilities
    peer_catalog = build_bounded_peer_catalog(training, config.candidate_k)
    peer_groups = {
        group_key: group.copy()
        for group_key, group in peer_catalog.groupby(GROUP_COLUMNS, dropna=False)
    }

    best_probabilities = np.full(len(testing), np.nan, dtype="float32")
    best_reductions = np.full(len(testing), np.nan, dtype="float32")
    candidate_counts = np.zeros(len(testing), dtype="uint16")

    complete_testing = testing[testing["has_complete_metadata"]].copy()
    complete_testing["__policy_row_position"] = complete_testing.index.astype("int64")

    for group_key, checkout_group in complete_testing.groupby(GROUP_COLUMNS, dropna=False):
        candidates = peer_groups.get(group_key)
        if candidates is None or candidates.empty:
            continue
        candidate_count = len(candidates)
        checkout_chunk_size = max(1, config.batch_size // candidate_count)
        candidate_values = {
            column: candidates[column].to_numpy()
            for column in [VARIANT_KEY, *PRODUCT_FEATURES]
        }
        for start in range(0, len(checkout_group), checkout_chunk_size):
            checkout_chunk = checkout_group.iloc[start : start + checkout_chunk_size]
            repeated_count = len(checkout_chunk) * candidate_count
            batch = pd.DataFrame(index=range(repeated_count))
            for column in [VARIANT_KEY, *PRODUCT_FEATURES]:
                batch[column] = np.tile(candidate_values[column], len(checkout_chunk))
            for column in CUSTOMER_FEATURES:
                batch[column] = np.repeat(checkout_chunk[column].to_numpy(), candidate_count)
            row_positions = np.repeat(
                checkout_chunk["__policy_row_position"].to_numpy(),
                candidate_count,
            )
            current_variants = np.repeat(checkout_chunk[VARIANT_KEY].to_numpy(), candidate_count)
            keep_mask = batch[VARIANT_KEY].to_numpy() != current_variants
            if not np.any(keep_mask):
                continue
            batch = batch.loc[keep_mask].reset_index(drop=True)
            row_positions = row_positions[keep_mask]
            for column in bundle["features"]:
                if column not in batch.columns:
                    batch[column] = 0
            unique_positions, counts = np.unique(row_positions, return_counts=True)
            candidate_counts[unique_positions.astype("int64")] = np.minimum(
                counts,
                np.iinfo(np.uint16).max,
            ).astype("uint16")
            probabilities = score_frame(bundle, batch).astype("float32")
            batch_result = pd.DataFrame(
                {"row_position": row_positions, "risk_probability": probabilities}
            )
            best_by_position = batch_result.groupby("row_position")["risk_probability"].min()
            positions = best_by_position.index.to_numpy(dtype="int64")
            best_values = best_by_position.to_numpy(dtype="float32")
            best_probabilities[positions] = best_values
            best_reductions[positions] = risk_probabilities[positions] - best_values

    artifact = pd.DataFrame(
        {
            "risk_probability": risk_probabilities,
            "prediction_margin": prediction_margin(risk_probabilities).astype("float32"),
            "is_returned": testing[TARGET].astype("bool").to_numpy(),
            "complete_metadata": testing["has_complete_metadata"].astype("bool").to_numpy(),
            "has_peer_candidate": candidate_counts > 0,
            "best_peer_risk_probability": best_probabilities,
            "best_peer_risk_reduction": best_reductions,
            "rescored_candidate_count": candidate_counts,
        }
    )
    manifest = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "feature_set": config.feature_set,
        "model_version": config.model_version,
        "candidate_strategy": (
            "same_brand_same_product_type_training_frequency_bounded_peer_rescore"
        ),
        "candidate_source": "official_training_catalog_only",
        "candidate_k": config.candidate_k,
        "uses_labels_for_candidate_pool": False,
        "test_rows": int(len(testing)),
        "artifact_rows": int(len(artifact)),
        "complete_metadata_rows": int(artifact["complete_metadata"].sum()),
        "rows_with_peer_candidate": int(artifact["has_peer_candidate"].sum()),
        "source_test_path": f"data/processed/{config.feature_set}_testing.pkl",
        "source_training_path": f"data/processed/{config.feature_set}_training.pkl",
        "artifact_path": str(config.artifact_path),
        "generated_at": datetime.now(UTC).isoformat(),
        "generation_seconds": round(time.monotonic() - started, 3),
        "notes": [
            "prediction_margin is abs(p - 0.5) * 2, a probability-margin proxy only.",
            "Peer risks are rescored under the current checkout customer features.",
            "Full artifact excludes IDs, brand, productType, and recommendation text.",
        ],
    }
    return artifact[POLICY_COLUMNS], manifest


def write_policy_artifact(
    artifact: pd.DataFrame,
    manifest: dict[str, Any],
    *,
    artifact_path: Path = POLICY_ARTIFACT_PATH,
    manifest_path: Path = POLICY_MANIFEST_PATH,
) -> None:
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(artifact_path, "wb") as file:
        artifact.to_pickle(file)
    manifest = {**manifest, "artifact_size_bytes": artifact_path.stat().st_size}
    manifest_path.write_text(json.dumps(manifest, indent=2))


def load_policy_artifact(path: Path = POLICY_ARTIFACT_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Policy simulation artifact is missing: {path}")
    return pd.read_pickle(path)


def simulate_from_artifact(
    artifact: pd.DataFrame,
    policy: PolicySettings,
) -> dict[str, Any]:
    evaluated = len(artifact)
    positives = int(artifact["is_returned"].sum())
    prompt_budget = int(np.floor(evaluated * policy.max_prompts_per_1000 / 1000))
    recommendations_allowed = (
        policy.allow_variant_recommendations or policy.allow_product_recommendations
    )
    eligible = artifact[
        (artifact["risk_probability"] >= policy.high_risk_threshold)
        & (artifact["prediction_margin"] >= policy.min_prediction_margin)
        & recommendations_allowed
        & artifact["has_peer_candidate"]
        & artifact["best_peer_risk_reduction"].notna()
        & (artifact["best_peer_risk_reduction"] >= policy.min_risk_reduction)
    ].copy()
    eligible_count = int(len(eligible))
    if prompt_budget <= 0 or eligible.empty:
        prompted = eligible.head(0)
    else:
        prompted = eligible.sort_values(
            ["risk_probability", "prediction_margin", "best_peer_risk_reduction"],
            ascending=[False, False, False],
            kind="mergesort",
        ).head(prompt_budget)
    true_positives = int(prompted["is_returned"].sum())
    false_positives = int(len(prompted) - true_positives)
    return {
        "evaluated_checkouts": evaluated,
        "estimated_prompts": int(len(prompted)),
        "eligible_checkouts": eligible_count,
        "prompt_budget": prompt_budget,
        "prompt_coverage": float(len(prompted) / evaluated) if evaluated else 0.0,
        "recall_at_policy": float(true_positives / positives) if positives else 0.0,
        "precision_at_policy": float(true_positives / len(prompted)) if len(prompted) else 0.0,
        "false_positives": false_positives,
        "user_disturbance_rate": float(len(prompted) / evaluated) if evaluated else 0.0,
    }


def build_policy_artifact_from_disk(
    bundle: dict[str, Any],
    config: PolicyArtifactConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    training = load_processed(config.feature_set, "training")
    testing = load_processed(config.feature_set, "testing")
    return compute_policy_artifact(
        bundle=bundle,
        training=training,
        testing=testing,
        config=config,
    )
