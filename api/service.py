from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path
from typing import Any

import pandas as pd

from api.schemas import (
    Alternative,
    PolicySettings,
    PolicySimulationResponse,
    PredictionResponse,
)
from src.data.dataset import CUSTOMER_KEY, VARIANT_KEY
from src.inference.policy_simulation import (
    POLICY_ARTIFACT_PATH,
    load_policy_artifact,
    simulate_from_artifact,
)
from src.inference.scenarios import (
    confidence_from_probability,
    find_alternative,
    load_calibrated_bundle,
    product_catalog,
    risk_level,
    score_frame,
    shap_top_factors,
)
from src.training.train import load_processed

MODEL_VERSION = "strict_no_leak_catboost_calibrated_v1"
NON_CAUSAL_DISCLAIMER = (
    "Policy metrics are offline model simulations, not verified causal reductions in returns."
)
PREDICTION_MARGIN_NOTE = (
    "The min_prediction_margin policy field controls prediction margin "
    "abs(p - 0.5) * 2; it is not an uncertainty estimate or confidence interval."
)
PEER_RECOMMENDATION_NOTE = (
    "Recommendations are limited to same-brand, same-product-type historical peers; "
    "inventory is not verified."
)


class ArtifactMissingError(RuntimeError):
    pass


class InferenceService:
    @cached_property
    def bundle(self) -> dict[str, Any]:
        model_path = Path("models/strict_no_leak_catboost_calibrated.joblib")
        if not model_path.exists():
            raise ArtifactMissingError(
                "Model artifact is missing. Run scripts/calibrate_models.py first."
            )
        return load_calibrated_bundle(model_path)

    @cached_property
    def testing_frame(self) -> pd.DataFrame:
        processed_path = Path("data/processed/strict_no_leak_testing.pkl")
        if not processed_path.exists():
            raise ArtifactMissingError(
                "Processed testing data is missing. Run scripts/build_datasets.py first."
            )
        return load_processed("strict_no_leak", "testing")

    @cached_property
    def complete_frame(self) -> pd.DataFrame:
        return self.testing_frame[self.testing_frame["has_complete_metadata"]].copy()

    @cached_property
    def catalog(self) -> pd.DataFrame:
        return product_catalog(self.complete_frame)

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "model_available": Path("models/strict_no_leak_catboost_calibrated.joblib").exists(),
            "processed_data_available": Path("data/processed/strict_no_leak_testing.pkl").exists(),
            "model_version": MODEL_VERSION,
        }

    def model_metrics_path(self) -> Path:
        metrics_path = Path("reports/metrics/overview_model_metrics.json")
        if not metrics_path.exists():
            raise ArtifactMissingError(
                "Overview metrics are missing. Run scripts/audit_overview_metrics.py first."
            )
        return metrics_path

    def model_explanations(self) -> dict[str, Any]:
        explanation_path = Path("reports/explanations/strict_no_leak_catboost_shap_summary.json")
        if not explanation_path.exists():
            raise ArtifactMissingError(
                "Model explanation summary is missing. Run scripts/generate_explanations.py first."
            )
        summary = json.loads(explanation_path.read_text())
        return {
            **summary,
            "model_version": MODEL_VERSION,
            "artifact_path": str(explanation_path),
        }

    def demo_scenarios(self) -> list[dict[str, Any]]:
        path = Path("data/samples/demo_scenarios.json")
        if not path.exists():
            raise ArtifactMissingError(
                "Demo scenarios are missing. Run scripts/generate_demo_scenarios.py first."
            )
        return json.loads(path.read_text())

    @cached_property
    def policy_simulation_frame(self) -> pd.DataFrame:
        if not POLICY_ARTIFACT_PATH.exists():
            raise ArtifactMissingError(
                "Policy simulation artifact is missing. "
                "Run scripts/build_policy_simulation.py first."
            )
        return load_policy_artifact(POLICY_ARTIFACT_PATH)

    def parse_id(self, value: str | int) -> int:
        return int(str(value))

    def find_checkout_row(self, user_id: str | int, variant_id: str | int) -> pd.Series:
        parsed_user_id = self.parse_id(user_id)
        parsed_variant_id = self.parse_id(variant_id)
        match = self.testing_frame[
            (self.testing_frame[CUSTOMER_KEY] == parsed_user_id)
            & (self.testing_frame[VARIANT_KEY] == parsed_variant_id)
        ]
        if match.empty:
            raise KeyError("No checkout row found for the requested user and variant.")
        return match.iloc[0]

    def predict(
        self,
        user_id: str | int,
        variant_id: str | int,
        policy: PolicySettings,
    ) -> PredictionResponse:
        row = self.find_checkout_row(user_id, variant_id)
        row_frame = pd.DataFrame([row])
        probability = float(score_frame(self.bundle, row_frame)[0])
        confidence = confidence_from_probability(probability)
        alternative = None
        if row["has_complete_metadata"] and policy.peer_recommendations_allowed():
            alternative_payload = find_alternative(
                self.bundle,
                row,
                self.catalog,
                probability,
                min_delta=policy.min_risk_reduction,
            )
            if alternative_payload is not None:
                alternative = Alternative(**alternative_payload)

        policy_reasons = self.policy_reasons(probability, confidence, alternative, policy)
        should_intervene = all(reason.startswith("pass:") for reason in policy_reasons)
        return PredictionResponse(
            user_id=str(row[CUSTOMER_KEY]),
            variant_id=str(row[VARIANT_KEY]),
            country=str(row["shippingCountry"]),
            product_type=str(row["productType"]),
            brand=str(row["brandDesc"]),
            risk_probability=probability,
            risk_level=risk_level(probability),
            confidence=confidence,
            should_intervene=should_intervene,
            top_factors=shap_top_factors(self.bundle, row_frame),
            policy_reasons=policy_reasons,
            alternative=alternative,
            model_version=MODEL_VERSION,
        )

    def policy_reasons(
        self,
        probability: float,
        confidence: float,
        alternative: Alternative | None,
        policy: PolicySettings,
    ) -> list[str]:
        reasons = []
        if probability >= policy.high_risk_threshold:
            reasons.append("pass: risk is above the intervention threshold")
        else:
            reasons.append("fail: risk is below the intervention threshold")

        if confidence >= policy.min_prediction_margin:
            reasons.append("pass: prediction margin meets the policy minimum")
        else:
            reasons.append("fail: prediction margin is below the policy minimum")

        if alternative is not None:
            reasons.append("pass: a lower-risk peer option is available")
        else:
            reasons.append("fail: no eligible lower-risk peer option is available")

        if policy.max_prompts_per_1000 > 0:
            reasons.append("pass: prompt frequency budget is nonzero")
        else:
            reasons.append("fail: prompt frequency budget is zero")
        return reasons

    def eligible_products(self, user_id: str | int) -> list[dict[str, Any]]:
        rows = self.testing_frame[self.testing_frame[CUSTOMER_KEY] == self.parse_id(user_id)]
        return [
            {
                "variant_id": str(row[VARIANT_KEY]),
                "product_type": str(row["productType"]),
                "brand": str(row["brandDesc"]),
                "country": str(row["shippingCountry"]),
                "has_complete_metadata": bool(row["has_complete_metadata"]),
            }
            for _, row in rows.head(50).iterrows()
        ]

    def simulate_policy(self, policy: PolicySettings) -> PolicySimulationResponse:
        metrics = simulate_from_artifact(self.policy_simulation_frame, policy)
        return PolicySimulationResponse(
            **metrics,
            artifact_rows=metrics["evaluated_checkouts"],
            disclaimer=(
                f"{NON_CAUSAL_DISCLAIMER} {PREDICTION_MARGIN_NOTE} "
                f"{PEER_RECOMMENDATION_NOTE}"
            ),
        )
