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

    def demo_scenarios(self) -> list[dict[str, Any]]:
        path = Path("data/samples/demo_scenarios.json")
        if not path.exists():
            raise ArtifactMissingError(
                "Demo scenarios are missing. Run scripts/generate_demo_scenarios.py first."
            )
        return json.loads(path.read_text())

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
        if row["has_complete_metadata"] and (
            policy.allow_variant_recommendations or policy.allow_product_recommendations
        ):
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

        if confidence >= policy.min_confidence:
            reasons.append("pass: model confidence meets the policy minimum")
        else:
            reasons.append("fail: model confidence is below the policy minimum")

        if alternative is not None:
            reasons.append("pass: a lower-risk alternative is available")
        else:
            reasons.append("fail: no eligible lower-risk alternative is available")

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
        scenarios = self.demo_scenarios()
        positives = sum(1 for scenario in scenarios if scenario["actual_return_label"] == 1)
        prompted = [
            scenario
            for scenario in scenarios
            if scenario["risk_probability"] >= policy.high_risk_threshold
            and scenario["confidence"] >= policy.min_confidence
            and scenario["alternative"] is not None
            and policy.max_prompts_per_1000 > 0
        ]
        true_positives = sum(1 for scenario in prompted if scenario["actual_return_label"] == 1)
        false_positives = sum(1 for scenario in prompted if scenario["actual_return_label"] == 0)
        precision = true_positives / len(prompted) if prompted else 0.0
        recall = true_positives / positives if positives else 0.0
        coverage = len(prompted) / len(scenarios) if scenarios else 0.0
        return PolicySimulationResponse(
            evaluated_checkouts=len(scenarios),
            estimated_prompts=len(prompted),
            prompt_coverage=coverage,
            recall_at_policy=recall,
            precision_at_policy=precision,
            false_positives=false_positives,
            user_disturbance_rate=coverage,
            disclaimer=NON_CAUSAL_DISCLAIMER,
        )
