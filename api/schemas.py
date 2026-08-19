from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PolicySettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    high_risk_threshold: float = Field(default=0.6, ge=0, le=1)
    # Probability-margin gate: abs(p - 0.5) * 2.
    # This is not an uncertainty estimate or interval.
    min_prediction_margin: float = Field(default=0.3, ge=0, le=1)
    max_prompts_per_1000: int = Field(default=150, ge=0, le=1000)
    # Compatibility fields: both controls map to the same supported peer pool.
    # The MVP only supports same-brand, same-product-type historical peers.
    allow_variant_recommendations: bool = True
    allow_product_recommendations: bool = True
    min_risk_reduction: float = Field(default=0.1, ge=0, le=1)

    def peer_recommendations_allowed(self) -> bool:
        return self.allow_variant_recommendations or self.allow_product_recommendations


class PredictionRequest(BaseModel):
    user_id: str
    variant_id: str
    policy: PolicySettings = Field(default_factory=PolicySettings)


class AlternativeRequest(BaseModel):
    user_id: str
    variant_id: str
    policy: PolicySettings = Field(default_factory=PolicySettings)


class PolicySimulationRequest(BaseModel):
    policy: PolicySettings = Field(default_factory=PolicySettings)


class TopFactor(BaseModel):
    feature: str
    impact: float
    direction: str


class Alternative(BaseModel):
    variant_id: str
    product_type: str
    brand: str
    risk_probability: float
    relative_risk_change: float
    reason: str
    candidate_type: str
    risk_basis: str
    inventory_status: str
    disclaimer: str


class PredictionResponse(BaseModel):
    user_id: str
    variant_id: str
    country: str
    product_type: str
    brand: str
    risk_probability: float
    risk_level: str
    confidence: float
    should_intervene: bool
    top_factors: list[TopFactor]
    policy_reasons: list[str]
    alternative: Alternative | None
    model_version: str


class PolicySimulationResponse(BaseModel):
    evaluated_checkouts: int
    estimated_prompts: int
    eligible_checkouts: int = 0
    prompt_budget: int = 0
    artifact_rows: int = 0
    prompt_coverage: float
    recall_at_policy: float
    precision_at_policy: float
    false_positives: int
    user_disturbance_rate: float
    disclaimer: str
