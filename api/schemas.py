from __future__ import annotations

from pydantic import BaseModel, Field


class PolicySettings(BaseModel):
    high_risk_threshold: float = Field(default=0.6, ge=0, le=1)
    min_confidence: float = Field(default=0.3, ge=0, le=1)
    max_prompts_per_1000: int = Field(default=150, ge=0, le=1000)
    allow_variant_recommendations: bool = True
    allow_product_recommendations: bool = True
    min_risk_reduction: float = Field(default=0.1, ge=0, le=1)


class PredictionRequest(BaseModel):
    user_id: int
    variant_id: int
    policy: PolicySettings = Field(default_factory=PolicySettings)


class AlternativeRequest(BaseModel):
    user_id: int
    variant_id: int
    policy: PolicySettings = Field(default_factory=PolicySettings)


class PolicySimulationRequest(BaseModel):
    policy: PolicySettings = Field(default_factory=PolicySettings)


class TopFactor(BaseModel):
    feature: str
    impact: float
    direction: str


class Alternative(BaseModel):
    variant_id: int
    product_type: str
    brand: str
    risk_probability: float
    relative_risk_change: float
    reason: str


class PredictionResponse(BaseModel):
    user_id: int
    variant_id: int
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
    prompt_coverage: float
    recall_at_policy: float
    precision_at_policy: float
    false_positives: int
    user_disturbance_rate: float
    disclaimer: str

