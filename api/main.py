from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    AlternativeRequest,
    PolicySimulationRequest,
    PolicySimulationResponse,
    PredictionRequest,
    PredictionResponse,
)
from api.service import ArtifactMissingError, InferenceService

app = FastAPI(title="BeforeReturn API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
service = InferenceService()


def api_error(error: Exception, status_code: int = 500) -> HTTPException:
    return HTTPException(status_code=status_code, detail=str(error))


@app.get("/health")
def health() -> dict:
    return service.health()


@app.get("/demo-scenarios")
def demo_scenarios() -> list[dict]:
    try:
        return service.demo_scenarios()
    except ArtifactMissingError as error:
        raise api_error(error, 503) from error


@app.get("/users/{user_id}/eligible-products")
def eligible_products(user_id: str) -> list[dict]:
    try:
        return service.eligible_products(user_id)
    except ArtifactMissingError as error:
        raise api_error(error, 503) from error


@app.post("/predict-return-risk", response_model=PredictionResponse)
def predict_return_risk(request: PredictionRequest) -> PredictionResponse:
    try:
        return service.predict(request.user_id, request.variant_id, request.policy)
    except ArtifactMissingError as error:
        raise api_error(error, 503) from error
    except KeyError as error:
        raise api_error(error, 404) from error


@app.post("/recommend-alternatives", response_model=PredictionResponse)
def recommend_alternatives(request: AlternativeRequest) -> PredictionResponse:
    try:
        return service.predict(request.user_id, request.variant_id, request.policy)
    except ArtifactMissingError as error:
        raise api_error(error, 503) from error
    except KeyError as error:
        raise api_error(error, 404) from error


@app.post("/simulate-policy", response_model=PolicySimulationResponse)
def simulate_policy(request: PolicySimulationRequest) -> PolicySimulationResponse:
    try:
        return service.simulate_policy(request.policy)
    except ArtifactMissingError as error:
        raise api_error(error, 503) from error


@app.get("/model-metrics")
def model_metrics() -> dict:
    try:
        return {
            "primary": service.health()["model_version"],
            "metrics_path": "reports/metrics/strict_no_leak_catboost_calibrated.json",
        }
    except ArtifactMissingError as error:
        raise api_error(error, 503) from error
