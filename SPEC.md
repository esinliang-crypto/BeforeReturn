# BeforeReturn MVP Spec

## Product Scope

BeforeReturn predicts return risk at checkout for an anonymous user-product variant pair using ASOS GraphReturns historical data. It decides whether to intervene through explicit policy rules, then recommends safer alternatives only when supported by the data.

## Core Demo Flow

1. Open Overview and understand the business problem, dataset limitations, and model metrics.
2. Enter Checkout Simulator.
3. Select one stable demo scenario.
4. Request return risk evaluation.
5. Review risk probability, risk level, confidence, top risk factors, and policy reasons.
6. If policy allows intervention, compare the original choice against lower-risk alternatives.
7. Open Policy Console and adjust thresholds to observe offline policy metrics.

## Required Stable Scenarios

- High risk, high confidence, eligible lower-risk alternative exists.
- High risk, low confidence, no intervention.
- Low risk, no intervention.

## Model Scope

- Logistic Regression baseline.
- CatBoost main model.
- Leakage-controlled historical aggregate features.
- Probability calibration.
- SHAP or equivalent feature attribution.
- Frozen inference artifacts for demo and API use.

## API Contract

- `GET /health`
- `GET /demo-scenarios`
- `GET /users/{user_id}/eligible-products`
- `POST /predict-return-risk`
- `POST /recommend-alternatives`
- `POST /simulate-policy`
- `GET /model-metrics`

Prediction responses must include:

- `risk_probability`
- `risk_level`
- `confidence`
- `should_intervene`
- `top_factors`
- `policy_reasons`
- `model_version`

## Frontend Scope

- Overview
- Checkout Simulator
- Safer Alternative
- Policy Console

UI text is English. Documentation may be Chinese or English.

## Non-Goals for MVP

- GNN, GraphSAGE, PyTorch Geometric, Node2Vec, Transformer, RAG, LLM chat, agents, online training, scraping, and causal claims.

