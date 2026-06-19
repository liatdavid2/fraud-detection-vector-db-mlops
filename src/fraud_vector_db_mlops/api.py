from __future__ import annotations

from functools import lru_cache
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from fraud_vector_db_mlops.config import get_settings
from fraud_vector_db_mlops.milvus_store import MilvusVectorStore


class PredictRequest(BaseModel):
    features: dict[str, Any] = Field(..., description="Raw application features")


class PredictResponse(BaseModel):
    fraud_probability: float
    decision: str
    risk_level: str
    reason_codes: list[str]
    vector_features: dict[str, float]
    similar_cases: list[dict[str, Any]]
    model_path: str


app = FastAPI(
    title="Fraud Vector DB MLOps API",
    description="Fraud detection API with vector similarity reasoning.",
    version="0.1.0",
)


@lru_cache(maxsize=1)
def load_model() -> Any:
    settings = get_settings()
    if not settings.model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {settings.model_path}. Run: python -m fraud_vector_db_mlops.train"
        )
    return joblib.load(settings.model_path)


def decision_from_probability(probability: float) -> tuple[str, str]:
    settings = get_settings()
    if probability >= settings.decline_threshold:
        return "decline", "high"
    if probability >= settings.review_threshold:
        return "manual_review", "medium"
    return "approve", "low"


@app.get("/health")
def health() -> dict[str, str]:
    settings = get_settings()
    model_status = "available" if settings.model_path.exists() else "missing"
    return {"status": "ok", "model": model_status}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    settings = get_settings()
    try:
        model = load_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    df = pd.DataFrame([request.features])
    scored = model.score_with_context(df)[0]
    probability = float(scored["fraud_probability"])
    decision, risk_level = decision_from_probability(probability)

    return PredictResponse(
        fraud_probability=probability,
        decision=decision,
        risk_level=risk_level,
        reason_codes=scored["reason_codes"],
        vector_features=scored["vector_features"],
        similar_cases=scored["similar_cases"],
        model_path=str(settings.model_path),
    )


@app.post("/similar-cases")
def similar_cases(request: PredictRequest, top_k: int = 10) -> dict[str, Any]:
    try:
        model = load_model()
        df = pd.DataFrame([request.features])
        embedding = model.transform_to_embeddings(df)[0]
        store = MilvusVectorStore(vector_dim=model.embedding_dim)
        results = store.search(embedding, top_k=top_k)
        return {"source": "milvus", "similar_cases": [r.__dict__ for r in results]}
    except Exception as exc:
        # Fallback to in-model nearest neighbors if Milvus is not running.
        model = load_model()
        df = pd.DataFrame([request.features])
        local_results = model.similarity_to_training(df, top_k=top_k)[0]
        return {
            "source": "local_model_fallback",
            "warning": str(exc),
            "similar_cases": [r.__dict__ for r in local_results],
        }
