from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from mcp.server.fastmcp import FastMCP

from fraud_vector_db_mlops.api import (
    build_model_explanation,
    decision_from_probability,
    explain_model_prediction,
    load_model,
)
from fraud_vector_db_mlops.config import get_settings
from fraud_vector_db_mlops.milvus_store import MilvusVectorStore


mcp = FastMCP("fraud-vector-db-mlops")


@mcp.tool()
def predict_fraud(features: dict[str, Any]) -> dict[str, Any]:
    """Predict fraud risk and return a human-review alert with real SHAP explanation."""
    settings = get_settings()
    model = load_model()

    df = pd.DataFrame([features])
    scored = model.score_with_context(df)[0]

    probability = float(scored["fraud_probability"])
    decision, risk_level, alert = decision_from_probability(probability)

    explanation = explain_model_prediction(
        model=model,
        df=df,
        top_n=5,
    )

    return {
        "fraud_probability": probability,
        "decision": decision,
        "risk_level": risk_level,
        "alert": alert,
        "reason_codes": scored["reason_codes"],
        "model_explanation": build_model_explanation(
            probability=probability,
            decision=decision,
            risk_level=risk_level,
            alert=alert,
            explanation=explanation,
        ),
        "model_path": str(settings.model_path),
    }


@mcp.tool()
def find_similar_fraud_cases(
    features: dict[str, Any],
    top_k: int = 50,
) -> dict[str, Any]:
    """Search Milvus for historically similar confirmed fraud cases."""
    top_k = min(max(top_k, 1), 50)

    model = load_model()
    df = pd.DataFrame([features])
    embedding = model.transform_to_embeddings(df)[0]

    store = MilvusVectorStore(vector_dim=model.embedding_dim)
    results = store.search(embedding, top_k=top_k)

    all_cases = [r.__dict__ for r in results]
    fraud_cases = [
        case
        for case in all_cases
        if int(case.get("label", 0)) == 1
    ]

    return {
        "source": "milvus",
        "top_k_searched": top_k,
        "fraud_only": True,
        "similar_cases": fraud_cases,
    }


@mcp.tool()
def get_latest_training_summary() -> dict[str, Any]:
    """Return latest training summary from reports/best_model.json."""
    settings = get_settings()
    path: Path = settings.reports_path / "best_model.json"

    if not path.exists():
        return {
            "status": "missing",
            "message": "best_model.json was not found. Run training first.",
        }

    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    mcp.run()