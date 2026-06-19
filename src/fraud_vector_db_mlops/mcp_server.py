from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

PROJECT_ROOT = Path(
    os.environ.get(
        "FRAUD_PROJECT_ROOT",
        Path(__file__).resolve().parents[2],
    )
)
os.chdir(PROJECT_ROOT)

API_BASE_URL = os.environ.get("FRAUD_API_BASE_URL", "http://localhost:8000")

mcp = FastMCP("fraud-vector-db-mlops")


def post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        url=url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


@mcp.tool()
def predict_fraud(features: dict[str, Any]) -> dict[str, Any]:
    """Call the Fraud FastAPI /predict endpoint and return fraud risk, alert, and SHAP explanation."""
    return post_json(
        url=f"{API_BASE_URL}/predict",
        payload={"features": features},
    )


@mcp.tool()
def find_similar_fraud_cases(
    features: dict[str, Any],
    top_k: int = 50,
) -> dict[str, Any]:
    """Call the Fraud FastAPI /similar-cases endpoint and return similar confirmed fraud cases."""
    top_k = min(max(top_k, 1), 50)

    return post_json(
        url=f"{API_BASE_URL}/similar-cases?top_k={top_k}",
        payload={"features": features},
    )


@mcp.tool()
def get_latest_training_summary() -> dict[str, Any]:
    """Return latest training summary from reports/best_model.json."""
    path = PROJECT_ROOT / "reports" / "best_model.json"

    if not path.exists():
        return {
            "status": "missing",
            "message": "reports/best_model.json was not found. Run training first.",
        }

    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    mcp.run()