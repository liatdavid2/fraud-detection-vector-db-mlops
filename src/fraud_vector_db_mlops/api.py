from __future__ import annotations

from functools import lru_cache
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from scipy import sparse

from fraud_vector_db_mlops.config import get_settings
from fraud_vector_db_mlops.milvus_store import MilvusVectorStore


class PredictRequest(BaseModel):
    features: dict[str, Any] = Field(..., description="Raw application features")


class PredictResponse(BaseModel):
    fraud_probability: float
    decision: str
    risk_level: str
    alert: bool
    reason_codes: list[str]
    similar_cases: list[dict[str, Any]] | None = None
    model_explanation: dict[str, Any]
    model_path: str


app = FastAPI(
    title="Fraud Vector DB MLOps API",
    description=(
        "Fraud detection API with vector similarity reasoning, "
        "human-review alerts, and real SHAP / feature-importance explanations."
    ),
    version="0.1.0",
)


FEATURE_DISPLAY_NAMES = {
    "income": "Income",
    "name_email_similarity": "Name-email similarity",
    "prev_address_months_count": "Previous address duration",
    "current_address_months_count": "Current address duration",
    "customer_age": "Customer age",
    "days_since_request": "Time since request",
    "intended_balcon_amount": "Intended balance amount",
    "payment_type": "Payment type",
    "zip_count_4w": "Recent ZIP-area activity",
    "velocity_6h": "6-hour application velocity",
    "velocity_24h": "24-hour application velocity",
    "velocity_4w": "4-week application velocity",
    "bank_branch_count_8w": "Bank branch activity",
    "date_of_birth_distinct_emails_4w": "Emails linked to date of birth",
    "employment_status": "Employment status",
    "credit_risk_score": "Credit risk score",
    "email_is_free": "Free email indicator",
    "housing_status": "Housing status",
    "phone_home_valid": "Home phone validity",
    "phone_mobile_valid": "Mobile phone validity",
    "bank_months_count": "Bank account age",
    "has_other_cards": "Other cards indicator",
    "proposed_credit_limit": "Requested credit limit",
    "foreign_request": "Foreign request",
    "source": "Application source",
    "session_length_in_minutes": "Session length",
    "device_os": "Device operating system",
    "keep_alive_session": "Keep-alive session",
    "device_distinct_emails_8w": "Emails from same device",
    "device_fraud_count": "Previous fraud count on device",
    "month": "Application month",
}


@lru_cache(maxsize=1)
def load_model() -> Any:
    settings = get_settings()

    if not settings.model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {settings.model_path}. "
            "Run: python -m fraud_vector_db_mlops.train"
        )

    return joblib.load(settings.model_path)


def decision_from_probability(probability: float) -> tuple[str, str, bool]:
    """The model does not decline applications.

    It only creates alerts for human review.
    Final business decision remains with a human fraud analyst.
    """
    settings = get_settings()

    if probability >= settings.decline_threshold:
        return "manual_review", "high", True

    if probability >= settings.review_threshold:
        return "manual_review", "medium", True

    return "approve", "low", False


def feature_display_name(feature: str) -> str:
    return FEATURE_DISPLAY_NAMES.get(feature, feature.replace("_", " ").title())


def json_safe_value(value: Any) -> Any:
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, np.generic):
        return value.item()

    return value

def fraud_similar_cases_only(
    similar_cases: list[dict[str, Any]],
    top_n: int = 5,
) -> list[dict[str, Any]] | None:
    fraud_cases = [
        case
        for case in similar_cases
        if int(case.get("label", 0)) == 1
    ]

    if not fraud_cases:
        return None

    return fraud_cases[:top_n]

def prepare_model_matrix(model: Any, df: pd.DataFrame) -> tuple[Any, list[str]]:
    """Apply the same preprocessing used during training.

    Returns:
        X_model: transformed feature matrix used by the classifier.
        feature_names: transformed feature names.
    """
    if getattr(model, "preprocessor_", None) is None:
        raise RuntimeError("Model preprocessor is missing.")

    if getattr(model, "feature_columns_", None) is not None:
        df_prepared = model._prepare_features(df)
    else:
        df_prepared = df.copy()

    X_processed = model.preprocessor_.transform(df_prepared)

    try:
        feature_names = list(model.preprocessor_.get_feature_names_out())
    except Exception:
        feature_names = [f"feature_{i}" for i in range(X_processed.shape[1])]

    if getattr(model, "use_vector_features", False):
        embeddings = model.transform_to_embeddings(df)
        vector_features, _ = model._vector_features(embeddings, exclude_self=False)

        X_model = sparse.hstack(
            [X_processed, sparse.csr_matrix(vector_features)],
            format="csr",
        )

        feature_names = feature_names + list(getattr(model, "vector_feature_names_", []))
    else:
        X_model = X_processed

    if len(feature_names) < X_model.shape[1]:
        feature_names += [
            f"extra_feature_{i}"
            for i in range(len(feature_names), X_model.shape[1])
        ]

    if len(feature_names) > X_model.shape[1]:
        feature_names = feature_names[: X_model.shape[1]]

    return X_model, feature_names


def to_dense_one_row(X_model: Any) -> np.ndarray:
    if sparse.issparse(X_model):
        return X_model.toarray()

    return np.asarray(X_model)


def original_feature_name(transformed_name: str, model: Any) -> str:
    """Map transformed feature names back to original input feature names.

    Examples:
        num__income -> income
        cat__payment_type_AE -> payment_type
    """
    name = transformed_name

    if "__" in name:
        name = name.split("__", 1)[1]

    numeric_columns = getattr(model, "numeric_columns_", []) or []
    categorical_columns = getattr(model, "categorical_columns_", []) or []
    vector_feature_names = getattr(model, "vector_feature_names_", []) or []

    if name in numeric_columns:
        return name

    if name in vector_feature_names:
        return name

    for col in sorted(categorical_columns, key=len, reverse=True):
        if name == col or name.startswith(f"{col}_"):
            return col

    return name


def aggregate_to_original_features(
    values: np.ndarray,
    transformed_feature_names: list[str],
    model: Any,
    df: pd.DataFrame,
    value_name: str,
    signed: bool,
) -> list[dict[str, Any]]:
    """Aggregate transformed / one-hot values back to original features."""
    grouped: dict[str, dict[str, Any]] = {}

    for transformed_name, raw_value in zip(
        transformed_feature_names,
        values,
        strict=False,
    ):
        original_name = original_feature_name(transformed_name, model)
        contribution = float(raw_value)

        if original_name not in grouped:
            grouped[original_name] = {
                "feature": original_name,
                "feature_name": feature_display_name(original_name),
                "value": json_safe_value(df.iloc[0].get(original_name)),
                value_name: 0.0,
                "abs_value": 0.0,
            }

        if signed:
            grouped[original_name][value_name] += contribution
            grouped[original_name]["abs_value"] += abs(contribution)
        else:
            grouped[original_name][value_name] += abs(contribution)
            grouped[original_name]["abs_value"] += abs(contribution)

    rows = list(grouped.values())

    for row in rows:
        val = float(row[value_name])
        row[value_name] = round(val, 6)
        row["abs_value"] = round(float(row["abs_value"]), 6)

        if signed:
            if val > 0:
                row["effect"] = "increases_fraud_risk"
                row["explanation"] = (
                    f"{row['feature_name']} contributed toward higher fraud risk."
                )
            elif val < 0:
                row["effect"] = "decreases_fraud_risk"
                row["explanation"] = (
                    f"{row['feature_name']} contributed toward lower fraud risk."
                )
            else:
                row["effect"] = "neutral"
                row["explanation"] = (
                    f"{row['feature_name']} had little contribution to this prediction."
                )
        else:
            row["effect"] = "global_importance"
            row["explanation"] = (
                f"{row['feature_name']} is one of the most important features "
                "used by the trained model."
            )

    rows.sort(key=lambda item: item["abs_value"], reverse=True)
    return rows


def catboost_real_shap(
    model: Any,
    df: pd.DataFrame,
    top_n: int,
) -> dict[str, Any]:
    """Use CatBoost native real SHAP values."""
    from catboost import Pool

    X_model, transformed_feature_names = prepare_model_matrix(model, df)
    X_dense = to_dense_one_row(X_model)

    pool = Pool(X_dense, feature_names=transformed_feature_names)

    shap_matrix = model.model_.get_feature_importance(
        type="ShapValues",
        data=pool,
    )

    # CatBoost returns [feature_shap_values..., expected_value]
    shap_values = np.asarray(shap_matrix)[0, :-1]
    expected_value = float(np.asarray(shap_matrix)[0, -1])

    rows = aggregate_to_original_features(
        values=shap_values,
        transformed_feature_names=transformed_feature_names,
        model=model,
        df=df,
        value_name="shap_value",
        signed=True,
    )

    return {
        "method": "catboost_real_shap_values",
        "method_note": (
            "Real SHAP values computed using CatBoost native "
            "get_feature_importance(type='ShapValues'). Values are in model-output space."
        ),
        "expected_value": round(expected_value, 6),
        "top_contributions": rows[:top_n],
    }


def tree_explainer_real_shap(
    model: Any,
    df: pd.DataFrame,
    top_n: int,
) -> dict[str, Any]:
    """Use SHAP TreeExplainer for XGBoost / LightGBM tree models."""
    import shap

    X_model, transformed_feature_names = prepare_model_matrix(model, df)
    X_dense = to_dense_one_row(X_model)

    explainer = shap.TreeExplainer(model.model_)
    shap_values_raw = explainer.shap_values(X_dense)

    if isinstance(shap_values_raw, list):
        shap_values = np.asarray(shap_values_raw[-1])
    else:
        shap_values = np.asarray(shap_values_raw)

    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, -1]

    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(1, -1)

    values = shap_values[0]

    expected_value = getattr(explainer, "expected_value", None)
    if isinstance(expected_value, list):
        expected_value = expected_value[-1]
    if isinstance(expected_value, np.ndarray):
        expected_value = expected_value.tolist()

    rows = aggregate_to_original_features(
        values=values,
        transformed_feature_names=transformed_feature_names,
        model=model,
        df=df,
        value_name="shap_value",
        signed=True,
    )

    return {
        "method": "shap_tree_explainer",
        "method_note": (
            "Real SHAP values computed with shap.TreeExplainer. "
            "Values are in model-output space."
        ),
        "expected_value": json_safe_value(expected_value),
        "top_contributions": rows[:top_n],
    }


def real_model_feature_importance(
    model: Any,
    df: pd.DataFrame,
    top_n: int,
    error: str | None = None,
) -> dict[str, Any]:
    """Fallback: use the trained model's real global feature importance."""
    X_model, transformed_feature_names = prepare_model_matrix(model, df)
    classifier = model.model_

    if hasattr(classifier, "feature_importances_"):
        importances = np.asarray(classifier.feature_importances_)

    elif hasattr(classifier, "get_feature_importance"):
        importances = np.asarray(classifier.get_feature_importance())

    elif hasattr(classifier, "coef_"):
        coef = np.asarray(classifier.coef_)
        importances = np.abs(coef[0] if coef.ndim > 1 else coef)

    else:
        raise RuntimeError("Model does not expose feature_importances_ or coef_.")

    if len(importances) > X_model.shape[1]:
        importances = importances[: X_model.shape[1]]

    if len(importances) < X_model.shape[1]:
        importances = np.pad(
            importances,
            (0, X_model.shape[1] - len(importances)),
            constant_values=0,
        )

    rows = aggregate_to_original_features(
        values=importances,
        transformed_feature_names=transformed_feature_names,
        model=model,
        df=df,
        value_name="importance",
        signed=False,
    )

    return {
        "method": "real_model_feature_importance",
        "method_note": (
            "Global feature importance taken directly from the trained model "
            "using feature_importances_ / get_feature_importance / coef_."
        ),
        "fallback_reason": error,
        "top_contributions": rows[:top_n],
    }


def explain_model_prediction(
    model: Any,
    df: pd.DataFrame,
    top_n: int = 5,
) -> dict[str, Any]:
    """Return real SHAP when possible, otherwise real model feature importance."""
    classifier = model.model_
    classifier_module = classifier.__class__.__module__.lower()
    classifier_name = classifier.__class__.__name__.lower()

    try:
        if "catboost" in classifier_module or "catboost" in classifier_name:
            return catboost_real_shap(model=model, df=df, top_n=top_n)

        return tree_explainer_real_shap(model=model, df=df, top_n=top_n)

    except Exception as exc:
        return real_model_feature_importance(
            model=model,
            df=df,
            top_n=top_n,
            error=str(exc),
        )


def build_model_explanation(
    probability: float,
    decision: str,
    risk_level: str,
    alert: bool,
    explanation: dict[str, Any],
) -> dict[str, Any]:
    if alert and risk_level == "high":
        summary = (
            f"High-risk fraud alert. The model estimated fraud probability "
            f"of {probability:.3f}. This application should be reviewed by "
            "a human fraud analyst; the model does not make the final decision."
        )
    elif alert:
        summary = (
            f"Medium-risk fraud alert. The model estimated fraud probability "
            f"of {probability:.3f}. This application should be reviewed manually."
        )
    else:
        summary = (
            f"Low-risk application. The model estimated fraud probability "
            f"of {probability:.3f}."
        )

    return {
        "summary": summary,
        "prediction_probability": round(float(probability), 6),
        **explanation,
    }


@app.get("/health")
def health() -> dict[str, str]:
    settings = get_settings()
    model_status = "available" if settings.model_path.exists() else "missing"

    return {
        "status": "ok",
        "model": model_status,
    }


@app.post("/predict", response_model=PredictResponse, response_model_exclude_none=True)
def predict(request: PredictRequest) -> PredictResponse:
    settings = get_settings()

    try:
        model = load_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    df = pd.DataFrame([request.features])

    scored = model.score_with_context(df)[0]
    probability = float(scored["fraud_probability"])
    decision, risk_level, alert = decision_from_probability(probability)

    try:
        explanation = explain_model_prediction(
            model=model,
            df=df,
            top_n=5,
        )
    except Exception as exc:
        explanation = {
            "method": "explanation_failed",
            "method_note": "SHAP and feature importance explanation failed.",
            "error": str(exc),
            "top_contributions": [],
        }

    return PredictResponse(
        fraud_probability=probability,
        decision=decision,
        risk_level=risk_level,
        alert=alert,
        reason_codes=scored["reason_codes"],
        similar_cases=fraud_similar_cases_only(scored["similar_cases"], top_n=5),
        model_explanation=build_model_explanation(
            probability=probability,
            decision=decision,
            risk_level=risk_level,
            alert=alert,
            explanation=explanation,
        ),
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

        all_cases = [r.__dict__ for r in results]
        fraud_cases = fraud_similar_cases_only(all_cases, top_n=top_k)

        response = {"source": "milvus"}

        if fraud_cases is not None:
            response["similar_cases"] = fraud_cases

        return response

    except Exception as exc:
        # Fallback to in-model nearest neighbors if Milvus is not running.
        model = load_model()
        df = pd.DataFrame([request.features])
        local_results = model.similarity_to_training(df, top_k=top_k)[0]

        all_cases = [r.__dict__ for r in local_results]
        fraud_cases = fraud_similar_cases_only(all_cases, top_n=top_k)

        response = {
            "source": "local_model_fallback",
            "warning": str(exc),
        }

        if fraud_cases is not None:
            response["similar_cases"] = fraud_cases

        return response