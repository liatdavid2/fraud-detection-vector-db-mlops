from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from fraud_vector_db_mlops.config import get_settings
from fraud_vector_db_mlops.data import load_dataset
from fraud_vector_db_mlops.milvus_store import try_index_model_embeddings
from fraud_vector_db_mlops.model import FraudVectorModel
from fraud_vector_db_mlops.validation import save_validation_report, validate_dataframe


def temporal_or_stratified_split(
    df: pd.DataFrame, target: str, test_size: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df.drop(columns=[target])
    y = df[target].astype(int)

    if "month" in df.columns and df["month"].nunique() > 1:
        cutoff = df["month"].quantile(1 - test_size)
        train_mask = df["month"] <= cutoff
        test_mask = ~train_mask
        if test_mask.sum() > 100 and train_mask.sum() > 100:
            return X[train_mask], X[test_mask], y[train_mask], y[test_mask]

    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def plot_reports(y_true: pd.Series, y_proba: np.ndarray, y_pred: np.ndarray, reports_path: Path) -> None:
    reports_path.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(values_format="d")
    plt.title("Fraud Detection Confusion Matrix")
    plt.tight_layout()
    plt.savefig(reports_path / "confusion_matrix.png")
    plt.close()

    PrecisionRecallDisplay.from_predictions(y_true, y_proba)
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(reports_path / "pr_curve.png")
    plt.close()

    RocCurveDisplay.from_predictions(y_true, y_proba)
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(reports_path / "roc_curve.png")
    plt.close()




def configure_mlflow(settings) -> str:
    """Use configured MLflow server, with a local fallback for smoke tests."""
    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)
        return settings.mlflow_tracking_uri
    except Exception as exc:
        fallback = f"file:{settings.project_root / 'mlruns'}"
        print(f"MLflow server unavailable, using local tracking store: {fallback}. Reason: {exc}")
        mlflow.set_tracking_uri(fallback)
        mlflow.set_experiment(settings.mlflow_experiment_name)
        return fallback

def train(skip_milvus: bool = False) -> dict[str, float]:
    settings = get_settings()
    df, target = load_dataset(max_rows=settings.max_train_rows)

    checks = validate_dataframe(df, target)
    save_validation_report(checks, settings.reports_path / "validation_report.json")

    X_train, X_test, y_train, y_test = temporal_or_stratified_split(
        df, target, settings.test_size, settings.random_state
    )

    train_ids = (
        X_train["application_id"].astype(str)
        if "application_id" in X_train.columns
        else pd.Series([f"train-{i}" for i in range(len(X_train))])
    )

    model = FraudVectorModel(
        target_column=target,
        embedding_dim=settings.embedding_dim,
        n_neighbors=settings.n_neighbors,
        random_state=settings.random_state,
    )

    tracking_uri = configure_mlflow(settings)

    with mlflow.start_run(run_name="hybrid-vector-fraud-model"):
        mlflow.log_param("tracking_uri_used", tracking_uri)
        mlflow.log_params(
            {
                "target": target,
                "rows": len(df),
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "fraud_rate": float(df[target].mean()),
                "embedding_dim": settings.embedding_dim,
                "n_neighbors": settings.n_neighbors,
                "model_family": "xgboost_if_available_else_logistic_regression",
            }
        )

        model.fit(X_train, y_train, application_ids=train_ids)
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= settings.review_threshold).astype(int)

        metrics = {
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
            "average_precision": float(average_precision_score(y_test, y_proba)),
            "precision_at_review_threshold": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall_at_review_threshold": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_at_review_threshold": float(f1_score(y_test, y_pred, zero_division=0)),
            "review_threshold": float(settings.review_threshold),
        }

        save_json(metrics, settings.reports_path / "metrics.json")
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        save_json(report, settings.reports_path / "classification_report.json")
        plot_reports(y_test, y_proba, y_pred, settings.reports_path)

        sample_predictions = X_test.head(200).copy()
        sample_predictions["actual_fraud"] = y_test.head(200).values
        sample_predictions["fraud_probability"] = y_proba[: len(sample_predictions)]
        sample_predictions.to_csv(settings.reports_path / "training_sample_predictions.csv", index=False)

        joblib.dump(model, settings.model_path)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(settings.model_path), artifact_path="model")
        for artifact in [
            "metrics.json",
            "classification_report.json",
            "confusion_matrix.png",
            "pr_curve.png",
            "roc_curve.png",
            "validation_report.json",
            "training_sample_predictions.csv",
        ]:
            mlflow.log_artifact(str(settings.reports_path / artifact), artifact_path="reports")

        train_probabilities = model.predict_proba(X_train)[:, 1]
        if not skip_milvus:
            indexed = try_index_model_embeddings(model, probabilities=train_probabilities)
            mlflow.log_param("milvus_indexed", indexed)
        else:
            mlflow.log_param("milvus_indexed", False)

    print("Training completed.")
    print(json.dumps(metrics, indent=2))
    print(f"Model saved to: {settings.model_path}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-milvus", action="store_true", help="Train without writing embeddings to Milvus")
    args = parser.parse_args()
    train(skip_milvus=args.skip_milvus)


if __name__ == "__main__":
    main()
