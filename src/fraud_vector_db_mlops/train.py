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


def fraud_top_percent_metrics(
    y_true: pd.Series,
    y_proba: np.ndarray,
    top_percent: float = 0.05,
) -> dict[str, float]:
    y_true_arr = np.asarray(y_true).astype(int)
    y_proba_arr = np.asarray(y_proba)

    n_review = max(1, int(len(y_true_arr) * top_percent))
    top_indices = np.argsort(y_proba_arr)[::-1][:n_review]

    reviewed_true = y_true_arr[top_indices]
    fraud_in_review = int(reviewed_true.sum())
    total_fraud = max(int(y_true_arr.sum()), 1)
    suffix = int(top_percent * 100)

    return {
        f"review_rate_top_{suffix}pct": float(top_percent),
        f"review_count_top_{suffix}pct": float(n_review),
        f"precision_at_top_{suffix}pct": float(fraud_in_review / n_review),
        f"recall_at_top_{suffix}pct": float(fraud_in_review / total_fraud),
        f"fraud_captured_at_top_{suffix}pct": float(fraud_in_review / total_fraud),
    }


def find_best_threshold(y_true: pd.Series, y_proba: np.ndarray) -> dict[str, float]:
    best = {
        "best_threshold": 0.5,
        "best_precision": 0.0,
        "best_recall": 0.0,
        "best_f1": 0.0,
    }

    for threshold in np.arange(0.01, 1.00, 0.01):
        y_pred = (y_proba >= threshold).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        if f1 > best["best_f1"]:
            best = {
                "best_threshold": float(threshold),
                "best_precision": float(precision),
                "best_recall": float(recall),
                "best_f1": float(f1),
            }

    return best


def evaluate_fraud_model(
    y_true: pd.Series,
    y_proba: np.ndarray,
    default_threshold: float,
) -> dict[str, float]:
    y_pred = (y_proba >= default_threshold).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "average_precision": float(average_precision_score(y_true, y_proba)),
        "precision_at_default_threshold": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall_at_default_threshold": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_at_default_threshold": float(f1_score(y_true, y_pred, zero_division=0)),
        "default_threshold": float(default_threshold),
    }
    metrics.update(find_best_threshold(y_true, y_proba))
    metrics.update(fraud_top_percent_metrics(y_true, y_proba, top_percent=0.05))
    return metrics


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

    candidates = [
        ("xgboost_tabular", "xgboost", False),
        ("xgboost_vector", "xgboost", True),
        ("lightgbm_tabular", "lightgbm", False),
        ("lightgbm_vector", "lightgbm", True),
        ("catboost_tabular", "catboost", False),
        ("catboost_vector", "catboost", True),
    ]

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    all_results: list[dict[str, object]] = []
    best_model: FraudVectorModel | None = None
    best_metrics: dict[str, float] | None = None
    best_name: str | None = None
    best_y_proba: np.ndarray | None = None
    best_y_pred: np.ndarray | None = None

    for experiment_name, classifier_name, use_vector_features in candidates:
        with mlflow.start_run(run_name=experiment_name):
            mlflow.log_params(
                {
                    "target": target,
                    "rows": len(df),
                    "train_rows": len(X_train),
                    "test_rows": len(X_test),
                    "fraud_rate": float(df[target].mean()),
                    "embedding_dim": settings.embedding_dim,
                    "n_neighbors": settings.n_neighbors,
                    "classifier_name": classifier_name,
                    "use_vector_features": use_vector_features,
                }
            )

            model = FraudVectorModel(
                target_column=target,
                embedding_dim=settings.embedding_dim,
                n_neighbors=settings.n_neighbors,
                random_state=settings.random_state,
                classifier_name=classifier_name,
                use_vector_features=use_vector_features,
            )

            model.fit(X_train, y_train, application_ids=train_ids)
            y_proba = model.predict_proba(X_test)[:, 1]
            metrics = evaluate_fraud_model(
                y_true=y_test,
                y_proba=y_proba,
                default_threshold=settings.review_threshold,
            )
            y_pred = (y_proba >= metrics["best_threshold"]).astype(int)

            row = {
                "experiment_name": experiment_name,
                "classifier_name": classifier_name,
                "use_vector_features": use_vector_features,
                **metrics,
            }
            all_results.append(row)

            mlflow.log_metrics(metrics)

            is_better = best_metrics is None or (
                metrics["average_precision"], metrics["recall_at_top_5pct"]
            ) > (
                best_metrics["average_precision"], best_metrics["recall_at_top_5pct"]
            )
            if is_better:
                best_model = model
                best_metrics = metrics
                best_name = experiment_name
                best_y_proba = y_proba
                best_y_pred = y_pred

    if best_model is None or best_metrics is None or best_name is None:
        raise RuntimeError("No model was trained successfully.")
    if best_y_proba is None or best_y_pred is None:
        raise RuntimeError("Best model predictions are missing.")

    comparison_df = pd.DataFrame(all_results).sort_values(
        by=["average_precision", "recall_at_top_5pct"],
        ascending=False,
    )
    comparison_path = settings.reports_path / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    save_json(best_metrics, settings.reports_path / "metrics.json")
    save_json(
        {
            "best_model": best_name,
            "selection_metric": "average_precision_then_recall_at_top_5pct",
            "metrics": best_metrics,
        },
        settings.reports_path / "best_model.json",
    )

    report = classification_report(y_test, best_y_pred, output_dict=True, zero_division=0)
    save_json(report, settings.reports_path / "classification_report.json")
    plot_reports(y_test, best_y_proba, best_y_pred, settings.reports_path)

    sample_predictions = X_test.head(200).copy()
    sample_predictions["actual_fraud"] = y_test.head(200).values
    sample_predictions["fraud_probability"] = best_y_proba[: len(sample_predictions)]
    sample_predictions.to_csv(settings.reports_path / "training_sample_predictions.csv", index=False)

    joblib.dump(best_model, settings.model_path)

    train_probabilities = best_model.predict_proba(X_train)[:, 1]
    if not skip_milvus:
        indexed = try_index_model_embeddings(best_model, probabilities=train_probabilities)
    else:
        indexed = False

    with mlflow.start_run(run_name="best-model-artifacts"):
        mlflow.log_params(
            {
                "best_model": best_name,
                "milvus_indexed": indexed,
                "selection_metric": "average_precision_then_recall_at_top_5pct",
            }
        )
        mlflow.log_metrics(best_metrics)
        mlflow.log_artifact(str(settings.model_path), artifact_path="model")
        for artifact in [
            "metrics.json",
            "best_model.json",
            "model_comparison.csv",
            "classification_report.json",
            "confusion_matrix.png",
            "pr_curve.png",
            "roc_curve.png",
            "validation_report.json",
            "training_sample_predictions.csv",
        ]:
            path = settings.reports_path / artifact
            if path.exists():
                mlflow.log_artifact(str(path), artifact_path="reports")

    print("Training completed.")
    print(f"Best model: {best_name}")
    print(json.dumps(best_metrics, indent=2))
    print(f"Model saved to: {settings.model_path}")
    print(f"Model comparison saved to: {comparison_path}")
    return best_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-milvus", action="store_true", help="Train without writing embeddings to Milvus")
    args = parser.parse_args()
    train(skip_milvus=args.skip_milvus)


if __name__ == "__main__":
    main()