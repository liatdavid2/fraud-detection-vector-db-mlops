from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    project_root: Path
    data_path: Path
    processed_data_path: Path
    model_path: Path
    reports_path: Path
    dataset_name: str
    target_column: str
    mlflow_tracking_uri: str
    mlflow_experiment_name: str
    milvus_host: str
    milvus_port: str
    milvus_collection: str
    milvus_vector_dim: int
    review_threshold: float
    decline_threshold: float
    test_size: float
    random_state: int
    n_neighbors: int
    embedding_dim: int
    max_train_rows: int


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for path in [current, *current.parents]:
        if (path / "pyproject.toml").exists() or (path / "README.md").exists():
            return path
    return current


def load_yaml_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_settings(config_path: str | Path = "configs/config.yaml") -> Settings:
    load_dotenv()
    root = find_project_root()
    cfg = load_yaml_config(root / config_path)

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    mlflow_cfg = cfg.get("mlflow", {})
    milvus_cfg = cfg.get("milvus", {})

    def env(name: str, default: Any) -> str:
        return os.getenv(name, str(default))

    settings = Settings(
        project_root=root,
        data_path=root / env("DATA_PATH", data_cfg.get("raw_path", "data/raw")),
        processed_data_path=root / env(
            "PROCESSED_DATA_PATH", data_cfg.get("processed_path", "data/processed")
        ),
        model_path=root / env("MODEL_PATH", "models/fraud_vector_model.joblib"),
        reports_path=root / env("REPORTS_PATH", "reports"),
        dataset_name=env(
            "DATASET_NAME",
            data_cfg.get("dataset_name", "sgpjesus/bank-account-fraud-dataset-neurips-2022"),
        ),
        target_column=env("TARGET_COLUMN", data_cfg.get("target_column", "fraud_bool")),
        mlflow_tracking_uri=env(
            "MLFLOW_TRACKING_URI", mlflow_cfg.get("tracking_uri", "http://localhost:5000")
        ),
        mlflow_experiment_name=env(
            "MLFLOW_EXPERIMENT_NAME", mlflow_cfg.get("experiment_name", "fraud-vector-db-mlops")
        ),
        milvus_host=env("MILVUS_HOST", milvus_cfg.get("host", "localhost")),
        milvus_port=env("MILVUS_PORT", milvus_cfg.get("port", "19530")),
        milvus_collection=env(
            "MILVUS_COLLECTION", milvus_cfg.get("collection_name", "fraud_cases")
        ),
        milvus_vector_dim=int(env("MILVUS_VECTOR_DIM", milvus_cfg.get("vector_dim", 32))),
        review_threshold=float(env("REVIEW_THRESHOLD", model_cfg.get("threshold_review", 0.35))),
        decline_threshold=float(env("DECLINE_THRESHOLD", model_cfg.get("threshold_decline", 0.75))),
        test_size=float(env("TEST_SIZE", cfg.get("test_size", 0.2))),
        random_state=int(env("RANDOM_STATE", cfg.get("random_state", 42))),
        n_neighbors=int(env("N_NEIGHBORS", model_cfg.get("n_neighbors", 20))),
        embedding_dim=int(env("EMBEDDING_DIM", model_cfg.get("embedding_dim", 32))),
        max_train_rows=int(env("MAX_TRAIN_ROWS", data_cfg.get("max_train_rows", 200000))),
    )

    settings.data_path.mkdir(parents=True, exist_ok=True)
    settings.processed_data_path.mkdir(parents=True, exist_ok=True)
    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    settings.reports_path.mkdir(parents=True, exist_ok=True)
    return settings
