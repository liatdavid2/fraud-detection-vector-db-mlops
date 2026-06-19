from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, normalize


@dataclass
class SimilarCase:
    application_id: str
    label: int
    similarity: float


class FraudVectorModel:
    """Hybrid fraud model: tabular ML plus nearest-neighbor fraud similarity features."""

    def __init__(
        self,
        target_column: str,
        embedding_dim: int = 32,
        n_neighbors: int = 20,
        random_state: int = 42,
    ) -> None:
        self.target_column = target_column
        self.embedding_dim = embedding_dim
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.feature_columns_: list[str] | None = None
        self.numeric_columns_: list[str] | None = None
        self.categorical_columns_: list[str] | None = None
        self.preprocessor_: ColumnTransformer | None = None
        self.svd_: TruncatedSVD | None = None
        self.nn_: NearestNeighbors | None = None
        self.model_: Any | None = None
        self.train_embeddings_: np.ndarray | None = None
        self.train_labels_: np.ndarray | None = None
        self.train_application_ids_: list[str] | None = None
        self.vector_feature_names_ = [
            "neighbor_fraud_rate_top_5",
            "neighbor_fraud_rate_top_20",
            "fraud_neighbors_top_20",
            "nearest_neighbor_similarity",
            "nearest_fraud_similarity",
            "avg_similarity_to_fraud_neighbors",
        ]

    def _build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
        categorical_cols = [c for c in X.columns if c not in numeric_cols]
        self.numeric_columns_ = numeric_cols
        self.categorical_columns_ = categorical_cols

        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=True, max_categories=30),
                ),
            ]
        )
        return ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numeric_cols),
                ("cat", categorical_pipeline, categorical_cols),
            ],
            remainder="drop",
            sparse_threshold=0.3,
        )

    @staticmethod
    def _ensure_sparse(matrix: Any) -> sparse.csr_matrix:
        if sparse.issparse(matrix):
            return matrix.tocsr()
        return sparse.csr_matrix(matrix)

    def _build_classifier(self) -> Any:
        try:
            from xgboost import XGBClassifier

            return XGBClassifier(
                n_estimators=250,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.8,
                eval_metric="logloss",
                tree_method="hist",
                random_state=self.random_state,
                n_jobs=2,
            )
        except Exception:
            return LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                n_jobs=2,
                random_state=self.random_state,
            )

    def _fit_embeddings(self, X_processed: sparse.csr_matrix) -> np.ndarray:
        n_features = X_processed.shape[1]
        n_components = max(2, min(self.embedding_dim, n_features - 1))
        self.svd_ = TruncatedSVD(n_components=n_components, random_state=self.random_state)
        embeddings = self.svd_.fit_transform(X_processed)
        if embeddings.shape[1] < self.embedding_dim:
            pad = np.zeros((embeddings.shape[0], self.embedding_dim - embeddings.shape[1]))
            embeddings = np.hstack([embeddings, pad])
        embeddings = normalize(embeddings[:, : self.embedding_dim], norm="l2")
        return np.asarray(embeddings, dtype="float32")

    def transform_to_embeddings(self, X: pd.DataFrame) -> np.ndarray:
        if self.preprocessor_ is None or self.svd_ is None or self.feature_columns_ is None:
            raise RuntimeError("Model is not fitted.")
        X = self._prepare_features(X)
        X_processed = self._ensure_sparse(self.preprocessor_.transform(X))
        embeddings = self.svd_.transform(X_processed)
        if embeddings.shape[1] < self.embedding_dim:
            pad = np.zeros((embeddings.shape[0], self.embedding_dim - embeddings.shape[1]))
            embeddings = np.hstack([embeddings, pad])
        embeddings = normalize(embeddings[:, : self.embedding_dim], norm="l2")
        return np.asarray(embeddings, dtype="float32")

    def _fit_nn(self, embeddings: np.ndarray) -> None:
        n = min(max(self.n_neighbors + 1, 2), len(embeddings))
        self.nn_ = NearestNeighbors(n_neighbors=n, metric="cosine")
        self.nn_.fit(embeddings)

    def _vector_features(
        self, embeddings: np.ndarray, exclude_self: bool = False
    ) -> tuple[np.ndarray, list[list[SimilarCase]]]:
        if self.nn_ is None or self.train_labels_ is None or self.train_application_ids_ is None:
            raise RuntimeError("Nearest-neighbor index is not fitted.")

        n_query_neighbors = min(self.n_neighbors + int(exclude_self), len(self.train_labels_))
        distances, indices = self.nn_.kneighbors(embeddings, n_neighbors=n_query_neighbors)

        features: list[list[float]] = []
        similar_cases: list[list[SimilarCase]] = []

        for row_distances, row_indices in zip(distances, indices, strict=False):
            if exclude_self and len(row_indices) > 1:
                row_distances = row_distances[1:]
                row_indices = row_indices[1:]

            labels = self.train_labels_[row_indices]
            similarities = 1 - row_distances
            top5 = min(5, len(labels))
            top20 = min(20, len(labels))

            fraud_mask = labels == 1
            fraud_sims = similarities[fraud_mask]

            nearest_fraud_similarity = float(fraud_sims.max()) if len(fraud_sims) else 0.0
            avg_similarity_to_fraud = float(fraud_sims.mean()) if len(fraud_sims) else 0.0

            features.append(
                [
                    float(labels[:top5].mean()) if top5 else 0.0,
                    float(labels[:top20].mean()) if top20 else 0.0,
                    float(labels[:top20].sum()) if top20 else 0.0,
                    float(similarities[0]) if len(similarities) else 0.0,
                    nearest_fraud_similarity,
                    avg_similarity_to_fraud,
                ]
            )

            similar_cases.append(
                [
                    SimilarCase(
                        application_id=self.train_application_ids_[int(idx)],
                        label=int(self.train_labels_[int(idx)]),
                        similarity=float(sim),
                    )
                    for idx, sim in zip(row_indices[:10], similarities[:10], strict=False)
                ]
            )

        return np.asarray(features, dtype="float32"), similar_cases

    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.feature_columns_ is None:
            raise RuntimeError("Feature columns are not initialized.")
        X = X.copy()
        for col in self.feature_columns_:
            if col not in X.columns:
                X[col] = np.nan
        return X[self.feature_columns_]

    def fit(self, X: pd.DataFrame, y: pd.Series, application_ids: pd.Series | None = None) -> "FraudVectorModel":
        self.feature_columns_ = [
            c for c in X.columns if c != self.target_column and c != "application_id"
        ]
        X = self._prepare_features(X)
        self.preprocessor_ = self._build_preprocessor(X)
        X_processed = self._ensure_sparse(self.preprocessor_.fit_transform(X))

        embeddings = self._fit_embeddings(X_processed)
        self.train_embeddings_ = embeddings
        self.train_labels_ = np.asarray(y).astype(int)
        if application_ids is None:
            self.train_application_ids_ = [f"train-{i}" for i in range(len(X))]
        else:
            self.train_application_ids_ = application_ids.astype(str).tolist()

        self._fit_nn(embeddings)
        vector_features, _ = self._vector_features(embeddings, exclude_self=True)
        X_augmented = sparse.hstack([X_processed, sparse.csr_matrix(vector_features)], format="csr")

        self.model_ = self._build_classifier()
        self.model_.fit(X_augmented, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.preprocessor_ is None or self.model_ is None:
            raise RuntimeError("Model is not fitted.")
        X_prepared = self._prepare_features(X)
        X_processed = self._ensure_sparse(self.preprocessor_.transform(X_prepared))
        embeddings = self.transform_to_embeddings(X)
        vector_features, _ = self._vector_features(embeddings, exclude_self=False)
        X_augmented = sparse.hstack([X_processed, sparse.csr_matrix(vector_features)], format="csr")
        proba = self.model_.predict_proba(X_augmented)
        return np.asarray(proba)

    def score_with_context(self, X: pd.DataFrame) -> list[dict[str, Any]]:
        probas = self.predict_proba(X)[:, 1]
        embeddings = self.transform_to_embeddings(X)
        vector_features, similar_cases = self._vector_features(embeddings, exclude_self=False)

        results: list[dict[str, Any]] = []
        for i, fraud_probability in enumerate(probas):
            vf = dict(zip(self.vector_feature_names_, vector_features[i].tolist(), strict=False))
            reasons = self.reason_codes(float(fraud_probability), vf)
            results.append(
                {
                    "fraud_probability": float(fraud_probability),
                    "vector_features": vf,
                    "reason_codes": reasons,
                    "similar_cases": [case.__dict__ for case in similar_cases[i]],
                }
            )
        return results

    @staticmethod
    def reason_codes(fraud_probability: float, vector_features: dict[str, float]) -> list[str]:
        reasons: list[str] = []
        if fraud_probability >= 0.75:
            reasons.append("High model fraud probability")
        if vector_features.get("neighbor_fraud_rate_top_20", 0) >= 0.25:
            reasons.append("High fraud rate among similar historical cases")
        if vector_features.get("nearest_fraud_similarity", 0) >= 0.85:
            reasons.append("Very similar to at least one known fraud case")
        if vector_features.get("fraud_neighbors_top_20", 0) >= 3:
            reasons.append("Multiple fraud cases found in nearest-neighbor cluster")
        if not reasons:
            reasons.append("No strong fraud similarity signal detected")
        return reasons

    def get_training_embeddings_frame(self) -> pd.DataFrame:
        if self.train_embeddings_ is None or self.train_labels_ is None or self.train_application_ids_ is None:
            raise RuntimeError("Model is not fitted.")
        df = pd.DataFrame(self.train_embeddings_)
        df.insert(0, "application_id", self.train_application_ids_)
        df["label"] = self.train_labels_
        return df

    def similarity_to_training(self, X: pd.DataFrame, top_k: int = 10) -> list[list[SimilarCase]]:
        embeddings = self.transform_to_embeddings(X)
        _, similar = self._vector_features(embeddings, exclude_self=False)
        return [cases[:top_k] for cases in similar]
