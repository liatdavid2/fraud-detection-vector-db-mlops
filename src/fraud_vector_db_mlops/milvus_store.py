from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from fraud_vector_db_mlops.config import get_settings


@dataclass
class MilvusSearchResult:
    application_id: str
    label: int
    similarity: float
    distance: float


class MilvusVectorStore:
    def __init__(self, collection_name: str | None = None, vector_dim: int | None = None) -> None:
        self.settings = get_settings()
        self.collection_name = collection_name or self.settings.milvus_collection
        self.vector_dim = vector_dim or self.settings.milvus_vector_dim
        self.collection: Any | None = None

    def connect(self) -> None:
        from pymilvus import connections

        connections.connect(
            alias="default",
            host=self.settings.milvus_host,
            port=self.settings.milvus_port,
        )

    def create_collection(self, drop_existing: bool = False) -> None:
        from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, utility

        self.connect()
        if utility.has_collection(self.collection_name):
            if drop_existing:
                utility.drop_collection(self.collection_name)
            else:
                self.collection = Collection(self.collection_name)
                self.collection.load()
                return

        fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="application_id", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="label", dtype=DataType.INT64),
            FieldSchema(name="fraud_probability", dtype=DataType.FLOAT),
            FieldSchema(name="split", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.vector_dim),
        ]
        schema = CollectionSchema(fields=fields, description="Fraud cases vector index")
        self.collection = Collection(name=self.collection_name, schema=schema)
        index_params = {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 200},
        }
        self.collection.create_index(field_name="embedding", index_params=index_params)
        self.collection.load()

    def upsert_embeddings(
        self,
        embeddings: np.ndarray,
        application_ids: list[str],
        labels: list[int] | np.ndarray,
        probabilities: list[float] | np.ndarray | None = None,
        split: str = "train",
        drop_existing: bool = True,
    ) -> None:
        self.create_collection(drop_existing=drop_existing)
        if self.collection is None:
            raise RuntimeError("Milvus collection is not initialized.")

        probabilities = probabilities if probabilities is not None else np.zeros(len(application_ids))
        data = [
            application_ids,
            [int(x) for x in labels],
            [float(x) for x in probabilities],
            [split for _ in application_ids],
            embeddings.astype("float32").tolist(),
        ]
        self.collection.insert(data)
        self.collection.flush()
        self.collection.load()

    def search(self, embedding: np.ndarray, top_k: int = 10) -> list[MilvusSearchResult]:
        self.create_collection(drop_existing=False)
        if self.collection is None:
            raise RuntimeError("Milvus collection is not initialized.")
        results = self.collection.search(
            data=embedding.astype("float32").reshape(1, -1).tolist(),
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=top_k,
            output_fields=["application_id", "label", "fraud_probability", "split"],
        )
        output: list[MilvusSearchResult] = []
        for hit in results[0]:
            entity = hit.entity
            distance = float(hit.distance)
            output.append(
                MilvusSearchResult(
                    application_id=str(entity.get("application_id")),
                    label=int(entity.get("label")),
                    similarity=distance,
                    distance=distance,
                )
            )
        return output


def try_index_model_embeddings(model: Any, probabilities: np.ndarray | None = None) -> bool:
    try:
        store = MilvusVectorStore(vector_dim=model.embedding_dim)
        if model.train_embeddings_ is None or model.train_application_ids_ is None or model.train_labels_ is None:
            return False
        store.upsert_embeddings(
            embeddings=model.train_embeddings_,
            application_ids=model.train_application_ids_,
            labels=model.train_labels_,
            probabilities=probabilities,
            split="train",
            drop_existing=True,
        )
        print("Milvus indexing completed.")
        return True
    except Exception as exc:
        print(f"Milvus indexing skipped: {exc}")
        return False
