"""Dense retrieval against the delivered ``nea.concepts`` collection.

The collection is built once by ``scripts/database/build_milvus_concepts.py``
and is read-only here: one concept per row, ``BAAI/bge-small-zh-v1.5`` at 512
dimensions, HNSW + COSINE.  This module only embeds a query and returns the
matching concepts as :class:`~nea.data.Chunk` rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Sequence

from nea.data import Chunk, KnowledgeBase
from nea.domain_models import HuggingFaceDomainModels
from nea.storage.vector import VectorSearchHit

DEFAULT_URI = "http://localhost:19530"
DEFAULT_DB = "nea"
DEFAULT_COLLECTION = "concepts"
VECTOR_FIELD = "vector"
PRIMARY_FIELD = "id"


def fp16_payload(vector: Sequence[float]):
    """Typed ndarray makes PyMilvus infer FLOAT16 for both insert and search."""
    import numpy as np
    half = np.asarray(vector, dtype="<f2")
    if half.shape != (512,) or not np.isfinite(half).all():
        raise ValueError("Milvus vector must be finite and 512-dimensional")
    return half


@dataclass
class MilvusConceptStore:
    """Read-only dense recall over ``<db_name>.<collection>``."""

    knowledge_base: KnowledgeBase
    models: HuggingFaceDomainModels
    uri: str = DEFAULT_URI
    db_name: str = DEFAULT_DB
    collection: str = DEFAULT_COLLECTION

    def __post_init__(self) -> None:
        if (self.uri, self.db_name, self.collection) != (DEFAULT_URI, DEFAULT_DB, DEFAULT_COLLECTION):
            raise ValueError("Only localhost nea.concepts is supported")
        from pymilvus import MilvusClient

        self._client = MilvusClient(uri=self.uri, db_name=self.db_name)
        self._by_id = {chunk.chunk_id: chunk for chunk in self.knowledge_base.chunks}

    def healthcheck(self) -> dict[str, Any]:
        stats = self._client.get_collection_stats(collection_name=self.collection)
        indexes = self._client.list_indexes(collection_name=self.collection)
        index = self._client.describe_index(
            collection_name=self.collection, index_name=indexes[0]
        )
        return {
            "uri": self.uri,
            "db_name": self.db_name,
            "collection": self.collection,
            "row_count": int(stats.get("row_count", 0)),
            "index_type": str(index.get("index_type", "")),
            "metric_type": str(index.get("metric_type", "")),
        }

    def _embed_query(self, query: str):
        vectors = self.models.embed([query], purpose="retrieval")
        if not vectors or not vectors[0]:
            raise RuntimeError("query embedding produced no vector")
        return fp16_payload(vectors[0])

    def search(
        self,
        query: str,
        *,
        top_k: int,
        query_id: str = "query",
        level_filter: str = "",
    ) -> list[VectorSearchHit]:
        results = self._client.search(
            collection_name=self.collection,
            data=[self._embed_query(query)],
            anns_field=VECTOR_FIELD,
            limit=top_k,
            filter=level_filter,
            output_fields=[PRIMARY_FIELD],
            search_params={"metric_type": "COSINE", "params": {}},
        )
        hits: list[VectorSearchHit] = []
        for row in results[0] if results else []:
            entity = row.get("entity") or row
            concept_id = str(entity.get(PRIMARY_FIELD) or row.get("id", ""))
            chunk = self._by_id.get(concept_id)
            if chunk is None:
                raise RuntimeError(
                    f"Milvus returned concept {concept_id!r} that is absent from "
                    "knowledge_base/concepts.jsonl; the collection and the files "
                    "are out of sync"
                )
            hits.append(
                VectorSearchHit(
                    chunk=chunk,
                    dense_score=float(row.get("distance", 0.0)),
                    query_id=query_id,
                )
            )
        return hits
