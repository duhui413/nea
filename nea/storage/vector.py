"""Milvus search result."""
from dataclasses import dataclass
from nea.data import Chunk
@dataclass(frozen=True)
class VectorSearchHit:
    chunk: Chunk
    dense_score: float
    query_id: str
