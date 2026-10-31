"""Shared ordinary RAG context for all experiment arms."""
import hashlib
import json
import math
from typing import Any, Callable, Mapping, Sequence

def _chunk_id(row: Any) -> str:
    chunk = row.get("chunk") if isinstance(row, Mapping) else getattr(row, "chunk", None)
    if chunk is not None:
        return str(chunk.get("chunk_id") if isinstance(chunk, Mapping)
                   else getattr(chunk, "chunk_id"))
    return str(row.get("chunk_id") if isinstance(row, Mapping) else getattr(row, "chunk_id"))

def _field(row: Any, key: str, default: Any = None) -> Any:
    return row.get(key, default) if isinstance(row, Mapping) else getattr(row, key, default)

def _public_chunk(row: Any) -> dict[str, Any]:
    chunk = _field(row, "chunk", row)
    return {"chunk_id": _chunk_id(row),
            "concept_id": str(_field(chunk, "concept_id", "")),
            "concept_name": str(_field(chunk, "concept_name", "")),
            "content": str(_field(chunk, "content", ""))}

def select_shared_rag_context(
    candidate_pool: Sequence[Any], *, top_k: int = 8,
) -> dict[str, Any]:
    """Choose the same ordinary top-eight evidence for all arms."""
    if top_k != 8:
        raise ValueError("Main RAG context must contain exactly eight chunks")
    if len(candidate_pool) < top_k:
        raise ValueError("candidate pool has fewer than eight chunks")
    ids = [_chunk_id(row) for row in candidate_pool]
    if not all(ids) or len(set(ids)) != len(ids):
        raise ValueError("candidate pool needs eight or more distinct chunk IDs")
    # Existing ordinary-RAG rank comes from the shared retriever.  Equal
    # scores break on raw chunk ID, never on arm or input enumeration.
    def key(row: Any) -> tuple[float, float, str]:
        score = float(_field(row, "rerank_score", _field(row, "score", 0)) or 0)
        dense = float(_field(row, "dense_score", 0) or 0)
        if not math.isfinite(score) or not math.isfinite(dense):
            raise ValueError("retrieval scores must be finite")
        return (-score, -dense, _chunk_id(row))
    ranked = sorted(candidate_pool, key=key)
    strategy = "shared-ordinary-rag"
    selected = [_public_chunk(row) for row in ranked[:top_k]]
    pool_fingerprint = hashlib.sha256(
        json.dumps([_public_chunk(row) for row in sorted(candidate_pool, key=_chunk_id)],
                   ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {"strategy": strategy, "candidate_pool_sha256": pool_fingerprint,
            "chunk_ids": [row["chunk_id"] for row in selected], "chunks": selected}