"""Target grounding: deterministic candidates with explicit abstention."""

from __future__ import annotations

import json
from decimal import Decimal, ROUND_HALF_EVEN
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from nea.method.graph import Graph
from nea.providers.config import LOCAL_MODEL_DEVICE, LOCAL_MODEL_DTYPE


TOP_K = 10
MAX_TARGETS = 2
_SIX = Decimal("0.000001")


class GroundingEmbedder(Protocol):
    def embed(self, texts: Sequence[str], *, device: str, dtype: str) -> Sequence[Sequence[float]]: ...


def concept_text(node: Mapping[str, Any]) -> str:
    return "\n".join((str(node["name"]), str(node.get("chapter", "")), str(node.get("explain", ""))[:300]))


def _q6(value: float) -> str:
    return format(Decimal(repr(float(np.float32(value)))).quantize(_SIX, rounding=ROUND_HALF_EVEN), ".6f")


class CandidateIndex:
    """Encode concepts in CUDA fp16; accumulate cosine in fp32 for stable ranking."""

    def __init__(self, graph: Graph, embedder: GroundingEmbedder):
        self.ids = sorted(graph.nodes)
        vectors = embedder.embed([concept_text(graph.nodes[node_id]) for node_id in self.ids],
                                 device=LOCAL_MODEL_DEVICE, dtype=LOCAL_MODEL_DTYPE)
        import torch
        from nea.domain_models import configure_local_runtime
        configure_local_runtime()
        matrix = torch.as_tensor(vectors, dtype=torch.float16, device="cuda")
        if matrix.ndim != 2 or matrix.shape[0] != len(self.ids) or not torch.isfinite(matrix).all().item():
            raise ValueError("embedder returned malformed concept vectors")
        self.matrix = self._normalize(matrix)
        self.embedder = embedder
        self.last_scores = None

    @staticmethod
    def _normalize(vectors):
        import torch
        wide = vectors.float()
        norms = torch.linalg.vector_norm(wide, dim=-1, keepdim=True)
        return (wide / torch.where(norms == 0, torch.ones_like(norms), norms)).half()

    def rank(self, question: str, k: int = TOP_K) -> list[dict[str, str]]:
        import torch
        vector = torch.as_tensor(
            self.embedder.embed([question], device=LOCAL_MODEL_DEVICE, dtype=LOCAL_MODEL_DTYPE)[0],
            dtype=torch.float16, device="cuda")
        if vector.shape != (self.matrix.shape[1],) or not torch.isfinite(vector).all().item():
            raise ValueError("embedder returned a malformed question vector")
        normalized = self._normalize(vector)
        # FP16 multiplication inputs, FP32 accumulation/output. No low precision
        # reduction or TF32; configure_local_runtime fixes deterministic cuBLAS.
        scores = torch.mm(self.matrix, normalized[:, None], out_dtype=torch.float32).flatten()
        if not torch.isfinite(scores).all().item():
            raise ValueError("grounding produced non-finite scores")
        self.last_scores = scores
        ranked = sorted(((-Decimal(_q6(score)), node_id, _q6(score))
                         for node_id, score in zip(self.ids, scores.cpu().tolist())))
        return [{"concept_id": node_id, "score_q6": q6} for _, node_id, q6 in ranked[:k]]


def grounding_schema(candidate_ids: Sequence[str] | None = None) -> dict[str, Any]:
    item: dict[str, Any] = {"type": "string", "minLength": 1}
    if candidate_ids is not None:
        if any(not isinstance(node, str) or not node for node in candidate_ids):
            raise ValueError("grounding schema requires nonempty candidate IDs")
        if candidate_ids:
            item["enum"] = sorted(set(candidate_ids))
    return {"type": "object", "additionalProperties": False,
            "required": ["grounded_target_ids"],
            "properties": {"grounded_target_ids": {
                "type": "array", "minItems": 0, "maxItems": (MAX_TARGETS if candidate_ids is None or candidate_ids else 0),
                "uniqueItems": True, "items": item}}}


def validate_grounding_structure(text: str) -> dict[str, Any]:
    """Transport-level structure check (G3): JSON object with a string list."""
    parsed = json.loads(text)
    if not isinstance(parsed, dict) or set(parsed) != {"grounded_target_ids"}:
        raise ValueError("grounding response must be {grounded_target_ids: [...]} only")
    ids = parsed["grounded_target_ids"]
    if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
        raise ValueError("grounded_target_ids must be a list of strings")
    return parsed


def select_targets(parsed: Mapping[str, Any] | None, candidates: Sequence[Mapping[str, str]],
                   *, failure: str | None = None) -> dict[str, Any]:
    """Keep a valid 0-2 candidate selection; on failure abstain with a reason."""
    candidate_ids = [row["concept_id"] for row in candidates]
    reason = failure
    chosen: list[str] = []
    if reason is None and parsed is not None:
        raw = list(parsed.get("grounded_target_ids", []))
        if not 0 <= len(raw) <= MAX_TARGETS:
            reason = "target_count_out_of_range"
        elif len(set(raw)) != len(raw):
            reason = "duplicate_target"
        elif not set(raw) <= set(candidate_ids):
            reason = "target_outside_candidates"
        else:
            chosen = raw
    elif reason is None:
        reason = "no_response"
    if reason is not None:
        return {"candidates": list(candidates), "grounded_target_ids": [],
                "grounding_source": "abstained", "fallback_reason": reason}
    return {"candidates": list(candidates), "grounded_target_ids": chosen,
            "grounding_source": "llm" if chosen else "abstained", "fallback_reason": None}
