"""Local-only embedding/reranking models for the current experiment."""
from __future__ import annotations
import os
import re
import unicodedata
from pathlib import Path
from typing import Sequence
from nea.providers.config import (
    LOCAL_MODEL_PATHS as MODEL_PATHS, LOCAL_MODEL_DEVICE, LOCAL_MODEL_DTYPE,
    LOCAL_MODEL_SEED, LOCAL_MODEL_ATTENTION,
)


def configure_local_runtime() -> None:
    """Fixed CUDA FP16 execution; fail instead of silently changing precision."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Local model runtime requires CUDA FP16; CUDA is unavailable")
    torch.manual_seed(LOCAL_MODEL_SEED)
    torch.cuda.manual_seed_all(LOCAL_MODEL_SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cudnn.allow_tf32 = False

def model_local_path(models_dir: Path, model_id: str) -> Path:
    return models_dir.joinpath(*model_id.split("/"))

def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold().strip()

def lexical_tokens(text: str) -> set[str]:
    normalized = normalize_text(text)
    ascii_words = set(re.findall(r"[a-z0-9_]+", normalized))
    han_runs = re.findall(r"[\u3400-\u9fff]+", normalized)
    tokens = set(ascii_words)
    for run in han_runs:
        tokens.update(run)
        tokens.update(run[index : index + 2] for index in range(max(0, len(run) - 1)))
    return {token for token in tokens if token}

class HuggingFaceDomainModels:
    def __init__(self, models_dir: Path, device=LOCAL_MODEL_DEVICE, dtype=LOCAL_MODEL_DTYPE, strict=True):
        if not strict:
            raise ValueError("Only real local models are supported")
        if (device, dtype) != (LOCAL_MODEL_DEVICE, LOCAL_MODEL_DTYPE):
            raise ValueError("Local model runtime must use cuda/fp16")
        configure_local_runtime()
        self.models_dir, self.device, self.dtype = models_dir, device, dtype
        self._embedding = self._reranker = None

    def embed(self, texts: Sequence[str], *, purpose="retrieval") -> list[list[float]]:
        if purpose not in {"retrieval", "grounding"}:
            raise ValueError("Only retrieval and grounding embeddings are used by this experiment")
        if self._embedding is None:
            import torch
            from sentence_transformers import SentenceTransformer
            self._embedding = SentenceTransformer(str(model_local_path(self.models_dir, MODEL_PATHS["retrieval_embedding"])),
                device=self.device, local_files_only=True,
                model_kwargs={"dtype": torch.float16, "attn_implementation": LOCAL_MODEL_ATTENTION})
            self._embedding.half().eval()
        # Return Python values with FP16 quantisation; CUDA cosine and Milvus
        # explicitly reconstruct their half-precision input at the boundary.
        return self._embedding.encode(list(texts), normalize_embeddings=purpose == "retrieval",
                                      show_progress_bar=False).astype("float16").tolist()

    def rerank(self, query: str, documents: Sequence[str]) -> list[float]:
        if self._reranker is None:
            import torch
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(str(model_local_path(self.models_dir, MODEL_PATHS["reranker"])),
                device=self.device, local_files_only=True,
                model_kwargs={"dtype": torch.float16, "attn_implementation": LOCAL_MODEL_ATTENTION})
            self._reranker.half().eval()
        return [float(score) for score in self._reranker.predict([(query, doc) for doc in documents],
                                                               show_progress_bar=False)]
