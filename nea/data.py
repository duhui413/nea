"""Frozen concept records for retrieval and terminology scoring."""
from __future__ import annotations
import json
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]

@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    concept_id: str
    concept_name: str
    aliases: tuple[str, ...]
    content: str
    graph_level: int
    chapter: str

    def to_dict(self):
        return asdict(self)

class KnowledgeBase:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.chunks_by_concept = defaultdict(list)
        for chunk in self.chunks:
            self.chunks_by_concept[chunk.concept_name].append(chunk)

    @classmethod
    def load(cls, root: Path = Path("knowledge_base")):
        from nea.method.graph import load_graph, to_knowledge_base
        return to_knowledge_base(load_graph(root))
