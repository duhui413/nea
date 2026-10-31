"""Read-only knowledge-graph view for need-aware (levels from ``graph.jsonl`` only).

Only prerequisite edges carry learning order (node1 is a prerequisite of
node2).  Nothing here writes to or regenerates the frozen knowledge base.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class Graph:
    nodes: dict[str, dict[str, Any]]
    edges: tuple[dict[str, str], ...]
    _incoming: dict[str, tuple[dict[str, str], ...]] = field(default_factory=dict, repr=False)
    _labels: tuple[tuple[str, str], ...] = field(default=(), repr=False)

    def incoming(self, node_id: str) -> tuple[dict[str, str], ...]:
        return self._incoming.get(node_id, ())

    def level(self, node_id: str) -> int | None:
        return self.nodes[node_id]["level"] if node_id in self.nodes else None

    def name(self, node_id: str) -> str:
        return self.nodes[node_id]["name"]

    def labels(self) -> tuple[tuple[str, str], ...]:
        return self._labels


def graph_from_records(concepts: Iterable[Mapping[str, Any]], edges: Iterable[Mapping[str, Any]],
                       graph_rows: Iterable[Mapping[str, Any]]) -> Graph:
    levels: dict[str, int] = {}
    for row in graph_rows:
        level = row.get("graph_level")
        if type(level) is not int or level < 0:
            raise ValueError(f"graph row {row.get('id')!r} has no integer graph_level")
        levels[str(row["id"])] = level
    nodes: dict[str, dict[str, Any]] = {}
    for row in concepts:
        node_id = str(row["id"])
        if node_id in nodes:
            raise ValueError(f"duplicate concept id {node_id}")
        name = str(row.get("name", "")).strip()
        if not name:
            raise ValueError(f"concept {node_id} has no name")
        nodes[node_id] = {"id": node_id, "name": name,
                          "aliases": [str(alias) for alias in row.get("aliases") or [] if str(alias).strip()],
                          "chapter": str(row.get("chapter", "")),
                          "explain": str(row.get("explain", "")),
                          "level": levels.get(node_id)}
    prerequisite: list[dict[str, str]] = []
    for row in edges:
        if row.get("type") != "prerequisite":
            continue
        src, dst = str(row["node1"]), str(row["node2"])
        if src not in nodes or dst not in nodes:
            raise ValueError(f"edge {row.get('id')!r} refers to an unknown concept")
        if src != dst:
            prerequisite.append({"edge_id": str(row["id"]), "src": src, "dst": dst})
    prerequisite.sort(key=lambda edge: edge["edge_id"])
    incoming: dict[str, list[dict[str, str]]] = {}
    for edge in prerequisite:
        incoming.setdefault(edge["dst"], []).append(edge)
    labels: dict[str, str] = {}
    for node_id in sorted(nodes):
        for label in (nodes[node_id]["name"], *nodes[node_id]["aliases"]):
            key = label.strip().casefold()
            if len(key) >= 2 and key not in labels:
                labels[key] = node_id  # ambiguous label: smallest id wins, deterministically
    ordered_labels = tuple(sorted(labels.items(), key=lambda item: (-len(item[0]), item[0])))
    return Graph(nodes=nodes, edges=tuple(prerequisite),
                 _incoming={key: tuple(value) for key, value in incoming.items()},
                 _labels=ordered_labels)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_graph(root: Path) -> Graph:
    """Read concepts/edges/graph JSONL (read-only)."""
    return graph_from_records(_jsonl(root / "concepts.jsonl"), _jsonl(root / "edges.jsonl"),
                              _jsonl(root / "graph.jsonl"))


def link_concepts(text: str, graph: Graph) -> list[str]:
    """Longest-label, non-overlapping concept linking; ids in first-occurrence order."""
    folded = str(text).casefold()
    taken = [False] * len(folded)
    hits: list[tuple[int, str]] = []
    for label, node_id in graph.labels():
        start = folded.find(label)
        while start != -1:
            end = start + len(label)
            if not any(taken[start:end]):
                for index in range(start, end):
                    taken[index] = True
                hits.append((start, node_id))
            start = folded.find(label, start + 1)
    ordered: list[str] = []
    for _, node_id in sorted(hits):
        if node_id not in ordered:
            ordered.append(node_id)
    return ordered


def resolve_name(name: str, graph: Graph) -> str | None:
    """Map one concept name/alias to a library id, or ``None``."""
    key = str(name).strip().casefold()
    for label, node_id in graph.labels():
        if label == key:
            return node_id
    return None


def to_knowledge_base(graph: Graph) -> Any:
    """Use graph levels in the concept view consumed by retrieval and T scoring."""
    from nea.data import Chunk, KnowledgeBase
    return KnowledgeBase([Chunk(chunk_id=node_id, concept_id=node_id, concept_name=node["name"],
                                aliases=tuple(node["aliases"]), content=node["explain"],
                                graph_level=node["level"], chapter=node["chapter"])
                          for node_id, node in sorted(graph.nodes.items())])
