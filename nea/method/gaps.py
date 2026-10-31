"""Local candidates C2, query representation, gap detectors, order and slots.

All functions are pure.  G_q (the method's own gap set) feeds only internal
signals and templates; it is never a paper measurement denominator.
"""

from __future__ import annotations

import heapq
import json
import math
from typing import Any, Iterable, Mapping, Sequence

from nea.method.graph import Graph, resolve_name


EPSILON = 1e-8
GRANULARITY = (("detailed", 3), ("medium", 2), ("brief", 1))
LABELS = ("gap", "given", "not_needed", "uncertain")


def local_candidates(graph: Graph, target_ids: Sequence[str]) -> dict[str, Any]:
    """Union of 1-2 hop prerequisite predecessors of every grounded target."""
    targets = set(target_ids)
    if not targets or not targets <= set(graph.nodes):
        raise ValueError("grounded targets must be known concepts")
    candidates: set[str] = set()
    edge_ids: set[str] = set()
    one_hop: set[str] = set()
    for target in sorted(targets):
        for edge in graph.incoming(target):
            first = edge["src"]
            one_hop.add(first)
            candidates.add(first)
            edge_ids.add(edge["edge_id"])
            for second_edge in graph.incoming(first):
                candidates.add(second_edge["src"])
                edge_ids.add(second_edge["edge_id"])
    candidates -= targets
    return {"candidate_ids": sorted(candidates), "edge_ids": sorted(edge_ids),
            "one_hop_ids": sorted(one_hop - targets)}


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def query_representation(*, terminology: float, structure: float,
                         question_concept_ids: Iterable[str], c2_ids: Sequence[str]) -> dict[str, Any]:
    """T and S come from deterministic query components; P uses the local C2 denominator."""
    for name, value in (("T", terminology), ("S", structure)):
        if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    c2 = set(c2_ids)
    linked = set(question_concept_ids)
    prerequisite = 1.0 if not c2 else len(linked & c2) / (len(c2) + EPSILON)
    raw = (float(terminology) + prerequisite + float(structure)) / 3.0
    if raw < 1 / 3:
        name, sentences = GRANULARITY[0]
    elif raw < 2 / 3:
        name, sentences = GRANULARITY[1]
    else:
        name, sentences = GRANULARITY[2]
    return {"T": round(float(terminology), 6), "P": round(prerequisite, 6),
            "S": round(float(structure), 6), "raw": round(raw, 6),
            "z": round(sigmoid(raw), 6), "granularity": name, "sentences_per_gap": sentences}


def _result(detector: str, labels: Mapping[str, str], extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    gap = sorted(key for key, value in labels.items() if value == "gap")
    return {"detector": detector, "gap_ids": gap,
            "given_ids": sorted(key for key, value in labels.items() if value == "given"),
            "assumed_ids": sorted(key for key, value in labels.items() if value == "assumed"),
            "uncertain_ids": sorted(key for key, value in labels.items() if value == "uncertain"),
            "not_needed_ids": sorted(key for key, value in labels.items() if value == "not_needed"),
            "route": "decompose" if gap else "direct", **dict(extra or {})}


def detect_kp(c2: Mapping[str, Any]) -> dict[str, Any]:
    """KP arm: every local candidate is handed over as a gap (no selection)."""
    return _result("KP", {node: "gap" for node in c2["candidate_ids"]})


def detect_d2(c2: Mapping[str, Any], question_concept_ids: Iterable[str]) -> dict[str, Any]:
    """D2 rule: 1-hop prerequisites not mentioned in the question are gaps."""
    mentioned = set(question_concept_ids)
    labels = {}
    for node in c2["candidate_ids"]:
        if node in mentioned:
            labels[node] = "given"
        elif node in set(c2["one_hop_ids"]):
            labels[node] = "gap"
        else:
            labels[node] = "assumed"
    return _result("D2", labels)


def d1_schema() -> dict[str, Any]:
    return {"type": "object", "additionalProperties": False, "required": ["gap_concepts"],
            "properties": {"gap_concepts": {"type": "array", "maxItems": 8,
                                            "items": {"type": "string"}}}}


def validate_d1_structure(text: str) -> dict[str, Any]:
    parsed = json.loads(text)
    if not isinstance(parsed, dict) or set(parsed) != {"gap_concepts"}:
        raise ValueError("D1 response must be {gap_concepts: [...]} only")
    names = parsed["gap_concepts"]
    if not isinstance(names, list) or not all(isinstance(item, str) for item in names):
        raise ValueError("gap_concepts must be a list of names")
    return parsed


def detect_d1(parsed: Mapping[str, Any], graph: Graph, target_ids: Sequence[str]) -> dict[str, Any]:
    """D1: LLM names without KG candidates; unmappable names are dropped."""
    labels: dict[str, str] = {}
    dropped: list[str] = []
    for name in parsed["gap_concepts"]:
        node = resolve_name(name, graph)
        if node is None or node in set(target_ids):
            dropped.append(str(name))
        else:
            labels[node] = "gap"
    return _result("D1", labels, {"dropped_names": sorted(set(dropped))})


def d3_schema(candidate_ids: Sequence[str]) -> dict[str, Any]:
    if any(not isinstance(node, str) or not node for node in candidate_ids):
        raise ValueError("D3 schema candidate IDs must be nonempty strings")
    ids = sorted(set(candidate_ids))
    return {"type": "object", "additionalProperties": False, "required": ["labels"],
            "properties": {"labels": {"type": "object", "additionalProperties": False,
                                      "required": ids,
                                      "properties": {node: {"type": "string", "enum": list(LABELS)}
                                                     for node in ids}}}}


def validate_d3_structure(text: str, candidate_ids: Sequence[str]) -> dict[str, Any]:
    parsed = json.loads(text)
    if not isinstance(parsed, dict) or set(parsed) != {"labels"}:
        raise ValueError("D3 response must be {labels: {...}} only")
    labels = parsed["labels"]
    if not isinstance(labels, dict) or set(labels) != set(candidate_ids):
        raise ValueError("D3 must label exactly every local candidate")
    if any(value not in LABELS for value in labels.values()):
        raise ValueError("D3 labels must be gap/given/assumed")
    return parsed


def detect_d3(parsed: Mapping[str, Any]) -> dict[str, Any]:
    return _result("D3", dict(parsed["labels"]))


def order_gaps(gap_ids: Sequence[str], graph: Graph) -> dict[str, Any]:
    """Topological order over prerequisite edges among gaps; ties by level, id.

    Adjacent level differences must satisfy -1 <= delta <= 1 (formulas 13/14,
    delta_l = omega = 1).  Violations are recorded; the set is never changed.
    """
    gaps = set(gap_ids)
    edges = [edge for edge in graph.edges if edge["src"] in gaps and edge["dst"] in gaps]
    indegree = {node: 0 for node in gaps}
    children: dict[str, list[str]] = {node: [] for node in gaps}
    for edge in edges:
        indegree[edge["dst"]] += 1
        children[edge["src"]].append(edge["dst"])

    def key(node: str) -> tuple[int, int, str]:
        level = graph.level(node)
        return (0 if level is not None else 1, level if level is not None else 0, node)

    heap = [key(node) for node in gaps if indegree[node] == 0]
    heapq.heapify(heap)
    order: list[str] = []
    while heap:
        node = heapq.heappop(heap)[2]
        order.append(node)
        for child in sorted(children[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(heap, key(child))
    cyclic = sorted(gaps - set(order), key=key)
    order.extend(cyclic)
    violations = []
    for left, right in zip(order, order[1:]):
        low, high = graph.level(left), graph.level(right)
        if low is None or high is None:
            violations.append({"from": left, "to": right, "delta": None, "reason": "unknown_level"})
        elif not -1 <= high - low <= 1:
            violations.append({"from": left, "to": right, "delta": high - low, "reason": "level_jump"})
    return {"order": order, "order_violations": violations, "cycle_members": cyclic,
            "edge_ids": sorted(edge["edge_id"] for edge in edges)}


def example_needed(style: str) -> bool:
    """Only application-style questions carry the optional example slot."""
    return style == "application"


def slot_plan(*, route: str, ordered_gaps: Sequence[str], target_ids: Sequence[str],
              sentences_per_gap: int, style: str) -> list[dict[str, Any]]:
    """decompose: obj -> pre(gap) -> bridge -> target -> example?; direct: obj -> target -> example?"""
    if route not in {"direct", "decompose"}:
        raise ValueError("route must be direct or decompose")
    if route == "decompose" and not ordered_gaps:
        raise ValueError("decompose route needs gaps")
    slots: list[dict[str, Any]] = [{"slot": "obj", "sentences": "1-2"}]
    if route == "decompose":
        slots.extend({"slot": "pre", "concept": node, "sentences": sentences_per_gap}
                     for node in ordered_gaps)
        slots.append({"slot": "bridge", "sentences": 1})
    slots.extend({"slot": "target", "concept": node} for node in target_ids)
    if example_needed(style):
        slots.append({"slot": "example", "sentences": 1})
    return slots


def set_f1(predicted: Iterable[str], gold: Iterable[str]) -> dict[str, int]:
    """Counts for micro F1 of gap sets (tp, fp, fn)."""
    left, right = set(predicted), set(gold)
    return {"tp": len(left & right), "fp": len(left - right), "fn": len(right - left)}


def micro_f1(counts: Iterable[Mapping[str, int]]) -> float:
    tp = fp = fn = 0
    for row in counts:
        tp, fp, fn = tp + row["tp"], fp + row["fp"], fn + row["fn"]
    return 1.0 if tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)
