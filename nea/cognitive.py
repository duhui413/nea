"""Query terminology and structure formulas used by the paper experiment."""
from __future__ import annotations
import math
import re
from dataclasses import dataclass, asdict
from typing import Any, Iterable, Sequence
from nea.data import KnowledgeBase
from nea.domain_models import lexical_tokens, normalize_text

EPSILON = 1e-8


def clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def terminology_precision(
    question: str, kb: KnowledgeBase, models: Any | None = None,
) -> dict[str, Any]:
    """Formula (7): IDF-weighted candidate terminology precision."""
    tokens = lexical_tokens(question)
    ner_terms: set[str] = set()
    ner_evidence: list[dict[str, Any]] = []
    normalized_question = normalize_text(question)
    domain_phrases = sorted(
        {
            str(label)
            for concept, chunks in kb.chunks_by_concept.items()
            for label in (
                concept,
                *(alias for chunk in chunks for alias in chunk.aliases),
            )
            if normalize_text(str(label))
            and normalize_text(str(label)) in normalized_question
        }
    )
    surface_phrases = sorted(
        {
            match.group(0).strip()
            for match in re.finditer(
                r"[A-Za-z][A-Za-z0-9_-]*(?:\s+[A-Za-z][A-Za-z0-9_-]*){0,3}|[\u3400-\u9fff]{2,12}",
                question,
            )
            if match.group(0).strip()
        }
    )
    for phrase in (*domain_phrases, *surface_phrases):
        tokens.update(lexical_tokens(phrase))
    documents = [lexical_tokens(chunk.content) for chunk in kb.chunks]
    dictionary = {
        token
        for concept, chunks in kb.chunks_by_concept.items()
        for token in lexical_tokens(" ".join((concept, *(alias for chunk in chunks for alias in chunk.aliases), *(chunk.chapter for chunk in chunks))))
    }
    numerator = 0.0
    denominator = 0.0
    evidence: list[dict[str, float | str]] = []
    for token in sorted(tokens):
        df = sum(token in document for document in documents)
        idf = math.log((len(documents) + 1) / (df + 1)) + 1.0
        denominator += idf
        if token in dictionary:
            numerator += idf
        evidence.append({"term": token, "idf": round(idf, 8), "in_domain_dictionary": token in dictionary})
    value = numerator / (denominator + EPSILON)
    return {
        "value": clamp01(value),
        "numerator": numerator,
        "denominator": denominator,
        "terms": evidence,
        "ner_terms": sorted(ner_terms),
        "ner_evidence": ner_evidence,
        "noun_phrase_candidates": {
            "domain_exact_matches": domain_phrases,
            "deterministic_surface_spans": surface_phrases,
            "policy": "deterministic_rule_spans",
        },
    }

def question_structure_completeness(
    question: str, models: Any | None = None,
) -> dict[str, Any]:
    """Formula (9): four equally weighted rule/NER-detectable slots."""
    normalized = normalize_text(question)
    ner_rows: list[dict[str, Any]] = []
    ner_constraint = False
    slots = {
        "object": bool(re.search(r"[\u3400-\u9fffA-Za-z0-9]", question)),
        "goal": any(marker in normalized for marker in ("为什么", "如何", "说明", "解释", "why", "how", "explain")),
        "constraint": ner_constraint or any(marker in normalized for marker in ("条件", "假设", "安全", "攻击", "限制", "if", "under", "assumption")),
        "response_form": any(marker in normalized for marker in ("请", "用", "分", "步骤", "证明", "compare", "show")),
    }
    weights = {name: 0.25 for name in slots}
    return {
        "value": sum(weights[name] * float(present) for name, present in slots.items()),
        "slots": slots,
        "weights": weights,
        "ner_evidence": ner_rows,
        "decision_policy": "deterministic_rule_markers",
    }
