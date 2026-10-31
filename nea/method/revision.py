"""Candidate structural checks; semantic validation must run before acceptance."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from nea.method.citations import extract_citations
from nea.runtime.records import answer_sha256

from nea.pipeline.registry import REVISION_LIMITS


MAX_GROWTH = REVISION_LIMITS["max_growth"]
MAX_CALLS = REVISION_LIMITS["max_calls"]
MAX_LENGTH_RETRY = REVISION_LIMITS["length_retry"]
MAX_ACTIONS = REVISION_LIMITS["max_actions"]
ACTION_TYPES = ("add_gap", "compress_detour", "add_bridge")
CONDITION_MARKERS = ("当且仅当", "前提", "条件", "必须", "只有", "如果", "若", "假设", "除非")
_SENTENCE = re.compile(r"[^。！？!?\n]+[。！？!?]?")


def key_condition_sentences(answer: str) -> list[str]:
    """Sentences of the initial answer that state a condition; candidates for semantic checking, never protected verbatim."""
    found: list[str] = []
    for match in _SENTENCE.finditer(answer):
        sentence = match.group(0).strip()
        if sentence and any(marker in sentence for marker in CONDITION_MARKERS) and sentence not in found:
            found.append(sentence)
    return found


def max_chars(initial: str) -> int:
    """The one length limit: shown to the model and enforced (no slack)."""
    return int(len(initial) * MAX_GROWTH)


def constraints(initial: str, allowed_citations: Sequence[str]) -> dict[str, Any]:
    return {"max_chars": max_chars(initial), "parent_sha256": answer_sha256(initial),
            "conditions_to_check": key_condition_sentences(initial),
            "allowed_citations": sorted(set(allowed_citations)),
            "semantic_validation_required": True}


def select_actions(parsed: Mapping[str, Any], *, uncovered_gaps: Sequence[str],
                   detour_units: Sequence[int], weak_transitions: Sequence[int],
                   triggered: Mapping[str, bool]) -> list[dict[str, Any]]:
    """Keep executable, evidence-backed actions only (<= 3)."""
    allowed_types = set()
    if triggered.get("cer_g"):
        allowed_types.add("add_gap")
    if triggered.get("dr_g"):
        allowed_types.add("compress_detour")
    if triggered.get("rcc") or triggered.get("lts"):
        allowed_types.add("add_bridge")
    selected: list[dict[str, Any]] = []
    for row in parsed.get("actions", []):
        if not isinstance(row, Mapping):
            continue
        kind, instruction = row.get("type"), str(row.get("instruction", "")).strip()
        if kind not in allowed_types or not instruction:
            continue
        if kind == "add_gap" and row.get("concept_id") not in set(uncovered_gaps):
            continue
        if kind == "compress_detour" and row.get("unit_index") not in set(detour_units):
            continue
        if kind == "add_bridge" and row.get("unit_index") not in set(weak_transitions):
            continue
        action = {"type": kind, "instruction": instruction}
        for field in ("concept_id", "unit_index"):
            if field in row:
                action[field] = row[field]
        if action not in selected:
            selected.append(action)
        if len(selected) == MAX_ACTIONS:
            break
    return selected


def validate_revision(initial: str, text: str, limits: Mapping[str, Any]) -> tuple[bool, str]:
    """Check candidate structure only; never label a candidate accepted."""
    if not isinstance(text, str) or not text.strip():
        return False, "invalid_schema"
    if text.strip() == initial.strip():
        return False, "unchanged"
    limit = limits["max_chars"]
    if type(limit) is not int or limit != max_chars(initial):
        raise ValueError("revision limits do not belong to this initial answer")
    if len(text) > limit:
        return False, "length_exceeded"
    cited = set(extract_citations(text))
    if not cited <= set(limits["allowed_citations"]):
        return False, "unknown_citation"
    if limits.get("parent_sha256") != answer_sha256(initial):
        raise ValueError("revision parent mismatch")
    return True, "pending_semantic_validation"



def decision(records: Sequence[Mapping[str, Any]]) -> str:
    """``accepted`` | ``exhausted`` | ``continue`` for the next revise call."""
    if any(row["status"] == "accepted" for row in records):
        return "accepted"
    if len(records) >= MAX_CALLS:
        return "exhausted"
    if sum(row["status"] == "length_exceeded" for row in records) > MAX_LENGTH_RETRY:
        return "exhausted"
    return "continue"


def finalize(initial: str, records: Sequence[Mapping[str, Any]], *, reason_if_none: str | None = None,
             actions: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    initial_hash = answer_sha256(initial)
    # An accepted record must actually change the text; an identical one can
    # never mark the answer as revised (defence in depth for "unchanged").
    accepted = [row for row in records if row["status"] == "accepted"
                and answer_sha256(row["answer"]) != initial_hash]
    if accepted:
        answer = accepted[0]["answer"]
        return {"answer": answer, "answer_sha256": answer_sha256(answer), "parent_sha256": initial_hash,
                "revised": True, "fallback_reason": None, "calls": len(records),
                "candidate_records": list(records), "actions": list(actions)}
    return {"answer": initial, "answer_sha256": initial_hash, "parent_sha256": initial_hash,
            "revised": False,
            "fallback_reason": reason_if_none or ("no_valid_candidate" if records else "no_call"),
            "calls": len(records), "candidate_records": list(records), "actions": list(actions)}
