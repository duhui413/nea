"""Blind paper evaluation helpers. No provider, file, or network side effects.

The teacher receives only ``request["payload"]``.  Identity mappings and
transport metadata are deliberately separate from that blind payload.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import defaultdict
from typing import Any, Iterable, Mapping


RUBRIC_VERSION = "paper-rubric-v2"
SLOTS = ("B", "P", "L", "F", "SR")
# Match the executable paper arm registry; PE/RAG are presentation labels only.
BASES = ("pe", "rag")
TEACHERS = ("teacher1", "teacher2")
SEED = 20260927

from nea.prompts import DEFAULT_PROMPTS_DIR, PromptStore


def _rubric():
    return json.loads((DEFAULT_PROMPTS_DIR / "quality_rubric.json").read_text(encoding="utf-8"))


RUBRIC_ANCHORS = {dim: {int(k): v for k, v in anchors.items()}
                  for dim, anchors in _rubric()["anchors"].items()}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def rubric_payload() -> dict[str, Any]:
    """The complete six-dimensional paper rubric for one blind evaluation."""
    return _rubric()


RUBRIC_HASH = _hash(rubric_payload())


def _answer_rows(rows: Iterable[Mapping[str, Any]], *, require_six: bool = True) -> list[dict[str, str]]:
    found: dict[tuple[str, str, str, str], dict[str, str]] = {}
    questions: dict[str, str] = {}
    for source in rows:
        keys = ("qid", "generator", "base", "arm", "question", "answer")
        if any(not isinstance(source.get(key), str) or not source[key].strip() for key in keys):
            raise ValueError("answer fields must be nonempty strings")
        row = {key: source[key] for key in keys}
        if row["base"] not in BASES or row["arm"] not in SLOTS:
            raise ValueError("invalid answer identity or empty question/answer")
        old_question = questions.setdefault(row["qid"], row["question"])
        if old_question != row["question"]:
            raise ValueError(f"conflicting question text for {row['qid']}")
        key = tuple(row[k] for k in ("qid", "generator", "base", "arm"))
        if key in found:
            raise ValueError(f"duplicate evaluation slot: {key}")
        found[key] = row
    if require_six:
        groups: dict[tuple[str, str, str], set[str]] = defaultdict(set)
        for qid, generator, base, arm in found:
            groups[(qid, generator, base)].add(arm)
        for key, slots in groups.items():
            if slots != set(SLOTS):
                raise ValueError(f"incomplete five-slot evaluation group {key}: {sorted(slots)}")
    return [found[key] for key in sorted(found)]


_IDENTITY_LEAK = re.compile(
    r"(?<![\w])(?:ca[-_]kdag|pe_ca|rag_ca|c1[-_]i|c1[-_]f|"
    r"ablation|experiment_id|sample_id|program_reason|step[1-5])(?![\w])",
    re.IGNORECASE,
)


def _check_blind_content(question: str, answer: str) -> None:
    # The schema itself excludes arm identity; reject explicit identity text as
    # well. Do not ban single-letter slots, which are ordinary domain symbols.
    if _IDENTITY_LEAK.search(question) or _IDENTITY_LEAK.search(answer):
        raise ValueError("question or answer exposes method/experiment identity")


def build_teacher_queue(
    rows: Iterable[Mapping[str, Any]], batch_id: str, *, teachers: tuple[str, ...] = TEACHERS,
    seed: int = SEED,
) -> dict[str, Any]:
    """Mix five slots in one batch and map byte-identical answers to one call.

    Cache identity is exactly (qid, answer_hash, rubric_hash, teacher, batch).
    The queue's mapping is an audit artifact and must never enter a prompt.
    """
    if not batch_id or not teachers or len(set(teachers)) != len(teachers):
        raise ValueError("batch_id and distinct teachers are required")
    source = _answer_rows(rows)
    requests: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    mapping: list[dict[str, str]] = []
    for row in source:
        _check_blind_content(row["question"], row["answer"])
        answer_hash = hashlib.sha256(row["answer"].encode("utf-8")).hexdigest()
        candidate_id = _hash([row["qid"], answer_hash, RUBRIC_HASH])[:24]
        payload = {"question": row["question"], "candidate_id": candidate_id,
                   "candidate_answer": row["answer"], "rubric": rubric_payload()}
        if set(payload) != {"question", "candidate_id", "candidate_answer", "rubric"}:
            raise AssertionError("blind payload contains metadata")
        for teacher in teachers:
            cache_key = (row["qid"], answer_hash, RUBRIC_HASH, teacher, batch_id)
            if cache_key not in requests:
                requests[cache_key] = {
                    "cache_key": _hash(cache_key), "teacher": teacher, "batch_id": batch_id,
                    "rubric_hash": RUBRIC_HASH, "payload": payload,
                }
            mapping.append({**{k: row[k] for k in ("qid", "generator", "base", "arm")},
                            "teacher": teacher, "batch_id": batch_id,
                            "candidate_id": candidate_id, "cache_key": requests[cache_key]["cache_key"]})
    ordered = list(requests.values())
    random.Random(seed).shuffle(ordered)
    return {"rubric_hash": RUBRIC_HASH, "batch_id": batch_id,
            "requests": ordered, "mapping": mapping,
            "evaluation_slots": len(source), "teacher_calls": len(ordered)}


def import_teacher_scores(queue: Mapping[str, Any], results: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Validate paid results before expanding cached answers to all slots."""
    if queue.get("rubric_hash") != RUBRIC_HASH:
        raise ValueError("teacher queue uses a different rubric")
    expected = {r["cache_key"]: r for r in queue["requests"]}
    received: dict[str, Mapping[str, Any]] = {}
    for row in results:
        key = str(row["cache_key"])
        if key not in expected or key in received:
            raise ValueError(f"unknown or duplicate teacher cache key: {key}")
        request = expected[key]
        if (row.get("candidate_id") != request["payload"]["candidate_id"]
                or row.get("teacher") != request["teacher"]
                or row.get("rubric_hash") != request["rubric_hash"]
                or row.get("batch_id") != request["batch_id"]):
            raise ValueError("teacher identity/rubric/batch mismatch")
        scores = row.get("scores")
        if not isinstance(scores, dict) or set(scores) != set(RUBRIC_ANCHORS):
            raise ValueError("all six rubric dimensions are required")
        if any(type(value) is not int or not 1 <= value <= 5 for value in scores.values()):
            raise ValueError("rubric scores must be integers in 1..5")
        received[key] = row
    expanded = []
    for item in queue["mapping"]:
        if item["cache_key"] in received:
            expanded.append({**item, "scores": dict(received[item["cache_key"]]["scores"])})
    return expanded


def export_human_pairs(rows: Iterable[Mapping[str, Any]], *, seed: int = SEED,
                       per_cell: int = 10) -> dict[str, Any]:
    """Preselect F/P test questions by generator × base, before any scores."""
    source = _answer_rows(rows)
    by_cell: dict[tuple[str, str], dict[str, dict[str, dict[str, str]]]] = defaultdict(lambda: defaultdict(dict))
    for row in source:
        by_cell[(row["generator"], row["base"])][row["qid"]][row["arm"]] = row
    if len(by_cell) != 4:
        raise ValueError("human audit requires two generators × two bases")
    requests, mapping = [], []
    for cell in sorted(by_cell):
        eligible = sorted(qid for qid, arms in by_cell[cell].items() if {"F", "P"} <= arms.keys())
        if len(eligible) < per_cell:
            raise ValueError(f"fewer than {per_cell} F/P pairs in {cell}")
        selected = sorted(random.Random(f"{seed}:{cell}").sample(eligible, per_cell))
        for qid in selected:
            arms = by_cell[cell][qid]
            pair_id = _hash(["human", seed, *cell, qid])[:24]
            flipped = random.Random(f"{seed}:{pair_id}").randrange(2) == 1
            left_arm, right_arm = ("F", "P") if flipped else ("P", "F")
            requests.append({"pair_id": pair_id, "question": arms["F"]["question"],
                             "left": arms[left_arm]["answer"], "right": arms[right_arm]["answer"],
                             "fields": ["relevance", "correctness", "necessary_explanation", "reasoning_continuity", "overall_preference", "rationale"]})
            mapping.append({"pair_id": pair_id, "qid": qid, "generator": cell[0], "base": cell[1],
                            "left_arm": left_arm, "right_arm": right_arm})
    random.Random(seed).shuffle(requests)
    return {"protocol": "human-f-vs-p-v2", "seed": seed, "requests": requests, "mapping": mapping,
            "status": "NOT_RUN", "pairs": len(requests)}


_HUMAN_FIELDS = ("relevance", "correctness", "necessary_explanation", "reasoning_continuity", "overall_preference")


def import_human_reviews(export: Mapping[str, Any], reviews: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Retain both independent opinions and a third adjudication on disputes."""
    ids = {r["pair_id"] for r in export["requests"]}
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for source in reviews:
        row = dict(source)
        pair_id, rater = row.get("pair_id"), row.get("rater_id")
        if pair_id not in ids or not rater or rater in grouped[pair_id]:
            raise ValueError("unknown pair or duplicate/missing rater")
        if row.get("role") not in {"independent", "adjudicator"}:
            raise ValueError("human role must be independent or adjudicator")
        if any(row.get(k) not in {"left", "tie", "right"} for k in _HUMAN_FIELDS):
            raise ValueError("every human comparison needs left/tie/right")
        if not str(row.get("rationale", "")).strip():
            raise ValueError("human comparison requires rationale")
        grouped[pair_id][rater] = row
    complete, disputes, pending = 0, 0, []
    for pair_id in sorted(ids):
        values = list(grouped[pair_id].values())
        independent = [r for r in values if r["role"] == "independent"]
        adjudicators = [r for r in values if r["role"] == "adjudicator"]
        if len(independent) > 2 or len(adjudicators) > 1:
            raise ValueError("requires exactly two independent raters and at most one adjudicator")
        if len(independent) < 2:
            pending.append(pair_id)
            continue
        disagree = any(independent[0][k] != independent[1][k] for k in _HUMAN_FIELDS)
        disputes += int(disagree)
        if disagree and not adjudicators:
            pending.append(pair_id)
        elif not disagree and adjudicators:
            raise ValueError("third rater is only for a disagreement")
        else:
            complete += 1
    return {"status": "COMPLETE" if not pending else "NOT_RUN", "complete_pairs": complete,
            "disputed_pairs": disputes, "pending_pair_ids": pending,
            "raw_reviews": [grouped[p][r] for p in sorted(grouped) for r in sorted(grouped[p])]}
