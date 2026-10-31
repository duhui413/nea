"""Independent mechanism measurements over explicit, auditable evidence.

This module never infers that a Chinese definition is correct from a keyword.
The evidence producer must supply a validated semantic judgement and the exact
answer span.  Missing independent question gold or unvalidated evidence is a
missing result, not a zero or an optimistic score. These measurements remain
separate from the method-internal signals.
"""

from __future__ import annotations

import hashlib
import json
import math
from statistics import fmean
from typing import Any, Mapping, Sequence


METRIC_VERSION = "ca-v2-mechanism-1"
SLOT_ORDER = ("obj", "pre", "bridge", "target", "example")
CHINESE_SLOTS = {
    "任务": "obj", "问题": "obj", "前置": "pre", "先修": "pre",
    "桥接": "bridge", "承接": "bridge", "目标": "target", "示例": "example",
}
VALID_EVIDENCE_SOURCES = frozenset({"human_gold", "validated_detector"})
MIN_LEVEL_COVERAGE = 0.50


def canonical_hash(value: Any) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _na(reason: str, *, coverage: float = 0.0, unknown_count: int = 0) -> dict[str, Any]:
    return {"value": None, "applicable": False, "coverage": coverage,
            "unknown_count": unknown_count, "reason": reason}


def _invalid(missing: Sequence[str], calibration_hash: str) -> dict[str, Any]:
    return {
        "metric_version": METRIC_VERSION, "status": "NOT_VALIDATED",
        "missing": sorted(set(missing)), "calibration_hash": calibration_hash,
        "cer": _na("missing_validated_gold"), "rcc": _na("missing_validated_gold"),
        "lts": _na("missing_validated_gold"),
    }


def _slot(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    normalized = CHINESE_SLOTS.get(text, text)
    return normalized if normalized in SLOT_ORDER else None


def _unit_index(value: Any, unit_count: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < unit_count:
        raise ValueError("support unit_index is outside the answer units")
    return value


def _bounded(value: Any, name: str) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return number


def _normalise(value: float, row: Mapping[str, Any]) -> float:
    low, high = float(row["minimum"]), float(row["maximum"])
    if not math.isfinite(low) or not math.isfinite(high) or high < low:
        raise ValueError("normalizer maximum is below minimum")
    return 0.0 if high - low <= 1e-8 else max(0.0, min(1.0, (value - low) / (high - low)))


def score_metrics(
    question_gold: Mapping[str, Any] | None,
    answer_evidence: Mapping[str, Any] | None,
    *,
    normalizers: Mapping[str, Mapping[str, Any]] | None = None,
    calibration_hash: str = "",
) -> dict[str, Any]:
    """Score one answer without deriving the denominator from answer or arm.

    ``question_gold`` is frozen before viewing candidates. ``answer_evidence``
    supplies explicit support judgements and adjacent-pair evidence.  A plain
    concept mention, a definition keyword, or a low-level model similarity is
    never sufficient to mark the meaning and its role correct.
    """
    missing: list[str] = []
    if question_gold is None or question_gold.get("status") != "ADJUDICATED":
        missing.append("independent_question_gold")
    if answer_evidence is None or answer_evidence.get("status") not in {"ADJUDICATED", "PREDICTED_VALIDATED"}:
        missing.append("validated_answer_evidence")
    if missing:
        return _invalid(missing, calibration_hash)
    assert question_gold is not None and answer_evidence is not None
    if str(question_gold.get("qid", "")) != str(answer_evidence.get("qid", "")):
        raise ValueError("question and answer qid mismatch")
    if not str(answer_evidence.get("answer_hash", "")):
        raise ValueError("answer evidence needs a stable answer_hash")
    source = str(answer_evidence.get("evidence_source", ""))
    if source not in VALID_EVIDENCE_SOURCES:
        return _invalid(["validated_evidence_source"], calibration_hash)
    if source == "validated_detector" and not answer_evidence.get("validity_hash"):
        return _invalid(["detector_validity_hash"], calibration_hash)
    units = list(answer_evidence.get("units") or [])
    if not all(isinstance(unit, str) and unit.strip() for unit in units):
        raise ValueError("answer units must be nonempty strings")
    required = list(question_gold.get("necessary_prerequisite_ids") or [])
    if not all(isinstance(concept, str) and concept for concept in required) or len(required) != len(set(required)):
        raise ValueError("necessary_prerequisite_ids must be unique concept IDs")
    target = question_gold.get("target_concept_id")
    if not isinstance(target, str) or not target:
        raise ValueError("independent question gold needs target_concept_id")
    if target in required:
        raise ValueError("the target cannot be its own necessary prerequisite")
    given = dict(question_gold.get("given_evidence") or {})
    if not set(given).issubset(required):
        raise ValueError("given evidence must refer to a necessary prerequisite")
    question_text = str(question_gold.get("question", ""))
    for row in given.values():
        span = str(row.get("question_span", "")) if isinstance(row, Mapping) else ""
        if not span or not question_text or span not in question_text:
            raise ValueError("given evidence must quote the independent question")
    support = dict(answer_evidence.get("support") or {})
    if not set(support).issubset(required):
        raise ValueError("answer support contains a non-necessary concept")

    concept_rows: list[dict[str, Any]] = []
    for concept in required:
        given_row = given.get(concept)
        given_valid = (
            isinstance(given_row, Mapping)
            and bool(str(given_row.get("question_span", "")).strip())
            and given_row.get("correct_meaning") is True
            and given_row.get("question_role") is True
        )
        rows = support.get(concept) or []
        if not isinstance(rows, list):
            raise ValueError("support for each concept must be a list")
        accepted: list[int] = []
        mentioned: list[int] = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("support row must be an object")
            index = _unit_index(row.get("unit_index"), len(units))
            span = str(row.get("answer_span", "")).strip()
            if not span or span not in units[index]:
                raise ValueError("support span must occur in its cited unit")
            mentioned.append(index)
            if row.get("correct_meaning") is True and row.get("question_role") is True:
                accepted.append(index)
        concept_rows.append({
            "concept_id": concept, "given": given_valid,
            "mention_units": sorted(set(mentioned)),
            "supported_units": sorted(set(accepted)),
            "value": 1.0 if given_valid or accepted else 0.0,
        })
    cer = _na("no_necessary_prerequisites", coverage=1.0) if not required else {
        "value": round(fmean(row["value"] for row in concept_rows), 6),
        "applicable": True, "coverage": 1.0, "unknown_count": 0,
        "denominator": len(required), "supported_count": sum(row["value"] == 1.0 for row in concept_rows),
        "concepts": concept_rows,
    }

    pair_input = list(answer_evidence.get("pairs") or [])
    if len(units) >= 2 and len(pair_input) != len(units) - 1:
        raise ValueError("each adjacent unit pair needs one explicit evidence row")
    if len(units) < 2 and pair_input:
        raise ValueError("single-unit answers cannot carry adjacent pairs")
    pairs: list[dict[str, Any]] = []
    for index, pair in enumerate(pair_input):
        if not isinstance(pair, Mapping) or pair.get("left_index") != index or pair.get("right_index") != index + 1:
            raise ValueError("pair indices must identify each adjacent pair in order")
        similarity = _bounded(pair.get("semantic_support"), "semantic_support")
        left_slot, right_slot = _slot(pair.get("left_slot")), _slot(pair.get("right_slot"))
        slot_known = left_slot is not None and right_slot is not None
        slot_forward = slot_known and 0 <= SLOT_ORDER.index(right_slot) - SLOT_ORDER.index(left_slot) <= 1
        # A monotone slot label alone is not a valid reasoning bridge.
        bridge = pair.get("transition_supported") is True
        rcc_pair = 0.5 * similarity + 0.5 * float(slot_forward and bridge)
        left_level, right_level = pair.get("left_level"), pair.get("right_level")
        levels_known = type(left_level) is int and type(right_level) is int
        hierarchy_penalty = None
        if levels_known:
            delta = right_level - left_level
            hierarchy_penalty = max(0.0, min(1.0, max(0, delta - 1) + max(0, -delta - 1)))
        pairs.append({
            "left_index": index, "right_index": index + 1,
            "semantic_support": round(similarity, 6),
            "left_slot": left_slot, "right_slot": right_slot,
            "slot_known": slot_known, "transition_supported": bridge,
            "rcc_pair": round(rcc_pair, 6), "left_level": left_level,
            "right_level": right_level, "hierarchy_penalty": hierarchy_penalty,
        })
    slot_coverage = sum(row["slot_known"] for row in pairs) / len(pairs) if pairs else 0.0
    rcc = _na("fewer_than_two_units") if not pairs else {
        "value": round(fmean(row["rcc_pair"] for row in pairs), 6) if slot_coverage >= MIN_LEVEL_COVERAGE else None,
        "applicable": slot_coverage >= MIN_LEVEL_COVERAGE,
        "coverage": round(slot_coverage, 6),
        "unknown_count": sum(not row["slot_known"] for row in pairs),
        "reason": None if slot_coverage >= MIN_LEVEL_COVERAGE else "insufficient_slot_coverage",
        "pairs": pairs,
    }
    linked = [float(row["hierarchy_penalty"]) for row in pairs if row["hierarchy_penalty"] is not None]
    unknown_count = len(pairs) - len(linked)
    if not pairs:
        lts = _na("fewer_than_two_units")
    elif not linked:
        lts = _na("no_linked_level_pairs", unknown_count=unknown_count)
    else:
        semantic = [row["semantic_support"] for row in pairs]
        avg = fmean(semantic)
        raw = {
            "variance": fmean((value - avg) ** 2 for value in semantic),
            "discontinuity": fmean(1.0 - value for value in semantic),
            "hierarchy": fmean(linked),
        }
        level_coverage = len(linked) / len(pairs)
        low_h = sum(linked) / len(pairs)
        high_h = (sum(linked) + unknown_count) / len(pairs)
        if normalizers is None:
            value = None
            reason = "evaluation_scale_missing"
            low_value = high_value = None
        else:
            if set(raw) - set(normalizers):
                raise ValueError("evaluation scale lacks an LTS component")
            normal_v = _normalise(raw["variance"], normalizers["variance"])
            normal_n = _normalise(raw["discontinuity"], normalizers["discontinuity"])
            normal_h = _normalise(raw["hierarchy"], normalizers["hierarchy"])
            low_value = round((normal_v + normal_n + _normalise(low_h, normalizers["hierarchy"])) / 3, 6)
            high_value = round((normal_v + normal_n + _normalise(high_h, normalizers["hierarchy"])) / 3, 6)
            value = round((normal_v + normal_n + normal_h) / 3, 6) if level_coverage >= MIN_LEVEL_COVERAGE else None
            reason = None if value is not None else "insufficient_level_coverage"
        lts = {
            "value": value, "applicable": value is not None,
            "coverage": round(level_coverage, 6),
            "unknown_count": unknown_count, "raw_components": {key: round(val, 6) for key, val in raw.items()},
            "unknown_as_zero_hierarchy": round(low_h, 6),
            "unknown_as_one_hierarchy": round(high_h, 6),
            "value_if_unknown_zero": low_value, "value_if_unknown_one": high_value,
            "reason": reason,
        }
    return {
        "metric_version": METRIC_VERSION, "status": "VALIDATED", "missing": [],
        "qid": str(question_gold["qid"]), "answer_hash": str(answer_evidence.get("answer_hash", "")),
        "necessary_set_hash": canonical_hash(sorted(required)),
        "calibration_hash": calibration_hash, "evidence_source": source,
        "detector_validity_hash": str(answer_evidence.get("validity_hash", "")),
        "cer": cer, "rcc": rcc, "lts": lts,
    }
