"""Blind measurements for the formal eight-arm protocol; no asset or API I/O."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
from typing import Any, Mapping, Sequence

from nea.method.templates import array, obj, string
from nea.providers import config
from nea.runtime.records import answer_sha256
from nea.runtime.structured import validate_json_schema

PROTOCOL = "nea-formal8-v1"
SELECTION_SALT = "formal8|20261006|"
CONCEPT_STATES = ("absent", "mentioned", "explained")
DEFECT_TYPES = ("term_gap", "step_gap", "logical_jump", "term_excess", "step_excess")
AUDIT_ARMS = ("B", "L", "SR", "SRL", "FB", "F")
ROLE_STAGE = {"formal_concept_measurement": "measure", "formal_defect_audit": "audit",
              "formal_perturbation_rewrite": "perturb"}


def role_schema(role: str, payload: Mapping[str, Any] | None = None) -> dict | None:
    """Static schemas; per-instance identity checks stay in local validation."""
    if role == "formal_concept_measurement":
        return obj({"concepts": array(obj({"concept_id": string(),
            "state": string(*CONCEPT_STATES), "reason": string()}))})
    if role == "formal_defect_audit":
        defect = obj({"type": string(*DEFECT_TYPES), "span": {"type": "string"},
                      "question_anchor": {"type": "string"}, "reason": string()})
        return {"anyOf": [obj({"status": string("issues"), "defects": array(defect, minItems=1)}),
                          obj({"status": string("no_issue", "uncertain"),
                               "defects": array(defect, maxItems=0)})]}
    if role == "formal_perturbation_rewrite":
        return obj({"applicable": {"type": "boolean"}, "before_span": {"type": "string"},
                    "after_span": {"type": "string"}, "term": {"type": "string"},
                    "reason": string()})
    return None


def cross_family_teacher(generator: str) -> str:
    models = config.experiment_role_models_for(generator)
    family = config.experiment_provider_name_for_model(models["student1"])
    return next(role for role in ("teacher1", "teacher2")
                if config.experiment_provider_name_for_model(models[role]) != family)


def blind_concept_payload(question: str, answer: str, labels: Mapping[str, str],
                          concepts: Mapping[str, Mapping[str, Any]]) -> dict:
    """Pass reference content, never gold label classes, arms, or method state."""
    return {"question": question, "candidate_answer": answer,
            "concepts": [{"concept_id": key, "name": str(concepts[key]["name"]),
                          "reference": str(concepts[key].get("explain", concepts[key].get("content", "")))}
                         for key in sorted(labels)]}


def sanitize_judge_localization(parsed: Mapping[str, Any], answer: str, question: str,
                                allowed_evidence_ids: Sequence[str] = ()) -> dict:
    """Preserve all valid scores/reasons; erase only unsupported location fields."""
    result = deepcopy(dict(parsed))
    rows = list(result.get("findings", ())) + list(result.get("reasons", {}).values())
    counts = {"span": 0, "question_anchor": 0, "evidence_ids": 0, "unlocated_rows": 0}
    for row in rows:
        for field, source in (("span", answer), ("question_anchor", question)):
            value = row.get(field, "")
            if value and (not value.strip() or value not in source):
                row[field] = ""
                counts[field] += 1
        evidence = row.get("evidence_ids", [])
        known = [key for key in evidence if key in allowed_evidence_ids]
        counts["evidence_ids"] += len(evidence) - len(known)
        row["evidence_ids"] = known
        counts["unlocated_rows"] += not bool(row.get("span") or row.get("question_anchor"))
    result["localization"] = {"policy": "soft-exact-v1", "invalid_fields": sum(counts[key]
        for key in ("span", "question_anchor", "evidence_ids")), "counts": counts,
        "checked_rows": len(rows)}
    return result


def validate_formal_payload(job: Any, parsed: Mapping[str, Any]) -> dict:
    """Called after JSON Schema validation by formal_transport only."""
    role = job.arg("schema_name")
    result = deepcopy(dict(parsed))
    if role == "cove_verification_questions":
        ids = [row["question_id"] for row in result["verification_questions"]]
        if len(ids) != len(set(ids)):
            raise ValueError("verification question IDs must be unique")
    if role == "cove_verification_answer":
        ids = [row["question_id"] for row in result["verification_answers"]]
        if len(ids) != len(set(ids)) or set(ids) != set(job.arg("verification_question_ids", ())):
            raise ValueError("verification answers must cover exactly the requested question IDs")
    if role == "answer_quality_evaluation":
        return sanitize_judge_localization(result, job.arg("locatable_answer", ""),
            job.arg("locatable_question", ""), job.arg("allowed_evidence_ids", ()))
    if role == "formal_concept_measurement":
        ids = [row["concept_id"] for row in result["concepts"]]
        if len(ids) != len(set(ids)) or set(ids) != set(job.arg("measurement_concept_ids", ())):
            raise ValueError("measurement must classify exactly the supplied concept IDs")
        result["concepts"].sort(key=lambda row: row["concept_id"])
    if role == "formal_defect_audit":
        # Invalid locations remain observable and count as localization misses.
        answer, question = job.arg("locatable_answer", ""), job.arg("locatable_question", "")
        invalid = 0
        for row in result["defects"]:
            for key, source in (("span", answer), ("question_anchor", question)):
                if row[key] and (not row[key].strip() or row[key] not in source):
                    row[key] = ""
                    invalid += 1
        result["localization"] = {"invalid_fields": invalid}
    if role == "formal_perturbation_rewrite":
        from nea.pipeline.formal_perturbation import apply_model_edit
        apply_model_edit(job.arg("locatable_answer", ""), job.arg("perturbation_kind"), result,
                         key=job.qid, duplicate=job.arg("duplicate_paragraph", ""))
    return result


def concept_metrics(labels: Mapping[str, str], parsed: Mapping[str, Any]) -> dict:
    schema = role_schema("formal_concept_measurement")
    validate_json_schema(parsed, schema)
    classified = {item["concept_id"]: item["state"] for item in parsed["concepts"]}
    if len(classified) != len(parsed["concepts"]) or set(classified) != set(labels):
        raise ValueError("measurement classifications differ from the reference concept set")
    result = {}
    for metric, kinds in (("NCov", {"gap"}), ("ROE", {"given", "assumed"})):
        selected = [key for key, label in labels.items() if label in kinds]
        numerator = sum(classified[key] == "explained" for key in selected)
        result[metric] = {"value": numerator / len(selected) if selected else None,
                          "numerator": numerator, "denominator": len(selected),
                          "applicable": bool(selected), "reason": None if selected else "empty_denominator"}
    return result


def summarize_concept_metrics(rows: Sequence[Mapping[str, Any]]) -> dict:
    """Question counts include NA questions, without turning NA into zero."""
    selected = [row for row in rows if row.get("primary", True)]
    result = {"question_count": len({row["qid"] for row in selected})}
    for metric in ("NCov", "ROE"):
        valid = [row for row in selected if row.get("status") == "ok" and row.get(metric)]
        eligible = [row for row in valid if row[metric]["value"] is not None]
        na = [row for row in valid if row[metric]["value"] is None]
        result[metric] = {"value": (sum(row[metric]["value"] for row in eligible) / len(eligible)
                                     if eligible else None), "positions": len(eligible),
                          "question_count": len({row["qid"] for row in eligible}),
                          "na_positions": len(na), "na_question_count": len({row["qid"] for row in na}),
                          "missing_positions": len(selected) - len(valid)}
    return result


def classification_agreement(rows: Sequence[Mapping[str, Any]]) -> dict:
    """Unweighted Cohen kappa and raw three-class agreement, unique answers only."""
    groups = {}
    for row in rows:
        if row.get("status") == "ok" and row.get("parsed"):
            key = (row["qid"], row["generator"], row["answer_sha256"])
            groups.setdefault(key, {})[row["teacher"]] = {
                item["concept_id"]: item["state"] for item in row["parsed"]["concepts"]}
    pairs = []
    for answers in groups.values():
        if set(answers) == {"teacher1", "teacher2"}:
            left, right = answers["teacher1"], answers["teacher2"]
            pairs.extend((left[key], right[key]) for key in sorted(left.keys() & right.keys()))
    matrix = {left: {right: 0 for right in CONCEPT_STATES} for left in CONCEPT_STATES}
    for left, right in pairs:
        matrix[left][right] += 1
    n = len(pairs)
    observed = sum(left == right for left, right in pairs) / n if n else None
    lcount, rcount = Counter(left for left, _ in pairs), Counter(right for _, right in pairs)
    expected = sum(lcount[state] * rcount[state] for state in CONCEPT_STATES) / n ** 2 if n else None
    return {"concept_pairs": n, "agreement": observed,
            "cohen_kappa": ((observed - expected) / (1 - expected)
                            if n and expected != 1 else None), "confusion_matrix": matrix,
            "complete_answer_pairs": sum(set(item) == {"teacher1", "teacher2"} for item in groups.values())}


def selection_hash(qid: str) -> str:
    return hashlib.sha256((SELECTION_SALT + qid).encode("utf8")).hexdigest()


def select_t90(rows: Sequence[Mapping[str, Any]], per_style: int = 30) -> list[str]:
    selected = []
    unique = {row["qid"]: row for row in rows}
    for style in ("direct", "prerequisite", "application"):
        group = [row for row in unique.values() if row.get("style") == style]
        selected.extend(row["qid"] for row in sorted(group,
            key=lambda row: (selection_hash(row["qid"]), row["qid"]))[:per_style])
    return selected


def select_perturbation_sources(rows: Sequence[Mapping[str, Any]], limit: int = 200) -> list[dict]:
    """Require recorded no_issue diagnoses; ties use full answer identity."""
    unique = {}
    for row in rows:
        diagnosis = row.get("diagnosis") or {}
        if diagnosis.get("status") != "no_issue" or diagnosis.get("actions", []):
            continue
        if row.get("split", "test") != "test" or not row.get("answer"):
            continue
        key = (row["qid"], row["generator"], row["base"], answer_sha256(row["answer"]))
        unique.setdefault(key, dict(row))
    ordered = sorted(unique, key=lambda key: (selection_hash(key[0]), *key))
    return [unique[key] for key in ordered[:limit]]
